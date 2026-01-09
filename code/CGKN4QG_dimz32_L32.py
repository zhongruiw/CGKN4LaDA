'''
UNet + ResBlock for AE (Deep res block: proj uses Resblock); dimz32 with low-rank g2
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector
import time
from torchviz import make_dot

# device = "cuda:1"
device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_long.npz")

pos = QG_Data["xy_obs"]
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1)
psi = QG_Data["psi_noisy"]

# u1 = torch.tensor(pos_unit, dtype=torch.float).view(pos.shape[0], -1)
u1 = torch.tensor(pos_unit, dtype=torch.float) # shape (Nt, L, 4), keep tracers parallel
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 128, 128, 2)

# Train / Test
Ntrain = 80000
Nval = 10000
Ntest = 10000
L_total = 1024 # total number of tracers in training dataset
L = 32 # number of tracers used in data assimilation

train_u1 = u1[:Ntrain]
train_u2 = u2[:Ntrain]
val_u1 = u1[Ntrain:Ntrain+Nval, :L]
val_u2 = u2[Ntrain:Ntrain+Nval]
test_u1 = u1[Ntrain+Nval:Ntrain+Nval+Ntest, :L]
test_u2 = u2[Ntrain+Nval:Ntrain+Nval+Ntest]

############################################################
################# CGKN: AutoEncoder + CGN  #################
############################################################
def unit2xy(xy_unit):
    cos0 = xy_unit[..., 0]
    sin0 = xy_unit[..., 1]
    cos1 = xy_unit[..., 2]
    sin1 = xy_unit[..., 3]
    x = torch.atan2(sin0, cos0) # range [-pi, pi)
    y = torch.atan2(sin1, cos1) # range [-pi, pi)

    return x, y

class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # manual padding
            bias=bias
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        H, W = x.shape[-2:]
        pad_h = self._compute_padding(H, self.kernel_size[0], self.stride[0])
        pad_w = self._compute_padding(W, self.kernel_size[1], self.stride[1])

        # Apply circular padding manually
        x = nnF.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='circular')
        x = self.conv(x)
        return x

    def _compute_padding(self, size, k, s):
        if size % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (size % s), 0)
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        return (pad_top, pad_bottom)

class CircularConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # we'll manually manage padding
            bias=bias
        )
        # compute circular padding width
        self.pad_w = int(0.5 + (self.kernel_size[1] - 1.) / (2. * self.stride[1]))
        self.pad_h = int(0.5 + (self.kernel_size[0] - 1.) / (2. * self.stride[0]))

    def forward(self, x):
        # Apply circular padding manually before transposed convolution
        x_padded = nnF.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='circular')
        out = self.deconv(x_padded)

        # Crop the output (equivalent to cropping `crop=stride * pad`)
        crop_h = (self.stride[0]+1) * self.pad_h
        crop_w = (self.stride[1]+1) * self.pad_w
        out = out[:, :, crop_h:out.shape[2]-crop_h, crop_w:out.shape[3]-crop_w]
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(32, out_channels)
        self.conv = CircularConv2d(in_channels, out_channels, 3, 1, 1)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(groups, out_channels)
        self.skip = (in_channels == out_channels)
        if not self.skip:
            self.proj = CircularConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        if self.skip:
            return x + y
        else:
            return self.proj(x) + y

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(32, out_channels)
        self.down = nn.Sequential(
            CircularConv2d(in_channels, out_channels, 4, 2, 1), # downsample via stride=2
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            )

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="bilinear"):
        super().__init__()
        groups = min(32, out_channels)
        self.up = nn.Sequential(
            CircularConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            )

    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    """
    UNet-like encoder with residual blocks.
    in:  (B, 2, H, W)
    out: (B, C_latent, H/4, W/4)  # for depth=2
    """
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=16, depth=2):
        super().__init__()
        self.pre = nn.Sequential(
            ResBlock(in_channels, hidden_channels),
            ResBlock(hidden_channels, hidden_channels),
            ResBlock(hidden_channels, hidden_channels),
            )
        ch_in = hidden_channels
        downs = []
        for i in range(depth):
            ch_out = ch_in * 2  # gradually increase channels
            downs += [
                ResBlock(ch_in, ch_out),
                ResBlock(ch_out, ch_out),
                DownBlock(ch_out, ch_out),
                ResBlock(ch_out, ch_out),
            ]
            ch_in = ch_out
        self.down = nn.Sequential(*downs)
        # self.proj = CircularConv2d(ch_in, out_channels, kernel_size=1) if ch_in != out_channels else nn.Identity()
        self.proj = ResBlock(ch_in, out_channels)
        self.post = nn.Sequential(
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
            # ResBlock(out_channels, out_channels),
            )

    def forward(self, u):  # (B, H, W, 2)
        x = u.permute(0, 3, 1, 2).contiguous()  # -> (B, 2, H, W)
        z = self.pre(x)
        z = self.down(z)
        z = self.proj(z)
        z = self.post(z)
        return z

class Decoder(nn.Module):
    """
    Mirrors Encoder. Returns to (B, H, W, 2).
    """
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=16, depth=2):
        super().__init__()
        self.pre = nn.Sequential(
            ResBlock(in_channels, in_channels),
            ResBlock(in_channels, in_channels),
            # ResBlock(in_channels, in_channels),
            )
        ch_out_proj = hidden_channels * 2**depth
        # self.proj = CircularConv2d(in_channels, ch_out_proj, kernel_size=1) if ch_out_proj != in_channels else nn.Identity()
        self.proj = ResBlock(in_channels, ch_out_proj)
        ups = []
        ch_in = ch_out_proj
        for i in range(depth):
            ch_out = ch_in // 2
            ups += [
                ResBlock(ch_in, ch_in),
                UpBlock(ch_in, ch_in),
                ResBlock(ch_in, ch_in),
                ResBlock(ch_in, ch_out),
            ]
            ch_in = ch_out
        self.up = nn.Sequential(*ups) if ups else nn.Identity()
        self.post = nn.Sequential(
            ResBlock(hidden_channels, hidden_channels),
            ResBlock(hidden_channels, hidden_channels),
            ResBlock(hidden_channels, out_channels),
            )

    def forward(self, z):
        x = self.pre(z)
        x = self.proj(x)
        x = self.up(x)
        x = self.post(x)
        u = x.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, 2)
        return u

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=16, depth=2):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, hidden_channels, depth)
        self.decoder = Decoder(out_channels, in_channels, hidden_channels, depth)

# class Encoder(nn.Module):
#     """
#     UNet-like encoder with residual blocks.
#     in:  (B, 2, H, W)
#     out: (B, C_latent, H/4, W/4)  # for depth=2
#     """
#     def __init__(self, in_channels=2, base=32, depth=2, latent_channels=2, res_per_stage=1, gn_groups=8):
#         super().__init__()
#         chs = [base * (2**i) for i in range(depth+1)]  # e.g., [32, 64, 128] if depth=2
#         self.enc_stages = nn.ModuleList()

#         self.inc = CircularConv2d(in_channels, chs[0], 3, 1, 1)
#         enc_stages = []
#         for d in range(depth):                   # stages 0..depth-1 with downsamples
#             blocks = [ResBlock(chs[d], chs[d], groups=gn_groups) for _ in range(res_per_stage)]
#             blocks += [DownBlock(chs[d], chs[d+1])]
#             self.enc_stages.append(nn.Sequential(*blocks))

#         self.bottle_enc = nn.Sequential(
#             ResBlock(chs[-1], chs[-1], groups=gn_groups),
#             CircularConv2d(chs[-1], latent_channels, 1, 1, 0)  # channel squeeze to latent
#         )

#     def forward(self, u):  # (B, H, W, 2)
#         x = u.permute(0, 3, 1, 2).contiguous()  # -> (B, 2, H, W)
#         x = self.inc(x)
#         for stage in self.enc_stages:
#             for m in stage[:-1]:
#                 x = m(x)
#             x = stage[-1](x)      # downsample
#         z = self.bottle_enc(x)
#         return z

# class Decoder(nn.Module):
#     """
#     Mirrors Encoder. Returns to (B, H, W, 2).
#     """
#     def __init__(self, out_channels=2, base=32, depth=2, latent_channels=2, res_per_stage=1, up_mode="bilinear", gn_groups=8):
#         super().__init__()
#         chs = [base * (2**i) for i in range(depth+1)]  # [32, 64, 128] if depth=2
#         self.dec_stages = nn.ModuleList()

#         self.bottle_dec = nn.Sequential(
#             CircularConv2d(latent_channels, chs[-1], 1, 1, 0),
#             ResBlock(chs[-1], chs[-1], groups=gn_groups))

#         for d in reversed(range(depth)):
#             blocks = [UpBlock(chs[d+1], chs[d], mode=up_mode)]
#             blocks += [ResBlock(chs[d], chs[d], groups=gn_groups) for _ in range(res_per_stage)]
#             self.dec_stages.append(nn.Sequential(*blocks))

#         # self.outc = nn.Sequential(
#         #     CircularConv2d(chs[0], chs[0], 3, 1, 1),
#         #     nn.SiLU(),
#         #     CircularConv2d(chs[0], out_channels, 1, 1, 0)
#         # )
#         self.outc = CircularConv2d(chs[0], out_channels, 3, 1, 1)

#     def forward(self, z):
#         x = self.bottle_dec(z)
#         for stage in self.dec_stages:
#             x = stage(x)
#         u = self.outc(x)
#         return u.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, 2)

# class AutoEncoder(nn.Module):
#     """
#     Periodic UNet-like AutoEncoder.
#     - Input:  (B, H, W, C_in)
#     - Output: (B, H, W, C_out)  (default C_out = C_in)
#     """
#     def __init__(self,
#                  in_channels=2,
#                  out_channels=2,
#                  base=16,
#                  depth=2,
#                  latent_channels=2,
#                  res_per_stage=1,
#                  up_mode="bilinear",
#                  gn_groups=8,
#                  ):
#         super().__init__()
#         self.encoder = Encoder(in_channels, base, depth, latent_channels, res_per_stage, gn_groups)
#         self.decoder = Decoder(out_channels, base, depth, latent_channels, res_per_stage, up_mode, gn_groups)
#         self.apply(self._init_weights)
#         # make the very last conv small to avoid early overshoot
#         for m in self.decoder.outc.modules():
#             if isinstance(m, nn.Conv2d):
#                 with torch.no_grad():
#                     m.weight.mul_(0.01)

#     @staticmethod
#     def _init_weights(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             if m.bias is not None: nn.init.zeros_(m.bias)
#         if isinstance(m, nn.ConvTranspose2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             if m.bias is not None: nn.init.zeros_(m.bias)


    # def forward(self, u):  # (B, H, W, C_in)
    #     z = self.encoder(u)
    #     recon = self.decoder(z)
    #     return recon, z 

# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.enc = nn.Sequential(
#             CircularConv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
#             CircularConv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
#             CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConv2d(32, 2, kernel_size=3, stride=1, padding=1)
#         )

#     def forward(self, u):                           # u:(B, H, W, 2)
#         u = u.permute(0, 3, 1, 2)                   # → (B, 2, H, W)
#         out = self.enc(u)                           # → (B, 2, d1, d2)
#         # print(out.shape)
#         return out#.squeeze(1)                      # → (B, 2, d1, d2)

# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dec = nn.Sequential(
#             CircularConvTranspose2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
#             CircularConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1)
#         )

#     def forward(self, z):                  # z: (B, 2, d1, d2)
#         u = self.dec(z)                    # (B, 2, 64, 64)
#         return u.permute(0, 2, 3, 1)       # (B, 64, 64, 2)
        
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6, rank_g2=64):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.f1_size = (dim_u1, 1)
        self.g1_size = (dim_u1, dim_z)
        self.f2_size = (dim_z, 1)
        self.g2_size = (dim_z, dim_z)
        self.use_pos_encoding = use_pos_encoding
        self.num_frequencies = num_frequencies
        # Homogeneous Tracers
        in_dim = 2  # (x1, x2)
        out_dim_f1 = 2*2
        out_dim_g1 = 2*2*dim_z
        self.in_dim = in_dim
        # self.out_dim = in_dim*2 + in_dim*2*dim_z
        if use_pos_encoding:
            in_dim = 2 + 2 * 2 * num_frequencies  # sin/cos for each x1 and x2
        # self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(), 
        #                          nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
        #                          nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
        #                          nn.Linear(32, 16), nn.LayerNorm(16), nn.SiLU(),
        #                          nn.Linear(16, self.out_dim))

        self.f1_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, out_dim_f1)
        )

        self.g1_net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, out_dim_g1)
        )

        self.f2_param = nn.Parameter(1/dim_z**0.5 * torch.rand(dim_z, 1))
        # self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))
        # g2 params
        self.U_raw = nn.Parameter(torch.randn(dim_z, rank_g2) / (dim_z ** 0.5))
        self.V_raw = nn.Parameter(torch.randn(dim_z, rank_g2) / (dim_z ** 0.5))
        self.s_raw = nn.Parameter(torch.full((rank_g2,), float(1.0)))
        self.diag = nn.Parameter(torch.ones(dim_z))  # learnable residual D

    def forward(self, x): # x shape(Nt, L, 2)
        Nt, L = x.shape[:2]
        if self.use_pos_encoding:
            x = self.positional_encoding(x.view(-1,2)) # (Nt * L, in_dim)
            x = x.view(Nt, L, -1)
        # out = self.net(x) # (Nt, L, out_dim)
        # f1 = out[:, :, :self.in_dim*2].reshape(Nt, *self.f1_size) # (Nt, L*4, 1)
        # g1 = out[:, :, self.in_dim*2:].reshape(Nt, *self.g1_size) # (Nt, L*4, dim_z)
        f1 = self.f1_net(x).view(Nt, *self.f1_size)   # (Nt, L*4, 1)
        g1 = self.g1_net(x).view(Nt, *self.g1_size)   # (Nt, L*4, dim_z)
        g1 = nnF.softmax(g1 / 1, dim=-1) # sharp attention 
        f2 = self.f2_param.expand(Nt, -1, -1)
        # g2 = self.g2_param.repeat(Nt, 1, 1)

        # Project to Stiefel (orthonormal columns)
        U, _ = torch.linalg.qr(self.U_raw, mode='reduced')       # (d, r), U^T U = I
        V, _ = torch.linalg.qr(self.V_raw, mode='reduced')       # (d, r), V^T V = I
        # Nonnegative “singular values”
        sigma = nnF.softplus(self.s_raw) + 1e-6 # (r,)
        g2 = ((U * sigma) @ V.T + torch.diag(self.diag)).expand(Nt, -1, -1)

        return [f1, g1, f2, g2]

    def positional_encoding(self, x):
        """
        Input: x of shape (B, 2)
        Output: encoded x of shape (B, 2+ 4*num_frequencies)
        """
        freqs = 2 ** torch.arange(self.num_frequencies, device=x.device) * np.pi  # (F,)
        # freqs = torch.logspace(0, np.log10(np.pi * 2**self.num_frequencies), self.num_frequencies, device=x.device)
        x1 = x[:, 0:1] * freqs                 # (B, F)
        x2 = x[:, 1:2] * freqs                 # (B, F)
        encoded = torch.cat([x, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)  # shape: (B, 2+4F)
        return encoded

class CGKN(nn.Module):
    def __init__(self, autoencoder, cgn):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn

    def forward(self, u1, z): # u1: (Nt, L, 4), z: (Nt, dim_z)
        # Matrix-form Computation
        x, y = unit2xy(u1)
        pos = torch.stack([x, y], dim=-1) # (Nt, L, 2)
        f1, g1, f2, g2 = self.cgn(pos) # f1: (Nt, L*4, 1), g1: (Nt, L*4, dim_z), f2: (Nt, dim_z, 1), g2: (Nt, dim_z, dim_z)
        z = z.unsqueeze(-1) # (Nt, dim_z, 1)
        u1_pred = f1 + g1@z # (Nt, L*4, 1)
        z_pred = f2 + g2@z  # (Nt, dim_z, 1)
        return u1_pred.view(*u1.shape), z_pred.squeeze(-1)


########################################################
################# Train cgkn (Stage1)  #################
########################################################
dim_u1 = L*4
dim_u2 = 64*64*2
z_h, z_w = (32, 32)
latent_channels = 2
dim_z = z_h*z_w*latent_channels

# Stage1: Train cgkn with loss_forecast + loss_ae + loss_forecast_z
epochs = 500
train_batch_size = 200
val_batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u1[:-1], train_u2[:-1], train_u1[1:], train_u2[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
val_tensor = torch.utils.data.TensorDataset(val_u1[:-1], val_u2[:-1], val_u1[1:], val_u2[1:])
val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=val_batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
loss_history = {
    "train_forecast_u1": [],
    "train_forecast_u2": [],
    "train_ae": [],
    "train_forecast_z": [],
    "train_physics": [],
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    "val_ae": [],
    "val_forecast_z": [],
    }
best_val_loss = float('inf')

autoencoder = AutoEncoder(in_channels=2, out_channels=latent_channels, hidden_channels=32, depth=1).to(device)
cgn = CGN(dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6).to(device)
cgkn = CGKN(autoencoder, cgn).to(device)
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)

# CGKN: Number of Parameters
print(f'f1 #parameters:      {parameters_to_vector(cgkn.cgn.f1_net.parameters()).numel():,}')
print(f'g1 #parameters:      {parameters_to_vector(cgkn.cgn.g1_net.parameters()).numel():,}')
print(f"f2 #parameters: {parameters_to_vector([cgkn.cgn.f2_param]).numel():,}")
# print(f"g2 #parameters: {parameters_to_vector([cgkn.cgn.g2_param]).numel():,}")
print(f"g2 #parameters: {parameters_to_vector([cgkn.cgn.U_raw, cgkn.cgn.V_raw, cgkn.cgn.s_raw, cgkn.cgn.diag]).numel():,}")

# CGKN: Number of Parameters
cgn_params = parameters_to_vector(cgkn.cgn.parameters()).numel()
encoder_params = parameters_to_vector(cgkn.autoencoder.encoder.parameters()).numel()
decoder_params = parameters_to_vector(cgkn.autoencoder.decoder.parameters()).numel()
total_params = cgn_params + encoder_params + decoder_params
print(f'cgn #parameters:      {cgn_params:,}')
print(f'encoder #parameters:  {encoder_params:,}')
print(f'decoder #parameters:  {decoder_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')

"""
for ep in range(1, epochs+1):
    # Training 
    cgkn.train()
    start_time = time.time()

    train_loss_forecast_u1 = 0.
    train_loss_forecast_u2 = 0.
    train_loss_ae = 0.
    train_loss_forecast_z = 0.
    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # randomly choosing tracers
        tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
        u1_initial = torch.index_select(u1_initial, dim=1, index=tracer_idx) # (batch, L, 4)
        u1_next    = torch.index_select(u1_next,    dim=1, index=tracer_idx) # (batch, L, 4)

        # AutoEncoder
        z_initial = cgkn.autoencoder.encoder(u2_initial)
        u2_initial_ae = cgkn.autoencoder.decoder(z_initial)
        loss_ae = nnF.mse_loss(u2_initial, u2_initial_ae)

        #  State Forecast
        z_initial_flat = z_initial.reshape(-1, dim_z)
        u1_pred, z_flat_pred = cgkn(u1_initial, z_initial_flat)
        z_pred = z_flat_pred.view(-1, latent_channels, z_h, z_w)
        u2_pred = cgkn.autoencoder.decoder(z_pred)
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)

        z_next = cgkn.autoencoder.encoder(u2_next)
        loss_forecast_z = nnF.mse_loss(z_next, z_pred)

        loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
        train_loss_ae += loss_ae.item()
        train_loss_forecast_z += loss_forecast_z.item()
    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    train_loss_ae /= train_num_batches
    train_loss_forecast_z /= train_num_batches
    loss_history["train_forecast_u1"].append(train_loss_forecast_u1)
    loss_history["train_forecast_u2"].append(train_loss_forecast_u2)
    loss_history["train_ae"].append(train_loss_ae)
    loss_history["train_forecast_z"].append(train_loss_forecast_z)
    end_time = time.time()

    # Validation
    if ep % 10 == 0:
        cgkn.eval()
        val_loss_u1 = 0.
        val_loss_u2 = 0.
        val_loss_z = 0.
        with torch.no_grad():
            for u1_initial, u2_initial, u1_next, u2_next in val_loader:
                u1_initial, u2_initial, u1_next, u2_next = map(lambda x: x.to(device), [u1_initial, u2_initial, u1_next, u2_next])

                # # randomly choosing tracers
                # tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
                # u1_initial = torch.index_select(u1_initial, dim=1, index=tracer_idx) # (batch, L, 4)
                # u1_next    = torch.index_select(u1_next,    dim=1, index=tracer_idx) # (batch, L, 4)

                z_initial = cgkn.autoencoder.encoder(u2_initial)
                z_initial_flat = z_initial.reshape(-1, dim_z)
                u1_pred, z_flat_pred = cgkn(u1_initial, z_initial_flat)
                z_pred = z_flat_pred.view(-1, latent_channels, z_h, z_w)
                u2_pred = cgkn.autoencoder.decoder(z_pred)
                z_next = cgkn.autoencoder.encoder(u2_next)

                val_loss_u1 += nnF.mse_loss(u1_next, u1_pred).item()
                val_loss_u2 += nnF.mse_loss(u2_next, u2_pred).item()
                val_loss_z += nnF.mse_loss(z_next, z_pred).item()
        val_loss_u1 /= len(val_loader)
        val_loss_u2 /= len(val_loader)
        val_loss_z /= len(val_loader)
        val_loss_total = val_loss_u1 + val_loss_u2 + val_loss_z
        loss_history["val_forecast_u1"].append(val_loss_u1)
        loss_history["val_forecast_u2"].append(val_loss_u2)
        loss_history["val_forecast_z"].append(val_loss_z)
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            checkpoint = {
                'epoch': ep,
                'model_state_dict': cgkn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_total
            }
            torch.save(checkpoint, r"../model/CGKN_resblock_dimz32_stage1.pt")
            status = "✅"
        else:
            status = ""
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f} "
              f"train_z: {train_loss_forecast_z:.4f}  ae: {train_loss_ae:.4f} | "
              f"val_u1: {val_loss_u1:.4f}  val_u2: {val_loss_u2:.4f}  val_z: {val_loss_z:.4f}  val_total: {val_loss_total:.4f} "
              f"{status}"
              )
    else:
        loss_history["val_forecast_u1"].append(np.nan)
        loss_history["val_forecast_u2"].append(np.nan)
        loss_history["val_forecast_z"].append(np.nan)
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f} "
              f"train_z: {train_loss_forecast_z:.4f}  ae: {train_loss_ae:.4f} "
              )

np.savez(r"../model/CGKN_resblock_dimz32_stage1_loss_history.npz", **loss_history)
"""
checkpoint = torch.load("../model/CGKN_resblock_dimz32_stage1.pt", map_location=device, weights_only=True)
cgkn.load_state_dict(checkpoint['model_state_dict'])

############################################
########### Test cgkn (stage 1) ############
############################################

# # CGKN for One-Step Prediction
# batch_size = 100
# test_u1 = test_u1.to(device)
# test_u2 = test_u2.to(device)
# test_u1_preds = []
# test_u2_preds = []
# cgkn.eval()
# with torch.no_grad():
#     for i in range(0, Ntest, batch_size):
#         test_u1_batch = test_u1[i:i+batch_size]
#         test_u2_batch = test_u2[i:i+batch_size]
#         test_z_flat_batch = cgkn.autoencoder.encoder(test_u2_batch).view(batch_size, dim_z)
#         test_u1_pred_batch, test_z_flat_pred_batch = cgkn(test_u1_batch, test_z_flat_batch)
#         test_z_pred_batch = test_z_flat_pred_batch.view(batch_size, 2, z_h, z_w)
#         test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_pred_batch)
#         test_u1_preds.append(test_u1_pred_batch)
#         test_u2_preds.append(test_u2_pred_batch)
#     test_u1_pred = torch.cat(test_u1_preds, dim=0)
#     test_u2_pred = torch.cat(test_u2_preds, dim=0)
# MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
# print("MSE1:", MSE1.item())
# MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
# print("MSE2:", MSE2.item())
# test_u1 = test_u1.to("cpu")
# test_u2 = test_u2.to("cpu")
# np.save(r"../data/CGKN_resblock_dimz32_xy_unit_OneStepPrediction_stage1.npy", test_u1_pred.to("cpu"))
# np.save(r"../data/CGKN_resblock_dimz32_psi_OneStepPrediction_stage1.npy", test_u2_pred.to("cpu"))


#################################################################
################# Noise Coefficient & CGFilter  #################
#################################################################
def compute_sigma_hat(train_u1, train_u2, cgkn, dim_u1, dim_z, 
                      batch_size=100, device="cuda"):
    Ntrain = train_u1.size(0)
    train_u1 = train_u1.to(device)
    train_u2 = train_u2.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, Ntrain, batch_size):
            batch_u1 = train_u1[i:i+batch_size]
            batch_u2 = train_u2[i:i+batch_size]
            batch_z_flat = cgkn.autoencoder.encoder(batch_u2).view(batch_u2.size(0), dim_z)
            batch_u1_pred, _ = cgkn(batch_u1, batch_z_flat)
            preds.append(batch_u1_pred)
        train_u1_pred = torch.cat(preds, dim=0)
    sigma_hat = torch.zeros(dim_u1 + dim_z, device="cpu")
    sigma_hat[:dim_u1] = torch.sqrt(torch.mean((train_u1[1:] - train_u1_pred[:-1])**2, dim=0)).view(-1).to("cpu")
    sigma_hat[dim_u1:] = 0.1  # sigma2 manually set
    return sigma_hat

sigma_hat = compute_sigma_hat(train_u1[:, :L], train_u2, cgkn, dim_u1, dim_z, batch_size=100, device=device)
torch.save(sigma_hat, "../data/CGKN_L32_resblock_dimz32_sigma_hat.pt")
# sigma_hat = torch.load("../data/CGKN_resblock_dimz32_sigma_hat.pt", weights_only=True)

def CGFilter(cgkn, sigma, u1, mu0, R0):
    # u1: (Nt, L, 4, 1)
    # mu0: (dim_z, 1)
    # R0: (dim_z, dim_z)
    device = u1.device
    Nt = u1.shape[0]
    u1_flat = u1.view(Nt, -1, 1)
    dim_u1 = u1_flat.shape[1]
    dim_z = mu0.shape[0]
    s1, s2 = torch.diag(sigma[:dim_u1]), torch.diag(sigma[dim_u1:])
    mu_pred = torch.zeros((Nt, dim_z, 1)).to(device)
    R_pred = torch.zeros((Nt, dim_z, dim_z)).to(device)
    mu_pred[0] = mu0
    R_pred[0] = R0
    for n in range(1, Nt):
        x, y = unit2xy(u1[n-1].permute(2,0,1))
        pos = torch.stack([x, y], dim=-1) # (1, L, 2)
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(pos)]

        K0 = torch.linalg.solve(s1 @ s1.T + g1 @ R0 @ g1.T, g1 @ R0 @ g2.T).T
        mu1 = f2 + g2@mu0 + K0@(u1_flat[n]-f1-g1@mu0)
        R1 = g2@R0@g2.T + s2@s2.T - K0@g1@R0@g2.T
        R1 = 0.5 * (R1 + R1.T)

        # mu1 = f2 + g2@mu0 + K@(u1_flat[n]-f1-g1@mu0)
        # R1 = g2@R0@g2.T + s2@s2.T - K@g1@R0@g2.T
        mu_pred[n] = mu1
        R_pred[n] = R1
        mu0 = mu1
        R0 = R1
    return (mu_pred, R_pred)

# def CGFilter_batch(cgkn, sigma, u1, mu0, R0):
#     # u1:  (B, Nt, L, 4, 1)
#     # mu0: (B, dim_z, 1)
#     # R0:  (B, dim_z, dim_z)
#     device = u1.device
#     B, Nt = u1.shape[:2]
#     u1_flat = u1.view(B, Nt, -1, 1)  # (B, Nt, dim_u1, 1)
#     dim_u1 = u1_flat.shape[2]
#     dim_z = mu0.shape[1]
#     s1 = torch.diag(sigma[:dim_u1]).to(device)       # (dim_u1, dim_u1)
#     s2 = torch.diag(sigma[dim_u1:]).to(device)       # (dim_z, dim_z)
#     s1_cov = s1 @ s1.T  # (dim_u1, dim_u1)
#     s2_cov = s2 @ s2.T  # (dim_z, dim_z)
#     mu_pred = torch.zeros((B, Nt, dim_z, 1), device=device)
#     R_pred = torch.zeros((B, Nt, dim_z, dim_z), device=device)
#     mu_pred[:, 0] = mu0
#     R_pred[:, 0] = R0
#     for n in range(1, Nt):
#         x, y = unit2xy(u1[:, n - 1].squeeze(-1))  # u1[:, n-1]: (B, L, 4, 1)
#         pos = torch.stack([x, y], dim=-1)  # (B, L, 2)
#         f1, g1, f2, g2 = cgkn.cgn(pos)  # shapes: (B, dim_u1, 1), (B, dim_u1, dim_z), ...

#         # Kalman gain: solve per batch
#         S = s1_cov + torch.bmm(g1, torch.bmm(R0, g1.transpose(1, 2)))      # (B, dim_u1, dim_u1)
#         Cross = torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2)))           # (B, dim_u1, dim_z)
#         K = torch.linalg.solve(S, Cross).transpose(1, 2)                   # (B, dim_z, dim_u1)
#         innov = u1_flat[:, n] - f1 - torch.bmm(g1, mu0)                    # (B, dim_u1, 1)
#         mu1 = f2 + torch.bmm(g2, mu0) + torch.bmm(K, innov)                # (B, dim_z, 1)
#         R1 = torch.bmm(g2, torch.bmm(R0, g2.transpose(1, 2))) + s2_cov \
#              - torch.bmm(K, torch.bmm(g1, torch.bmm(R0, g2.transpose(1, 2))))  # (B, dim_z, dim_z)
#         R1 = 0.5 * (R1 + R1.transpose(1, 2))  # Ensure symmetry

#         mu_pred[:, n] = mu1
#         R_pred[:, n] = R1
#         mu0 = mu1
#         R0 = R1
#     return mu_pred, R_pred


def CGFilter_batch(cgkn, sigma, u1, mu0, R0, jitter: float = 1e-6):
    """
    Same outputs as original:
      mu_pred: (B, Nt, dim_z, 1)
      R_pred:  (B, Nt, dim_z, dim_z)
    Memory-lean: in-place diag adds, in-place symmetrize, no eye(), minimal .contiguous().
    """
    device = u1.device
    dtype  = u1.dtype
    B, Nt  = u1.shape[:2]

    # Flatten once
    u1_flat = u1.view(B, Nt, -1, 1)            # (B, Nt, dim_u1, 1)
    dim_u1  = u1_flat.shape[2]
    dim_z   = mu0.shape[1]

    # Keep noise as vectors; square for variances
    s1_sq = (sigma[:dim_u1].to(device=device, dtype=dtype))**2  # (dim_u1,)
    s2_sq = (sigma[dim_u1:].to(device=device, dtype=dtype))**2  # (dim_z,)

    # # Outputs (same as original)
    mu_pred = torch.empty((B, Nt, dim_z, 1), device=device, dtype=dtype)
    R_pred  = None
    # R_pred  = torch.empty((B, Nt, dim_z, dim_z), device=device, dtype=dtype)
    mu_pred[:, 0] = mu0
    # R_pred[:, 0]  = R0

    for n in range(1, Nt):
        # Controls -> positions
        x, y = unit2xy(u1[:, n - 1].squeeze(-1))    # (B, L), (B, L)
        pos  = torch.stack([x, y], dim=-1)          # (B, L, 2) (no contiguous needed here)

        # Model terms
        f1, g1, f2, g2 = cgkn.cgn(pos)              # f1:(B,dim_u1,1) g1:(B,dim_u1,dim_z) f2:(B,dim_z,1) g2:(B,dim_z,dim_z)

        # ---- Innovation covariance S = g1 R0 g1^T + diag(s1^2) ----
        R0_g1T = torch.bmm(R0, g1.transpose(1, 2))  # (B, dim_z, dim_u1)
        S      = torch.bmm(g1, R0_g1T)              # (B, dim_u1, dim_u1)

        # In-place diagonal add (no diag_embed, no new tensor)
        S.diagonal(dim1=-2, dim2=-1).add_(s1_sq)

        # # In-place numeric symmetrization: S <- (S + S^T)/2 without extra large temporaries
        S = 0.5 * (S + S.transpose(1, 2))
        # Tiny in-place jitter on the diagonal (no identity)
        S.diagonal(dim1=-2, dim2=-1).add_(jitter)

        # ---- Cross term and gain ----
        R0_g2T = torch.bmm(R0, g2.transpose(1, 2))  # (B, dim_z, dim_z)
        Cross  = torch.bmm(g1, R0_g2T)              # (B, dim_u1, dim_z)

        L = torch.linalg.cholesky(S)
        X = torch.cholesky_solve(Cross, L)          # (B, dim_u1, dim_z)
        K = X.transpose(1, 2)                        # (B, dim_z, dim_u1) (view, no copy)

        # ---- Innovation, mean update ----
        innov = u1_flat[:, n] - f1 - torch.bmm(g1, mu0)   # (B, dim_u1, 1)
        mu1   = f2 + torch.bmm(g2, mu0) + torch.bmm(K, innov)

        # ---- Covariance update: R1 = g2 R0 g2^T + diag(s2^2) - K·Cross ----
        R1 = torch.bmm(g2, R0_g2T) - torch.bmm(K, Cross)  # (B, dim_z, dim_z)
        R1.diagonal(dim1=-2, dim2=-1).add_(s2_sq)
        R1 = 0.5 * (R1 + R1.transpose(1, 2)) 

        # Save & roll
        mu_pred[:, n] = mu1
        # R_pred[:, n]  = R1
        mu0, R0 = mu1, R1

    return mu_pred, R_pred



# Print allocated and reserved GPU memory in MB
print('stage1 loaded:')
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
      f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


########################################################
################# Train cgkn (Stage2)  #################
########################################################
torch.manual_seed(1)
np.random.seed(1)

# Stage 2: Train cgkn with loss_forecast + loss_da + loss_ae + loss_forecast_z
short_steps = 2
long_steps = 40
cut_point = 20

epochs = 500
train_batch_size = 200
train_batch_size_da = 10
val_batch_size = 10
val_per_epochs = 1
val_per_itrs = int(Ntrain / train_batch_size * val_per_epochs)
train_num_batches = int(Ntrain / train_batch_size)
Niters = epochs * train_num_batches
loss_history = {
    "train_forecast_u1": [],
    "train_forecast_u2": [],
    "train_ae": [],
    "train_forecast_z": [],
    "train_da": [],
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    "val_ae": [],
    "val_forecast_z": [],
    "val_da": [],
    }
best_val_loss = float('inf')
optimizer = torch.optim.Adam(cgkn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for itr in range(1, Niters+1):
    # Training
    start_time = time.time()
    cgkn.train()

    # === Short Forecast ===
    head_indices_short = torch.from_numpy(np.random.choice(Ntrain-short_steps+1, size=train_batch_size, replace=False))
    u1_short = torch.stack([train_u1[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (short_steps, B, L, 2)
    u2_short = torch.stack([train_u2[idx:idx + short_steps] for idx in head_indices_short], dim=1).to(device) # (short_steps, B, Nx, Nx, 2)

    # randomly choosing tracers
    tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
    u1_short = torch.index_select(u1_short, dim=2, index=tracer_idx)     # (short_steps, B, L, 4)

    # AutoEncoder
    z_short = cgkn.autoencoder.encoder(u2_short.view(-1, *u2_short.shape[2:])) # (short_steps*B, 2, 16, 16)
    u2_ae_short = cgkn.autoencoder.decoder(z_short).view(*u2_short.shape)      # (short_steps, B, Nx, Nx, 2)
    loss_ae = nnF.mse_loss(u2_short, u2_ae_short)

    # State Prediction
    z_short = z_short.view(short_steps, train_batch_size, *z_short.shape[1:]) # (short_steps, B, 2, 16, 16)
    z_flat_short = z_short.reshape(short_steps, train_batch_size, dim_z)      # (short_steps, B, dim_z)
    u1_short_pred = [u1_short[0]]
    z_flat_short_pred = [z_flat_short[0]]
    for n in range(1, short_steps):
        u1_short_pred_n, z_short_pred_n = cgkn(u1_short_pred[-1], z_flat_short_pred[-1])
        u1_short_pred.append(u1_short_pred_n)
        z_flat_short_pred.append(z_short_pred_n)
    u1_short_pred = torch.stack(u1_short_pred, dim=0)
    z_flat_short_pred = torch.stack(z_flat_short_pred, dim=0)
    loss_forecast_z = nnF.mse_loss(z_flat_short[1:], z_flat_short_pred[1:])

    z_short_pred = z_flat_short_pred.reshape(short_steps, train_batch_size, 2, z_h, z_w)
    u2_short_pred = cgkn.autoencoder.decoder(z_short_pred.view(-1, 2, z_h, z_w)).view(*u2_short.shape)
    loss_forecast_u1 = nnF.mse_loss(u1_short[1:], u1_short_pred[1:])
    loss_forecast_u2 = nnF.mse_loss(u2_short[1:], u2_short_pred[1:])

    # print('forecast loss finished:')
    # # Print allocated and reserved GPU memory in MB
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
    #       f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # === DA Loss ===
    head_idx_long = torch.from_numpy(np.random.choice(Ntrain - long_steps + 1, size=train_batch_size_da, replace=False))
    u1_long = torch.stack([train_u1[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, L, 4)
    u2_long = torch.stack([train_u2[i:i + long_steps] for i in head_idx_long]).to(device)  # (B, Nt, H, W, 2)
    # randomly choosing tracers
    u1_long = torch.index_select(u1_long, dim=2, index=tracer_idx)     # (short_steps, B, L, 4)
    mu0 = torch.zeros(train_batch_size_da, dim_z, 1, device=device)
    R0 = 0.1 * torch.eye(dim_z, device=device).expand(train_batch_size_da, dim_z, dim_z)
        

    # print('DA prepared:')
    # # Print allocated and reserved GPU memory in MB
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
    #       f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    mu_z_flat_pred_long, _ = CGFilter_batch(cgkn, sigma_hat.to(device), u1_long.unsqueeze(-1), mu0, R0)  # (B, Nt, dim_z, 1)

    # print('DA finished:')
    # # Print allocated and reserved GPU memory in MB
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
    #       f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    mu_z_pred_long = mu_z_flat_pred_long[:, cut_point:].squeeze(-1).reshape(-1, 2, z_h, z_w)  # (B*(Nt-cut), 2, z_h, z_w)


    mu_pred_long = cgkn.autoencoder.decoder(mu_z_pred_long)  # (B*(Nt-cut), H, W, 2)
    mu_pred_long = mu_pred_long.view(train_batch_size_da, long_steps - cut_point, *mu_pred_long.shape[1:])  # (B, Nt-cut, ...)
    u2_long_target = u2_long[:, cut_point:]  # (B, Nt-cut, H, W, 2)
    loss_da = nnF.mse_loss(mu_pred_long, u2_long_target)

    loss_total = loss_forecast_u1 + loss_forecast_u2 + loss_ae + loss_forecast_z + loss_da

    if torch.isnan(loss_total):
        print(itr, "nan")
        continue

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    loss_history["train_forecast_u1"].append(loss_forecast_u1.item())
    loss_history["train_forecast_u2"].append(loss_forecast_u2.item())
    loss_history["train_forecast_z"].append(loss_forecast_z.item())
    loss_history["train_ae"].append(loss_ae.item())
    loss_history["train_da"].append(loss_da.item())
    end_time = time.time()

    # Validation
    if itr % val_per_itrs == 0:
        cgkn.eval()
        val_loss_forecast_u1 = 0.
        val_loss_forecast_u2 = 0.
        val_loss_forecast_z = 0.
        val_loss_ae = 0.
        val_loss_physics = 0.
        val_loss_da = 0.
        val_total_samples = 0
        with torch.no_grad():
            for i in range(0, Nval - short_steps + 1, val_batch_size):
                B = min(val_batch_size, Nval - short_steps + 1 - i)
                u1_batch = torch.stack([val_u1[i+j:i+j+short_steps] for j in range(B)], dim=1).to(device)
                u2_batch = torch.stack([val_u2[i+j:i+j+short_steps] for j in range(B)], dim=1).to(device)

                # # randomly choosing tracers
                # tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
                # u1_batch = torch.index_select(u1_batch, dim=2, index=tracer_idx)     # (short_steps, B, L, 4)

                # AE loss
                z_val = cgkn.autoencoder.encoder(u2_batch.view(-1, *u2_batch.shape[2:]))
                u2_val_ae = cgkn.autoencoder.decoder(z_val).view_as(u2_batch)
                val_loss_ae += nnF.mse_loss(u2_batch, u2_val_ae, reduction='mean').item() * B

                # Forecast loss
                z_val = z_val.view(short_steps, B, *z_val.shape[1:])
                z_flat_val = z_val.view(short_steps, B, -1)
                u1_preds = [u1_batch[0]]
                z_preds = [z_flat_val[0]]
                for n in range(1, short_steps):
                    u1_pred, z_pred = cgkn(u1_preds[-1], z_preds[-1])
                    u1_preds.append(u1_pred)
                    z_preds.append(z_pred)
                u1_preds = torch.stack(u1_preds)
                z_preds = torch.stack(z_preds)
                val_loss_forecast_z += nnF.mse_loss(z_flat_val[1:], z_preds[1:], reduction='mean').item() * B

                z_pred_reshaped = z_preds.view(short_steps, B, 2, z_h, z_w)
                u2_preds = cgkn.autoencoder.decoder(z_pred_reshaped.view(-1, 2, z_h, z_w)).view_as(u2_batch)
                val_loss_forecast_u1 += nnF.mse_loss(u1_batch[1:], u1_preds[1:], reduction='mean').item() * B
                val_loss_forecast_u2 += nnF.mse_loss(u2_batch[1:], u2_preds[1:], reduction='mean').item() * B

                val_total_samples += B
            val_loss_forecast_u1 = val_loss_forecast_u1 / val_total_samples
            val_loss_forecast_u2 = val_loss_forecast_u2 / val_total_samples
            val_loss_forecast_z = val_loss_forecast_z / val_total_samples
            val_loss_ae = val_loss_ae / val_total_samples

            del u1_batch, u2_batch, z_val, z_flat_val, u2_val_ae, u1_preds, z_preds, z_pred_reshaped
            torch.cuda.empty_cache()

            # DA loss
            val_u1_gpu = val_u1[None, ..., None].to(device)
            val_u2_gpu = val_u2.to(device)
            val_mu_z_pred = CGFilter_batch(cgkn, sigma_hat.to(device), val_u1_gpu, mu0=torch.zeros(1, dim_z, 1, device=device), R0=0.1*torch.eye(dim_z, device=device).unsqueeze(0))[0].squeeze((0, -1)).reshape(-1, 2, z_h, z_w)
            val_mu_pred = cgkn.autoencoder.decoder(val_mu_z_pred)
            val_loss_da = nnF.mse_loss(val_u2_gpu[cut_point:], val_mu_pred[cut_point:]).item()

            # Free DA intermediates
            del val_mu_z_pred, val_mu_pred, val_u1_gpu, val_u2_gpu
            torch.cuda.empty_cache()
            
            loss_history["val_forecast_u1"].append(val_loss_forecast_u1)
            loss_history["val_forecast_u2"].append(val_loss_forecast_u2)
            loss_history["val_forecast_z"].append(val_loss_forecast_z)
            loss_history["val_ae"].append(val_loss_ae)
            loss_history["val_da"].append(val_loss_da)

            val_loss_total = val_loss_da
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                checkpoint = {
                    'itr': itr,
                    'model_state_dict': cgkn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_total,
                }
                torch.save(checkpoint, "../model/CGKN_resblock_dimz32_stage2.pt")
                status = "✅"
            else:
                status = ""
            print(f"ep {int(itr/train_num_batches):d} time {end_time - start_time:.4f} | "
                  f"train_u1: {loss_forecast_u1:.4f}  train_u2: {loss_forecast_u2:.4f} "
                  f"train_z: {loss_forecast_z:.4f}  ae: {loss_ae:.4f} da: {loss_da:.4f} | "
                  f"val_u1: {val_loss_forecast_u1:.4f}  val_u2: {val_loss_forecast_u2:.4f}  val_z: {val_loss_forecast_z:.4f} "
                  f"val_da: {val_loss_da:.4f} val_total: {val_loss_total:.4f} "
                  f"{status}"
                  )
    else:
        loss_history["val_forecast_u1"].append(np.nan)
        loss_history["val_forecast_u2"].append(np.nan)
        loss_history["val_forecast_z"].append(np.nan)
        loss_history["val_ae"].append(np.nan)
        loss_history["val_da"].append(np.nan)
        if itr % train_num_batches == 0:
            print(f"ep {int(itr/train_num_batches):d} time {end_time - start_time:.4f} | "
                  f"train_u1: {loss_forecast_u1:.4f}  train_u2: {loss_forecast_u2:.4f} "
                  f"train_z: {loss_forecast_z:.4f}  ae: {loss_ae:.4f} da: {loss_da:.4f} | "
                  )
np.savez(r"../model/CGKN_resblock_dimz32_loss_history_stage2.npz", **loss_history)
"""

checkpoint = torch.load("../model/CGKN_resblock_dimz32_stage2.pt", map_location=device, weights_only=True)
cgkn.load_state_dict(checkpoint['model_state_dict'])

#####################################################################################
################# DA Uncertainty Quantification via Residual Analysis ###############
#####################################################################################
# # CGKN for Data Assimilation (Training Data)
# def run_da(cgkn, train_u1, train_u2, sigma_hat, L_total, L, dim_z, z_h, z_w, cut_point, batch_steps=1000, device="cuda"):
#     cgkn.eval()
#     train_u1 = train_u1.to(device)
#     train_u2 = train_u2.to(device)
#     train_mu_preds = []
#     with torch.no_grad():
#         for i in range(0, train_u1.size(0), batch_steps):
#             tracer_idx = torch.randperm(L_total, device=device)[:L]  # choose L tracers
#             train_u1_batch = train_u1[i:i+batch_steps, tracer_idx]   # (batch, L, features)
#             train_mu_z_flat_pred_batch = CGFilter(
#                 cgkn, sigma_hat.to(device),
#                 train_u1_batch.unsqueeze(-1),
#                 mu0=torch.zeros(dim_z, 1, device=device),
#                 R0=0.1*torch.eye(dim_z, device=device)
#             )[0].squeeze(-1)
#             train_mu_z_pred_batch = train_mu_z_flat_pred_batch.reshape(-1, 2, z_h, z_w)
#             train_mu_pred_batch = cgkn.autoencoder.decoder(train_mu_z_pred_batch)
#             train_mu_preds.append(train_mu_pred_batch)
#         train_mu_pred = torch.cat(train_mu_preds, dim=0)
#     mse = nnF.mse_loss(train_u2[cut_point:], train_mu_pred[cut_point:]).item()
#     print("MSE2_DA (training):", mse)
#     return train_mu_pred.to('cpu')

# train_mu_pred = run_da(cgkn, train_u1, train_u2, sigma_hat, L_total, L, dim_z, z_h, z_w, cut_point, batch_steps=1000, device=device)

# # train_u1 = train_u1.to(device)
# # # train_mu_preds = []
# # cgkn.eval()
# # with torch.no_grad():
# #     # for i in range(0, Ntest, batch_steps):
# #     #     # randomly choosing tracers
# #     #     tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
# #     #     u1_short = torch.index_select(u1_short, dim=2, index=tracer_idx)     # (short_steps, B, L, 4)
# #     #     train_u1_batch = train_u1[i:i+batch_size]

# #     train_mu_z_flat_pred = CGFilter(cgkn, sigma_hat.to(device), train_u1[:, :L].unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
# #     train_mu_z_pred = train_mu_z_flat_pred.reshape(-1, 2, z_h, z_w)
# #     train_mu_pred = cgkn.autoencoder.decoder(train_mu_z_pred)
# # print("MSE2_DA (training):", nnF.mse_loss(train_u2[cut_point:], train_mu_pred[cut_point:]).item())
# # torch.save(train_mu_pred, "../data/CGKN_resblock_dimz32_train_mu_pred.pt")
# train_mu_pred = torch.load("../data/CGKN_resblock_dimz32_train_mu_pred.pt")

# # Target Variable: Residual (std of posterior mean)
# train_mu_std = torch.abs(train_u2[cut_point:] - train_mu_pred[cut_point:])

class UncertaintyNet(nn.Module):
    def __init__(self, dim_u1, dim_u2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_u1, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, 100), nn.SiLU(),
                                 nn.Linear(100, dim_u2))

    def forward(self, x):
        out = self.net(x)
        return out

# uncertainty_net = UncertaintyNet(dim_u1, dim_u2).to(device)
# # epochs = 500
# # train_batch_size = 2000
# # train_tensor = torch.utils.data.TensorDataset(train_u1[cut_point:, :L], train_mu_std)
# # train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=train_batch_size)
# # train_num_batches = len(train_loader)
# # Niters = epochs * train_num_batches
# # train_loss_uncertainty_history = []
# # optimizer = torch.optim.Adam(uncertainty_net.parameters(), lr=1e-3)
# # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
# # for ep in range(1, epochs+1):
# #     uncertainty_net.train()
# #     start_time = time.time()
# #     train_loss_uncertainty = 0.
# #     for x_batch, y_batch in train_loader:
# #         x_batch = x_batch.to(device).reshape(x_batch.size(0), dim_u1)
# #         y_batch = y_batch.to(device).reshape(y_batch.size(0), dim_u2)
# #         optimizer.zero_grad()
# #         preds = uncertainty_net(x_batch)
# #         loss = nnF.mse_loss(preds, y_batch)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         train_loss_uncertainty += loss.item()
# #     train_loss_uncertainty /= train_num_batches
# #     train_loss_uncertainty_history.append(train_loss_uncertainty)
# #     end_time = time.time()
# #     print("ep", ep,
# #           " time:", round(end_time - start_time, 4),
# #           " loss uncertainty:", round(train_loss_uncertainty, 4))

# # torch.save({
# #     'model_state_dict': uncertainty_net.state_dict(),
# #     'optimizer_state_dict': optimizer.state_dict(),
# #     'epoch': epochs,
# #     'loss': train_loss_uncertainty_history,
# # }, "../model/UQNet_resblock_dimz32.pt")

# checkpoint = torch.load("../model/UQNet_resblock_dimz32.pt", map_location=device)
# uncertainty_net.load_state_dict(checkpoint['model_state_dict'])

############################################
################ Test cgkn #################
############################################

# # CGKN for One-Step Prediction
# batch_size = 100
# test_u1 = test_u1.to(device)
# test_u2 = test_u2.to(device)
# test_u1_preds = []
# test_u2_preds = []
# cgkn.eval()
# with torch.no_grad():
#     for i in range(0, Ntest, batch_size):
#         test_u1_batch = test_u1[i:i+batch_size]
#         test_u2_batch = test_u2[i:i+batch_size]
#         test_z_flat_batch = cgkn.autoencoder.encoder(test_u2_batch).view(batch_size, dim_z)
#         test_u1_pred_batch, test_z_flat_pred_batch = cgkn(test_u1_batch, test_z_flat_batch)
#         test_z_pred_batch = test_z_flat_pred_batch.view(batch_size, 2, z_h, z_w)
#         test_u2_pred_batch = cgkn.autoencoder.decoder(test_z_pred_batch)
#         test_u1_preds.append(test_u1_pred_batch)
#         test_u2_preds.append(test_u2_pred_batch)
#     test_u1_pred = torch.cat(test_u1_preds, dim=0)
#     test_u2_pred = torch.cat(test_u2_preds, dim=0)
# MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1])
# print("MSE1:", MSE1.item())
# MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1])
# print("MSE2:", MSE2.item())
# test_u1 = test_u1.to("cpu")
# test_u2 = test_u2.to("cpu")

# np.save(r"../data/CGKN_resblock_dimz32_xy_unit_OneStepPrediction.npy", test_u1_pred.to("cpu"))
# np.save(r"../data/CGKN_resblock_dimz32_psi_OneStepPrediction.npy", test_u2_pred.to("cpu"))

# CGKN for Data Assimilation
test_u1 = test_u1.to(device)
test_u2 = test_u2.to(device)
cgkn.eval()
t0 = time.time()
with torch.no_grad():
    test_mu_z_flat_pred = CGFilter(cgkn, sigma_hat.to(device), test_u1.unsqueeze(-1), mu0=torch.zeros(dim_z, 1).to(device), R0=0.1*torch.eye(dim_z).to(device))[0].squeeze(-1)
    test_mu_z_pred = test_mu_z_flat_pred.reshape(-1, 2, z_h, z_w)
    test_mu_pred = cgkn.autoencoder.decoder(test_mu_z_pred)
t1 = time.time()
MSE2_DA = nnF.mse_loss(test_u2[cut_point:], test_mu_pred[cut_point:])
print("MSE2_DA:", MSE2_DA.item())
print('DA time (hrs):', (t1 - t0) / 3600)
test_u1 = test_u1.to("cpu")
test_u2 = test_u2.to("cpu")
np.save(r"../data/CGKN_L32_resblock_dimz32_psi_DA.npy", test_mu_pred.to("cpu"))

# # uncertainty_net for Uncertainty Quantification
# test_u1 = test_u1.to(device)
# uncertainty_net.eval()
# with torch.no_grad():
#     test_mu_std_pred = uncertainty_net(test_u1.reshape(-1, dim_u1)).reshape(-1,*test_u2.shape[1:]).cpu()
# np.save(r"../data/CGKN_L64_resblock_dimz32_psi_DA_std.npy", test_mu_std_pred)