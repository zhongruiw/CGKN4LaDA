import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils import parameters_to_vector
import time
from torchviz import make_dot

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

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
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1):
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
            bias=True
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
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1):
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
            bias=True
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            CircularConv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConv2d(32, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, u):                           # u:(B, H, W, 2)
        u = u.permute(0, 3, 1, 2)                   # → (B, 2, H, W)
        out = self.enc(u)                           # → (B, 2, d1, d2)
        # print(out.shape)
        return out#.squeeze(1)                      # → (B, 2, d1, d2)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            CircularConvTranspose2d(2, 32, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.SiLU(),
            CircularConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            CircularConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z):                  # z: (B, 2, d1, d2)
        u = self.dec(z)                    # (B, 2, 64, 64)
        return u.permute(0, 2, 3, 1)       # (B, 64, 64, 2)
        
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

class CGN(nn.Module):
    def __init__(self, dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6):
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
        self.g2_param = nn.Parameter(1/dim_z * torch.rand(dim_z, dim_z))

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
        f2 = self.f2_param.repeat(Nt, 1, 1)
        g2 = self.g2_param.repeat(Nt, 1, 1)
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

#################################################################
########################### CGFilter  ###########################
#################################################################

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
Ls = [32, 64, 128] # number of tracers used in data assimilation
for L in Ls:
    print('Number of tracers:', L)

    test_u1 = u1[Ntrain+Nval:Ntrain+Nval+Ntest, :L]
    test_u2 = u2[Ntrain+Nval:Ntrain+Nval+Ntest]
    sigma_hat = torch.load("../data/CGKN_L{:d}_sigma_hat.pt".format(L), weights_only=True)

    dim_u1 = L*4
    dim_u2 = 64*64*2
    z_h, z_w = (16, 16)
    dim_z = z_h*z_w*2

    autoencoder = AutoEncoder().to(device)
    cgn = CGN(dim_u1, dim_z, use_pos_encoding=True, num_frequencies=6).to(device)
    cgkn = CGKN(autoencoder, cgn).to(device)

    torch.manual_seed(1)
    np.random.seed(1)

    # Stage 2
    cut_point = 20
    checkpoint = torch.load("../model/CGKN_L1024_complex_long_sepf1g1_stage2.pt", map_location=device, weights_only=True)
    cgkn.load_state_dict(checkpoint['model_state_dict'])

    ############################################
    ################ Test cgkn #################
    ############################################
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
    test_u1 = test_u1.to("cpu")
    test_u2 = test_u2.to("cpu")

    print('Time used (hrs):', (t1-t0)/3600)