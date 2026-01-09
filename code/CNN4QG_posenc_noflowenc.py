import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import time

device = "cuda:1"
torch.manual_seed(0)
np.random.seed(0)

###############################################
################# Data Import #################
###############################################

QG_Data = np.load(r"../data/qg_data_long.npz")
pos = QG_Data["xy_obs"]
pos_unit = np.stack([np.cos(pos[:, :, 0]), np.sin(pos[:, :, 0]), np.cos(pos[:, :, 1]), np.sin(pos[:, :, 1])], axis=-1)
psi = QG_Data["psi_noisy"]
u1 = torch.tensor(pos_unit, dtype=torch.float) # shape (Nt, L, 4), keep tracers parallel
u2 = torch.tensor(psi, dtype=torch.float) # shape (Nt, 64, 64, 2)

# Train / Test
Ntrain = 80000
Nval = 10000
Ntest = 10000
L_total = 1024 # total number of tracers in training dataset
L = 128 # number of tracers used in data assimilation

train_u1 = u1[:Ntrain]
train_u2 = u2[:Ntrain]
val_u1 = u1[Ntrain:Ntrain+Nval, :L]
val_u2 = u2[Ntrain:Ntrain+Nval]
test_u1 = u1[Ntrain+Nval:Ntrain+Nval+Ntest, :L]
test_u2 = u2[Ntrain+Nval:Ntrain+Nval+Ntest]

################################################################
############################### CNN ############################
################################################################
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

class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(CircularConv2d(2, 32, 3, 1, 1), nn.SiLU(),
                                 CircularConv2d(32, 64, 3, 1, 1), nn.SiLU(),
                                 CircularConv2d(64, 128, 3, 1, 2), nn.SiLU(),
                                 CircularConv2d(128, 128, 3, 1, 1), nn.SiLU(),
                                 CircularConv2d(128, 64, 3, 1, 2), nn.SiLU(),
                                 CircularConv2d(64, 32, 3, 1, 1), nn.SiLU(),
                                 CircularConv2d(32, 2, 3, 1, 1))
    def forward(self, x):
        # x shape(B, 2, y, x)
        out = self.seq(x)
        out = out.permute(0, 2, 3, 1)
        return out

class SolNet(nn.Module):
    def __init__(self, flow_net, flow_dim=256, use_pos_encoding=True, num_frequencies=6):
        super().__init__()
        
        self.use_pos_encoding = use_pos_encoding
        self.num_frequencies = num_frequencies
        if use_pos_encoding:
            in_dim = 2 + 2 * 2 * num_frequencies # sin/cos for each x and y
        else:
            in_dim = 2
        out_dim = 4

        # Flow prediction net
        self.flow_net = flow_net

        # Fully connected layers to fuse flow + position
        self.predictor = nn.Sequential(
            nn.Linear(flow_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, out_dim)  # output: dx, dy or dcos x, dsin x, dcos y, dsin y
        )

        # Gamma and Beta networks for FiLM
        self.gamma_net = nn.Sequential(
            nn.Linear(in_dim, flow_dim), nn.SiLU()
        )
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, flow_dim), nn.SiLU()
        )

    def forward(self, tracer_positions, flow_field):
        """
        tracer_positions: (B, L, 4)
        flow_field: (B, 64, 64, 2)
        Output: (B, L, 4)
        """
        B, L, _ = tracer_positions.shape
        flow_field = flow_field.permute(0, 3, 1, 2)  # (B, 2, 64, 64)

        # Predict flow fields
        flow_next = self.flow_net(flow_field)  # (B, 64, 64, 2)

        # Predict tracer positions
        # Encode tracer positions (optional)
        x, y = unit2xy(tracer_positions)
        xy = torch.stack([x, y], dim=-1) # (B, L, 2)
        if self.use_pos_encoding:
            xy_encoded = self.positional_encoding(xy.view(-1,2)) # (B * L, 2)
            tracer_positions_encoded = xy_encoded.view(B, L, -1)
        else:
            tracer_positions_encoded = xy

        flow_expanded = flow_field[:,0:1].reshape(B, 1, -1).expand(B, L, -1)  # (B, L, flow_dim)
        gamma = self.gamma_net(tracer_positions_encoded)      # (B, L, flow_dim)
        beta = self.beta_net(tracer_positions_encoded)        # (B, L, flow_dim)
        flow_modulated = flow_expanded * gamma + beta         # (B, L, flow_dim) FiLM modulation
        pos_next = tracer_positions + self.predictor(flow_modulated) # (B, L, 4)
        
        return pos_next, flow_next

    def positional_encoding(self, x):
        """
        Input: x of shape (B, 2)
        Output: encoded x of shape (B, 2+ 4*num_frequencies)
        """
        freqs = 2 ** torch.arange(self.num_frequencies, device=x.device) * np.pi  # (F,)
        x1 = x[:, 0:1] * freqs                 # (B, F)
        x2 = x[:, 1:2] * freqs                 # (B, F)
        encoded = torch.cat([x, torch.sin(x1), torch.cos(x1), torch.sin(x2), torch.cos(x2)], dim=1)  # shape: (B, 2+4F)
        return encoded


##############################################
################# Train CNN  #################
##############################################
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
    "val_forecast_u1": [],
    "val_forecast_u2": [],
    }
best_val_loss = float('inf')

flow_net = FlowNet()
solnet = SolNet(flow_net, flow_dim=64*64).to(device)
optimizer = torch.optim.Adam(solnet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)

# Number of Parameters
cnn_params = parameters_to_vector(solnet.flow_net.parameters()).numel()
gamma_params = parameters_to_vector(solnet.gamma_net.parameters()).numel()
beta_params = parameters_to_vector(solnet.beta_net.parameters()).numel()
predictor_params = parameters_to_vector(solnet.predictor.parameters()).numel()
total_params = cnn_params + gamma_params + beta_params + predictor_params
print(f'flow cnn #parameters:      {cnn_params:,}')
print(f'gamma #parameters:  {gamma_params:,}')
print(f'beta #parameters:  {beta_params:,}')
print(f'tracer predictor #parameters:  {predictor_params:,}')
print(f'TOTAL #parameters:    {total_params:,}')
"""
for ep in range(1, epochs+1):
    solnet.train()
    start_time = time.time()

    train_loss_forecast_u1 = 0.
    train_loss_forecast_u2 = 0.
    train_loss_ae = 0.
    for u1_initial, u2_initial, u1_next, u2_next in train_loader:
        u1_initial, u2_initial, u1_next, u2_next = u1_initial.to(device), u2_initial.to(device), u1_next.to(device), u2_next.to(device)

        # randomly choosing tracers
        tracer_idx = torch.randperm(L_total, device=device)[:L]              # unique
        u1_initial = torch.index_select(u1_initial, dim=1, index=tracer_idx) # (batch, L, 4)
        u1_next    = torch.index_select(u1_next,    dim=1, index=tracer_idx) # (batch, L, 4)

        u1_pred, u2_pred = solnet(u1_initial, u2_initial)
        loss_forecast_u1 = nnF.mse_loss(u1_next, u1_pred)
        loss_forecast_u2 = nnF.mse_loss(u2_next, u2_pred)
        # loss_ae = nnF.mse_loss(u2_initial[..., 0:1], u2_decoded)
        loss_total = loss_forecast_u1 + loss_forecast_u2# + loss_ae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        train_loss_forecast_u1 += loss_forecast_u1.item()
        train_loss_forecast_u2 += loss_forecast_u2.item()
        # train_loss_ae += loss_ae.item()
    train_loss_forecast_u1 /= train_num_batches
    train_loss_forecast_u2 /= train_num_batches
    # train_loss_ae /= train_num_batches
    loss_history["train_forecast_u1"].append(train_loss_forecast_u1)
    loss_history["train_forecast_u2"].append(train_loss_forecast_u2)
    end_time = time.time()

    # Validation
    if ep % 10 == 0:
        solnet.eval()
        val_loss_u1 = 0.
        val_loss_u2 = 0.
        with torch.no_grad():
            for u1_initial, u2_initial, u1_next, u2_next in val_loader:
                u1_initial, u2_initial, u1_next, u2_next = map(lambda x: x.to(device), [u1_initial, u2_initial, u1_next, u2_next])

                u1_pred, u2_pred = solnet(u1_initial, u2_initial)
                val_loss_u1 += nnF.mse_loss(u1_next, u1_pred).item()
                val_loss_u2 += nnF.mse_loss(u2_next, u2_pred).item()
        val_loss_u1 /= len(val_loader)
        val_loss_u2 /= len(val_loader)
        val_loss_total = val_loss_u1 + val_loss_u2
        loss_history["val_forecast_u1"].append(val_loss_u1)
        loss_history["val_forecast_u2"].append(val_loss_u2)
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            checkpoint = {
                'epoch': ep,
                'model_state_dict': solnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_total
            }
            torch.save(checkpoint, r"../model/CNN.pt")
            status = "✅"
        else:
            status = ""
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f} | "
              f"val_u1: {val_loss_u1:.4f}  val_u2: {val_loss_u2:.4f}  val_total: {val_loss_total:.4f} "
              f"{status}"
              )
    else:
        loss_history["val_forecast_u1"].append(np.nan)
        loss_history["val_forecast_u2"].append(np.nan)
        print(f"ep {ep} time {end_time - start_time:.4f} | "
              f"train_u1: {train_loss_forecast_u1:.4f}  train_u2: {train_loss_forecast_u2:.4f}"
              )

np.savez(r"../model/CNN_loss_history.npz", **loss_history)

"""
############################################
################# Test CNN #################
############################################
checkpoint = torch.load("../model/CNN.pt", map_location=device, weights_only=True)
solnet.load_state_dict(checkpoint['model_state_dict'])

# # One-Step Prediction
# batch_size = 100
# test_u1_preds = []
# test_u2_preds = []
# solnet.eval()
# with torch.no_grad():
#     for i in range(0, Ntest, batch_size):
#         test_u1_batch = test_u1[i:i+batch_size].to(device)
#         test_u2_batch = test_u2[i:i+batch_size].to(device)
#         test_u1_pred_batch, test_u2_pred_batch = solnet(test_u1_batch, test_u2_batch)
#         test_u1_preds.append(test_u1_pred_batch)
#         test_u2_preds.append(test_u2_pred_batch)
#     test_u1_pred = torch.cat(test_u1_preds, dim=0)
#     test_u2_pred = torch.cat(test_u2_preds, dim=0)
#     # test_u1_pred, test_u2_pred = solnet(test_u1, test_u2)
# MSE1 = nnF.mse_loss(test_u1[1:], test_u1_pred[:-1].to('cpu'))
# print("MSE1:", MSE1.item())
# MSE2 = nnF.mse_loss(test_u2[1:], test_u2_pred[:-1].to('cpu'))
# print("MSE2:", MSE2.item())

# np.save(r"../data/CNN_xy_unit_OneStepPrediction.npy", test_u1_pred.to("cpu"))
# np.save(r"../data/CNN_psi_OneStepPrediction.npy", test_u2_pred.to("cpu"))

# Multi-Step Prediction
si = 9000
steps = 50
u1_initial = test_u1[si:si+1].to(device)
u2_initial = test_u2[si:si+1].to(device)
test_u1_pred = torch.zeros(steps+1, *test_u1.shape[1:], device=device)
test_u2_pred = torch.zeros(steps+1, *test_u2.shape[1:], device=device)
test_u1_pred[0] = u1_initial[0]
test_u2_pred[0] = u2_initial[0]
with torch.no_grad():
    for n in range(1, steps+1):
        u1_pred, u2_pred = solnet(u1_initial, u2_initial)
        test_u1_pred[n] = u1_pred[0]
        test_u2_pred[n] = u2_pred[0]
        u1_initial = u1_pred
        u2_initial = u2_pred

np.save(r"../data/CNN_L64_xy_unit_10StepPrediction.npy", test_u1_pred.to("cpu"))
np.save(r"../data/CNN_L64_psi_10StepPrediction.npy", test_u2_pred.to("cpu"))

