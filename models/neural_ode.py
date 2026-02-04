"""
Neural ODE model components: Encoder, ODE function, ODE block, Decoder, and Forecaster.

Designed for 1D time series forecasting where an encoder maps a recent window
into a latent state `h0`, `odeint` integrates the latent forward in (continuous)
time using a learned derivative `f(h,t)`, and a decoder maps latent states back
to the observation space.

Dropout layers are included inside `ODEFunc`/encoder/decoder so Monte Carlo
Dropout can be used later for uncertainty estimation (keep `model.train()` during
inference to sample.)
"""
from typing import Optional

import torch
import torch.nn as nn

try:
    # prefer adjoint if available for memory savings
    from torchdiffeq import odeint_adjoint as odeint
except Exception:
    from torchdiffeq import odeint


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        # x: [batch, seq_len] or flattened features
        return self.net(x)


class ODEFunc(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, t, h):
        # odeint expects signature func(t, y) or func(y, t) depending on version; torchdiffeq uses (t, y)
        return self.net(h)


class ODEBlock(nn.Module):
    def __init__(self, odefunc: ODEFunc, solver_times: Optional[torch.Tensor] = None):
        super().__init__()
        self.odefunc = odefunc
        self.solver_times = solver_times

    def forward(self, h0, t_steps: Optional[torch.Tensor] = None):
        # h0: [batch, latent_dim]
        # t_steps: 1D times tensor of shape [T] (relative times). If None, use default [0,1]
        if t_steps is None:
            t_steps = torch.tensor([0.0, 1.0], device=h0.device, dtype=h0.dtype)

        # odeint expects initial state shaped like h0; it will integrate along t_steps
        # We need to transpose batch & features to match odeint's expectations: odeint returns [T, batch, dim]
        # use a fixed-step Runge-Kutta 4 solver for robustness and determinism
        try:
            traj = odeint(self.odefunc, h0, t_steps, method='rk4')
        except TypeError:
            # fallback if method keyword not supported in this torchdiffeq version
            traj = odeint(self.odefunc, h0, t_steps)
        # return trajectory as [batch, T, dim]
        traj = traj.permute(1, 0, 2)
        return traj


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        # h: [batch, latent_dim] or [batch, T, latent_dim]
        if h.dim() == 3:
            b, T, d = h.shape
            out = self.net(h.reshape(b * T, d))
            return out.reshape(b, T, 1)
        else:
            return self.net(h)


class NeuralODEForecaster(nn.Module):
    def __init__(self, seq_len: int, encoder_hidden: int, latent_dim: int, odefunc_hidden: int = 64,
                 decoder_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = Encoder(input_size=seq_len, hidden=encoder_hidden, latent_dim=latent_dim, dropout=dropout)
        self.odefunc = ODEFunc(latent_dim=latent_dim, hidden=odefunc_hidden, dropout=dropout)
        self.odeblock = ODEBlock(self.odefunc)
        self.decoder = Decoder(latent_dim=latent_dim, hidden=decoder_hidden)

    def forward(self, x_window, t_steps: Optional[torch.Tensor] = None):
        # x_window: [batch, seq_len, 1] or [batch, seq_len]
        if x_window.dim() == 3:
            x_flat = x_window.view(x_window.size(0), -1)
        else:
            x_flat = x_window

        h0 = self.encoder(x_flat)
        traj = self.odeblock(h0, t_steps=t_steps)  # [batch, T, latent_dim]
        out = self.decoder(traj)  # [batch, T, 1]
        return out
