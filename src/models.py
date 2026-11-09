"""FNO (written from scratch, ~40 lines of model code) and a small 1D U-Net baseline.

Task for both: initial condition u(x, 0) of shape (B, nx)  ->  whole trajectory u(x, t_k), k = 0..nt-1,
of shape (B, nt, nx). Time is produced as output channels (one forward pass, no rollout).
Both respect the periodic boundary: the FNO via FFT, the U-Net via circular padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Multiply the lowest `modes` Fourier coefficients by a learned complex matrix (per mode)."""

    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.w = nn.Parameter(scale * torch.randn(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x):                       # x: (B, C_in, nx)
        xf = torch.fft.rfft(x, dim=-1)          # (B, C_in, nx//2+1)
        out = torch.zeros(x.shape[0], self.w.shape[1], xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :self.modes] = torch.einsum("bim,iom->bom", xf[..., :self.modes], self.w)
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class FNO1d(nn.Module):
    def __init__(self, nt=201, modes=24, width=64, layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)         # input features: u(x,0) and the coordinate x
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, nt))

    def forward(self, u0):                      # (B, nx) -> (B, nt, nx)
        B, nx = u0.shape
        grid = torch.linspace(0, 1, nx, device=u0.device).expand(B, nx)
        h = self.lift(torch.stack([u0, grid], dim=-1)).permute(0, 2, 1)   # (B, width, nx)
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = F.gelu(h)
        return self.head(h.permute(0, 2, 1)).permute(0, 2, 1)             # (B, nt, nx)


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, 5, padding=2, padding_mode="circular"), nn.GELU(),
        nn.Conv1d(c_out, c_out, 5, padding=2, padding_mode="circular"), nn.GELU())


class UNet1d(nn.Module):
    """Small baseline: 3 down / 3 up levels (nx must be divisible by 8), circular padding, skip concat."""

    def __init__(self, nt=201, base=32):
        super().__init__()
        c = [base, 2 * base, 4 * base, 8 * base]
        self.enc = nn.ModuleList([_block(1, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.mid = _block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose1d(c[3], c[2], 2, stride=2),
                                 nn.ConvTranspose1d(c[2], c[1], 2, stride=2),
                                 nn.ConvTranspose1d(c[1], c[0], 2, stride=2)])
        self.dec = nn.ModuleList([_block(2 * c[2], c[2]), _block(2 * c[1], c[1]), _block(2 * c[0], c[0])])
        self.out = nn.Conv1d(c[0], nt, 1)

    def forward(self, u0):                      # (B, nx) -> (B, nt, nx)
        h, skips = u0.unsqueeze(1), []
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = F.avg_pool1d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], dim=1))
        return self.out(h)


def n_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())


def rel_l2(pred, true, dims):
    """Relative L2 per sample, norm taken over `dims` (e.g. (1,2) whole trajectory, (2,) per timestep)."""
    return torch.sqrt(((pred - true) ** 2).sum(dim=dims) / (true ** 2).sum(dim=dims))
