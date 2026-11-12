"""M7: a local-rule time-stepper ("FluxStepper").

One step of 0.01 is  u_next = u - dt/dx * (F_i - F_{i-1}),  where F_i is the time-averaged flux through the face between cells i and i+1.
F is a physics baseline (Rusanov flux + central diffusion with the verified nu/pi) plus a LEARNED correction that sees only a local window
(a periodic CNN with receptive field +-12 cells). Because the update is a difference of face fluxes on a periodic grid, total mass sum(u) is
conserved exactly (to round-off) for ANY network weights. The network never sees the whole wave, only neighbourhoods, so it can be applied to
waves it has never seen and to other grid sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inference import NU_EFF


class FluxStepper(nn.Module):
    def __init__(self, ch=32, layers=6, ksize=5, dt=0.01, nu=NU_EFF):
        super().__init__()
        self.dt, self.nu = dt, nu
        convs = [nn.Conv1d(2, ch, ksize, padding=ksize // 2, padding_mode="circular")]
        for _ in range(layers - 1):
            convs.append(nn.Conv1d(ch, ch, ksize, padding=ksize // 2, padding_mode="circular"))
        self.convs = nn.ModuleList(convs)
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)          # start as the pure physics baseline
        nn.init.zeros_(self.out.bias)
        self.receptive = 1 + (ksize - 1) * layers

    def flux(self, u):                           # u (B, N) -> flux at face i+1/2, shape (B, N)
        dx = 1.0 / u.shape[-1]
        up = torch.roll(u, -1, -1)
        alpha = torch.maximum(u.abs(), up.abs())
        f0 = 0.25 * (u * u + up * up) - 0.5 * alpha * (up - u) - self.nu * (up - u) / dx
        h = torch.stack([u, 0.5 * u * u], 1)
        for c in self.convs:
            h = F.gelu(c(h))
        return f0 + self.out(h)[:, 0]

    def step(self, u):
        dx = 1.0 / u.shape[-1]
        fb = self.flux(u)
        return u - (self.dt / dx) * (fb - torch.roll(fb, 1, -1))

    def forward(self, u0, n_steps, clamp=False):  # (B, N) -> (B, n_steps + 1, N), frame 0 is the input itself
        """clamp=True applies the maximum principle of viscous Burgers: u stays inside [min u0, max u0] for all time. It is physically valid,
        and only acts as a safety net against blow-up (it can change mass if it ever triggers, so evaluation reports both ways)."""
        lo, hi = u0.min(-1, keepdim=True).values, u0.max(-1, keepdim=True).values
        out, u = [u0], u0
        for _ in range(n_steps):
            u = self.step(u)
            if clamp:
                u = torch.minimum(torch.maximum(u, lo), hi)
            out.append(u)
        return torch.stack(out, 1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
