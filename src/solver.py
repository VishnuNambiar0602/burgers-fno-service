"""Small 1D viscous Burgers solver used as an independent reference.

    u_t + d/dx( a * u^2 / 2 ) = nu * u_xx,   periodic on [0, L)

Finite volume, so the update telescopes and total mass sum(u)*dx is conserved to round-off:
  * numerical flux at each cell face: Rusanov (local Lax-Friedrichs) on MUSCL-reconstructed
    left/right states with a minmod limiter (2nd order away from shocks, no oscillations at them)
  * diffusion: standard central difference nu * (u[i+1] - 2u[i] + u[i-1]) / dx^2
  * time: SSP-RK3 with a CFL-limited step (advective and diffusive limits)
`a` exists only so the data's equation convention can be matched (a=1 is the textbook form; PDEBench's
convention is checked against the data before this solver is used as a reference).
"""
import numpy as np


def _minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def _rhs(u, dx, nu, a):
    """d u / dt for u of shape (..., nx), periodic along the last axis."""
    du_l = u - np.roll(u, 1, axis=-1)
    du_r = np.roll(u, -1, axis=-1) - u
    slope = _minmod(du_l, du_r)
    uL = u + 0.5 * slope                       # state just left of face i+1/2 (from cell i)
    uR = np.roll(u - 0.5 * slope, -1, axis=-1)  # state just right of face i+1/2 (from cell i+1)
    fL, fR = 0.5 * a * uL**2, 0.5 * a * uR**2
    speed = np.maximum(np.abs(a * uL), np.abs(a * uR))
    F = 0.5 * (fL + fR) - 0.5 * speed * (uR - uL)    # face flux F[i] = F_{i+1/2}
    conv = -(F - np.roll(F, 1, axis=-1)) / dx
    diff = nu * (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / dx**2
    return conv + diff


def solve(u0, nu, t_out, L=1.0, a=1.0, cfl=0.4):
    """Integrate u0 (shape (..., nx), cell values on a uniform periodic grid of length L).

    t_out: increasing 1-D array of output times, t_out[0] may be 0. Returns (..., len(t_out), nx).
    """
    u = np.array(u0, dtype=np.float64, copy=True)
    nx = u.shape[-1]
    dx = L / nx
    out = np.empty(u.shape[:-1] + (len(t_out), nx))
    t = 0.0
    for k, t_target in enumerate(t_out):
        while t < t_target - 1e-12:
            umax = max(float(np.abs(u).max()) * abs(a), 1e-8)
            dt = cfl * min(dx / umax, dx**2 / (2 * nu) if nu > 0 else np.inf)
            dt = min(dt, t_target - t)
            k1 = u + dt * _rhs(u, dx, nu, a)
            k2 = 0.75 * u + 0.25 * (k1 + dt * _rhs(k1, dx, nu, a))
            u = u / 3.0 + 2.0 / 3.0 * (k2 + dt * _rhs(k2, dx, nu, a))
            t += dt
        out[..., k, :] = u
    return out


def mass(u, L=1.0):
    """Total mass sum(u)*dx over the last axis."""
    return u.sum(axis=-1) * (L / u.shape[-1])
