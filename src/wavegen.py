"""Starting-wave generators for M7 (periodic, 256 cell-centred points on [0,1)).

TRAIN styles (the model may train on these):  fourier, drawn, steps, bumps, mixed
UNSEEN styles (held out of ALL training, used only to test generalisation):  gp, saw, fm, packet

Every wave is rescaled so that max|u| is at most about 3.0 (the PDEBench training maximum is 3.35).
"""
import numpy as np

NX = 256
X = (np.arange(NX) + 0.5) / NX


def _scale(u, rng, lo=0.3, hi=3.0):
    m = np.abs(u).max(-1, keepdims=True) + 1e-9
    return u * (rng.uniform(lo, hi, size=(len(u), 1)) / m)


def _lowpass(u, k):
    F = np.fft.rfft(u)
    F[k + 1:] = 0
    return np.fft.irfft(F, n=len(u))


# ------------------------------------------------------------------ styles the model trains on
def fourier(rng, n):
    out = np.empty((n, NX))
    for i in range(n):
        kmax = int(rng.integers(2, 25))
        p = rng.uniform(0.3, 2.0)
        k = np.arange(1, kmax + 1)
        a = rng.standard_normal(kmax) / k ** p
        ph = rng.uniform(0, 2 * np.pi, kmax)
        out[i] = (a[:, None] * np.sin(2 * np.pi * k[:, None] * X[None] + ph[:, None])).sum(0) + rng.normal(0, 0.4)
    return _scale(out, rng)


def drawn(rng, n):
    out = np.empty((n, NX))
    for i in range(n):
        m = int(rng.integers(4, 17))
        px = np.sort(rng.uniform(0, 1, m))
        u = np.interp(X, px, rng.uniform(-2.5, 2.5, m), period=1.0) + 0.03 * rng.standard_normal(NX)
        if rng.random() < 0.5:
            u = _lowpass(u, int(rng.integers(4, 65)))
        out[i] = u
    return _scale(out, rng)


def steps(rng, n):
    out = np.empty((n, NX))
    for i in range(n):
        k = int(rng.integers(1, 5))
        cuts = np.sort(rng.uniform(0, 1, k + 1))
        lev = rng.uniform(-2.5, 2.5, k + 2)
        w = rng.choice([0.0, 0.01, 0.03])
        u = np.full(NX, lev[0])
        for j, c in enumerate(cuts):
            jump = lev[j + 1] - lev[j]
            u = u + jump * ((X > c).astype(float) if w == 0 else 0.5 * (1 + np.tanh((X - c) / w)))
        out[i] = u
    return _scale(out, rng)


def bumps(rng, n):
    out = np.empty((n, NX))
    for i in range(n):
        u = np.full(NX, rng.normal(0, 0.3))
        for _ in range(int(rng.integers(1, 5))):
            c, w, a = rng.uniform(0, 1), rng.uniform(0.02, 0.15), rng.uniform(-2.5, 2.5)
            d = np.minimum(np.abs(X - c), 1 - np.abs(X - c))
            u = u + a * np.exp(-(d / w) ** 2)
        out[i] = u
    return _scale(out, rng)


def mixed(rng, n):
    fs = [fourier, drawn, steps, bumps]
    out = np.empty((n, NX))
    for i in range(n):
        a, b = rng.choice(4, 2, replace=False)
        out[i] = rng.uniform(0.3, 1.0) * fs[a](rng, 1)[0] + rng.uniform(0.3, 1.0) * fs[b](rng, 1)[0]
    return _scale(out, rng)


# ------------------------------------------------------------------ UNSEEN styles (never used for training)
def gp(rng, n):                       # smooth random fields with a power-law spectrum
    out = np.empty((n, NX))
    for i in range(n):
        alpha = rng.uniform(1.0, 4.0)
        k = np.arange(1, 65)
        c = (rng.standard_normal(64) + 1j * rng.standard_normal(64)) * k ** (-alpha / 2)
        F = np.zeros(NX // 2 + 1, dtype=complex)
        F[1:65] = c
        out[i] = np.fft.irfft(F, n=NX) * NX + rng.normal(0, 0.3)
    return _scale(out, rng)


def saw(rng, n):                      # sawtooth and triangle waves
    out = np.empty((n, NX))
    for i in range(n):
        f, ph = int(rng.integers(1, 7)), rng.uniform(0, 1)
        fr = (f * X + ph) % 1.0
        u = 2 * fr - 1 if rng.random() < 0.5 else 2 * np.abs(2 * fr - 1) - 1
        out[i] = u * rng.uniform(0.5, 2.5) + rng.normal(0, 0.3)
    return _scale(out, rng)


def fm(rng, n):                       # frequency-modulated sine
    out = np.empty((n, NX))
    for i in range(n):
        f, g, beta, ph = int(rng.integers(1, 4)), int(rng.integers(1, 3)), rng.uniform(1, 6), rng.uniform(0, 2 * np.pi)
        out[i] = np.sin(2 * np.pi * f * X + beta * np.sin(2 * np.pi * g * X + ph)) + rng.normal(0, 0.3)
    return _scale(out, rng)


def packet(rng, n):                   # a smooth swell plus a localised high-frequency wave packet
    out = np.empty((n, NX))
    for i in range(n):
        c, w, k = rng.uniform(0, 1), rng.uniform(0.05, 0.2), int(rng.integers(8, 31))
        d = np.minimum(np.abs(X - c), 1 - np.abs(X - c))
        out[i] = rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * X + rng.uniform(0, 6.28)) + rng.uniform(0.3, 1.0) * np.exp(-(d / w) ** 2) * np.sin(2 * np.pi * k * X + rng.uniform(0, 6.28))
    return _scale(out, rng)


TRAIN_MIX = [("fourier", fourier, 0.25), ("drawn", drawn, 0.30), ("steps", steps, 0.15), ("bumps", bumps, 0.15), ("mixed", mixed, 0.15)]
UNSEEN = {"gp": gp, "saw": saw, "fm": fm, "packet": packet}


def train_mix(rng, n):
    """n waves drawn from the TRAIN styles with the mixture proportions above; returns (waves, style names)."""
    counts = rng.multinomial(n, [p for _, _, p in TRAIN_MIX])
    ws, names = [], []
    for (name, fn, _), c in zip(TRAIN_MIX, counts):
        if c:
            ws.append(fn(rng, int(c)))
            names += [name] * int(c)
    u = np.concatenate(ws)
    idx = rng.permutation(len(u))
    return u[idx], [names[i] for i in idx]
