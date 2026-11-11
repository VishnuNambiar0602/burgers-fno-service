"""Starting waves for agents: the same presets as the web page. 'random' and 'noise' use numpy's generator, so a given seed is
reproducible here but is NOT identical to the same seed on the web page (which uses a different generator)."""
import numpy as np

NX = 256

PRESET_INFO = {
    "sine": {"label": "Smooth wave", "description": "One gentle wave: u = sin(2 pi x).", "stress_test": False},
    "two": {"label": "Two waves", "description": "0.8 sin(4 pi x + 0.5) + 0.5 sin(10 pi x + 1.2): two waves of different speeds, sharper fronts.", "stress_test": False},
    "bump": {"label": "Bump", "description": "One smooth Gaussian bump centred at x = 0.5.", "stress_test": False},
    "square": {"label": "Square", "description": "A square wave (+1 on 0.25 < x < 0.75, -1 elsewhere): sudden jumps.", "stress_test": False},
    "random": {"label": "Random mix", "description": "A random mix of 1 to 3 sine waves (wavenumber 1-8) plus a small mean. Use 'seed' for a different one.", "stress_test": False},
    "tall": {"label": "Too tall", "description": "5 sin(2 pi x): higher than anything in training (largest 3.35). Raises the out-of-distribution warning.", "stress_test": True},
    "noise": {"label": "Noise", "description": "White noise with standard deviation 0.3. Raises the warning.", "stress_test": True},
    "wiggly": {"label": "Very wiggly", "description": "sin(2 pi 40 x): 40 oscillations, far finer than the training waves. Raises the warning.", "stress_test": True},
}


def grid(nx=NX):
    return (np.arange(nx) + 0.5) / nx


def sine_mix(components, mean=0.0, nx=NX):
    """u(x) = mean + sum amp * sin(2 pi k x + phase) on the cell-centred periodic grid."""
    x = grid(nx)
    u = np.full(nx, float(mean))
    for c in components:
        u = u + float(c["amp"]) * np.sin(2 * np.pi * float(c["k"]) * x + float(c.get("phase", 0.0)))
    return u


def preset(key, seed=1, nx=NX):
    x = grid(nx)
    if key == "sine":
        return sine_mix([{"amp": 1, "k": 1, "phase": 0}], 0, nx)
    if key == "two":
        return sine_mix([{"amp": 0.8, "k": 2, "phase": 0.5}, {"amp": 0.5, "k": 5, "phase": 1.2}], 0, nx)
    if key == "bump":
        return np.exp(-((x - 0.5) / 0.08) ** 2)
    if key == "square":
        return np.where((x > 0.25) & (x < 0.75), 1.0, -1.0)
    if key == "random":
        r = np.random.default_rng(int(seed))
        comps, tot = [], 0.0
        for _ in range(int(r.integers(1, 4))):
            a = 0.3 + 0.9 * r.random()
            tot += a
            comps.append({"amp": a, "k": int(r.integers(1, 9)), "phase": 2 * np.pi * r.random()})
        s = min(1.0, 2.4 / tot)                       # keep the largest value inside the training range
        for c in comps:
            c["amp"] *= s
        return sine_mix(comps, (r.random() - 0.5) * 0.8, nx)
    if key == "tall":
        return sine_mix([{"amp": 5, "k": 1, "phase": 0}], 0, nx)
    if key == "noise":
        return 0.3 * np.random.default_rng(int(seed)).standard_normal(nx)
    if key == "wiggly":
        return sine_mix([{"amp": 1, "k": 40, "phase": 0}], 0, nx)
    raise KeyError(key)
