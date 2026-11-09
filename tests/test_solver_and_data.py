import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import h5py
import numpy as np

from src.solver import solve, mass
from src.data import make_split, read_samples


def _grid(nx, L=1.0):
    return (np.arange(nx) + 0.5) * L / nx


def test_pure_diffusion_matches_exact_decay():
    # a=0 turns Burgers into the heat equation: sin(2 pi x) decays as exp(-nu (2 pi)^2 t)
    nu, t_end, nx = 0.01, 0.5, 128
    x = _grid(nx)
    u = solve(np.sin(2 * np.pi * x), nu, np.array([t_end]), a=0.0)[0]
    exact = np.exp(-nu * (2 * np.pi) ** 2 * t_end) * np.sin(2 * np.pi * x)
    assert np.abs(u - exact).max() < 2e-3


def test_mass_is_conserved_through_a_shock():
    nx = 256
    x = _grid(nx)
    u0 = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
    out = solve(u0, 0.002, np.linspace(0, 1.0, 11))
    m = mass(out)
    assert np.abs(m - m[0]).max() < 1e-12


def test_grid_refinement_converges():
    # coarse (128) vs fine (512) vs finer (2048) reference: error shrinks with resolution
    t = np.array([0.5])
    def run(nx):
        x = _grid(nx)
        return solve(np.sin(2 * np.pi * x), 0.01, t)[0]
    ref = run(2048)
    def err(nx):
        r = nx_ratio = 2048 // nx
        return np.abs(run(nx) - ref.reshape(nx, r).mean(axis=1)).mean()
    assert err(512) < err(128)


def test_split_is_disjoint_deterministic_and_prefix_stable():
    a, b = make_split(10000, seed=0), make_split(10000, seed=0)
    assert a == b
    assert len(a["test"]) == 1000 and len(a["val"]) == 500 and len(a["train_order"]) == 8500
    assert not set(a["test"]) & set(a["train_order"])
    assert make_split(10000, seed=1)["test"] != a["test"]


def test_read_samples_order_and_stride(tmp_path):
    data = np.arange(7 * 5 * 8, dtype=np.float32).reshape(7, 5, 8)
    p = tmp_path / "toy.hdf5"
    with h5py.File(p, "w") as f:
        f["tensor"] = data
    got = read_samples(str(p), [5, 1, 3], x_stride=2, t_stride=2, chunk=2)
    assert np.array_equal(got, data[[5, 1, 3]][:, ::2, ::2])
