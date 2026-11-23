import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "CNN function")]

import numpy as np
from scipy.sparse.linalg import eigsh

from axy_lgf.grid import SquareGrid
from axy_lgf.operators import assemble_reaction_diffusion
from cases_aniso import FAMILIES, make_case
from fdlgf_tensor import AnisoBackground, FDLGFTensor
from ops_aniso import assemble


def test_isotropic_tensor_equals_scalar_flux_stencil():
    n = 15
    t = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(t, t, indexing="xy")
    a = np.exp(np.sin(3 * X) + Y**2)
    q = 1 + X[1:-1, 1:-1]
    A = assemble(a, 0 * a, a, q, 1 / (n + 1))
    B = assemble_reaction_diffusion(a[1:-1, 1:-1], SquareGrid(n, 0.0, 1.0), q, a_nodes=a)
    assert abs(A - B).max() < 1e-9 * abs(B).max()


def test_symmetric_positive_definite_all_families():
    for fam in FAMILIES:
        c = make_case(fam, 31, 1e6)
        assert abs(c.A - c.A.T).max() < 1e-12 * abs(c.A).max()
        assert eigsh(c.A, k=1, sigma=0, which="LM")[0][0] > 0.5      # >= min q = 0.8


def test_background_exact_on_aligned_with_constant_q():
    n, h = 31, 1 / 32
    k1, k2 = np.linspace(1, 50, n + 2), np.full(n + 2, 3.0)
    q = np.full((n, n), 0.7)
    K11, K22 = np.tile(k1, (n + 2, 1)), np.tile(k2, (n + 2, 1))
    A = assemble(K11, 0 * K11, K22, q, h)
    u = np.random.default_rng(0).standard_normal((n, n))
    assert np.abs(AnisoBackground(k1, k2, q, h)((A @ u.ravel()).reshape(n, n)) - u).max() < 1e-10


def test_fdlgf_tensor_symmetric_and_exact_when_mask_covers_grid():
    c = make_case("disk", 31, 1e4)
    G = AnisoBackground(*c.bg, c.q, c.h)
    P = FDLGFTensor(c.A, G, c.mask, w=1 / 16)
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2, 31, 31))
    assert abs(np.sum(P(x) * y) - np.sum(x * P(y))) < 1e-10 * np.sum(np.abs(P(x) * y))
    full = FDLGFTensor(c.A, G, np.ones((31, 31), bool))
    assert np.abs(full(c.f) - c.u_h).max() < 1e-10
