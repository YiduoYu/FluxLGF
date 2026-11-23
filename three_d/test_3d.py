"""Checks for the 3D code. torch_env: python -m pytest -q test_3d.py (BoomerAMG test needs petsc4py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from cases3d import make_case
from ops3d import apply_op, assemble
from solvers3d import GMG3, LGF3, DSTTridiag3, Method2_3D

rng = np.random.default_rng(0)


def test_assembled_matrix_matches_matrix_free_operator():
    c = make_case("D_exp+rand", 15, 1e4)
    u = rng.standard_normal((15,) * 3)
    assert np.allclose(c.A @ u.ravel(), apply_op(u, c.a_nodes, c.q, c.grid.h).ravel(), rtol=1e-12, atol=0)
    assert abs(c.A - c.A.T).max() == 0


def test_lgf_exact_for_constant_coefficient():
    n, h = 15, 1 / 16
    A = assemble(np.full((n + 2,) * 3, 3.7), 2.5, h)
    u = rng.standard_normal((n,) * 3)
    assert np.linalg.norm(LGF3(n, h, 3.7, 2.5)((A @ u.ravel()).reshape(u.shape)) - u) < 1e-12 * np.linalg.norm(u)


def test_background_inverse_exact_for_layers():
    n, h = 15, 1 / 16
    a1 = 1e6 ** (np.arange(n + 2) * h)
    q = np.broadcast_to(1 + np.arange(1, n + 1) * h, (n, n, n))      # q = q(x): exact
    A = assemble(np.broadcast_to(a1, (n + 2,) * 3).copy(), q, h)
    u = rng.standard_normal((n,) * 3)
    assert np.linalg.norm(DSTTridiag3(a1, q, h)((A @ u.ravel()).reshape(u.shape)) - u) < 1e-9 * np.linalg.norm(u)


@pytest.mark.parametrize("build", ["gmg", "m2"])
def test_preconditioners_symmetric_positive(build):
    c = make_case("C_exp+incl", 15, 1e6)
    P = GMG3(c.A, 15) if build == "gmg" else Method2_3D(c.A, c.bg_a1, c.mask, c.q, c.grid.h, 1 / 32)
    x, y = rng.standard_normal((2, 15, 15, 15))
    assert abs(np.sum(x * P(y)) - np.sum(y * P(x))) < 1e-11 * np.linalg.norm(P(x)) * np.linalg.norm(y)
    assert np.sum(x * P(x)) > 0


def test_gmg_weight_is_six_sevenths_on_laplacian():
    A = assemble(np.ones((17,) * 3), 0.0, 1 / 16)
    assert np.isclose(GMG3(A, 15).smooth[0][0] * A.diagonal()[0], 6 / 7)


def test_boomeramg_symmetric():
    pytest.importorskip("petsc4py")
    from solvers3d import BoomerAMG3
    c = make_case("C_exp+incl", 15, 1e6)
    P = BoomerAMG3(c.A)
    x, y = rng.standard_normal((2, 15, 15, 15))
    assert abs(np.sum(x * P(y)) - np.sum(y * P(x))) < 1e-11 * np.linalg.norm(P(x)) * np.linalg.norm(y)
