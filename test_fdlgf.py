"""Checks for FD-LGF. Run from anywhere:  /opt/anaconda3/envs/hypre_env/bin/python -m pytest test_method2.py"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "CNN function")]  # axy_lgf package lives there

import numpy as np

from axy_lgf.grid import SquareGrid
from axy_lgf.operators import assemble_reaction_diffusion
from cases import make_case
from fdlgf import FDLGFPreconditioner, solve_fdlgf_pcg, split_coefficient


def _field(n, a_of_xy, q_of_xy):
    grid = SquareGrid(n, 0.0, 1.0)
    nodes = np.arange(n + 2) * grid.h
    X, Y = np.meshgrid(nodes, nodes, indexing="xy")
    a_nodes = a_of_xy(X, Y)
    a = np.ascontiguousarray(a_nodes[1:-1, 1:-1])
    x, y = grid.mesh()
    q = q_of_xy(x, y) * np.ones_like(x)
    A = assemble_reaction_diffusion(a, grid, q, a_nodes=a_nodes).tocsr()
    return grid, a, a_nodes, q, A


def _exactness(a_of_xy, q_of_xy, n=31):
    grid, a, a_nodes, q, A = _field(n, a_of_xy, q_of_xy)
    P = FDLGFPreconditioner(A, a_nodes, q, grid.h)
    u = np.random.default_rng(0).standard_normal((n, n))
    return P, np.linalg.norm(P((A @ u.ravel()).reshape(n, n)) - u) / np.linalg.norm(u)


def test_exact_for_x_only_coefficient_dst_path():
    P, err = _exactness(lambda X, Y: 1e6**X, lambda x, y: 1 + x**2)
    assert P.G.kind.startswith("DST") and not P.solves and err < 1e-9


def test_exact_for_y_only_coefficient_transposed_path():
    P, err = _exactness(lambda X, Y: 1e6**Y, lambda x, y: 1 + y**2)
    assert P.G.kind == "DST-x + tridiag-y" and err < 1e-9


def test_exact_for_separable_coefficient_fdm_path():
    # q proportional to a2(y): the separable reaction term R = mean_j(q / a2) is then exact
    P, err = _exactness(lambda X, Y: 1e3**X * 1e2 ** np.sin(np.pi * Y), lambda x, y: 1e2 ** np.sin(np.pi * y))
    assert P.G.kind == "fast diagonalisation" and err < 1e-9


def test_split_recovers_inclusions_and_layers():
    # C at n=127: discs cover 63 of 129 nodes of the worst column (< half); at n=63 it is 33 of 65 and the fit fails
    for case, n in (("C_exp+incl", 127), ("D_exp+rand", 63)):
        c = make_case(case, n, 1e4)
        a1, a2, mask = split_coefficient(c.a_nodes)
        assert np.array_equal(mask[1:-1, 1:-1], c.mask)
        assert np.allclose(a1 * a2[0], c.bg_a1) and np.ptp(np.log(a2)) < 1e-12


def test_symmetric_positive_with_features():
    c = make_case("C_exp+incl", 31, 1e6)
    P = FDLGFPreconditioner(c.A, c.a_nodes, c.q, c.grid.h)
    x, y = np.random.default_rng(1).standard_normal((2, 31, 31))
    assert abs(np.sum(x * P(y)) - np.sum(P(x) * y)) < 1e-10 * np.linalg.norm(P(x)) * np.linalg.norm(y)
    assert np.sum(x * P(x)) > 0


def test_converges_fast_on_layers_with_inclusions():
    c = make_case("C_exp+incl", 63, 1e6)
    res = solve_fdlgf_pcg(c.a, c.f, c.grid, q=c.q, a_nodes=c.a_nodes, tol=1e-10)
    assert res.converged and res.iterations <= 12
