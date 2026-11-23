"""The copied baselines must reproduce axy_lgf's originals exactly.
torch_env: python -m pytest -q test_baselines.py (BoomerAMG test skipped without petsc4py)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "CNN function")]

import numpy as np
import pytest

from axy_lgf.multigrid import solve_multigrid_pcg
from axy_lgf.solvers import solve_axy_pcg
from baselines import solve_gmg_pcg, solve_lgf_pcg
from cases import make_case


@pytest.mark.parametrize("case", ["A_exp", "C_exp+incl"])
def test_lgf_and_gmg_identical_to_axy_lgf(case):
    """GMG with omega=1.0 is axy_lgf's original; the benchmark uses the default omega=1.6 (textbook 4/5)."""
    c = make_case(case, 63, 1e4)
    kw = dict(q=c.q, a_nodes=c.a_nodes, tol=1e-10, maxiter=300)
    for new, old in ((solve_lgf_pcg(c.a, c.f, c.grid, **kw), solve_axy_pcg(c.a, c.f, c.grid, **kw)),
                     (solve_gmg_pcg(c.a, c.f, c.grid, omega=1.0, **kw),
                      solve_multigrid_pcg(c.a, c.f, c.grid, method="gmg", **kw))):
        assert new.iterations == old.iterations and np.array_equal(new.u, old.u)


def test_gmg_default_weight_is_four_fifths_and_spd():
    from axy_lgf.grid import SquareGrid
    from axy_lgf.operators import assemble_reaction_diffusion
    from baselines import GeometricVcycle
    n = 31
    A = assemble_reaction_diffusion(np.ones((n, n)), SquareGrid(n, 0.0, 1.0), 0.0, a_nodes=np.ones((n + 2, n + 2)))
    G = GeometricVcycle(A, n)
    assert np.isclose(G.smoothers[0][0] * A.diagonal()[0], 0.8)
    x, y = np.random.default_rng(0).standard_normal((2, n, n))
    assert abs(np.sum(x * G(y)) - np.sum(y * G(x))) < 1e-12 * np.linalg.norm(G(x)) * np.linalg.norm(y)
    assert np.sum(x * G(x)) > 0


def test_boomeramg_converges():
    pytest.importorskip("petsc4py")
    from baselines.boomeramg import solve_boomeramg_pcg
    c = make_case("C_exp+incl", 63, 1e4)
    res = solve_boomeramg_pcg(c.a, c.f, c.grid, q=c.q, a_nodes=c.a_nodes, tol=1e-10)
    assert res.converged and res.iterations <= 15
