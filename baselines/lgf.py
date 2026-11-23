"""传统 LGF: PCG preconditioned by the Dirichlet lattice Green's function of a0(-Delta_h) + sigma I.

Copied from axy_lgf.lgf_inverse.DirichletLGF and axy_lgf.solvers.solve_axy_pcg (same defaults:
a0 = mean(a), sigma = mean(q)). The inverse is exact for constant coefficients and applied with a 2D DST-I.
"""

from time import perf_counter

import numpy as np
from scipy.fft import dstn, idstn

from axy_lgf.operators import apply_reaction_diffusion, reaction_field
from axy_lgf.solvers import _pcg, _validate_problem


def laplacian_eigenvalues(n, h):
    """Eigenvalues of -Delta_h (5-point, Dirichlet) for modes (p, q), p, q = 1..n."""
    lam1d = (2.0 / h**2) * (1.0 - np.cos(np.arange(1, n + 1) * np.pi / (n + 1)))
    return lam1d[:, None] + lam1d[None, :]


class DirichletLGF:
    """u = (a0(-Delta_h) + sigma I)^-1 rhs on n x n arrays."""

    def __init__(self, grid, a0=1.0, sigma=0.0):
        self.denominator = a0 * laplacian_eigenvalues(grid.n, grid.h) + sigma

    def __call__(self, rhs):
        return idstn(dstn(rhs, type=1, norm="ortho") / self.denominator, type=1, norm="ortho")


def solve_lgf_pcg(a, f, grid, *, q=0.0, a_nodes=None, tol=1e-10, maxiter=500, u0=None, a0=None):
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    a0 = float(np.mean(a)) if a0 is None else float(a0)
    qq = reaction_field(q, grid)
    pinv = DirichletLGF(grid, a0=a0, sigma=float(np.mean(qq)))
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, qq, a_nodes=a_nodes)
    result = _pcg(action, pinv, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=a0)
    result.setup_time = setup_time
    return result
