"""GMG: PCG preconditioned by one geometric V-cycle. Copied from axy_lgf.multigrid (GeometricVcycle,
solve_multigrid_pcg(method="gmg")).

Bilinear interpolation P, restriction P^T, Galerkin coarse matrices P^T A P, two damped-Jacobi sweeps before
and after the coarse correction, Cholesky on the coarsest 3 x 3 grid. Needs n = 2^k - 1.

Jacobi weight omega_level = omega / (max row sum of D^-1 |A|). The default omega = 1.6 gives the textbook 4/5 on the
5-point Laplacian (row sum 2; minimises the high-frequency smoothing factor, mu = 3/5 — Briggs, Henson & McCormick
2000; Trottenberg et al. 2001). Since lambda_max(D^-1 A) <= row sum, omega_level * lambda_max <= 1.6 < 2: the smoother
converges and the V-cycle stays SPD. omega = 1.0 reproduces axy_lgf.multigrid exactly (weight 1/2 on the Laplacian).
"""

from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, kron

from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion
from axy_lgf.solvers import _pcg, _validate_problem


def geometric_interpolation(n):
    """Bilinear interpolation from ((n-1)/2)^2 to n^2 interior nodes (boundary values zero)."""
    nc = (n - 1) // 2
    j = np.arange(nc)
    rows = np.concatenate([2 * j, 2 * j + 1, 2 * j + 2])
    cols = np.tile(j, 3)
    weights = np.repeat([0.5, 1.0, 0.5], nc)
    p1 = coo_matrix((weights, (rows, cols)), shape=(n, nc)).tocsr()
    return kron(p1, p1, format="csr")


class GeometricVcycle:
    def __init__(self, matrix, n, omega=1.6):
        if n < 3 or (n + 1) & n:
            raise ValueError("GMG requires n=2^k-1 with k>=2")
        self.matrices, self.transfers, self.smoothers = [matrix.tocsr()], [], []
        while n > 3:
            aa = self.matrices[-1]
            diagonal = aa.diagonal()
            bound = float(np.max(np.asarray(abs(aa).sum(axis=1)).ravel() / diagonal))
            self.smoothers.append((omega / bound) / diagonal)
            pp = geometric_interpolation(n)
            self.transfers.append(pp)
            self.matrices.append((pp.T @ aa @ pp).tocsr())
            n = (n - 1) // 2
        self.coarse_factor = cho_factor(self.matrices[-1].toarray(), lower=True)

    def _cycle(self, level, rhs):
        if level == len(self.matrices) - 1:
            return cho_solve(self.coarse_factor, rhs)
        aa, pp, scale = self.matrices[level], self.transfers[level], self.smoothers[level]
        u = np.zeros_like(rhs)
        for _ in range(2):
            u += scale * (rhs - aa @ u)
        u += pp @ self._cycle(level + 1, pp.T @ (rhs - aa @ u))
        for _ in range(2):
            u += scale * (rhs - aa @ u)
        return u

    def __call__(self, rhs):
        return self._cycle(0, rhs.ravel()).reshape(rhs.shape)


def solve_gmg_pcg(a, f, grid, *, q=0.0, a_nodes=None, tol=1e-10, maxiter=500, u0=None, omega=1.6):
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    pinv = GeometricVcycle(assemble_reaction_diffusion(a, grid, q, a_nodes=a_nodes), grid.n, omega=omega)
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    result = _pcg(action, pinv, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=float(np.mean(a)))
    result.setup_time = setup_time
    return result
