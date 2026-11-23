"""hypre BoomerAMG (via petsc4py) as the AMG baseline. Needs /opt/anaconda3/envs/hypre_env.

Drop-in for solve_multigrid_pcg(method="amg"): same assembly, same shared _pcg loop and stopping rule,
only the V-cycle is hypre's (PETSc defaults: Falgout coarsening, classical interpolation,
symmetric hybrid SOR/Jacobi smoothing, direct coarse solve).
"""

from time import perf_counter

import numpy as np
from petsc4py import PETSc

from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion
from axy_lgf.solvers import _pcg, _validate_problem


class BoomerAMG:
    """One BoomerAMG V-cycle per call; acts on flat vectors (M @ r) or n x n arrays (M(r))."""

    def __init__(self, matrix):
        A = matrix.tocsr()
        self.mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        self.pc = PETSc.PC().create()
        self.pc.setOperators(self.mat)
        self.pc.setType("hypre")
        self.pc.setHYPREType("boomeramg")
        self.pc.setUp()
        self.x, self.y = self.mat.createVecs()

    def __matmul__(self, r):
        self.x.setArray(r)
        self.pc.apply(self.x, self.y)
        return self.y.getArray().copy()

    def __call__(self, r):
        return (self @ r.ravel()).reshape(r.shape)


def solve_boomeramg_pcg(a, f, grid, *, q=0.0, a_nodes=None, tol=1e-10, maxiter=500, u0=None):
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    pinv = BoomerAMG(assemble_reaction_diffusion(a, grid, q, a_nodes=a_nodes))
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, q, a_nodes=a_nodes)
    result = _pcg(action, pinv, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=float(np.mean(a)))
    result.setup_time = setup_time
    return result
