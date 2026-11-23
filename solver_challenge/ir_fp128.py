"""Mixed-precision iterative refinement for the real SolverChallenge26_01 matrix.

The official harness can be built with SC26_SOLUTION_FP128 (solution vector in __float128, residual checked in float128),
so the float64 representation floor of ||b - Ax||/||b|| (~1e-2, H39) does not bind. Here x is kept in double-double
(~32 digits, enough for the check), r = b - A x is computed in double-double, and each correction A d = r is solved in
float64 by GMRES + BoomerAMG (amg) or GMRES + the H41 block-triangular preconditioner (bt-v).
usage: python ir_fp128.py A.npz b.npy [inner_rtol] [amg|bt-v|bt-g]
"""
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

SPLIT = 134217729.0                                             # 2^27 + 1 (Dekker)


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def two_prod(a, b):
    p = a * b
    ca = SPLIT * a; ah = ca - (ca - a); al = a - ah
    cb = SPLIT * b; bh = cb - (cb - b); bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


class DDResidual:
    """r = b - A (xh + xl) in double-double, rows stored as ELL slots."""

    def __init__(self, A, b):
        n = A.shape[0]
        nnz = np.diff(A.indptr)
        self.k = int(nnz.max())
        self.cols = np.zeros((self.k, n), dtype=np.int64)
        self.vals = np.zeros((self.k, n))
        pos = np.arange(A.nnz) - np.repeat(A.indptr[:-1], nnz)
        rows = np.repeat(np.arange(n), nnz)
        self.cols[pos, rows] = A.indices
        self.vals[pos, rows] = A.data
        self.b = b

    def __call__(self, xh, xl):
        sh = np.zeros_like(self.b); sl = np.zeros_like(self.b)
        for s in range(self.k):
            a, c = self.vals[s], self.cols[s]
            p, e = two_prod(a, xh[c])
            sh, t = two_sum(sh, p)
            sl += t + e + a * xl[c]
        rh, t = two_sum(self.b, -sh)
        return rh + (t - sl)


def inner_solver(A, rtol, kind="amg"):
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if kind in ("bt-v", "bt-g"):
        from bt_elim import BTElim
        pc.setType("python"); pc.setPythonContext(BTElim(A, exact=False, fdm=kind == "bt-g"))
    else:
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        from bt_elim import amg_defaults
        amg_defaults("")
    ksp.setTolerances(rtol=rtol, max_it=500)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp, M.createVecs()


if __name__ == "__main__":
    A = sp.load_npz(sys.argv[1]).tocsr()
    b = np.load(sys.argv[2])
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    kind = sys.argv[4] if len(sys.argv) > 4 else "amg"
    nb = np.linalg.norm(b)
    res = DDResidual(A, b)
    x_amg = np.load(sys.argv[2].replace("b.npy", "x_amg.npy"))
    print(f"check: dd residual of the saved BoomerAMG iterate {np.linalg.norm(res(x_amg, 0 * x_amg)) / nb:.3e}"
          " (exact Fraction check gave 7.3e-3)")
    t0 = perf_counter()
    ksp, (dv, rv) = inner_solver(A, rtol, kind)
    t1 = perf_counter()
    xh = np.zeros_like(b); xl = np.zeros_like(b)
    r = b.copy()
    for step in range(1, 11):
        rv.setArray(r)
        ksp.solve(rv, dv)
        d = dv.getArray()
        xh, t = two_sum(xh, d); xl += t
        xh, xl = two_sum(xh, xl)
        r = res(xh, xl)
        rel = np.linalg.norm(r) / nb
        blk = [np.linalg.norm(r[k * len(b) // 3:(k + 1) * len(b) // 3]) / nb for k in range(3)]
        print(f"step {step}: inner its {ksp.getIterationNumber():3d}  ||b-Ax||/||b|| (dd) {rel:.3e}  "
              f"per block {blk[0]:.1e} {blk[1]:.1e} {blk[2]:.1e}  "
              f"float64-rounded x {np.linalg.norm(res(xh, 0 * xl)) / nb:.1e}  t {perf_counter() - t0:.1f} s", flush=True)
        if rel < 1e-9:
            break
    print(f"setup {t1 - t0:.1f} s, total {perf_counter() - t0:.1f} s (dd residual evaluations included)")
