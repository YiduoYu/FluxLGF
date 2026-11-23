"""Exact residual of the real matrix for a float64 vector x: rows whose rounding error could matter
(eps * (|A||x|)_i > 1e-3 * tol_abs) are summed exactly with fractions.Fraction; the rest in float64.
Also writes the BoomerAMG-3D GMRES iterate if no x is given.   usage: python exact_residual.py A.npz b.npy [x.npy]"""
import sys
from fractions import Fraction

import numpy as np
import scipy.sparse as sp


def exact_residual(A, b, x, tol_abs):
    r = b - A @ x
    absAx = abs(A) @ np.abs(x)
    risky = np.nonzero(np.finfo(float).eps * absAx * 50 > 1e-3 * tol_abs)[0]
    for i in risky:
        lo, hi = A.indptr[i], A.indptr[i + 1]
        s = Fraction(b[i]) - sum(Fraction(a) * Fraction(x[j]) for a, j in zip(A.data[lo:hi], A.indices[lo:hi]))
        r[i] = float(s)
    return r, risky.size


if __name__ == "__main__":
    A = sp.load_npz(sys.argv[1]).tocsr()
    b = np.load(sys.argv[2])
    if len(sys.argv) > 3:
        x = np.load(sys.argv[3])
    else:
        from petsc4py import PETSc
        M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create(); ksp.setOperators(M); ksp.setType("gmres"); ksp.setGMRESRestart(50)
        ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        pc = ksp.getPC(); pc.setType("hypre")
        ksp.setTolerances(rtol=1e-10, max_it=300); ksp.setFromOptions()
        xv, bv = M.createVecs(); bv.setArray(b); ksp.solve(bv, xv)
        print("GMRES+BoomerAMG its", ksp.getIterationNumber(), "reason", ksp.getConvergedReason(), "est", ksp.getResidualNorm() / np.linalg.norm(b))
        x = xv.getArray().copy(); np.save(sys.argv[2].replace("b.npy", "x_amg.npy"), x)
    nb = np.linalg.norm(b); N = A.shape[0] // 3
    r64 = b - A @ x
    r, nrisky = exact_residual(A, b, x, 1e-8 * nb)
    print(f"float64 residual {np.linalg.norm(r64)/nb:.3e}; exact residual {np.linalg.norm(r)/nb:.3e} ({nrisky} rows summed exactly)")
    print("exact residual per block:", [f"{np.linalg.norm(r[k*N:(k+1)*N])/nb:.2e}" for k in range(3)])
    print("float64 residual per block:", [f"{np.linalg.norm(r64[k*N:(k+1)*N])/nb:.2e}" for k in range(3)])
    print("|x| per block:", [f"{np.abs(x[k*N:(k+1)*N]).min():.2e}..{np.abs(x[k*N:(k+1)*N]).max():.2e}" for k in range(3)])
    floor = np.finfo(float).eps * (abs(A) @ np.abs(x))
    print(f"rounding floor eps*|| |A||x| || / ||b|| = {np.linalg.norm(floor)/nb:.2e}; rows with floor_i > 1e-8||b||/sqrt(n): {np.sum(floor > 1e-8*nb/np.sqrt(A.shape[0]))}")
