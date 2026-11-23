"""H43 regime sweep on our own 3T matrices (gen3t.py): amg-3d vs bt-v and its generalisations.
GMRES(50), right preconditioning, unpreconditioned residual, rtol 1e-8, x0 = 0, max 500 iterations, single thread.
Also logs solution-level regime numbers from the best converged solution: ||A21 x1|| / ||A22 x2|| and the share of the
electron off-diagonal part ||(A22 - D22) x2|| / ||D22 x2||.   usage: python run_gen.py OUT.csv m [F,G ...]
"""
import csv
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

import os

from bt_elim import BTElim, BTJoint, amg_defaults
from gen3t import coefficient_diagnostics, generate

METHODS = tuple(os.environ.get("RUN_METHODS", "amg-3d,bt-v,bt-v2,bt-v2x2").split(","))


def run(A, b, name, tol=1e-8, maxit=500):
    t0 = perf_counter()
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if name == "amg-3d":
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        ksp.setOptionsPrefix("g_amg_")
        amg_defaults("g_amg_")
    elif name in ("bt-j", "bt-jn"):                                     # H44 joint electron-ion
        pc.setType("python")
        pc.setPythonContext(BTJoint(A, nodal=name == "bt-jn"))
    else:
        pc.setType("python")
        pc.setPythonContext(BTElim(A, electron_amg=name in ("bt-v2", "bt-v2x2"), sweeps=2 if name == "bt-v2x2" else 1))
    ksp.setTolerances(rtol=tol, atol=0.0, max_it=maxit)
    ksp.setFromOptions()
    ksp.setUp()
    x, rhs = M.createVecs()
    rhs.setArray(b)
    t1 = perf_counter()
    ksp.solve(rhs, x)
    t2 = perf_counter()
    xs = x.getArray().copy()
    r = b - A @ xs
    d = A.diagonal()
    return dict(method=name, its=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                reported=ksp.getResidualNorm() / np.linalg.norm(b), true=np.linalg.norm(r) / np.linalg.norm(b),
                scaled=np.linalg.norm(r / d) / np.linalg.norm(b / d), setup_s=t1 - t0, solve_s=t2 - t1,
                total_s=t2 - t0), xs


if __name__ == "__main__":
    out, m = sys.argv[1], int(sys.argv[2])
    cases = [tuple(map(float, c.split(","))) for c in sys.argv[3:]] or \
        [(F, G) for F in (2e-11, 1e-6, 1e-3, 1e-1, 1.0) for G in (2e-4, 1e-2, 1.0, 1e2)]
    with open(out, "w", newline="") as fh:
        wr = None
        for F, G in cases:
            A, b = generate(m, F, G)
            cf, cg = coefficient_diagnostics(A)
            N = A.shape[0] // 3
            best, rows = None, []
            for name in METHODS:
                row, xs = run(A, b, name)
                if row["reason"] > 0 and (best is None or row["true"] < best[0]):
                    best = (row["true"], xs)
                row.update(m=m, F=F, G=G, coef_feedback=cf, coef_eoff=cg)
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row) + ["sol_feedback", "sol_eoff"])
                    wr.writeheader()
                rows.append(row)
                print(f"m={m} F={F:g} G={G:g} {name:8s} its {row['its']:3d} reason {row['reason']:2d} "
                      f"rep {row['reported']:.1e} scaled {row['scaled']:.1e} total {row['total_s']:.2f} s", flush=True)
            sf = se = float("nan")
            if best is not None:
                x1, x2 = best[1][:N], best[1][N:2 * N]
                A22 = A[N:2 * N, N:2 * N]; d22 = A22.diagonal()
                sf = np.linalg.norm(A[N:2 * N, :N] @ x1) / np.linalg.norm(A22 @ x2)
                se = np.linalg.norm((A22 - sp.diags(d22)) @ x2) / np.linalg.norm(d22 * x2)
            for row in rows:
                row.update(sol_feedback=sf, sol_eoff=se)
                wr.writerow(row)
            fh.flush()
            print(f"   regime: coef feedback {cf:.1e} e-offdiag {cg:.1e} | solution feedback {sf:.1e} e-offdiag {se:.1e}",
                  flush=True)
