"""H56: a time-step sequence (fixed geometry, drifting k(T) field, new right-hand side each step) solved by
  rebuild : bt-fd / bt-fdd rebuilt every step + GMRES(m)                      (what H53/H55 would do)
  frozen  : preconditioner built at step 0 and kept + GMRES(m)               (factorisation reuse)
  rec<k>  : frozen preconditioner + GCRO-DR recycling k vectors               (Krylov recycling)
  amg     : BoomerAMG rebuilt every step, PETSc GMRES(50)                    (yardstick only)
  amgf    : BoomerAMG set up at step 0 and reused, PETSc GMRES(50)          (yardstick only)
usage: seq.py BASE STEPS METHODS GEOMETRY [tol] [restart]
  BASE fd | fdd; GEOMETRY m:n:p:aspect:width:delta:seed (gen_gen); step s uses phase 0.15 s, rhs seed 100 + s."""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real  # noqa: E402
from btfd import BTFD  # noqa: E402
from gcrodr import Recycler  # noqa: E402
from gen_gen import generate  # noqa: E402

base, S, methods, geom = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-8
restart = int(sys.argv[6]) if len(sys.argv) > 6 else 50   # m=128 (n = 6.3M): 30, so V, Z = P V and U, C, Y fit in RAM
m_, n_, p_, a_, w_, d_, sd_ = geom.split(":")
mats = lambda s: generate(int(m_), int(n_), float(p_), float(a_), float(w_), float(d_), int(sd_),  # noqa: E731
                          phase=0.15 * s, rhs_seed=100 + s)
multi = "defl" if base == "fdd" else False
P0 = BTFD(*mats(0)[:1], multi=multi, tol=0.1)
print(f"{geom} {base}: feature S3 {len(P0.p3.F) / P0.N:.1%} A11 {len(P0.p1.F) / P0.N:.1%}", flush=True)
del P0
for meth in methods:
    tot, its_all, tset, worst = 0.0, [], 0.0, 0.0
    rec = None
    for s in range(S):
        A, b = mats(s)
        t0 = perf_counter()
        if meth == "amgf":                                   # BoomerAMG set up at step 0 and reused (fair yardstick)
            from petsc4py import PETSc
            from bt_elim import amg_defaults
            Mp = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
            if s == 0:
                ksp = PETSc.KSP().create(); ksp.setType("gmres"); ksp.setGMRESRestart(50)
                ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                ksp.getPC().setType("hypre"); ksp.getPC().setHYPREType("boomeramg"); ksp.setOptionsPrefix("amgf_")
                amg_defaults("amgf_")
                ksp.setTolerances(rtol=tol, atol=0.0, max_it=500); ksp.setFromOptions()
                ksp.setOperators(Mp); ksp.setUp(); tset += perf_counter() - t0
            else:
                ksp.getPC().setReusePreconditioner(True); ksp.setOperators(Mp)
            xv, rv = Mp.createVecs(); rv.setArray(b); ksp.solve(rv, xv)
            its_all.append(ksp.getIterationNumber()); tot += perf_counter() - t0
            continue
        if meth == "amg":
            r = run_real.run(A, b, "amg-3d", tol)
            its_all.append(r["its"]); tot += r["total_s"]; tset += r["setup_s"]
            continue
        if meth == "rebuild" or s == 0:
            P = BTFD(A, multi=multi, tol=0.1)                 # tol 0.1: smooth k(T) drift stays background (H56)
            tset += perf_counter() - t0
        k = int(meth[3:]) if meth.startswith("rec") else 0
        if rec is None or meth == "rebuild":
            rec = Recycler(A, P, m=restart, k=k)
        else:
            rec.set_matrix(A)
        x, its, hist = rec.solve(b, tol)
        d = A.diagonal()
        true = np.linalg.norm((b - A @ x) / d) / np.linalg.norm(b / d)        # diagonally scaled true residual
        tot += perf_counter() - t0
        its_all.append(its)
        worst = max(worst, true)
    print(f"{geom} {base} {meth:8s} its/step {its_all} total {tot:.1f} s (setup {tset:.1f})" + (f" max scaled true res {worst:.1e}" if not meth.startswith("amg") else ""), flush=True)
