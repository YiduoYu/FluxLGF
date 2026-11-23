"""H56 locked test: time-step sequences with setup reuse. Same Krylov for both methods (PETSc GMRES(50), right
preconditioning, x0 = 0, solver-reported rtol 1e-8, single thread):
  bt-fd   preconditioner (tol 0.1 matching) built at step 0 from A_0 and kept for all steps
  amg     BoomerAMG (AMG3D settings) set up at step 0 and kept (PCSetReusePreconditioner)
Each step generates the matrix once (not timed) and solves it with both methods.
usage: seq_lock.py OUT.csv STEPS GEOMETRY      GEOMETRY = m:n:p:aspect:width:delta:seed (gen_gen; phase 0.15 s,
rhs seed 100 + s)"""
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from bt_elim import amg_defaults  # noqa: E402
from btfd import BTFD  # noqa: E402
from gen_gen import generate  # noqa: E402

out, S, geom = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m_, n_, p_, a_, w_, d_, sd_ = geom.split(":")


def mats(s):
    return generate(int(m_), int(n_), float(p_), float(a_), float(w_), float(d_), int(sd_), phase=0.15 * s,
                    rhs_seed=100 + s)


def new_ksp(prefix):
    ksp = PETSc.KSP().create()
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(rtol=1e-8, atol=0.0, max_it=500)
    ksp.setOptionsPrefix(prefix)
    return ksp


ksps, rows = {}, []
for s in range(S):
    A, b = mats(s)
    d = A.diagonal()
    Mp = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    for meth in ("bt-fd", "amg"):
        t0 = perf_counter()
        if s == 0:
            ksp = new_ksp(meth.replace("-", "") + "_")
            pc = ksp.getPC()
            if meth == "amg":
                pc.setType("hypre"); pc.setHYPREType("boomeramg")
                amg_defaults("amg_")
            else:
                P = BTFD(A, multi=False, tol=0.1)
                pc.setType("python"); pc.setPythonContext(P)
                fF = max(len(P.p3.F), len(P.p1.F)) / P.N
            ksp.setFromOptions()
            ksp.setOperators(Mp)
            ksp.setUp()
            ksps[meth] = ksp
        else:
            ksp = ksps[meth]
            ksp.getPC().setReusePreconditioner(True)
            ksp.setOperators(Mp)
        t1 = perf_counter()
        xv, rv = Mp.createVecs(); rv.setArray(b)
        ksp.solve(rv, xv)
        t2 = perf_counter()
        x = xv.getArray()
        rows.append(dict(geometry=geom, step=s, method=meth, its=ksp.getIterationNumber(),
                         reason=int(ksp.getConvergedReason()), setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0,
                         scaled=np.linalg.norm((b - A @ x) / d) / np.linalg.norm(b / d), feature=fF))
        print(f"{geom} step {s} {meth:6s} its {rows[-1]['its']:3d} reason {rows[-1]['reason']} setup {t1 - t0:.1f} "
              f"solve {t2 - t1:.1f} s scaled {rows[-1]['scaled']:.1e}", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tot = {mm: sum(r["total_s"] for r in rows if r["method"] == mm) for mm in ("bt-fd", "amg")}
print(f"{geom} TOTAL bt-fd {tot['bt-fd']:.1f} s, amg {tot['amg']:.1f} s, ratio {tot['bt-fd'] / tot['amg']:.2f}, "
      f"feature {fF:.1%}", flush=True)
