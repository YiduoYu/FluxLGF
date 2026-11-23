"""Audit item 4: H56 G8-G11 again with each method in its own process, so one method's factors and Krylov space cannot
push the other into swap. Solver settings, sequences and metric are those of seq_lock.py (solver-reported rtol 1e-8,
total = setup at step 0 + all solves; matrix generation and PETSc matrix construction reported, not in the total).
Fixed before the run, on 2026-11-05: one pass; method order alternates to balance drift (G8 amg first, G9 bt-fd first,
G10 amg first, G11 bt-fd first); every run is kept, nothing is rejected or trimmed. A method that swaps on its own is a
real cost on this machine, so pageouts are reported next to the time, not used to discard it. Not a re-lock of H56, and
equal reported rtol is not equal accuracy (audit item 1): this checks the time multiples only.
usage: seq_isolated.py all | seq_isolated.py OUT_STEM METHOD STEPS GEOMETRY"""
import csv
import hashlib
import os
import re
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
GEOMETRIES = (("G8", "128:10:2:1:0:2e-2:1", ("amg", "bt-fd")), ("G9", "128:10:2:1:0:2e-2:2", ("bt-fd", "amg")),
              ("G10", "128:10:8:0.5:2:2e-2:3", ("amg", "bt-fd")), ("G11", "128:20:2:1:0:2e-2:1", ("bt-fd", "amg")))


def counters():
    txt = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Pageouts:\s+(\d+)", txt)[1]), int(re.search(r"Swapouts:\s+(\d+)", txt)[1])


def worker(stem, meth, S, geom):
    import numpy as np
    from petsc4py import PETSc
    sys.path[:0] = [str(HERE), str(HERE.parent / "solver_challenge")]
    from bt_elim import amg_defaults
    from btfd import BTFD
    from gen_gen import generate
    from provenance import feature_stats, save
    m_, n_, p_, a_, w_, d_, sd_ = geom.split(":")
    ksp, P, rows = None, None, []
    for s in range(S):
        t = perf_counter()
        A, b = generate(int(m_), int(n_), float(p_), float(a_), float(w_), float(d_), int(sd_), phase=0.15 * s, rhs_seed=100 + s)
        gen_s = perf_counter() - t
        before = counters()
        t = perf_counter()
        Mp = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        t0 = perf_counter()
        mat_s = t0 - t
        if s == 0:
            ksp = PETSc.KSP().create()
            ksp.setType("gmres"); ksp.setGMRESRestart(50)
            ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setTolerances(rtol=1e-8, atol=0.0, max_it=500)
            ksp.setOptionsPrefix(meth.replace("-", "") + "_")
            pc = ksp.getPC()
            if meth == "amg":
                pc.setType("hypre"); pc.setHYPREType("boomeramg")
                amg_defaults("amg_")
            else:
                P = BTFD(A, multi=False, tol=0.1)
                pc.setType("python"); pc.setPythonContext(P)
            ksp.setFromOptions()
            ksp.setOperators(Mp)
            ksp.setUp()
        else:
            ksp.getPC().setReusePreconditioner(True)
            ksp.setOperators(Mp)
        t1 = perf_counter()
        xv, rv = Mp.createVecs(); rv.setArray(b)
        ksp.solve(rv, xv)
        t2 = perf_counter()
        after = counters()
        x = xv.getArray()
        d = A.diagonal()
        h = hashlib.sha256(A.data.data); h.update(b.data)
        rows.append(dict(geometry=geom, step=s, method=meth, its=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                         setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0, gen_s=gen_s, mat_s=mat_s,
                         scaled=np.linalg.norm((b - A @ x) / d) / np.linalg.norm(b / d), pageouts=after[0] - before[0],
                         swapouts=after[1] - before[1], maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                         input_sha=h.hexdigest()[:16]))
        r = rows[-1]
        print(f"{geom} step {s} {meth:6s} its {r['its']:3d} reason {r['reason']} setup {r['setup_s']:.1f} solve {r['solve_s']:.1f} s "
              f"scaled {r['scaled']:.1e} pageouts {r['pageouts']} rss {r['maxrss_gb']:.1f} GB", flush=True)
        with open(f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save(stem, ksp, A, b, features=feature_stats(P) if P else None)


if __name__ == "__main__":
    if sys.argv[1] != "all":
        worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        out = HERE / "results" / "h56_isolated"
        out.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        for name, geom, order in GEOMETRIES:
            for meth in order:
                stem = out / f"{name}_{meth}"
                with open(f"{stem}.log", "w") as fh:
                    subprocess.run([sys.executable, __file__, str(stem), meth, "10", geom], env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
                print(name, meth, "done", flush=True)
