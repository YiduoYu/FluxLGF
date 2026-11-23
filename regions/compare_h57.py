"""H57 (protocol_h57.md): same-accuracy comparison. Every method must bring the double-double original residual
||b - Ax|| / ||b|| below 1e-8 by iterative refinement (x and r in double-double, each correction by GMRES(50) to 1e-6).
Sequences: the preconditioner is built at step 0 and kept; x starts from 0 at every step. Each run is its own process,
single thread; order shuffled with a fixed seed; every run is kept.
total_s = setup at step 0 + inner solves + dd residuals (+ hi/lo updates); gen_s, mat_s, prep_s (DD tables) are common
costs reported beside it. t4_s is the same total stopped at the first residual below 1e-4 in each step.
usage: compare_h57.py all | compare_h57.py STEM KIND STEPS real|GEOMETRY     (GEOMETRY = m:n:p:aspect:width:delta:seed)"""
import csv
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "h57"
TUNED = "-ir_amg_pc_hypre_boomeramg_strong_threshold 0.25"       # fastest setting of the audit's real-matrix screening
SEQ_ARMS = (("amg-base", "amg-3d", ""), ("amg-tuned", "amg-3d", TUNED), ("bt-v", "bt-v", ""), ("bt-j", "bt-j", ""),
            ("bt-fd", "bt-fd-t01", ""), ("bt-fd-chol", "bt-fd-chol-t01", ""))
REAL_ARMS = SEQ_ARMS + (("bt-fd-1e-6", "bt-fd", ""), ("bt-fd-chol-1e-6", "bt-fd-chol", ""))
SEQUENCES = (("G1", "128:1:2:1:0:2e-2:1"), ("G4", "128:1:2:2:0:2e-2:1"), ("G7", "128:3:2:1:0:2e-2:1"),
             ("G8", "128:10:2:1:0:2e-2:1"), ("G9", "128:10:2:1:0:2e-2:2"), ("G10", "128:10:8:0.5:2:2e-2:3"))
REAL_BLOCKS = 5
STEPS = 10


def worker(stem, kind, steps, case):
    import numpy as np
    import scipy.sparse as sp
    from petsc4py import PETSc
    sys.path[:0] = [str(HERE), str(HERE.parent / "solver_challenge")]
    from btfd import BTFD
    from gen_gen import generate
    from ir_fp128 import DDResidual, two_sum
    from provenance import feature_stats, save
    from run_ir import X, inner
    from seq_isolated import counters

    def load(s):
        if case == "real":
            return sp.load_npz(X / "A.npz").tocsr(), np.load(X / "b.npy")
        m_, n_, p_, a_, w_, d_, sd_ = case.split(":")
        return generate(int(m_), int(n_), float(p_), float(a_), float(w_), float(d_), int(sd_), phase=0.15 * s, rhs_seed=100 + s)

    ksp, rows = None, []
    for s in range(steps):
        t = perf_counter()
        A, b = load(s)
        gen_s = perf_counter() - t
        t = perf_counter()
        res = DDResidual(A, b)
        prep_s = perf_counter() - t
        h = hashlib.sha256(A.data.data); h.update(b.data)
        before = counters()
        setup_s = mat_s = 0.0
        if s == 0:
            t = perf_counter()
            ksp, (dv, rv) = inner(A, 1e-6, kind)
            setup_s = perf_counter() - t
        else:
            t = perf_counter()
            M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
            mat_s = perf_counter() - t
            ksp.getPC().setReusePreconditioner(True)
            ksp.setOperators(M)
        nb, d = np.linalg.norm(b), A.diagonal()
        xh, xl, r = np.zeros_like(b), np.zeros_like(b), b.copy()
        its, reasons, rel = [], [], float("inf")
        solve_s = resid_s = 0.0
        t4 = float("nan")
        for _ in range(10):
            t1 = perf_counter()
            rv.setArray(r); ksp.solve(rv, dv)
            t2 = perf_counter()
            its.append(ksp.getIterationNumber()); reasons.append(int(ksp.getConvergedReason()))
            xh, e = two_sum(xh, dv.getArray()); xl += e
            xh, xl = two_sum(xh, xl)
            r = res(xh, xl)
            rel = np.linalg.norm(r) / nb
            solve_s += t2 - t1; resid_s += perf_counter() - t2
            if rel < 1e-4 and t4 != t4:
                t4 = solve_s + resid_s
            if rel < 1e-8 or not np.isfinite(rel):
                break
        after = counters()
        rows.append(dict(case=case, kind=kind, step=s, ir_steps=len(its), inner_its=sum(its), inner_each=" ".join(map(str, its)),
                         reasons=" ".join(map(str, reasons)), ok=bool(rel < 1e-8 and min(reasons) > 0), final_rel=rel,
                         final_scaled=np.linalg.norm(r / d) / np.linalg.norm(b / d), setup_s=setup_s, solve_s=solve_s, resid_s=resid_s,
                         total_s=setup_s + solve_s + resid_s, t4_s=setup_s + t4, gen_s=gen_s, prep_s=prep_s, mat_s=mat_s,
                         swapouts=after[1] - before[1], pageouts=after[0] - before[0],
                         maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, input_sha=h.hexdigest()[:16]))
        q = rows[-1]
        print(f"{case} step {s} {kind}: ir {q['ir_steps']} inner {q['inner_each']} rel {rel:.2e} scaled {q['final_scaled']:.1e} "
              f"setup {setup_s:.1f} solve {solve_s:.1f} resid {resid_s:.1f} swapouts {q['swapouts']} rss {q['maxrss_gb']:.1f}", flush=True)
        with open(f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(f"{stem}.npz", xh=xh, xl=xl)
    P = ksp.getPC().getPythonContext() if ksp.getPC().getType() == "python" else None
    save(stem, ksp, A, b, features=feature_stats(P) if isinstance(P, BTFD) else None)


def run_all():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(57)
    base = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    jobs = []
    for b in range(REAL_BLOCKS):
        arms = list(REAL_ARMS); rng.shuffle(arms)
        jobs += [(f"real{b}_{a}", kind, opts, 1, "real") for a, kind, opts in arms]
    for name, geom in SEQUENCES:
        arms = list(SEQ_ARMS); rng.shuffle(arms)
        jobs += [(f"{name}_{a}", kind, opts, STEPS, geom) for a, kind, opts in arms]
    for tag, kind, opts, steps, case in jobs:
        env = dict(base, PETSC_OPTIONS=opts) if opts else base
        t0 = perf_counter()
        with open(OUT / f"{tag}.log", "w") as fh:
            rc = subprocess.run([sys.executable, __file__, str(OUT / tag), kind, str(steps), case], env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        with open(OUT / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(dict(tag=tag, kind=kind, opts=opts, case=case, returncode=rc, process_s=perf_counter() - t0)) + "\n")
        print(tag, "rc", rc, f"{perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        run_all()
    else:
        worker(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
