"""BoomerAMG with the configurations a user would try, on the FluxMG suite, held-out and grouping cases and the real matrix.

Round 1 compared FluxMG with one setting only (amg-3d defaults at theta 0.25, called fast.25 below). This script gives BoomerAMG the
same 8 scalar settings as the SSR baselines (gensolver/krylov.AMG), plus, for systems with several fields, unknowns reordered cell by
cell (not timed) with numfunctions set: sys.u (unknown-based coarsening) and sys.n (nodal coarsening).
Same criterion and code path as FluxMG (try_one.solve_ir): double-double original residual 1e-8 by refinement, inner rtol 1e-6, x0 = 0,
at most MAX_ITS inner iterations and MAX_STEPS steps, single thread, CPU seconds including setup. One process per (case, setting).

usage: amg_configs.py all [REPEATS]          writes amg_configs.jsonl (appends), shuffled jobs
       amg_configs.py one CASE CFG           one run, prints JSON
"""
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "amg_configs.jsonl"
_B = dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=1)
CFG = {
    "fast.25": dict(_B, strong_threshold=0.25),
    "fast.5": dict(_B, strong_threshold=0.5),
    "fast.7": dict(_B, strong_threshold=0.7),
    "fast.9": dict(_B, strong_threshold=0.9),
    "noagg.25": dict(_B, agg_nl=0, strong_threshold=0.25),
    "noagg.5": dict(_B, agg_nl=0, strong_threshold=0.5),
    "falgout.25": dict(coarsen_type="Falgout", interp_type="classical", P_max=0, agg_nl=0, strong_threshold=0.25),
    "falgout.5": dict(coarsen_type="Falgout", interp_type="classical", P_max=0, agg_nl=0, strong_threshold=0.5),
}
SYS = {"sys.u": dict(_B, strong_threshold=0.25), "sys.n": dict(_B, strong_threshold=0.25, nodal_coarsen=1)}


def one(case, cfg):
    sys.path.insert(0, str(HERE))
    import numpy as np
    from petsc4py import PETSc
    from eval_fluxmg import MAX_ITS, MAX_STEPS
    from eval_grouping import get
    from try_one import solve_ir
    A, b, dims, F = get(case)
    c = dict({**CFG, **SYS}[cfg])
    if cfg.startswith("sys"):
        N = A.shape[0] // F
        perm = (np.arange(N)[:, None] + N * np.arange(F)[None, :]).ravel()      # new index cell * F + field
        A = A[perm][:, perm].tocsr(); b = b[perm]
        c["numfunctions"] = F
    theta = c["strong_threshold"]
    pre = f"ir{theta:g}_".replace(".", "p")                                       # the prefix solve_ir uses
    o = PETSc.Options()
    for k, v in c.items():
        o[pre + "pc_hypre_boomeramg_" + k] = v                                    # set before solve_ir: amg_defaults keeps them
    q = solve_ir(A, b, "amg", theta=theta, max_it=MAX_ITS, max_steps=MAX_STEPS)
    import resource
    print(json.dumps(dict(case=case, method="amg", cfg=cfg, ok=bool(q["ok"]), its=q["its"], rel=float(q["rel"]), time=q["cpu"],
                          build=q["setup"], rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)))


def cases():
    sys.path.insert(0, str(HERE))
    from eval_fluxmg import CASES, HOLDOUT
    from eval_grouping import CASES as GROUPING
    return [c for c, _ in CASES] + [c for c, _ in HOLDOUT] + list(GROUPING) + ["real"]


def fields(case):
    sys.path.insert(0, str(HERE))
    p = case.split(":")
    if p[0] in ("g3t", "gen", "real"):
        return 3
    if p[0] == "mg":
        return int(p[2]) + 1
    if p[0] == "blob":
        from eval_grouping import get
        return get(case)[3]
    return 1


def run(repeats):
    jobs = []
    for c in cases():
        cf = list(CFG) + (list(SYS) if fields(c) > 1 else [])
        jobs += [(c, k) for k in cf]
    jobs = jobs * repeats
    random.Random(1109).shuffle(jobs)
    failed = set()
    for n, (c, k) in enumerate(jobs):
        if (c, k) in failed:
            continue
        try:
            p = subprocess.run([sys.executable, "-u", __file__, "one", c, k], capture_output=True, text=True, timeout=900)
            line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else \
                json.dumps(dict(case=c, cfg=k, ok=False, crash=p.returncode, err=p.stderr[-300:]))
        except subprocess.TimeoutExpired:
            line = json.dumps(dict(case=c, cfg=k, ok=False, crash="timeout 900 s"))
        if not json.loads(line)["ok"]:
            failed.add((c, k))
        with open(OUT, "a") as f:
            f.write(line + "\n")
        print(n + 1, len(jobs), line[:140], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3])
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
