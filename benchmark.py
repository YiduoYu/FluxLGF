"""方法二 vs 传统 LGF, GMG, hypre BoomerAMG: iterations, accuracy, time.

Run (single-threaded BLAS, hypre_env has petsc4py):
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/anaconda3/envs/hypre_env/bin/python benchmark.py [n ...] [--only METHOD ...]
--only reruns just those methods and replaces their rows / traces in the existing results.
Outputs results/benchmark.csv (one row per run) and results/traces.json (error after every PCG iteration).
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "CNN function")]  # axy_lgf package lives there

import numpy as np

from axy_lgf.operators import apply_reaction_diffusion, l2_norm
from baselines import DirichletLGF, GeometricVcycle, solve_gmg_pcg, solve_lgf_pcg
from baselines.boomeramg import BoomerAMG, solve_boomeramg_pcg
from cases import CASES, make_case
from fdlgf import FDLGFPreconditioner, solve_fdlgf_pcg

CHIS, TOL, MAXITER = (1e2, 1e4, 1e6, 1e8), 1e-10, 1000
RINGS = {"M2 w=1/16": 1 / 16, "M2 w=1/32": 1 / 32, "M2 w=1/64": 1 / 64}
OUT = HERE / "results"


def methods(case):
    base = ["LGF", "GMG", "BoomerAMG", "M2 w=1/16"]
    return base + (["M2 w=1/32", "M2 w=1/64"] if case in ("C_exp+incl", "D_exp+rand") else [])


def solve(c, name, tol=TOL, maxiter=MAXITER):
    kw = dict(q=c.q, a_nodes=c.a_nodes, tol=tol, maxiter=maxiter)
    if name == "LGF":
        return solve_lgf_pcg(c.a, c.f, c.grid, **kw)
    if name == "GMG":
        return solve_gmg_pcg(c.a, c.f, c.grid, **kw)
    if name == "BoomerAMG":
        return solve_boomeramg_pcg(c.a, c.f, c.grid, **kw)
    return solve_fdlgf_pcg(c.a, c.f, c.grid, w=RINGS[name], split=c.split, **kw)


def preconditioner(c, name):
    if name == "LGF":
        return DirichletLGF(c.grid, a0=float(np.mean(c.a)), sigma=float(np.mean(c.q)))
    if name == "GMG":
        return GeometricVcycle(c.A, c.n)
    if name == "BoomerAMG":
        return BoomerAMG(c.A)
    return FDLGFPreconditioner(c.A, c.a_nodes, c.q, c.grid.h, w=RINGS[name], split=c.split)


def error_trace(c, P, maxiter):
    """Same PCG recurrence as axy_lgf.solvers._pcg (no stopping), recording ||u_k - u_h|| / ||u_h|| for k = 0..."""
    rel = lambda u: l2_norm(u - c.u_h, c.grid) / l2_norm(c.u_h, c.grid)
    u = c.grid.zeros()
    r = c.f.copy()
    z = P(r)
    p = z.copy()
    rz = float(np.sum(r * z))
    errs = [1.0]
    for _ in range(maxiter):
        Ap = apply_reaction_diffusion(p, c.a, c.grid, c.q, a_nodes=c.a_nodes)
        pAp = float(np.sum(p * Ap))
        if pAp <= 0:  # exact solve reached (e.g. the local solve covers the whole grid)
            break
        alpha = rz / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        errs.append(rel(u))
        if errs[-1] < 1e-13 or (errs[-1] < 1e-7 and len(errs) > 30 and errs[-1] > 0.5 * errs[-21]):  # round-off floor
            break
        z = P(r)
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return errs


def first_below(errs, level):
    return next((k for k, e in enumerate(errs) if e <= level), None)


def main(sizes, only=()):
    OUT.mkdir(exist_ok=True)
    rows, traces = [], {}
    if only:
        rows = [r for r in csv.DictReader((OUT / "benchmark.csv").open()) if r["method"] not in only]
        traces = {k: v for k, v in json.loads((OUT / "traces.json").read_text()).items()
                  if k.split("|")[3] not in only}
    for n in sizes:
        for case in CASES:
            for chi in CHIS:
                c = make_case(case, n, chi)
                rel = lambda u, ref: l2_norm(u - ref, c.grid) / l2_norm(ref, c.grid)
                for name in [m for m in methods(case) if not only or m in only]:
                    res = solve(c, name)
                    runs = [res]
                    if res.setup_time + res.solve_time < 5.0:
                        runs += [solve(c, name) for _ in range(2)]
                    setup, solve_t = median(r.setup_time for r in runs), median(r.solve_time for r in runs)
                    errs = error_trace(c, preconditioner(c, name), MAXITER if name == "LGF" else 300)
                    traces[f"{case}|{n}|{chi:.0e}|{name}"] = errs
                    per_it = solve_t / max(res.iterations, 1)
                    k8 = first_below(errs, 1e-8)
                    P = getattr(res, "preconditioner", None)
                    row = dict(
                        case=case, n=n, chi=chi, method=name, iterations=res.iterations, converged=res.converged,
                        alg_error=rel(res.u, c.u_h), total_error=rel(res.u, c.u_exact),
                        setup_ms=1e3 * setup, solve_ms=1e3 * solve_t, total_ms=1e3 * (setup + solve_t),
                        ms_per_it=1e3 * per_it, timed_runs=len(runs),
                        err_5=errs[min(5, len(errs) - 1)], err_10=errs[min(10, len(errs) - 1)],
                        its_err_1e6=first_below(errs, 1e-6), its_err_1e8=k8, its_err_1e10=first_below(errs, 1e-10),
                        best_err=min(errs),
                        ms_to_err_1e8=None if k8 is None else 1e3 * (setup + k8 * per_it),
                        local_fraction=getattr(P, "local_fraction", None), lu_nnz=getattr(P, "lu_nnz", None),
                        load1=os.getloadavg()[0])
                    rows.append(row)
                    print(f"{case:10s} n={n:4d} chi={chi:.0e} {name:10s} it={res.iterations:4d} "
                          f"conv={res.converged!s:5s} alg={row['alg_error']:.1e} tot={row['total_error']:.4e} "
                          f"k8={k8} setup={row['setup_ms']:8.1f} solve={row['solve_ms']:8.1f} ms "
                          f"load={row['load1']:.1f}", flush=True)
                with (OUT / "benchmark.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (OUT / "traces.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    args = sys.argv[1:]
    cut = args.index("--only") if "--only" in args else len(args)
    main([int(s) for s in args[:cut]] or [127, 255, 511, 1023], tuple(args[cut + 1:]))
