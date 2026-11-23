"""3D benchmark: 方法二 vs traditional LGF, GMG, BoomerAMG (same metrics as ../benchmark.py).

  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  /opt/anaconda3/envs/hypre_env/bin/python benchmark3d.py [n ...]
Outputs results/benchmark3d.csv, results/traces3d.json.
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from cases3d import CASES, make_case
from ops3d import apply_op, pcg, rel_l2
from solvers3d import GMG3, LGF3, BoomerAMG3, Method2_3D

CHIS, TOL, MAXITER = (1e2, 1e4, 1e6, 1e8), 1e-10, 500
RINGS = {"M2 w=1/32": 1 / 32, "M2 w=1/64": 1 / 64}
OUT = HERE / "results"


def methods(case):
    return ["LGF", "GMG", "BoomerAMG"] + (list(RINGS) if case in ("C_exp+incl", "D_exp+rand") else ["M2"])


def build(c, name):
    if name == "LGF":
        return LGF3(c.n, c.grid.h, float(np.mean(c.a_nodes[1:-1, 1:-1, 1:-1])), float(np.mean(c.q)))
    if name == "GMG":
        return GMG3(c.A, c.n)
    if name == "BoomerAMG":
        return BoomerAMG3(c.A)
    return Method2_3D(c.A, c.bg_a1, c.mask, c.q, c.grid.h, RINGS.get(name, 1 / 32))


def run(c, name, tol=TOL, maxiter=MAXITER):
    t = perf_counter()
    P = build(c, name)
    setup = perf_counter() - t
    res = pcg(lambda u: apply_op(u, c.a_nodes, c.q, c.grid.h), P, c.f, tol=tol, maxiter=maxiter)
    res.setup_time, res.P = setup, P
    return res


def error_trace(c, P, maxiter):
    """Same PCG recurrence without stopping; relative error to u_h after every iteration."""
    act = lambda u: apply_op(u, c.a_nodes, c.q, c.grid.h)
    u = np.zeros_like(c.f)
    r = c.f.copy()
    z = P(r)
    p = z.copy()
    rz = float(np.sum(r * z))
    errs = [1.0]
    for _ in range(maxiter):
        Ap = act(p)
        alpha = rz / float(np.sum(p * Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        errs.append(rel_l2(u, c.u_h))
        if errs[-1] < 1e-13 or (errs[-1] < 1e-7 and len(errs) > 30 and errs[-1] > 0.5 * errs[-21]):
            break
        z = P(r)
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return errs


def first_below(errs, level):
    return next((k for k, e in enumerate(errs) if e <= level), None)


def main(sizes):
    OUT.mkdir(exist_ok=True)
    rows, traces = [], {}
    for n in sizes:
        for case in CASES:
            for chi in CHIS:
                c = make_case(case, n, chi)
                if c.u_h is None:   # A: reference = 方法二 PCG run to round-off (checked against a direct solve, n <= 63)
                    c.u_h = run(c, "M2", tol=1e-15, maxiter=40).u
                for name in methods(case):
                    res = run(c, name)
                    runs = [res]
                    if res.setup_time + res.solve_time < 5.0:
                        runs += [run(c, name) for _ in range(2)]
                    setup, solve_t = median(r.setup_time for r in runs), median(r.solve_time for r in runs)
                    errs = error_trace(c, res.P, 300)
                    traces[f"{case}|{n}|{chi:.0e}|{name}"] = errs
                    per_it = solve_t / max(res.iterations, 1)
                    k8 = first_below(errs, 1e-8)
                    row = dict(
                        case=case, n=n, chi=chi, method=name, iterations=res.iterations, converged=res.converged,
                        alg_error=rel_l2(res.u, c.u_h), total_error=rel_l2(res.u, c.u_exact),
                        setup_ms=1e3 * setup, solve_ms=1e3 * solve_t, total_ms=1e3 * (setup + solve_t),
                        ms_per_it=1e3 * per_it, timed_runs=len(runs),
                        err_5=errs[min(5, len(errs) - 1)], err_10=errs[min(10, len(errs) - 1)],
                        its_err_1e6=first_below(errs, 1e-6), its_err_1e8=k8, its_err_1e10=first_below(errs, 1e-10),
                        best_err=min(errs), ms_to_err_1e8=None if k8 is None else 1e3 * (setup + k8 * per_it),
                        local_fraction=getattr(res.P, "local_fraction", None), lu_nnz=getattr(res.P, "lu_nnz", None),
                        load1=os.getloadavg()[0])
                    rows.append(row)
                    print(f"{case:10s} n={n:3d} chi={chi:.0e} {name:10s} it={res.iterations:4d} "
                          f"conv={res.converged!s:5s} alg={row['alg_error']:.1e} tot={row['total_error']:.4e} "
                          f"k8={k8} setup={row['setup_ms']:9.1f} solve={row['solve_ms']:9.1f} ms "
                          f"load={row['load1']:.1f}", flush=True)
                with (OUT / "benchmark3d.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (OUT / "traces3d.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [31, 63, 127])
