"""FD-LGF vs BoomerAMG vs GMG on the high-contrast families of cases_hc.py (no traditional LGF).

  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  /opt/anaconda3/envs/hypre_env/bin/python bench_hc.py OUT_NAME n [n ...] [--families f ...] [--chis c ...]
Writes results/OUT_NAME.csv and results/OUT_NAME_traces.json. Same PCG loop, stopping rule (‖r‖/‖f‖ ≤ 1e-10),
maxiter 1000, timing (median of 3 if a run < 5 s) and error trace as ../benchmark.py.
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent / "CNN function")]

from axy_lgf.operators import l2_norm
from baselines import GeometricVcycle, solve_gmg_pcg
from baselines.boomeramg import BoomerAMG, solve_boomeramg_pcg
from benchmark import error_trace, first_below
from cases_hc import FAMILIES, make_case
from fdlgf import solve_fdlgf_pcg

CHIS, TOL, MAXITER = (1e2, 1e4, 1e6, 1e8), 1e-10, 1000
METHODS = {"GMG": None, "BoomerAMG": None, "FD-LGF w=1/16": 1 / 16, "FD-LGF w=1/64": 1 / 64}


def solve(c, name):
    kw = dict(q=c.q, a_nodes=c.a_nodes, tol=TOL, maxiter=MAXITER)
    if name == "GMG":
        return solve_gmg_pcg(c.a, c.f, c.grid, **kw)
    if name == "BoomerAMG":
        return solve_boomeramg_pcg(c.a, c.f, c.grid, **kw)
    return solve_fdlgf_pcg(c.a, c.f, c.grid, w=METHODS[name], split=c.split, **kw)


def preconditioner(c, name, res):
    if name == "GMG":
        return GeometricVcycle(c.A, c.n)
    if name == "BoomerAMG":
        return BoomerAMG(c.A)
    return res.preconditioner


def main(out, sizes, families, chis):
    (HERE / "results").mkdir(exist_ok=True)
    rows, traces = [], {}
    for n in sizes:
        for fam in families:
            for chi in chis:
                c = make_case(fam, n, chi)
                rel = lambda u: l2_norm(u - c.u_h, c.grid) / l2_norm(c.u_h, c.grid)
                for name in METHODS:
                    res = solve(c, name)
                    runs = [res]
                    if res.setup_time + res.solve_time < 5.0:
                        runs += [solve(c, name) for _ in range(2)]
                    setup, solve_t = median(r.setup_time for r in runs), median(r.solve_time for r in runs)
                    errs = error_trace(c, preconditioner(c, name, res), 300)
                    traces[f"{fam}|{n}|{chi:.0e}|{name}"] = errs
                    per_it = solve_t / max(res.iterations, 1)
                    k8 = first_below(errs, 1e-8)
                    P = getattr(res, "preconditioner", None)
                    row = dict(case=fam, n=n, chi=chi, method=name, iterations=res.iterations, converged=res.converged,
                               alg_error=rel(res.u), total_error=rel(res.u), setup_ms=1e3 * setup, solve_ms=1e3 * solve_t,
                               total_ms=1e3 * (setup + solve_t), ms_per_it=1e3 * per_it,
                               err_5=errs[min(5, len(errs) - 1)], its_err_1e8=k8, best_err=min(errs),
                               ms_to_err_1e8=None if k8 is None else 1e3 * (setup + k8 * per_it),
                               feature_fraction=c.feature_fraction,
                               local_fraction=getattr(P, "local_fraction", None),
                               background=getattr(getattr(P, "G", None), "kind", None), load1=os.getloadavg()[0])
                    rows.append(row)
                    print(f"{fam:8s} n={n:4d} chi={chi:.0e} {name:14s} it={res.iterations:4d} conv={res.converged!s:5s} "
                          f"alg={row['alg_error']:.1e} k8={k8} setup={row['setup_ms']:8.1f} solve={row['solve_ms']:8.1f} "
                          f"ms feat={c.feature_fraction:.2f} loc={row['local_fraction'] or 0:.2f} load={row['load1']:.1f}",
                          flush=True)
                with (HERE / "results" / f"{out}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (HERE / "results" / f"{out}_traces.json").write_text(json.dumps(traces))


def _opt(args, flag, conv, default):
    if flag not in args:
        return default
    i = args.index(flag)
    vals = []
    for a in args[i + 1:]:
        if a.startswith("--"):
            break
        vals.append(conv(a))
    return tuple(vals)


if __name__ == "__main__":
    args = sys.argv[1:]
    first_flag = next((i for i, a in enumerate(args) if a.startswith("--")), len(args))
    main(args[0], [int(s) for s in args[1:first_flag]], _opt(args, "--families", str, FAMILIES),
         _opt(args, "--chis", float, CHIS))
