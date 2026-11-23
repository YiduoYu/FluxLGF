"""BoomerAMG (θ = 0.25 PETSc default, θ = 0.5) vs GMG (ω = 4/5) vs FD-LGF on the anisotropic families.

  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  /opt/anaconda3/envs/hypre_env/bin/python bench_aniso.py OUT n [n ...] [--families f ...] [--chis c ...]
Same PCG recurrence and stopping rule as ../three_d/ops3d.pcg (‖r‖/‖f‖ ≤ 1e-10, residual recomputed), maxiter 1000.
The algebraic error ‖u_k - u_h‖/‖u_h‖ is recorded inside the same run; its cost is excluded from solve time.
After the stopping rule fires, PCG keeps iterating for the error trace only (untimed, not counted) until the error is
< 1e-10 or stagnates, so its_err_1e8 is defined for every method (the residual rule can stop while the error is ~1e-6).
Timing is the median of 3 runs when a run takes < 5 s. FD-LGF w = 1/64 is skipped when there is no feature mask
(it would equal w = 1/16).
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "three_d"), str(HERE.parents[1] / "CNN function")]

import numpy as np

from baselines.gmg import GeometricVcycle
from cases_aniso import FAMILIES, make_case
from fdlgf_tensor import AnisoBackground, FDLGFTensor

CHIS, TOL, MAXITER = (1e2, 1e4, 1e6), 1e-10, 1000


def build(c, name):
    if name.startswith("BoomerAMG"):
        from solvers3d import BoomerAMG3
        return BoomerAMG3(c.A, float(name.split("=")[1]))
    if name == "GMG":
        return GeometricVcycle(c.A, c.n)
    return FDLGFTensor(c.A, AnisoBackground(*c.bg, c.q, c.h), c.mask, w=1 / int(name.split("/")[1]))


def pcg_trace(c, P):
    """PCG with the ops3d.pcg stopping rule; returns iterations, converged, solve time, error trace."""
    n = c.n
    action = lambda u: (c.A @ u.ravel()).reshape(n, n)
    rel = lambda u: float(np.linalg.norm(u - c.u_h) / np.linalg.norm(c.u_h))
    started, spent = perf_counter(), 0.0
    u = np.zeros_like(c.f)
    r = c.f.copy()
    z = P(r)
    p = z.copy()
    rz = float(np.sum(r * z))
    fn = float(np.linalg.norm(c.f))
    errs, it, converged, solve_time = [1.0], 0, False, None
    while len(errs) <= MAXITER:
        Ap = action(p)
        pAp = float(np.sum(p * Ap))
        if pAp <= 0:                     # exact solve reached during the trace extension
            break
        alpha = rz / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        t = perf_counter()
        errs.append(rel(u))
        spent += perf_counter() - t
        if solve_time is None:
            it += 1
            if np.linalg.norm(r) / fn <= TOL:
                r = c.f - action(u)
                if np.linalg.norm(r) / fn <= TOL:   # then keep iterating (restarted) for the error trace only
                    converged, solve_time = True, perf_counter() - started - spent
                z = P(r)
                p = z.copy()
                rz = float(np.sum(r * z))
                continue
            if it == MAXITER:
                break
        # trace extension (not timed): stop at 1e-10, or when the error has stagnated (round-off floor)
        elif errs[-1] < 1e-10 or len(errs) - it > 200 or (len(errs) - it > 20 and errs[-1] > 0.5 * errs[-21]):
            break
        z = P(r)
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    if solve_time is None:
        solve_time = perf_counter() - started - spent
    return it, converged, solve_time, errs


def first_below(errs, level):
    return next((k for k, e in enumerate(errs) if e <= level), None)


def run(c, name):
    t = perf_counter()
    P = build(c, name)
    setup = perf_counter() - t
    return (setup, *pcg_trace(c, P), P)


def main(out, sizes, families, chis):
    rows, traces = [], {}
    for n in sizes:
        for fam in families:
            for chi in chis:
                c = make_case(fam, n, chi)
                methods = ["BoomerAMG θ=0.25", "BoomerAMG θ=0.5", "GMG", "FD-LGF w=1/16"]
                if c.mask.any():
                    methods.append("FD-LGF w=1/64")
                for name in methods:
                    runs = [run(c, name)]
                    if runs[0][0] + runs[0][3] < 5.0:
                        runs += [run(c, name) for _ in range(2)]
                    setup, its, conv, _, errs, P = runs[0]
                    setup, solve_t = median(r[0] for r in runs), median(r[3] for r in runs)
                    k8, per_it = first_below(errs, 1e-8), solve_t / its
                    traces[f"{fam}|{n}|{chi:.0e}|{name}"] = errs
                    row = dict(case=fam, n=n, chi=chi, method=name, iterations=its, converged=conv,
                               alg_error=errs[its], best_err=min(errs), err_5=errs[min(5, len(errs) - 1)], its_err_1e8=k8,
                               setup_ms=1e3 * setup, solve_ms=1e3 * solve_t, total_ms=1e3 * (setup + solve_t),
                               ms_per_it=1e3 * per_it,
                               ms_to_err_1e8=None if k8 is None else 1e3 * (setup + k8 * per_it),
                               feature_fraction=c.feature_fraction, local_fraction=getattr(P, "local_fraction", None),
                               load1=os.getloadavg()[0])
                    rows.append(row)
                    print(f"{fam:8s} n={n:4d} chi={chi:.0e} {name:17s} it={its:4d} conv={conv!s:5s} "
                          f"alg={errs[its]:.1e} best={min(errs):.1e} k8={k8} setup={1e3 * setup:8.1f} solve={1e3 * solve_t:9.1f} ms "
                          f"loc={row['local_fraction'] or 0:.2f} load={row['load1']:.1f}", flush=True)
                with (HERE / "results" / f"{out}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (HERE / "results" / f"{out}_traces.json").write_text(json.dumps(traces))


def _opt(args, flag, conv, default):
    if flag not in args:
        return default
    vals = []
    for a in args[args.index(flag) + 1:]:
        if a.startswith("--"):
            break
        vals.append(conv(a))
    return tuple(vals)


if __name__ == "__main__":
    args = sys.argv[1:]
    first_flag = next((i for i, a in enumerate(args) if a.startswith("--")), len(args))
    main(args[0], [int(s) for s in args[1:first_flag]], _opt(args, "--families", str, FAMILIES),
         _opt(args, "--chis", float, CHIS))
