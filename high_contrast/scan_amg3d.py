"""EXPLORATORY: where does BoomerAMG struggle in 3D? BoomerAMG (θ = 0.5 and PETSc default 0.25) and GMG (ω = 6/7)
on the 3D families, n = 31, 63, 127; FD-LGF only where its local region stays small (local fraction < 0.4).

  OMP_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python scan_amg3d.py OUT n [n ...] [--chis c ...]
"""

import csv
import os
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "three_d")]

import numpy as np

from cases_hc3d import FAMILIES3, make_case3
from ops3d import apply_op, pcg, rel_l2
from solvers3d import GMG3, BoomerAMG3, Method2_3D

BUILD = {"BoomerAMG θ=0.5": lambda c: BoomerAMG3(c.A, 0.5), "BoomerAMG θ=0.25": lambda c: BoomerAMG3(c.A, 0.25),
         "GMG": lambda c: GMG3(c.A, c.n),
         "FD-LGF w=1/32": lambda c: Method2_3D(c.A, c.bg_a1, c.mask, c.q, c.grid.h, 1 / 32),
         "FD-LGF w=1/64": lambda c: Method2_3D(c.A, c.bg_a1, c.mask, c.q, c.grid.h, 1 / 64)}


def main(out, sizes, chis):
    rows = []
    for n in sizes:
        for fam in FAMILIES3:
            for chi in chis:
                c = make_case3(fam, n, chi)
                for name, build in BUILD.items():
                    if name.startswith("FD-LGF") and c.feature_fraction > 0.2:
                        continue          # local LU would cover most of the cube
                    t = perf_counter()
                    P = build(c)
                    setup = perf_counter() - t
                    res = pcg(lambda u: apply_op(u, c.a_nodes, c.q, c.grid.h), P, c.f, tol=1e-10, maxiter=500)
                    row = dict(case=fam, n=n, chi=chi, method=name, iterations=res.iterations, converged=res.converged,
                               alg_error=rel_l2(res.u, c.u_h), setup_ms=1e3 * setup, solve_ms=1e3 * res.solve_time,
                               total_ms=1e3 * (setup + res.solve_time), feature_fraction=c.feature_fraction,
                               local_fraction=getattr(P, "local_fraction", None), load1=os.getloadavg()[0])
                    rows.append(row)
                    print(f"{fam:8s} n={n:3d} chi={chi:.0e} {name:16s} it={res.iterations:4d} conv={res.converged!s:5s} "
                          f"alg={row['alg_error']:.1e} setup={row['setup_ms']:8.0f} solve={row['solve_ms']:8.0f} ms "
                          f"feat={c.feature_fraction:.2f} loc={row['local_fraction'] or 0:.2f}", flush=True)
                with (HERE / "results" / f"{out}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    cut = args.index("--chis") if "--chis" in args else len(args)
    main(args[0], [int(s) for s in args[1:cut]], tuple(float(s) for s in args[cut + 1:]) or (1e2, 1e4, 1e6, 1e8))
