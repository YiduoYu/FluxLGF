"""H38: agg3t-w vs BoomerAMG on nonsymmetric 3T surrogates at m = 128 (GMRES(50), right preconditioning).

  /opt/anaconda3/envs/hypre_env/bin/python h38_nonsym.py         -> results/h38.csv
emis (dt 1e-3, 10): amg-3d and agg3t-w at the two targets.
scaled (dt 1e-3, 10): agg3t-w; BoomerAMG 3D settings with PETSc diagonal scaling (plain BoomerAMG fails here, see
m = 64 in research-log). run_sc.solve restarts until the true residual meets the target; MAXIT = 500 here.
"""

import csv
from pathlib import Path

import run_sc
from surrogate3t import build

HERE = Path(__file__).resolve().parent
RUNS = [("emis", 1e-3, "amg-3d", 1e-4), ("emis", 1e-3, "amg-3d", 1e-8), ("emis", 1e-3, "agg3t-w", 1e-4),
        ("emis", 1e-3, "agg3t-w", 1e-8), ("emis", 10.0, "amg-3d", 1e-4), ("emis", 10.0, "amg-3d", 1e-8),
        ("emis", 10.0, "agg3t-w", 1e-4), ("emis", 10.0, "agg3t-w", 1e-8),
        ("scaled", 1e-3, "agg3t-w", 1e-4), ("scaled", 1e-3, "agg3t-w", 1e-8),
        ("scaled", 1e-3, "amg-3d-ds", 1e-4), ("scaled", 1e-3, "amg-3d-ds", 1e-8),
        ("scaled", 10.0, "agg3t-w", 1e-4), ("scaled", 10.0, "agg3t-w", 1e-8),
        ("scaled", 10.0, "amg-3d-ds", 1e-4), ("scaled", 10.0, "amg-3d-ds", 1e-8)]

rows, cache = [], {}
run_sc.SYMMETRIC = False
run_sc.MAXIT = 500
for variant, dt, solver, tol in RUNS:
    if (variant, dt) not in cache:
        cache.clear()
        cache[(variant, dt)] = build(128, dt, scaled=variant == "scaled", emission=variant == "emis")
    A, b = cache[(variant, dt)]
    row = dict(variant=variant, dt=dt, **run_sc.solve(solver, A, b, 128, tol))
    rows.append(row)
    print(row, flush=True)
    with (HERE / "results" / "h38.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
