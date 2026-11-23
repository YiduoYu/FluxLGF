"""Autoresearch eval: FluxMG against BoomerAMG (theta 0.25) on a suite of equation classes.

Same criterion for both: double-double original residual ||b - Ax|| / ||b|| < 1e-8 by refinement, inner rtol 1e-6, x0 = 0,
single thread. Time = setup + solves + residuals, in CPU seconds. BoomerAMG times are measured once and cached in
amg_baseline.json, because they do not depend on the constrained files; delete that file only to re-measure them.

SCORE (higher is better) = - mean over cases of log2(t_fluxmg / t_amg), with a failed case counted as ratio FAIL_RATIO.
  SCORE 0: on a geometric average as fast as BoomerAMG. SCORE 1: twice as fast.
Also printed: WINS (cases faster than BoomerAMG), FAILS (guard), equivalence of the C and numpy cycles (guard).
usage: eval_fluxmg.py [--baseline] [--real] [--holdout]
  --real adds the real SC26_01 matrix (about 2 minutes), for confirmation only
  --holdout runs only the held-out cases, never used to choose changes: the check against overfitting the suite
"""
import json
import math
import resource
import sys
from pathlib import Path
from time import process_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
import flux_mg  # noqa: E402
from try_one import load, solve_ir  # noqa: E402

FAIL_RATIO = 8.0
MAX_ITS = 200          # inner iterations per refinement step
MAX_STEPS = 6          # refinement steps; beyond these limits the case counts as failed

# (name, class). The multigroup cases stand in for the easy, middle and hard systems of Yue et al. 2026 (U13-2, U92-1, U107-4);
# they are our own structured-grid analogues, not their matrices.
CASES = (
    ("poisson:64", "constant"),
    ("aniso:64:100", "aligned anisotropy, one strong axis"),
    ("aniso:64:0.01", "aligned anisotropy, one weak axis"),
    ("lognorm3", "high contrast, log-normal"),
    ("tubes", "high contrast, conducting tubes"),
    ("insul", "high contrast, insulating inclusions"),
    ("perc3", "high contrast, percolating"),
    ("g3t:64:2e-11:2e-4", "3T, SC26_01-like"),
    ("g3t:64:2e-11:1", "3T, strong electron conduction"),
    ("g3t:64:1:1e-2", "3T, strong feedback"),
    ("gen:64:1:8:0.5:0:2e-2:1", "3T, oblate box"),
    ("gen:64:1:2:1:4:2e-2:1", "3T, graded transition"),
    ("gen:64:3:2:1:0:2e-2:1", "3T, 3 capsules"),
    ("gen:64:10:2:1:0:2e-2:1", "3T, 10 capsules"),
    ("gen:64:10:2:1:0:2e-2:2", "3T, 10 capsules, other seed"),
    ("gen:64:10:8:0.5:2:2e-2:3", "3T, 10 graded boxes"),
    ("mg:32:8:1e-2:1e2:0:7", "multigroup, AMG easy (like U13-2)"),
    ("mg:32:4:1:1:0:7", "multigroup, middle (like U92-1)"),
    ("mg:32:8:1e2:1e-2:1:7", "multigroup, AMG hard (like U107-4)"),
)
# Never look at these inside the loop. New seeds, new parameters, larger grids.
HOLDOUT = (
    ("gen:64:5:2:1:0:2e-2:4", "3T, 5 capsules, new seed"),
    ("gen:64:10:2:1:0:2e-2:5", "3T, 10 capsules, new seed"),
    ("gen:64:1:2:2:0:2e-2:6", "3T, prolate"),
    ("gen:64:2:1:1:1:1e-1:7", "3T, octahedra, 10 % field"),
    ("g3t:64:1e-4:1e-1", "3T, other coupling"),
    ("mg:32:4:1e2:1:1:8", "multigroup, new parameters"),
    ("mg:32:8:1:1e-2:0:9", "multigroup, new parameters"),
    ("aniso:64:1000", "aligned anisotropy, stronger"),
    ("gen:128:3:2:1:0:2e-2:1", "3T, 3 capsules, m = 128"),
    ("gen:128:10:2:1:0:2e-2:2", "3T, 10 capsules, m = 128"),
)
BASE = HERE / "amg_baseline.json"


def run_flux(case):
    A, b, dims, F = load(case)
    t0 = process_time()
    P = flux_mg.FluxMG(A, dims, F)
    build = process_time() - t0
    q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=MAX_ITS, max_steps=MAX_STEPS)
    return dict(ok=q["ok"], its=q["its"], time=build + q["cpu"], build=build, K=P.K)


def equivalence():
    """The C cycle and the numpy cycle must agree; returns the worst relative difference."""
    worst = 0.0
    for case in ("poisson:16", "aniso:16:100", "g3t:16:2e-11:2e-4", "mg:16:4:1:1:0:7"):
        A, _, dims, F = load(case)
        r = np.random.default_rng(3).standard_normal(A.shape[0])
        flux_mg.FAST = False
        z0 = flux_mg.FluxMG(A, dims, F)(r)
        flux_mg.FAST = True
        z1 = flux_mg.FluxMG(A, dims, F)(r)
        worst = max(worst, float(np.linalg.norm(z0 - z1) / np.linalg.norm(z0)))
    return worst


if __name__ == "__main__":
    cases = list(HOLDOUT) if "--holdout" in sys.argv else list(CASES) + ([("real", "real SC26_01")] if "--real" in sys.argv else [])
    base = json.loads(BASE.read_text()) if BASE.exists() else {}
    if "--baseline" in sys.argv or any(c not in base for c, _ in cases):
        for c, _ in cases:
            if c not in base or "--baseline" in sys.argv:
                A, b, _d, _F = load(c)
                t = []
                for _ in range(3):                                       # fastest of 3, so noise in the yardstick favours BoomerAMG
                    q = solve_ir(A, b, "amg", theta=0.25)
                    t.append(q["cpu"])
                base[c] = dict(time=min(t), its=q["its"], ok=q["ok"])
                print("baseline", c, base[c], flush=True)
        BASE.write_text(json.dumps(base, indent=1) + "\n")
    logs, wins, fails = [], 0, 0
    print(f"{'case':28s} {'class':38s} {'flux its':>8s} {'amg its':>7s} {'flux s':>7s} {'amg s':>6s} {'ratio':>6s}")
    for c, label in cases:
        try:
            r = run_flux(c)
        except Exception as e:                                           # a crash counts as a failed case, and is shown
            r = dict(ok=False, its=-1, time=float("nan"), build=0.0, K=-1)
            print("  crash:", c, repr(e)[:90])
        ratio = r["time"] / base[c]["time"] if r["ok"] else FAIL_RATIO
        ratio = min(ratio, FAIL_RATIO)
        logs.append(math.log2(ratio))
        wins += ratio < 1
        fails += not r["ok"]
        print(f"{c:28s} {label:38s} {r['its']:8d} {base[c]['its']:7d} {r['time']:7.2f} {base[c]['time']:6.2f} {ratio:6.2f}{'' if r['ok'] else ' FAIL'}", flush=True)
    print(f"WINS: {wins}/{len(cases)}")
    print(f"FAILS: {fails}")
    print(f"EQUIVALENCE: {equivalence():.1e}")
    print(f"PEAK_RSS_GB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f}")
    print(f"SCORE: {-sum(logs) / len(logs):.4f}")
