"""Held-out check of the grouping rule (round 2): the probe rule against the pairwise rule alone and against BoomerAMG.

The cases below were fixed before the rule was implemented and were not used to choose PROBE_TAU or PROBE_BOXES: new seeds,
new parameters, new group counts, m = 48 and 64. `blob` is mgrd with a small heat capacity only on a fraction of the cells.
Same criterion as eval_fluxmg.py: double-double original residual 1e-8, single thread, CPU time including setup.
usage: eval_grouping.py
"""
import math
import resource
import sys
from pathlib import Path
from time import process_time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
import flux_mg  # noqa: E402
from try_one import load, solve_ir  # noqa: E402

CASES = (
    "mg:32:6:10:0.03:0:21",
    "mg:32:6:10:3:1:21",
    "mg:32:12:3:0.03:1:22",
    "mg:32:12:30:0.3:0:22",
    "mg:48:8:0.1:0.01:0:23",
    "mg:48:8:10:1e-3:1:23",
    "mg:48:4:3:0.3:0:24",
    "mg:32:8:0.1:1:0:25",
    "mg:64:12:10:0.05:0:26",
    "mg:64:6:1:10:1:26",
    "blob:32:6:3:1e-2:0:31:0.1",
    "blob:32:6:3:1e-2:1:31:0.1",
    "blob:32:12:30:1e-2:0:32:0.03",
    "blob:48:8:1:1e-2:0:33:0.03",
)


def blob(m, ng, sbar, zeta, lam, seed, frac):
    """mgrd with c_v = 1e2 * sbar, except c_v = zeta * sbar on about frac of the cells."""
    from mgrd import generate
    A, b = generate(m, ng, sbar, 1e2, lam, seed)
    f = gaussian_filter(np.random.default_rng(seed + 1).standard_normal((m, m, m)), sigma=m / 12, mode="wrap").ravel()
    z, y, x = np.indices((m, m, m)) / m
    ball = (np.sqrt(x**2 + y**2 + z**2) < 0.35).ravel()
    drop = (1e2 - zeta) * sbar * np.where(ball, 10.0, 1.0) * (f > np.quantile(f, 1 - frac))
    A = (A - sp.diags(np.concatenate([np.zeros(ng * m**3), drop]))).tocsr()
    A.sort_indices()
    return A, b, (m, m, m), ng + 1


def get(case):
    if case.startswith("blob"):
        p = case.split(":")
        return blob(int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), int(p[6]), float(p[7]))
    return load(case)


def flux(A, b, dims, F, tau):
    flux_mg.PROBE_TAU = tau
    t0 = process_time()
    P = flux_mg.FluxMG(A, dims, F)
    build = process_time() - t0
    q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=200, max_steps=6)
    return dict(ok=q["ok"], its=q["its"], time=build + q["cpu"], s=P.P.structure())


if __name__ == "__main__":
    print(f"{'case':30s} {'amg its':>7s} {'amg s':>6s} | {'pair its':>8s} {'s':>6s} {'ratio':>5s} | {'rule its':>8s} {'s':>6s} {'ratio':>5s} | rule structure")
    lp, lr, wp, wr = [], [], 0, 0
    for c in CASES:
        A, b, dims, F = get(c)
        a = min((solve_ir(A, b, "amg", theta=0.25) for _ in range(2)), key=lambda q: q["cpu"])
        p, r = flux(A, b, dims, F, np.inf), flux(A, b, dims, F, 0.7)
        rp, rr = [min(q["time"] / a["cpu"], 8.0) if q["ok"] else 8.0 for q in (p, r)]
        lp.append(math.log2(rp)); lr.append(math.log2(rr)); wp += rp < 1; wr += rr < 1
        print(f"{c:30s} {a['its']:7d} {a['cpu']:6.2f} | {p['its']:8d} {p['time']:6.2f} {rp:5.2f} | {r['its']:8d} {r['time']:6.2f} {rr:5.2f} | {r['s']}", flush=True)
    print(f"PAIR  SCORE: {-sum(lp) / len(lp):.4f}  WINS: {wp}/{len(CASES)}")
    print(f"RULE  SCORE: {-sum(lr) / len(lr):.4f}  WINS: {wr}/{len(CASES)}")
    print(f"PEAK_RSS_GB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f}")
