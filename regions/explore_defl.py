"""H55 exploration: bt-fdd (multi-region background + island coarse space) vs bt-fd / bt-fdm / amg-3d.
Prints regions, coarse Schur S, feature size, linearity defect, timings. usage: explore_defl.py METHODS CASE [CASE ...]"""
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
import rsolve
if len(sys.argv) > 3 and sys.argv[2].startswith("form="):
    rsolve.DeflatedFDLGF.FORM = sys.argv.pop(2)[5:]
for case in sys.argv[2:]:
    A, b = run_real.load(case)
    t = perf_counter(); P = BTFD(A, multi="defl"); ts = perf_counter() - t
    for nm, p in (("S3", P.p3), ("A11", P.p1)):
        print(f"{case} {nm}: " + ", ".join(f"[{r['mask'].mean():.1%} k={r['kappa']:.2e}{' L' if r['largest'] else ''}]" for r in p.regions)
              + f"; F {len(p.F) / p.m**3:.2%}; K {p.K}, diag D {np.diag(p.D)}", flush=True)
    rng = np.random.default_rng(0); x, y = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[0])
    t = perf_counter(); px = P(x); ta = perf_counter() - t
    print(f"   setup {ts:.1f} s, one apply {ta:.2f} s, linearity {np.linalg.norm(P(x + y) - px - P(y)) / np.linalg.norm(px + P(y)):.1e}", flush=True)
    for name in sys.argv[1].split(","):
        for tol in (1e-4, 1e-8):
            r = run_real.run(A, b, name, tol)
            print(f"   {name:9s} {tol:g}: its {r['its']:3d} reason {r['reason']} setup {r['setup_s']:.1f} total {r['total_s']:.1f} s", flush=True)
