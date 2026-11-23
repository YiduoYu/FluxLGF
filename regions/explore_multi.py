"""H54 exploration: bt-fdm (multi-region background) vs bt-fd vs amg-3d. Prints regions, feature size, linearity, timings."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
from time import perf_counter
for case in sys.argv[1:]:
    A, b = run_real.load(case)
    t = perf_counter(); P = BTFD(A, multi=True); ts = perf_counter() - t
    for nm, p in (("S3", P.p3), ("A11", P.p1)):
        print(f"{case} {nm}: {len(p.regions)} regions " + ", ".join(f"[{r['mask'].mean():.1%} k={r['kappa']:.2e} r={r['r']:.2e} b={r['beta']:.1e}]" for r in p.regions)
              + f"; feature {p.core_fraction:.2%} -> {len(p.F) / p.m**3:.2%} with ring", flush=True)
    rng = np.random.default_rng(0); x, y = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[0])
    t = perf_counter(); px = P(x); ta = perf_counter() - t
    print(f"   setup {ts:.1f} s, one apply {ta:.2f} s, linearity {np.linalg.norm(P(x + y) - px - P(y)) / np.linalg.norm(px + P(y)):.1e}", flush=True)
    for name in ("amg-3d", "bt-fd:2", "bt-fdm:2"):
        for tol in (1e-4, 1e-8):
            r = run_real.run(A, b, name, tol)
            print(f"   {name:9s} {tol:g}: its {r['its']:3d} reason {r['reason']} setup {r['setup_s']:.1f} total {r['total_s']:.1f} s", flush=True)
