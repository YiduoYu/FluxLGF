"""H56 exploration: many inclusions. bt-fdd variants: MIN_REGION (every inclusion gets a region + coarse vector,
Vuik-style one deflation vector per inclusion), ring width, MUMPS BLR on the feature LU.
usage: explore_many.py VARIANTS CASE [CASE ...]   VARIANT = minreg/ring/blr, e.g. 0/2/0 (0 = default)"""
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real, rsolve
from btfd import BTFD
for case in sys.argv[2:]:
    A, b = run_real.load(case)
    for v in sys.argv[1].split(","):
        minreg, ring, blr = v.split("/")
        rsolve.MultiFDLGF.MIN_REGION = int(minreg) or None
        rsolve.MultiFDLGF.BLR = float(blr)
        t = perf_counter(); P = BTFD(A, ring=int(ring), multi="defl"); ts = perf_counter() - t
        x = np.random.default_rng(0).standard_normal(A.shape[0]); y = np.random.default_rng(1).standard_normal(A.shape[0])
        lin = np.linalg.norm(P(x + y) - P(x) - P(y)) / np.linalg.norm(P(x) + P(y))
        out = [run_real.run(A, b, f"bt-fdd:{ring}", t) for t in (1e-4, 1e-8)]
        print(f"{case} {v:10s} F {len(P.p3.F) / P.N:.1%}/{len(P.p1.F) / P.N:.1%} regions {len(P.p3.regions)}/{len(P.p1.regions)} "
              f"K {P.p3.K}/{P.p1.K} setup {ts:.1f} lin {lin:.0e} | " + " | ".join(f"{t:g}: {r['its']} its {r['total_s']:.1f} s r{r['reason']}" for t, r in zip((1e-4, 1e-8), out)), flush=True)
