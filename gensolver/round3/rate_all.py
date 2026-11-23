"""Stationary convergence factor of SSR (x <- x - M A x, power iteration from a random start) on every case of a suite.
A factor above 1 means the cycle amplifies some mode and FGMRES is only hiding it; that is how the excess-update bug was found.
usage: rate_all.py dev|holdout|weak|weakhold|weakhold2 [KEY=VALUE ...]      one subprocess per case (rate_all.py one CASE ...)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def one(case, kw):
    import numpy as np
    import ssr
    from problems_weak import load
    A, b, _ = load(case)
    A = A.tocsr()
    P = ssr.SSR(A, **kw)
    x = np.random.default_rng(0).standard_normal(A.shape[0])
    f = []
    for _ in range(25):
        x = x - P(A @ x)
        nx = np.linalg.norm(x); f.append(nx); x /= nx
    print("RESULT " + json.dumps(dict(rate=f[-1], rate5=float(np.exp(np.mean(np.log(f[-5:])))), lev=P.levels, cx=P.complexity)))


def parse(args):
    kw = {}
    for a in args:
        k, v = a.split("="); kw[k] = float(v) if "." in v or "e" in v else int(v)
    return kw


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], parse(sys.argv[3:]))
        sys.exit()
    which = sys.argv[1]
    if which in ("dev", "holdout"):
        from problems import DEV, HOLDOUT
        cases = DEV if which == "dev" else HOLDOUT
    else:
        from problems_weak import WEAK, WEAK_HOLD, WEAK_HOLD2
        cases = {"weak": WEAK, "weakhold": WEAK_HOLD, "weakhold2": WEAK_HOLD2}[which]
    for c, _ in cases:
        try:
            p = subprocess.run([sys.executable, __file__, "one", c, *sys.argv[2:]], capture_output=True, text=True, timeout=300)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
            q = json.loads(line[0][7:]) if line else None
        except subprocess.TimeoutExpired:
            q = None
        if q:
            print(f"{c:30s} rate {q['rate5']:10.4g}  lev {q['lev']:3d}  cx {q['cx']:5.2f}{'  AMPLIFIES' if q['rate5'] > 1 else ''}", flush=True)
        else:
            print(f"{c:30s} error", flush=True)
