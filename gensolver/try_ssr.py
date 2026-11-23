"""Quick look: try_ssr.py CASE [CASE...] [KEY=VALUE ...]   Levers are ssr.py constants. Prints iterations, time, complexity."""
import sys
from pathlib import Path
from time import process_time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from krylov import run  # noqa: E402
from problems import load  # noqa: E402
import ssr  # noqa: E402


if __name__ == "__main__":
    kw = {}
    cases = []
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("="); kw[k] = float(v) if "." in v or "e" in v else int(v)
        else:
            cases.append(a)
    for c in cases:
        A, b, info = load(c)
        t0 = process_time()
        P = ssr.SSR(A, **kw)
        build = process_time() - t0
        q = run(A, b, "py", P=P)
        print(f"{c:26s} n={A.shape[0]:7d} ok={q['ok']!s:5s} its={q['its']:4d} rel={q['rel']:.1e} build={build:6.2f} total={build + q['cpu']:6.2f} lev={P.levels:3d} cx={P.complexity:5.2f} tr={P.transfer:5.2f} rss={q['rss']:.2f}", flush=True)
