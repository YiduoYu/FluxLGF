"""SSR on one case of either suite: ssr_one.py CASE KEY=VALUE ...  prints RESULT json (same fields as eval_ssr.py one)."""
import json, sys
from pathlib import Path
from time import process_time
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ssr
from krylov import run
from problems_weak import load
case = sys.argv[1]
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("="); kw[k] = float(v) if "." in v or "e" in v else int(v)
A, b, _ = load(case)
t0 = process_time()
P = ssr.SSR(A, **kw)
build = process_time() - t0
q = run(A, b, "py", P=P)
q.update(time=build + q["cpu"], build=build, lev=P.levels, cx=P.complexity)
print("RESULT " + json.dumps(q))
