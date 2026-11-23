"""H56 exploration: does a looser match tolerance (approximate constant-coefficient background) rescue bt-fd / bt-fdd
when the coefficients vary smoothly (gen delta > 0)? usage: explore_tol.py TOLS METHODS CASE [CASE ...]"""
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
for case in sys.argv[3:]:
    A, b = run_real.load(case)
    for tol in map(float, sys.argv[1].split(",")):
        run_real.BTFD = partial(BTFD, tol=tol)
        for name in sys.argv[2].split(","):
            P = run_real.BTFD(A, multi="defl" if name.startswith("bt-fdd") else False) if name != "amg-3d" else None
            f = f"F {len(P.p3.F) / P.N:.1%}/{len(P.p1.F) / P.N:.1%}" if P else ""
            out = [run_real.run(A, b, name, t) for t in (1e-4, 1e-8)]
            print(f"{case} tol {tol:g} {name:9s} {f:14s} " + " | ".join(f"{t:g}: {r['its']} its {r['total_s']:.1f} s r{r['reason']}" for t, r in zip((1e-4, 1e-8), out)), flush=True)
            if name == "amg-3d":
                break
        if "amg-3d" in sys.argv[2] and sys.argv[2].split(",") == ["amg-3d"]:
            break
