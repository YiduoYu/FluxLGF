"""H56 exploration: cheaper feature factorisation (literature: MUMPS block low-rank, METIS nested dissection).
usage: explore_lu.py METHODS VARIANTS CASE [CASE ...]   VARIANT = blr/ordering, e.g. 0/0 (default), 1e-8/0, 0/5"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real, rsolve
for case in sys.argv[3:]:
    A, b = run_real.load(case)
    for v in sys.argv[2].split(","):
        blr, order = v.split("/")
        rsolve.ScalarFDLGF.BLR, rsolve.ScalarFDLGF.ORDERING = float(blr), int(order)
        for name in sys.argv[1].split(","):
            out = [run_real.run(A, b, name, t) for t in (1e-4, 1e-8)]
            print(f"{case} {v:8s} {name:9s} " + " | ".join(f"{t:g}: {r['its']} its setup {r['setup_s']:.1f} solve {r['solve_s']:.1f} total {r['total_s']:.1f} s r{r['reason']}" for t, r in zip((1e-4, 1e-8), out)), flush=True)
