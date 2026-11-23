"""Round 3 screening eval on DEV or WEAK: SSR (one process per case) against BoomerAMG fast.25 and against the best configuration per case.
usage: eval3.py dev|weak|weakhold [KEY=VALUE ...]      SCORE = -mean log2(t_ssr / t_ref), failure counts as ratio 8.
Baselines: amg_baseline.json for dev, amg_weak.json for weak sets (screening only; final timing uses the shuffled protocol).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
FAIL = 8.0
which = sys.argv[1]
kw = sys.argv[2:]
if which == "dev":
    from problems import DEV as CASES
    base = json.loads((HERE / "amg_baseline.json").read_text())
else:
    from problems_weak import WEAK, WEAK_HOLD, WEAK_HOLD2
    CASES = {"weak": WEAK, "weakhold": WEAK_HOLD, "weakhold2": WEAK_HOLD2}[which]
    base = json.loads((HERE / "amg_weak.json").read_text())
L1, L2, w1, w2, fails = [], [], 0, 0, 0
print(f"{'case':28s} {'its':>5s} {'lev':>4s} {'cx':>6s} {'build':>6s} {'time':>7s} | {'f.25':>6s} {'ratio':>6s} | {'best':>10s} {'s':>6s} {'ratio':>6s}")
for c, label in CASES:
    try:
        p = subprocess.run([sys.executable, str(HERE / "round3" / "ssr_one.py"), c, *kw], capture_output=True, text=True, timeout=150)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
        r = json.loads(line[0][7:]) if line else dict(ok=False, its=-1, time=float("nan"), build=0, lev=0, cx=0)
    except subprocess.TimeoutExpired:
        r = dict(ok=False, its=-1, time=float("nan"), build=0, lev=0, cx=0)
    b = base[c]
    ok = {k: q for k, q in b.items() if q["ok"]}
    best = min(ok, key=lambda k: ok[k]["cpu"]) if ok else None
    tb = ok[best]["cpu"] if best else None
    f = b.get("fast.25", {})
    tf = f["cpu"] if f.get("ok") else None
    def ratio(t):
        if not r["ok"]:
            return FAIL
        if t is None:
            return 1 / FAIL
        return min(max(r["time"] / t, 1 / FAIL), FAIL)
    r1, r2 = ratio(tf), ratio(tb)
    L1.append(math.log2(r1)); L2.append(math.log2(r2)); w1 += r1 < 1; w2 += r2 < 1; fails += not r["ok"]
    print(f"{c:28s} {r['its']:5d} {r['lev']:4d} {r['cx']:6.2f} {r['build']:6.2f} {r['time']:7.2f} | {str(tf and round(tf, 2)):>6s} {r1:6.2f} | {str(best):>10s} {str(tb and round(tb, 2)):>6s} {r2:6.2f}{'' if r['ok'] else ' FAIL'}", flush=True)
n = len(CASES)
print(f"SCORE_FAST25: {-sum(L1) / n:.4f}  wins {w1}/{n}")
print(f"SCORE_BEST: {-sum(L2) / n:.4f}  wins {w2}/{n}  fails {fails}")
