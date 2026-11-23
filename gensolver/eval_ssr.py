"""Autoresearch eval: SSR with one fixed configuration against tuned BoomerAMG on the generalization suite.

Accuracy: double-double original residual 1e-8 for both (krylov.py). Time: CPU seconds with setup. One process per case.
SCORE = -mean log2(t_ssr / t_ref), failure counts as ratio 8. Two references are printed:
  SCORE      t_ref = the single BoomerAMG configuration with the best score over the suite (a user would pick one)
  ORACLE     t_ref = the fastest BoomerAMG configuration on each case, including nodal for elasticity (unfair to SSR, kept as a guard)
usage: eval_ssr.py [--holdout] [--cases a,b] [KEY=VALUE ...]        KEY=VALUE overrides ssr.py levers for all cases
"""
import json
import math
import subprocess
import sys
from pathlib import Path
from time import process_time

HERE = Path(__file__).resolve().parent
FAIL = 8.0
TIME_LIMIT = 90


def one(case, kw):
    sys.path.insert(0, str(HERE))
    import ssr
    from krylov import run
    from problems import load
    A, b, _ = load(case)
    t0 = process_time()
    P = ssr.SSR(A, **kw)
    build = process_time() - t0
    q = run(A, b, "py", P=P)
    q.update(time=build + q["cpu"], build=build, lev=P.levels, cx=P.complexity)
    print("RESULT " + json.dumps(q))


def fixed_best(base, cases):
    names = set.intersection(*[set(base[c]) for c in cases])
    names = {k for k in names if not k.startswith("nodal")}
    def mean_log(k):
        return sum(math.log2(base[c][k]["cpu"]) if base[c][k]["ok"] else math.log2(FAIL * oracle(base, c)) for c in cases) / len(cases)
    return min(names, key=mean_log)


def oracle(base, c):
    return min(q["cpu"] for q in base[c].values() if q["ok"])


def main():
    sys.path.insert(0, str(HERE))
    from problems import DEV, HOLDOUT, STRETCH
    kw = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    cases = list(HOLDOUT if "--holdout" in sys.argv else DEV)
    for a in sys.argv[1:]:
        if a.startswith("--cases="):
            want = a[8:].split(",")
            cases = [c for c in cases if c[0] in want]
    base = json.loads((HERE / "amg_baseline.json").read_text())
    cases = [c for c in cases if c[0] in base and base[c[0]]]
    names = [c for c, _ in cases]
    ref = fixed_best(base, [c for c, _ in DEV if c in base and base[c]])
    print(f"fixed BoomerAMG reference: {ref}")
    print(f"{'case':26s} {'class':32s} {'its':>5s} {'lev':>4s} {'cx':>6s} {'build':>6s} {'time':>7s} | {'ref its':>7s} {'ref s':>6s} {'ratio':>6s} | {'best s':>6s} {'ratio':>6s}")
    logs, ologs, wins, owins, fails = [], [], 0, 0, 0
    for c, label in cases:
        try:
            p = subprocess.run([sys.executable, __file__, "one", c] + kw, capture_output=True, text=True, timeout=TIME_LIMIT)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
            r = json.loads(line[0][7:]) if line else dict(ok=False, its=-1, time=float("nan"), build=0, lev=0, cx=0)
        except subprocess.TimeoutExpired:
            r = dict(ok=False, its=-1, time=float("nan"), build=0, lev=0, cx=0)
        rq = base[c][ref]
        tref = rq["cpu"] if rq["ok"] else FAIL * oracle(base, c)
        topt = oracle(base, c)
        ratio = min(r["time"] / tref, FAIL) if r["ok"] else FAIL
        oratio = min(r["time"] / topt, FAIL) if r["ok"] else FAIL
        logs.append(math.log2(ratio)); ologs.append(math.log2(oratio))
        wins += ratio < 1; owins += oratio < 1; fails += not r["ok"]
        print(f"{c:26s} {label[:32]:32s} {r['its']:5d} {r['lev']:4d} {r['cx']:6.2f} {r['build']:6.2f} {r['time']:7.2f} | {rq['its']:7d} {tref:6.2f} {ratio:6.2f} | {topt:6.2f} {oratio:6.2f}{'' if r['ok'] else ' FAIL'}", flush=True)
    print(f"WINS_FIXED: {wins}/{len(cases)}  WINS_ORACLE: {owins}/{len(cases)}  FAILS: {fails}")
    print(f"ORACLE_SCORE: {-sum(ologs) / len(ologs):.4f}")
    print(f"SCORE: {-sum(logs) / len(logs):.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        kw = {}
        for a in sys.argv[3:]:
            k, v = a.split("=")
            kw[k] = float(v) if "." in v or "e" in v else int(v)
        one(sys.argv[2], kw)
    else:
        main()
