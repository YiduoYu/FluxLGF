"""Round 3 timed comparison under the project's rules: one process per run, shuffled order, repeats, same accuracy test.
Each case runs SSR with the given configuration, BoomerAMG fast.25, and the fastest BoomerAMG configuration of the baseline sweep
(amg_baseline.json for dev/holdout, amg_weak.json for weak/weakhold; includes AIR, systems and nodal where they apply).
Reported time is the minimum over repeats for each method.
usage: final3.py dev|holdout|weak|weakhold REPEATS OUTTAG [KEY=VALUE ...]      writes round3/final_<set>_<tag>.json
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
LIMIT = 200
FAIL = 8.0


def call(task, case, kw, weak):
    if task == "ssr":
        cmd = [sys.executable, str(HERE / "round3" / "ssr_one.py"), case, *kw]
    elif weak:
        cmd = [sys.executable, str(HERE / "baseline_weak.py"), "one", case, task]
    else:
        cmd = [sys.executable, str(HERE / "baseline.py"), "one", case, task]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=LIMIT)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
        if not line:
            return dict(ok=False, its=-1, t=None, err=p.stderr[-160:])
        q = json.loads(line[0][7:])
        q["t"] = q.get("time", q["cpu"])
        return q
    except subprocess.TimeoutExpired:
        return dict(ok=False, its=-1, t=None, err="timeout")


def main():
    which, rep, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    kw = sys.argv[4:]
    weak = which.startswith("weak")
    if weak:
        from problems_weak import WEAK, WEAK_HOLD, WEAK_HOLD2
        table = {"weak": WEAK, "weakhold": WEAK_HOLD, "weakhold2": WEAK_HOLD2}[which]
        base = json.loads((HERE / "amg_weak.json").read_text())
    else:
        from problems import DEV, HOLDOUT
        table = DEV if which == "dev" else HOLDOUT
        base = json.loads((HERE / "amg_baseline.json").read_text())
    rng = random.Random(11)
    out = {}
    path = HERE / "round3" / f"final_{which}_{tag}.json"
    for case, label in table:
        ok = {k: q for k, q in base[case].items() if q["ok"]}
        best = min(ok, key=lambda k: ok[k]["cpu"]) if ok else "fast.25"
        tasks = sorted({"ssr", "fast.25", best})
        r = rep if not case.startswith("elast") else 1
        runs = [t for t in tasks for _ in range(r)]
        rng.shuffle(runs)
        res = {t: [] for t in tasks}
        for t in runs:
            res[t].append(call(t, case, kw, weak))
        s = {}
        for t in tasks:
            good = [q for q in res[t] if q["ok"]]
            s[t] = dict(ok=bool(good), its=(good or res[t])[0]["its"], t=min(q["t"] for q in good) if good else None, times=[q["t"] for q in res[t]],
                        lev=res[t][0].get("lev"), cx=res[t][0].get("cx"), build=res[t][0].get("build"), rss=max(q.get("rss", 0) or 0 for q in res[t]))
        out[case] = dict(label=label, best=best, **s)
        path.write_text(json.dumps(out, indent=1) + "\n")

        def ratio(x):
            if not s["ssr"]["ok"]:
                return FAIL
            return min(s["ssr"]["t"] / s[x]["t"], FAIL) if s[x]["ok"] else 1 / FAIL
        print(f"{case:28s} ssr {s['ssr']['its']:5d} its {str(s['ssr']['t'] and round(s['ssr']['t'], 2)):>7s} s | fast.25 {s['fast.25']['its']:5d} {str(s['fast.25']['t'] and round(s['fast.25']['t'], 2)):>7s} {ratio('fast.25'):5.2f} | {best:10s} {s[best]['its']:5d} {str(s[best]['t'] and round(s[best]['t'], 2)):>7s} {ratio(best):5.2f}", flush=True)
    keys = list(out)

    def score(pick):
        L = []
        for c in keys:
            s = out[c]; x = pick(c)
            if not s["ssr"]["ok"]:
                L.append(math.log2(FAIL))
            elif not s[x]["ok"]:
                L.append(-math.log2(FAIL))
            else:
                L.append(math.log2(min(max(s["ssr"]["t"] / s[x]["t"], 1 / FAIL), FAIL)))
        return -sum(L) / len(L), sum(v < 0 for v in L)
    a, wa = score(lambda c: "fast.25")
    b, wb = score(lambda c: out[c]["best"])
    print(f"SCORE vs fast.25: {a:.3f}  wins {wa}/{len(keys)}")
    print(f"SCORE vs best per case: {b:.3f}  wins {wb}/{len(keys)}")
    print("FAILS ssr:", [c for c in keys if not out[c]["ssr"]["ok"]])


if __name__ == "__main__":
    main()
