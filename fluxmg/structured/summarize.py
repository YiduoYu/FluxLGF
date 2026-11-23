"""Summary of compare_structured.py --run output against the predictions P1-P5 in its docstring.
Per (case, method): median CPU time over repeats; a pair that failed once counts as failed.
usage: summarize.py RESULTS.jsonl [AMG_CONFIGS.jsonl]   the second file adds BoomerAMG best per case over its settings
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from eval_fluxmg import CASES, HOLDOUT  # noqa: E402
from eval_grouping import CASES as GROUPING  # noqa: E402

FAIL = 8.0


def load(path, key):
    t, bad, its, joint = defaultdict(list), set(), {}, set()
    for line in Path(path).read_text().splitlines():
        q = json.loads(line)
        k = (q["case"], q[key])
        if not q["ok"]:
            bad.add(k); continue
        t[k].append(q["time"]); its[k] = q["its"]
        if q.get("joint"):
            joint.add(k)
    med = {k: float(np.median(v)) for k, v in t.items() if k not in bad}
    return med, bad, its, joint


def main():
    med, bad, its, joint = load(sys.argv[1], "method")
    amgbest = {}
    if len(sys.argv) > 2:
        m2, b2, i2, _ = load(sys.argv[2], "cfg")
        for (c, k), v in m2.items():
            if c not in amgbest or v < amgbest[c][0]:
                amgbest[c] = (v, k, i2[(c, k)])
    groups = (("suite", [c for c, _ in CASES]), ("holdout", [c for c, _ in HOLDOUT]), ("grouping", list(GROUPING)), ("real", ["real"]))
    meths = ("flux", "pfmg", "pfmg_rb", "smg", "amg")
    p5 = []
    for name, cs in groups:
        print(f"\n== {name}")
        print(f"{'case':34s}" + "".join(f"{m:>15s}" for m in meths) + f"{'amg best':>22s}  flux/struct  flux/amgbest")
        lr, la = [], []
        for c in cs:
            row = []
            for m in meths:
                k = (c, m)
                row.append("FAIL" if k in bad else f"{med[k]:.2f}/{its[k]}{'*' if k in joint else ''}" if k in med else "-")
            st = [med[(c, m)] for m in ("pfmg", "pfmg_rb", "smg") if (c, m) in med]
            fl = med.get((c, "flux"))
            r = (fl / min(st)) if fl and st else (FAIL if not fl and st else 1 / FAIL if fl else None)
            if r is not None:
                lr.append(np.log2(r))
                if name in ("suite", "holdout"):
                    p5.append(r < 1)
            ab = amgbest.get(c)
            ra = (fl / ab[0]) if fl and ab else None
            if ra is not None:
                la.append(np.log2(ra))
            abs_ = f"{ab[0]:.2f}/{ab[2]} {ab[1]}" if ab else "-"
            print(f"{c:34s}" + "".join(f"{x:>15s}" for x in row) + f"{abs_:>22s}  {r if r is None else round(r, 2)!s:>10s}  {ra if ra is None else round(ra, 2)!s:>10s}")
        if lr:
            print(f"   SCORE vs best structured {-np.mean(lr):+.3f}, flux faster on {sum(x < 0 for x in lr)}/{len(lr)}")
        if la:
            print(f"   SCORE vs best BoomerAMG setting {-np.mean(la):+.3f}, flux faster on {sum(x < 0 for x in la)}/{len(la)}")
    print(f"\nP5 as registered (flux faster than best structured on >= 70 % of suite + held-out cases): {np.mean(p5):.0%} -> {'holds' if np.mean(p5) >= 0.7 else 'fails'}")
    print("* = a group of several fields solved by one BoomerAMG V-cycle inside the structured method (hybrid)")


if __name__ == "__main__":
    main()
