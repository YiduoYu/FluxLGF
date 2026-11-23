"""H52 figure: real SC26_01 times (H40 criterion medians of 3 passes; official criterion by IR) and gen3t m=128 ratios."""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
med = {}
for f in sorted(R.glob("h53_R*.csv")):
    for r in csv.DictReader(open(f)):
        med.setdefault((r["method"], float(r["tol"])), []).append(float(r["total_s"]))
med = {k: float(np.median(v)) for k, v in med.items()}
ir = {(r["method"], float(r["target"])): float(r["total_s"]) for r in csv.DictReader(open(R / "h53_IR.csv"))}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
names = [("amg-3d", "BoomerAMG", "0.55"), ("bt-v", "bt-v (AMG blocks)", "C1"), ("bt-fd:2", "bt-fd (no AMG)", "C0")]
for a, data, title, key in ((ax[0], med, "real SC26_01, solver-reported residual\n(median of 3)", lambda n, t: (n, t)),
                            (ax[1], ir, "real SC26_01, official ||b-Ax||/||b||\n(double-double refinement)",
                             lambda n, t: (n.replace(":2", ""), t))):
    x = np.arange(2)
    for i, (n, lab, col) in enumerate(names):
        v = [data[key(n, t)] for t in (1e-4, 1e-8)]
        a.bar(x + (i - 1) * 0.27, v, 0.27, label=lab, color=col)
        for xx, vv in zip(x + (i - 1) * 0.27, v):
            a.text(xx, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(x, ["1e-4", "1e-8"]); a.set_ylabel("total time (s)"); a.set_title(title, fontsize=9)
ax[0].legend(fontsize=7)
cases = sorted(R.glob("h53_G_*.csv"))
lab, r4, r8 = [], [], []
for f in cases:
    d = {(r["method"], float(r["tol"])): float(r["total_s"]) for r in csv.DictReader(open(f))}
    p = f.stem.replace("h53_G_", "").split("_")
    lab.append(f"{p[0]}/{p[1]}" + ("" if p[2] == "1" else f" c{p[2]}") + ("" if p[3] == "2e-11" else f" F{p[3]}"))
    r4.append(d[("bt-fd:2", 1e-4)] / d[("amg-3d", 1e-4)]); r8.append(d[("bt-fd:2", 1e-8)] / d[("amg-3d", 1e-8)])
x = np.arange(len(lab))
ax[2].bar(x - 0.2, r4, 0.4, label="1e-4"); ax[2].bar(x + 0.2, r8, 0.4, label="1e-8")
ax[2].axhline(1, color="k", lw=0.8); ax[2].set_xticks(x, lab, rotation=45, fontsize=7, ha="right")
ax[2].set_ylabel("bt-fd / BoomerAMG time"); ax[2].set_title("own 3T, m = 128 (12 matrices, 5 new)", fontsize=9); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "results53.png", dpi=160)
print(med); print(ir); print(list(zip(lab, np.round(r4, 2), np.round(r8, 2))))
