"""H55 figure: gen3t m=128 time ratios vs BoomerAMG (bt-fd and bt-fdd, both tolerances) and real SC26_01 times."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[2.3, 1]))
lab, rows = [], []
for f in sorted(R.glob("h55_G_*.csv")):
    d = {(r["method"].split(":")[0], float(r["tol"])): float(r["total_s"]) for r in csv.DictReader(open(f))}
    p = f.stem.replace("h55_G_", "").split("_")
    lab.append(f"{p[0]}/{p[1]}" + ("" if p[2] == "1" else f" c{p[2]}") + ("" if p[3] == "2e-11" else f" F{p[3]}"))
    rows.append([d[(m, t)] / d[("amg-3d", t)] for m in ("bt-fd", "bt-fdd") for t in (1e-4, 1e-8)])
rows = np.array(rows); x = np.arange(len(lab)); w = 0.2
for j, (l, c) in enumerate((("bt-fd 1e-4", "C0"), ("bt-fd 1e-8", "#9ecae1"), ("bt-fdd 1e-4", "C3"), ("bt-fdd 1e-8", "#fc9272"))):
    ax[0].bar(x + (j - 1.5) * w, rows[:, j], w, label=l, color=c)
ax[0].axhline(1, color="k", lw=0.8); ax[0].set_xticks(x, lab, rotation=45, fontsize=7, ha="right")
ax[0].set_ylabel("time / BoomerAMG time"); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("own 3T, m = 128 (12 matrices): bt-fdd 0.47-0.71x, bt-fd 0.34-1.80x", fontsize=9)
med = {}
for p in (1, 2, 3):
    for r in csv.DictReader(open(R / f"h55_R{p}.csv")):
        med.setdefault((r["method"].split(":")[0], float(r["tol"])), []).append(float(r["total_s"]))
med = {k: float(np.median(v)) for k, v in med.items()}
ir = {(r["method"], float(r["target"])): float(r["total_s"]) for r in csv.DictReader(open(R / "h55_IR.csv"))}
groups = [("reported 1e-4", lambda m: med[(m, 1e-4)]), ("official 1e-4", lambda m: ir[(m, 1e-4)]), ("official 1e-8", lambda m: ir[(m, 1e-8)])]
for i, (m, l, c) in enumerate((("amg-3d", "BoomerAMG", "0.55"), ("bt-fd", "bt-fd", "C0"), ("bt-fdd", "bt-fdd", "C3"))):
    v = [g(m) for _, g in groups]; xx = np.arange(3) + (i - 1) * 0.27
    ax[1].bar(xx, v, 0.27, label=l, color=c)
    for a_, b_ in zip(xx, v):
        ax[1].text(a_, b_, f"{b_:.1f}", ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(np.arange(3), [g for g, _ in groups], fontsize=8); ax[1].set_ylabel("total time (s)")
ax[1].set_title("real SC26_01", fontsize=9); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "results55.png", dpi=160)
print(med, ir)
