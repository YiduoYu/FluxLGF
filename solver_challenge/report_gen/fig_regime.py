"""Regime map of H43: outer iterations per method and time of the fastest bt variant relative to amg-3d.
usage: python fig_regime.py h43_m64.csv OUT.png"""
import csv, sys
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
rows = list(csv.DictReader(open(sys.argv[1])))
Fs = sorted({float(r["F"]) for r in rows}); Gs = sorted({float(r["G"]) for r in rows})
def grid(method, key):
    M = np.full((len(Fs), len(Gs)), np.nan)
    for r in rows:
        if r["method"] == method:
            v = float(r[key]) if int(r["reason"]) > 0 else np.nan
            M[Fs.index(float(r["F"])), Gs.index(float(r["G"]))] = v
    return M
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
titles = {"amg-3d": "整体 BoomerAMG", "bt-v": "bt-v", "bt-v2": "bt-v2（电子块用 AMG）", "bt-v2x2": "bt-v2×2（两次扫描）"}
for ax, mth in zip(axes[:4], titles):
    M = grid(mth, "its")
    im = ax.imshow(M, cmap="Blues", vmin=10, vmax=95, origin="lower")
    for i in range(len(Fs)):
        for j in range(len(Gs)):
            ax.text(j, i, "失败" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > 60) else "black")
    ax.set_title(titles[mth] + "：迭代数", fontsize=10)
best = np.nanmin(np.stack([grid(m, "total_s") for m in ("bt-v", "bt-v2", "bt-v2x2")]), axis=0)
ratio = best / grid("amg-3d", "total_s")
im = axes[4].imshow(ratio, cmap="RdBu_r", vmin=0.4, vmax=1.6, origin="lower")
for i in range(len(Fs)):
    for j in range(len(Gs)):
        axes[4].text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if ratio[i, j] > 1.4 else "black")
axes[4].set_title("最快 bt 变体时间 / BoomerAMG 时间", fontsize=10)
for ax in axes:
    ax.set_xticks(range(len(Gs))); ax.set_xticklabels([f"{g:g}" for g in Gs], fontsize=8)
    ax.set_yticks(range(len(Fs))); ax.set_yticklabels([f"{f:g}" for f in Fs], fontsize=8)
    ax.set_xlabel("电子扩散 G")
axes[0].set_ylabel("辐射反馈 F")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
