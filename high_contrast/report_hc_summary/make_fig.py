"""Figures for the high-contrast solver summary (2026-11-11).
Numbers are copied from existing results:
  fig_fluxmg: fluxmg/structured/results_20261109.jsonl + fluxmg/amg_configs.jsonl (summarize.py output)
  fig_fdlgf : FD_LGF/report/tables_3d.tex, case A (chi^x), chi = 1e8
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": ["Heiti TC", "sans-serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, AQUA, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3df"
FAIL = None

# case label, FluxMG s, best hypre structured s (None = all failed), best BoomerAMG setting s
rows = [
    ("Poisson 64³", 0.38, 0.19, 0.58),
    ("各向异性 100", 0.73, 0.21, 0.60),
    ("各向异性 0.01", 0.45, 0.26, 0.59),
    ("对数正态 lognorm3", 0.39, 0.20, 0.60),
    ("导电管道 tubes", 0.51, 0.23, 0.69),
    ("绝缘板 insul", 0.47, 0.24, 0.62),
    ("渗流两相 perc3", 0.72, 0.24, 0.71),
    ("3T 类 SC26_01", 1.26, 1.66, 2.55),
    ("3T 强电子传导", 2.07, 2.23, 3.21),
    ("3T 强反馈", 1.44, 1.92, 2.35),
    ("3T 扁方块", 1.28, 1.20, 2.61),
    ("3T 渐变过渡", 1.82, 1.41, 2.65),
    ("3T 3 胶囊", 1.52, FAIL, 2.56),
    ("3T 10 胶囊", 1.78, FAIL, 2.71),
    ("3T 10 胶囊(种子2)", 1.80, FAIL, 2.79),
    ("3T 10 渐变扁方块", 3.07, FAIL, 2.87),
    ("多群 易", 0.82, 0.55, 0.52),
    ("多群 中", 0.33, 0.25, 0.32),
    ("多群 难", 0.83, 1.49, 1.56),
    ("真实 SC26_01", 23.16, FAIL, 29.01),
]
rows = rows[::-1]
fig, ax = plt.subplots(figsize=(6.4, 5.6))
FAILX = 5.0
for i, (lab, fl, st, am) in enumerate(rows):
    ax.plot([fl / am], [i], "o", ms=6, color=BLUE, mec="white", mew=1, zorder=3)
    if st is None:
        ax.plot([FAILX], [i], "x", ms=6, mew=2, color=ORANGE, zorder=3)
    else:
        ax.plot([st / am], [i], "s", ms=5.5, color=ORANGE, mec="white", mew=1, zorder=3)
ax.axvline(1.0, color="#52514e", lw=1)
ax.set_xscale("log")
ax.set_xlim(0.2, 6.5)
ax.set_xticks([0.25, 0.5, 1, 2, 4, FAILX])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "失败"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_xlabel("时间 / BoomerAMG 最优设置的时间（对数轴，<1 表示更快）")
ax.plot([], [], "o", color=BLUE, label="FluxMG")
ax.plot([], [], "s", color=ORANGE, label="hypre PFMG/SMG 最优（× 为全部失败）")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
for y in (0.5, 3.5, 12.5):
    ax.axhline(y, color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(HERE / "fig_fluxmg.pdf")

# FD-LGF on chi^x, 3D, chi = 1e8, total time (s)
n = [31, 63, 127]
series = [("FD-LGF", [0.005, 0.041, 0.335], AQUA, "o"),
          ("BoomerAMG", [0.100, 1.23, 8.86], BLUE, "s"),
          ("GMG", [0.033, 0.257, 2.62], ORANGE, "^")]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for lab, t, c, m in series:
    ax.plot(n, t, "-", marker=m, color=c, lw=2, ms=6, mec="white", mew=1)
    ax.annotate(lab, (n[-1], t[-1]), xytext=(6, 0), textcoords="offset points", va="center", color="#0b0b0b", fontsize=8)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(n); ax.set_xticklabels([f"{k}³" for k in n])
ax.set_xlim(26, 190)
ax.set_xlabel("网格 n³"); ax.set_ylabel("总时间 (s，单线程)")
ax.grid(color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(HERE / "fig_fdlgf.pdf")
print("ok")
