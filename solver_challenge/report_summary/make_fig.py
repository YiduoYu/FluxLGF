"""m=128 total time to 1e-8 for the summary report (numbers from results/h37_*, h37b_*, h38.csv)."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
cases = ["对称\nΔt=1e-3", "对称\nΔt=10", "发射线性化\nΔt=1e-3", "发射线性化\nΔt=10", "行缩放\nΔt=1e-3", "行缩放\nΔt=10"]
# (time s, converged); None = not run
data = {"BoomerAMG 三维参数": [(13.0, 1), (17.2, 1), (15.0, 1), (375.7, 0), (49.8, 1), (361.4, 0)],
        "FD-LGF-3T（含 LGF）": [(23.7, 1), (33.0, 1), None, None, None, None],
        "agg3t-w（去 LGF）": [(7.4, 1), (10.0, 1), (8.6, 1), (11.9, 1), (28.0, 1), (41.9, 1)]}
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(figsize=(8.2, 3.6))
w = 0.26
for j, ((name, vals), c) in enumerate(zip(data.items(), colors)):
    for i, v in enumerate(vals):
        x = i + (j - 1) * w
        if v is None:
            ax.text(x, 1.15, "未测", ha="center", va="bottom", fontsize=6.5, color="#6b6a63", rotation=90)
            continue
        t, ok = v
        ax.bar(x, t, w * 0.92, color=c, hatch=None if ok else "///", edgecolor="white", linewidth=0.8,
               label=name if i == 0 else None)
        ax.text(x, t * 1.06, f"{t:.0f}" if t >= 100 else f"{t:.1f}", ha="center", va="bottom", fontsize=6.5)
        if not ok:
            ax.text(x, t * 1.45, "未收敛", ha="center", va="bottom", fontsize=6.5, color="#e34948")
ax.set_yscale("log")
ax.set_ylim(1, 1500)
ax.set_xticks(range(len(cases)), cases, fontsize=8)
ax.set_ylabel("到 $10^{-8}$ 的总时间 (s，对数轴)")
ax.grid(axis="y", color="#e6e5df", linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig("time128.png", dpi=200)
