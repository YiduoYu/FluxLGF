"""Figure for report 12: ceiling of each add-on family relative to BoomerAMG (H45)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
d = pd.read_csv("../results/h45_ceiling.csv")
cases = list(dict.fromkeys(d.case))
base = d[d.diag == "D0"].set_index("case")
def its(diag, setting):
    return d[(d.diag == diag) & (d.setting == setting)].set_index("case").loc[cases, "its"].values / base.loc[cases, "its"].values
d3 = d[(d.diag == "D3") & (d.its > 0)]
best = d3.groupby("case").total_s.min().loc[cases].values / base.loc[cases, "total_s"].values
series = [("D1 慢模态紧缩 k=16（迭代比）", its("D1", "k=16"), "#3b6fb6"),
          ("D2 初值相对误差 1%（迭代比）", its("D2", "eps=0.01"), "#d9822b"),
          ("D3 最优 AMG 参数（时间比）", best, "#4c9a5f")]
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(cases)); w = 0.26
for i, (lab, v, c) in enumerate(series):
    ax.bar(x + (i - 1) * w, v, w - 0.02, label=lab, color=c)
ax.axhline(1, color="0.3", lw=1)
ax.set_xticks(x, cases); ax.set_ylabel("相对 BoomerAMG（<1 更好）")
ax.set_ylim(0, 1.8); ax.legend(fontsize=8, loc="upper left", frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("ceiling.png", dpi=200)
