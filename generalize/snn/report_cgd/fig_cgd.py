import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
d = pd.concat([pd.read_csv(f"../results/h47_{n}.csv") for n in (511, 1023, 2047)])
col = {"circ": "#0072B2", "disk": "#E69F00", "band": "#009E73", "rot": "#CC79A7"}
mk = {511: "o", 1023: "s", 2047: "^"}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axhline(1, color="#888", lw=1, zorder=0)
for (f, n), g in d.groupby(["fam", "n"]):
    ax.scatter(g.base_its, g.ratio, c=col[f], marker=mk[n], s=40, edgecolors="white", linewidths=0.8,
               label=f"{f}, n={n}")
ax.set_xscale("log"); ax.set_xlabel("BoomerAMG-PCG 迭代数（对数）"); ax.set_ylabel("粗网格紧缩总时间 / BoomerAMG 总时间")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#ddd", lw=0.6)
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.rcParams["font.sans-serif"] = ["Songti SC"]
fig.tight_layout(); fig.savefig("cgd_ratio.png", dpi=200)
