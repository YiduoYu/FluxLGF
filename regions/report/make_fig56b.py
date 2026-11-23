"""H56 locked results: bt-fd / BoomerAMG total-time ratio against the feature fraction (10-step sequences, m = 128),
plus a post-hoc sensitivity column (setup + 10 x median step solve, removing swap outliers)."""
import csv
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
order = "128_1_2_1_0_2e-2_1 128_1_1_1_0_2e-2_1 128_1_8_0.5_0_2e-2_1 128_1_2_2_0_2e-2_1 128_1_2_1_4_2e-2_1 128_1_2_1_0_1e-1_1 128_3_2_1_0_2e-2_1 128_10_2_1_0_2e-2_1 128_10_2_1_0_2e-2_2 128_10_8_0.5_2_2e-2_3 128_20_2_1_0_2e-2_1".split()
rows = []
for i, g in enumerate(order, 1):
    r = list(csv.DictReader(open(R / f"h56_{g}.csv")))
    s = {}
    for m in ("bt-fd", "amg"):
        q = [x for x in r if x["method"] == m]
        s[m] = (sum(float(x["total_s"]) for x in q), float(q[0]["setup_s"]) + len(q) * median(float(x["solve_s"]) for x in q))
    rows.append((f"G{i}", 100 * float(r[0]["feature"]), s["bt-fd"][0] / s["amg"][0], s["bt-fd"][1] / s["amg"][1], s["bt-fd"][0], s["amg"][0]))
    print(rows[-1][0], f"F {rows[-1][1]:.1f}% ratio {rows[-1][2]:.2f} median-based {rows[-1][3]:.2f}")
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.axhline(1, color="#6b6a64", lw=1)
ax.text(31.5, 1.04, "BoomerAMG", ha="right", va="bottom", fontsize=8, color="#55544e")
for lab, sel, col, mk in (("one capsule", rows[:6], "#2a78d6", "o"), ("3-20 capsules", rows[6:], "#eb6834", "s")):
    ax.scatter([r[1] for r in sel], [r[2] for r in sel], s=46, color=col, marker=mk, edgecolor="white", linewidth=1.2, label=lab, zorder=3)
off = dict(G1=(8, -9), G6=(8, -1), G3=(-18, 2), G5=(7, 2), G2=(6, 2), G4=(6, -3), G7=(6, -3), G8=(6, 2), G9=(6, -8), G10=(6, -3), G11=(-22, -3))
for r in rows:
    ax.annotate(r[0], (r[1], r[2]), xytext=off[r[0]], textcoords="offset points", fontsize=8, color="#33322e")
ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 1, 2, 4], ["0.4", "0.6", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel("feature fraction of bt-fd, % of cells (with 2-cell ring)", fontsize=9)
ax.set_ylabel("bt-fd / BoomerAMG total time", fontsize=9); ax.tick_params(labelsize=8)
ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "results56b.png", dpi=170)
