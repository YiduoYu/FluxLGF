"""Time ratios of the locked SSR against BoomerAMG, from final_dev.log and final_holdout.log. Writes ssr_ratios.pdf."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
LABEL = {"poisson3d": "3D Poisson", "aniso": "对齐各向异性", "rot2d": "2D 旋转各向异性", "rot3d": "3D 旋转各向异性", "stretch": "拉伸网格", "lognorm3": "对数正态", "tubes": "导电管", "perc3": "渗流", "fourth": "四阶格式", "cd3d": "3D 旋流", "g3t": "3T（类 SC26_01）", "insul": "绝缘块"}
def label(c):
    p = c.split(":")
    if p[0] == "fe3d": return "非结构网格，跳跃 " + p[2] if p[2] != "1" else "非结构网格"
    if p[0] == "cd2d": return f"2D {'回流' if p[3] == 'rec' else '均匀流'} ε={p[2]}"
    if p[0] == "elast": return f"弹性 ν={p[2]}" + (f"，跳跃 {p[3]}" if p[3] != "1" else "")
    if p[0] == "graph": return {"ws": "小世界图", "ba": "无标度图", "rgg": "随机几何图"}[p[1]]
    if p[0] == "gen": return f"3T，{p[2]} 胶囊"
    if p[0] == "mg": return f"多群，{p[2]} 群"
    return LABEL[p[0]]
def read(f):
    out = []
    for l in (HERE / f).read_text().splitlines():
        m = re.match(r"(\S+)\s+ssr.*\|\s+fast\.25.*?([\d.]+) \|\s+\S+.*?([\d.]+)$", l)
        if m: out.append((label(m[1]), float(m[2]), float(m[3])))
    return sorted(out, key=lambda r: r[1])

plt.rcParams.update({"font.family": ["Heiti TC", "Arial Unicode MS"], "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
sets = [("开发集（23 个算例）", read("final_dev.log")), ("留出集（14 个算例）", read("final_holdout.log"))]
fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.9), gridspec_kw={"wspace": 0.62, "height_ratios": [1]})
for ax, (title, rows) in zip(axes, sets):
    y = range(len(rows))
    for i, r in zip(y, rows):
        ax.plot([r[1], r[2]], [i, i], color=GRID, lw=1.5, zorder=1)
    ax.scatter([r[2] for r in rows], y, s=26, color=ORANGE, marker="s", edgecolor="white", linewidth=0.8, zorder=3, label="对每个算例的最优配置")
    ax.scatter([r[1] for r in rows], y, s=30, color=BLUE, marker="o", edgecolor="white", linewidth=0.8, zorder=4, label="对 fast.25")
    ax.axvline(1, color=MUTED, lw=1, zorder=0)
    ax.set_xscale("log"); ax.set_xlim(0.18, 9.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"]); ax.minorticks_off()
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.tick_params(length=0, colors=INK); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK); ax.set_xlabel("时间比 SSR / BoomerAMG（左侧更快）", color=MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.subplots_adjust(left=0.175, right=0.985, top=0.885, bottom=0.09)
fig.savefig(HERE / "report" / "ssr_ratios.pdf")
fig.savefig(HERE / "report" / "ssr_ratios.png", dpi=170)
