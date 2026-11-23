"""Figures and LaTeX tables for the H34 report (report_aniso/)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "high_contrast"), str(HERE.parents[1] / "CNN function")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cases_aniso import FAMILIES, make_case
from make_hc_report import rng, table

OUT = HERE / "report_aniso"
OUT.mkdir(exist_ok=True)
NAME = {"aligned": "aligned 对齐（对照）", "rot30": "rot30 旋转 30°", "rot45": "rot45 旋转 45°",
        "circ": "circ 圆形磁力线", "disk": "disk 局部旋转圆盘", "band": "band 斜断层带"}
METH = ["BoomerAMG θ=0.25", "BoomerAMG θ=0.5", "GMG", "FD-LGF w=1/16", "FD-LGF w=1/64"]
STYLE = {"BoomerAMG θ=0.25": ("#eb6834", "s"), "BoomerAMG θ=0.5": ("#b8452a", "D"), "GMG": ("#1baf7a", "^"),
         "FD-LGF w=1/16": ("#2a78d6", "o"), "FD-LGF w=1/64": ("#6250d6", "P")}


def gallery():
    fig, axs = plt.subplots(1, 6, figsize=(16, 3.1), layout="constrained")
    for ax, fam in zip(axs, FAMILIES):
        c = make_case(fam, 127, 1e4)
        K11, K12, K22 = (k[1:-1, 1:-1] for k in c.K)
        aniso = np.abs(K11 - K22) + np.abs(K12) > 1e-9
        ax.imshow(aniso, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=0, vmax=2.5)
        theta = 0.5 * np.arctan2(2 * K12, K11 - K22)          # direction of the large eigenvalue
        s = slice(4, None, 10)
        t = np.linspace(0, 1, 129)[1:-1]
        X, Y = np.meshgrid(t, t, indexing="xy")
        ex, ey = np.where(aniso, np.cos(theta), 0), np.where(aniso, np.sin(theta), 0)
        ax.quiver(X[s, s], Y[s, s], ex[s, s], ey[s, s], color="#2a78d6", pivot="mid", headwidth=0, headlength=0,
                  headaxislength=0, scale=14, width=0.012)
        ax.set_title(fam, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    fig.savefig(OUT / "gallery.png", dpi=120)
    plt.close(fig)


def scaling_plot(d, chi):
    fig, axs = plt.subplots(1, 6, figsize=(16, 3.4), layout="constrained", sharey=True)
    for ax, fam in zip(axs, FAMILIES):
        for m in METH:
            g = d[(d.case == fam) & (d.method == m) & (d.chi == chi)].sort_values("n")
            if len(g):
                color, marker = STYLE[m]
                ax.plot(g.n, g.iterations, marker=marker, color=color, lw=2, ms=6, label=m, markeredgecolor="white")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(fam, fontsize=11)
        ax.set_xlabel("n")
        ax.grid(True, color="#e6e5e0")
    axs[0].set_ylabel(f"PCG iterations (χ = {chi:.0e})")
    handles, labels = axs[4].get_legend_handles_labels()                # disk has every method
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.savefig(OUT / f"scaling_{chi:.0e}.png", dpi=130)
    plt.close(fig)


def main():
    gallery()
    d = pd.concat([pd.read_csv(HERE / "results" / f) for f in ("scan.csv", "aniso1023.csv")])
    for chi in sorted(d.chi.unique()):
        scaling_plot(d, chi)
    big = d[d.n == 1023]

    def sci(v):
        m, e = f"{v:.1e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"

    kw = dict(names=NAME)
    parts = [
        table(big, None, lambda g: rng(g.iterations), r"$n=1023$：PCG 迭代次数（各 $\chi$ 的范围）", "tab:its", METH, **kw),
        table(big, None, lambda g: rng(g.its_err_1e8), r"$n=1023$：误差降到 $10^{-8}$ 所需迭代数", "tab:k8", METH, **kw),
        table(big, None, lambda g: rng(g.total_ms / 1e3, "{:.2f}"), r"$n=1023$：总时间 setup+solve（s，单线程）",
              "tab:time", METH, **kw),
        table(big, None, lambda g: rng(g.setup_ms / 1e3, "{:.2f}"), r"$n=1023$：setup 时间（s）", "tab:setup", METH, **kw),
        table(big, None, lambda g: sci(g.alg_error.max()), r"$n=1023$：停止时的代数误差（各 $\chi$ 最大值）",
              "tab:err", METH, **kw),
    ]
    for chi in sorted(d.chi.unique()):
        sub = d[d.chi == chi].copy()
        sub["case_n"] = sub.case + " n=" + sub.n.astype(str)
        parts.append(table(sub, None, lambda g: rng(g.iterations),
                           f"迭代次数随 $n$ 的变化（$\\chi=10^{{{int(np.log10(chi))}}}$）", f"tab:n{int(np.log10(chi))}",
                           METH, index="case_n"))
    (OUT / "tables.tex").write_text("\n\n".join(parts))


if __name__ == "__main__":
    main()
