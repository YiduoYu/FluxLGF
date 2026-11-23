"""Figures and LaTeX tables for the H33 report (report_hc/)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent / "CNN function")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cases_hc import FAMILIES, make_case

OUT = HERE / "report_hc"
OUT.mkdir(exist_ok=True)
NAME = {"sep2": "sep2 可分离", "chan": "chan 通道", "checker": "checker 棋盘", "lognorm": "lognorm 对数正态",
        "layered": "layered 类SPE10分层", "diag": "diag 斜条纹", "walls": "walls 绝缘墙", "perc": "perc 随机两相",
        "diagfine": "diagfine 细斜条纹", "rough": "rough 粗糙随机场"}
METH = ["BoomerAMG", "GMG", "FD-LGF w=1/16", "FD-LGF w=1/64"]
HEAD = {"BoomerAMG": "BoomerAMG", "GMG": "GMG", "FD-LGF w=1/16": r"FD-LGF $w{=}\frac1{16}$",
        "FD-LGF w=1/64": r"FD-LGF $w{=}\frac1{64}$", "BoomerAMG θ=0.5": r"BAMG $\theta{=}0.5$",
        "BoomerAMG θ=0.25": r"BAMG $\theta{=}0.25$", "FD-LGF w=1/32": r"FD-LGF $w{=}\frac1{32}$"}
STYLE = {"FD-LGF w=1/16": ("#2a78d6", "o"), "BoomerAMG": ("#eb6834", "s"), "GMG": ("#1baf7a", "^"),
         "FD-LGF w=1/64": ("#6250d6", "P")}


def gallery():
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.4), layout="constrained")
    for ax, fam in zip(axs.ravel(), FAMILIES):
        c = make_case(fam, 255, 1e8)
        im = ax.imshow(np.log10(c.a_nodes), origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=8)
        ax.contour(np.linspace(0, 1, 257), np.linspace(0, 1, 257), c.split[2].astype(float), levels=[0.5],
                   colors="white", linewidths=0.5)
        ax.set_title(f"{fam}  (features {c.feature_fraction:.0%})", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="log10 a  (χ = 1e8)")
    fig.savefig(OUT / "gallery.png", dpi=110)
    plt.close(fig)


def time_plot(d):
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    fams = list(FAMILIES)
    for k, m in enumerate(METH):
        color, marker = STYLE[m]
        sub = d[d.method == m].groupby("case").total_ms
        lo, hi = sub.min().reindex(fams), sub.max().reindex(fams)
        x = np.arange(len(fams)) + (k - 1.5) * 0.15
        ax.vlines(x, lo / 1e3, hi / 1e3, color=color, lw=2)
        ax.plot(x, hi / 1e3, marker, color=color, ms=6, label=m, markeredgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(range(len(fams)), fams)
    ax.set_ylabel("setup + solve [s]  (range over χ)")
    ax.grid(True, axis="y", color="#e6e5e0")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.savefig(OUT / "time1023.png", dpi=130)
    plt.close(fig)


def rng(s, fmt="{:.0f}"):
    s = s.dropna()
    if s.empty:
        return "--"
    a, b = fmt.format(s.min()), fmt.format(s.max())
    return a if a == b else f"{a}--{b}"


def sec(x):
    return f"{x / 1e3:.2f}"


def table(d, value, fmt, caption, label, methods, index="case", names=NAME):
    rows = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
            r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
            "系数 & " + " & ".join(HEAD[m] for m in methods) + r" \\", r"\midrule"]
    for fam in dict.fromkeys(d[index]):
        cells = []
        for m in methods:
            g = d[(d[index] == fam) & (d.method == m)]
            cells.append(fmt(g) if len(g) else "--")
        rows.append(names.get(fam, fam) + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def main():
    gallery()
    d = pd.read_csv(HERE / "results" / "hc1023.csv")
    s = pd.read_csv(HERE / "results" / "scan.csv")
    s3 = pd.read_csv(HERE / "results" / "scan3d.csv")
    time_plot(d)
    def sci(v):
        m, e = f"{v:.1e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"
    parts = [
        table(d, None, lambda g: rng(g.iterations), r"$n=1023$：PCG 迭代次数（4 个 $\chi$ 的范围）", "tab:its", METH),
        table(d, None, lambda g: rng(g.its_err_1e8), r"$n=1023$：误差降到 $10^{-8}$ 所需迭代数", "tab:k8", METH),
        table(d, None, lambda g: rng(g.total_ms / 1e3, "{:.2f}"), r"$n=1023$：总时间 setup+solve（s，单线程）", "tab:time", METH),
        table(d, None, lambda g: rng(g.setup_ms / 1e3, "{:.2f}"), r"$n=1023$：setup 时间（s）", "tab:setup", METH),
        table(d[d.method.str.startswith("FD")], None, lambda g: rng(g.local_fraction, "{:.2f}"),
              r"$n=1023$：FD-LGF 局部 LU 区域占网格的比例", "tab:loc", METH[2:]),
        table(d, None, lambda g: sci(g.alg_error.max()), r"$n=1023$：按残差规则停止时的代数误差（各 $\chi$ 最大值）", "tab:err", METH),
        table(s, None, lambda g: rng(g.iterations), r"探索性扫描（$n=255, 511$）：迭代次数范围", "tab:scan", METH),
    ]
    s3 = s3.copy()
    names3 = {"sheets": "sheets 薄层", "tubes": "tubes 管道网", "insul": "insul 绝缘板", "perc3": "perc3 随机两相",
              "lognorm3": "lognorm3 对数正态"}
    m3 = ["BoomerAMG θ=0.25", "BoomerAMG θ=0.5", "GMG", "FD-LGF w=1/32", "FD-LGF w=1/64"]
    for n in (63, 127):
        sub = s3[s3.n == n]
        parts.append(table(sub, None, lambda g: rng(g.iterations), f"三维探索性扫描：迭代次数（$n={n}$）",
                           f"tab:s3its{n}", m3, names=names3))
    parts.append(table(s3[s3.n == 127], None, lambda g: rng(g.total_ms / 1e3, "{:.1f}"),
                       r"三维探索性扫描：总时间（s，$n=127$，约 $2\times10^6$ 未知量）", "tab:s3time", m3, names=names3))
    (OUT / "tables_1023.tex").write_text("\n\n".join(parts[:6]))
    (OUT / "tables_scan.tex").write_text("\n\n".join(parts[6:]))


if __name__ == "__main__":
    main()
