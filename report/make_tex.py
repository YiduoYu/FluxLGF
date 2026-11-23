"""LaTeX tables (tables_2d.tex, tables_3d.tex, appendix_2d.tex, appendix_3d.tex) from the benchmark CSVs."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = {"2d": ROOT / "results" / "benchmark.csv", "3d": ROOT / "three_d" / "results" / "benchmark3d.csv"}
CASE = {"A_exp": r"A: $\chi^x$", "B_step": "B: 阶跃层", "C_exp+incl": "C: $\\chi^x$+规则夹杂",
        "D_exp+rand": "D: $\\chi^x$+随机夹杂"}
ORDER = ["LGF", "GMG", "BoomerAMG", "M2", "M2 w=1/16", "M2 w=1/32", "M2 w=1/64"]
HEAD = {"LGF": "传统 LGF", "GMG": "GMG", "BoomerAMG": "BoomerAMG", "M2": "方法二", "M2 w=1/16": r"M2 $w{=}\frac1{16}$",
        "M2 w=1/32": r"M2 $w{=}\frac1{32}$", "M2 w=1/64": r"M2 $w{=}\frac1{64}$"}


def sci(x):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def ms(x):
    if not np.isfinite(x):
        return "--"
    return f"{x / 1e3:.2f}\\,s" if x >= 1e3 else f"{x:.0f}"


def rng(s, fmt=lambda v: f"{v:.0f}"):
    s = s.dropna()
    if s.empty:
        return "--"
    lo, hi = s.min(), s.max()
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}--{fmt(hi)}"


def its_cell(g):
    txt = rng(g.iterations)
    fails = int((~g.converged.astype(str).eq("True")).sum())
    return txt + (f"$^{{\\times{fails}}}$" if fails else "")


def methods_of(d):
    return [m for m in ORDER if m in set(d.method)]


def grid_table(d, cell, caption, label):
    meths = methods_of(d)
    lines = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{ll" + "c" * len(meths) + "}", r"\toprule",
             "算例 & $n$ & " + " & ".join(HEAD[m] for m in meths) + r" \\", r"\midrule"]
    for case in CASE:
        sub = d[d.case == case]
        for i, n in enumerate(sorted(sub.n.unique())):
            cells = []
            for m in meths:
                g = sub[(sub.n == n) & (sub.method == m)]
                cells.append(cell(g) if len(g) else "")
            lines.append((CASE[case] if i == 0 else "") + f" & {n} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if case != "D_exp+rand" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def a_error_table(d, dim):
    a = d[(d.case == "A_exp") & (d.n == d.n.max())]
    meths = methods_of(a)
    lines = [r"\begin{table}[H]\centering\small",
             f"\\caption{{{dim}，算例 A（$n={a.n.max()}$）：相对连续解的总误差 $\\|u-u^*\\|/\\|u^*\\|$}}"
             f"\\label{{tab:aerr{dim}}}",
             r"\begin{tabular}{l" + "c" * len(meths) + "}", r"\toprule",
             r"$\chi$ & " + " & ".join(HEAD[m] for m in meths) + r" \\", r"\midrule"]
    for chi in sorted(a.chi.unique()):
        cells = [sci(a[(a.chi == chi) & (a.method == m)].total_error.iloc[0]) for m in meths]
        lines.append(f"$10^{{{int(np.log10(chi))}}}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def appendix(d, dim):
    lines = [r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{lrrlrcccrrr}",
             f"\\caption{{{dim} 全部结果。err：按残差规则停止时相对 $u_h$ 的代数误差；$k_8$：误差降到 $10^{{-8}}$ 所需迭代数；"
             r"时间单位 ms（单线程）。}\\",
             r"\toprule 算例 & $n$ & $\chi$ & 方法 & 迭代 & err & 第 5 步误差 & $k_8$ & setup & solve & 合计 \\ \midrule",
             r"\endfirsthead \toprule 算例 & $n$ & $\chi$ & 方法 & 迭代 & err & 第 5 步误差 & $k_8$ & setup & solve & 合计 \\ \midrule \endhead"]
    for _, r in d.sort_values(["case", "n", "chi"]).iterrows():
        k8 = "--" if pd.isna(r.its_err_1e8) else f"{int(r.its_err_1e8)}"
        it = f"{r.iterations}" + ("" if str(r.converged) == "True" else r"$^\times$")
        lines.append(f"{r.case[0]} & {r.n} & $10^{{{int(np.log10(r.chi))}}}$ & {HEAD[r.method]} & {it} & {sci(r.alg_error)} & "
                     f"{sci(r.err_5)} & {k8} & {r.setup_ms:.0f} & {r.solve_ms:.0f} & {r.total_ms:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines)


def main():
    for dim, path in SRC.items():
        if not path.exists():
            continue
        d = pd.read_csv(path)
        D = "二维" if dim == "2d" else "三维"
        cap, trace = (1000, 1000) if dim == "2d" else (500, 300)   # maxiter, error-trace length
        hard = d[d.chi == 1e8]
        parts = [
            grid_table(d, its_cell, f"{D}：PCG 迭代次数（$\\chi\\in\\{{10^2,10^4,10^6,10^8\\}}$ 的最小--最大值；"
                       rf"$^{{\times k}}$ 表示有 $k$ 个 $\chi$ 在 {cap} 次内未收敛）", f"tab:its{dim}"),
            grid_table(d, lambda g: rng(g.its_err_1e8), f"{D}：误差 $\\|u_k-u_h\\|/\\|u_h\\|\\le10^{{-8}}$ 所需迭代数"
                       f"（各 $\\chi$ 的范围；--：{trace} 次内未达到）", f"tab:k8{dim}"),
            grid_table(d, lambda g: sci(g.alg_error.max()),
                       f"{D}：按残差规则 $\\|r\\|/\\|f\\|\\le10^{{-10}}$ 停止时的代数误差 $\\|u-u_h\\|/\\|u_h\\|$"
                       "（各 $\\chi$ 中的最大值）", f"tab:err{dim}"),
            grid_table(hard, lambda g: ms(g.total_ms.iloc[0]), f"{D}：总时间 setup+solve（$\\chi=10^8$，单线程，ms）",
                       f"tab:time{dim}"),
            grid_table(hard, lambda g: ms(g.ms_to_err_1e8.iloc[0]) if g.ms_to_err_1e8.notna().any() else "--",
                       f"{D}：误差降到 $10^{{-8}}$ 的时间 setup$+k_8\\times$单步时间（$\\chi=10^8$，ms）", f"tab:t8{dim}"),
            a_error_table(d, D),
        ]
        (HERE / f"tables_{dim}.tex").write_text("\n\n".join(parts))
        (HERE / f"appendix_{dim}.tex").write_text(appendix(d, D))
        for png in path.parent.glob("*.png"):        # figures from make_figures.py
            shutil.copy(png, HERE / f"fig{dim}_{png.name}")


if __name__ == "__main__":
    main()
