"""Figures and tables for the H35/H36 report (report_sc/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_sc"
OUT.mkdir(exist_ok=True)
R = HERE / "results"
SHORT = {"FD-LGF background G (block means)": "原 FD-LGF 背景 G",
         "+ scalar Jacobi scaling (Green-Jacobi)": "+ 标量 Jacobi 缩放（GJ）",
         "+ region coarse space (box 4, decades)": "+ 分区粗空间",
         "3x3 point-block scaling, no coarse": "3×3 块缩放 GJ",
         "+ box coarse space (b = 4, LU)": "+ box 粗空间",
         "+ block-Jacobi smoothing (LU coarse)": "+ 块 Jacobi 光滑",
         "FD-LGF-3T: multilevel W-cycle": "FD-LGF-3T（多层）"}
NAMES = {"amg": "BoomerAMG 默认", "amg-3d": "BoomerAMG 三维参数", "fdlgf3t": "FD-LGF-3T", "fdlgf3t-fast": "FD-LGF-3T 调参",
         "bnn3t-fast": "去 LGF（BNN）", "agg3t-w": "agg3t-w"}


def ablation():
    d = pd.read_csv(R / "ablation.csv")
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    variants = list(dict.fromkeys(d.variant))
    colors = {16: "#9ec5f4", 32: "#4a90d9", 64: "#1c4f8f"}
    for k, m in enumerate(sorted(d.m.unique())):
        g = d[d.m == m].set_index("variant").reindex(variants)
        x = np.arange(len(variants)) + (k - 1) * 0.27
        ax.bar(x, g.iterations, width=0.25, color=colors[m], label=f"m = {m}")
        for xi, (it, ok) in zip(x, zip(g.iterations, g.converged)):
            ax.text(xi, it * 1.08, f"{it}" if ok else "×", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(variants)), [SHORT[v] for v in variants], fontsize=8)
    ax.set_ylabel("CG 迭代数（到 1e-8，× = 3000 次未收敛）")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#e6e5e0")
    fig.savefig(OUT / "ablation.png", dpi=130)
    plt.close(fig)


def table(rows, cols, caption, label, nl=2):
    lines = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\begin{tabular}{" + "l" * nl + "c" * (len(cols) - nl) + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"] + [" & ".join(map(str, r)) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    ablation()
    rob = pd.concat([pd.read_csv(f) for f in sorted(R.glob("rob64_dt*.csv"))])
    rob["dt"] = rob.case.str.replace("rob64_dt", "").astype(float)
    rows = []
    for dt, g in rob[rob.tol == 1e-8].sort_values("dt").groupby("dt", sort=True):
        a, f = g[g.solver == "amg-3d"].iloc[0], g[g.solver == "fdlgf3t"].iloc[0]
        rows.append([f"{dt:.0e}", a.iterations, f.iterations, f"{a.total_s:.2f}", f"{f.total_s:.2f}",
                     f"{f.total_s / a.total_s:.2f}"])
    t1 = table(rows, ["$\\Delta t$", "BAMG 迭代", "FD-LGF-3T 迭代", "BAMG (s)", "FD-LGF-3T (s)", "时间比"],
               r"$m=64$（78.6 万未知量）：时间步长扫描，到 $10^{-8}$", "tab:dt", nl=1)
    h = pd.concat([pd.read_csv(f) for f in sorted(R.glob("h36_dt*.csv"))])
    rows = []
    for (case, solver, tol), g in h.groupby(["case", "solver", "tol"], sort=False):
        r = g.iloc[0]
        rows.append([case.replace("h36_dt", "$\\Delta t=$"), NAMES[solver], f"{tol:.0e}", r.iterations,
                     f"{r.setup_s:.1f}", f"{r.solve_s:.1f}", f"{r.total_s:.1f}", f"{r.true_rel_res:.1e}"])
    t2 = table(rows, ["工况", "方法", "目标", "迭代", "setup (s)", "solve (s)", "总时间 (s)", "真实残差"],
               r"$m=128$（与赛题同规模：6,291,456 阶，52,133,888 非零元），单线程", "tab:m128")
    (OUT / "tables.tex").write_text(t1 + "\n\n" + t2)
    parts = []
    for tag, cap in (("h37", r"H37（锁定 e212a666）：$m=128$，新代码"), ("h37b", r"H37b（锁定 f3f5516d）：$m=128$")):
        h = pd.concat([pd.read_csv(f) for f in sorted(R.glob(f"{tag}_dt*.csv"))])
        rows = []
        for (case, solver, tol), g in h.groupby(["case", "solver", "tol"], sort=False):
            r = g.iloc[0]
            rows.append([case.replace(f"{tag}_dt", "$\\Delta t=$"), NAMES[solver], f"{tol:.0e}", r.iterations,
                         f"{r.setup_s:.1f}", f"{r.solve_s:.1f}", f"{r.total_s:.1f}"])
        parts.append(table(rows, ["工况", "方法", "目标", "迭代", "setup (s)", "solve (s)", "总时间 (s)"], cap,
                           f"tab:{tag}"))
    rob = pd.concat([pd.read_csv(f) for f in sorted(R.glob("rob64b_dt*.csv"))])
    rob["dt"] = rob.case.str.replace("rob64b_dt", "").astype(float)
    rows = []
    for dt, g in rob[rob.tol == 1e-8].sort_values("dt").groupby("dt", sort=True):
        a, f = g[g.solver == "amg-3d"].iloc[0], g[g.solver == "agg3t-w"].iloc[0]
        rows.append([f"{dt:.0e}", a.iterations, f.iterations, f"{a.total_s:.2f}", f"{f.total_s:.2f}",
                     f"{f.total_s / a.total_s:.2f}"])
    parts.append(table(rows, ["$\\Delta t$", "BAMG 迭代", "agg3t-w 迭代", "BAMG (s)", "agg3t-w (s)", "时间比"],
                       r"$m=64$：agg3t-w 的时间步长扫描，到 $10^{-8}$", "tab:dtb", nl=1))
    (OUT / "tables_speed.tex").write_text("\n\n".join(parts))
    h = pd.read_csv(R / "h38.csv")
    vname = {"emis": "发射线性化", "scaled": "行缩放"}
    sname = {"amg-3d": "BoomerAMG 三维参数", "amg-3d-ds": "BoomerAMG 三维 + 对角缩放", "agg3t-w": "agg3t-w"}
    rows = [[vname[r.variant], f"{r.dt:g}", sname[r.solver], f"{r.tol:.0e}",
             f"{r.iterations}" + ("" if r.reason > 0 else "（未收敛）"), r.restarts, f"{r.true_rel_res:.1e}",
             f"{r.total_s:.1f}"] for r in h.itertuples()]
    (OUT / "tables_nonsym.tex").write_text(table(
        rows, ["矩阵", "$\\Delta t$", "方法", "目标", "迭代", "重启", "真实残差", "总时间 (s)"],
        r"H38（锁定 2ed4bcd5）：$m=128$ 的非对称替代矩阵，GMRES(50) 右预条件，上限 500 次", "tab:h38"))


if __name__ == "__main__":
    main()
