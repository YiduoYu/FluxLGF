"""Figures and HTML tables from the benchmark results.
  python make_figures.py      2D: results/benchmark.csv + traces.json -> results/*.png, results/tables.html
  python make_figures.py 3d   3D: three_d/results/benchmark3d.csv + traces3d.json -> three_d/results/*.png"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CASES = ("A_exp", "B_step", "C_exp+incl", "D_exp+rand")
TITLES = {"A_exp": "A: a = χ^x", "B_step": "B: step layer", "C_exp+incl": "C: χ^x + 9 discs",
          "D_exp+rand": "D: χ^x + 20 random discs"}
# fixed categorical order (reference palette, light mode); marker = secondary encoding
STYLE = {"M2 w=1/16": ("#2a78d6", "o"), "BoomerAMG": ("#eb6834", "s"), "GMG": ("#1baf7a", "^"),
         "LGF": ("#eda100", "v"), "M2 w=1/32": ("#e87ba4", "D"), "M2 w=1/64": ("#6250d6", "P"), "M2": ("#2a78d6", "o")}
if sys.argv[1:] == ["3d"]:
    RES = HERE / "three_d" / "results"
    TITLES.update({"C_exp+incl": "C: χ^x + 27 balls", "D_exp+rand": "D: χ^x + 30 random balls"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load():
    suffix = "3d" if RES.parent.name == "three_d" else ""
    rows = list(csv.DictReader((RES / f"benchmark{suffix}.csv").open()))
    for r in rows:
        for k in ("n", "iterations"):
            r[k] = int(r[k])
        for k in ("chi", "alg_error", "total_error", "setup_ms", "solve_ms", "total_ms", "err_5", "best_err",
                  "ms_to_err_1e8", "its_err_1e8"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
    return rows, json.loads((RES / f"traces{suffix}.json").read_text())


def _axes(title_extra):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), layout="constrained")
    for ax, case in zip(axs, CASES):
        ax.set_title(f"{TITLES[case]}{title_extra}", color=INK, fontsize=10)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    return fig, axs


def _legend(fig, axs):
    handles = {}
    for ax in axs:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labels = [m for m in STYLE if m in handles]
    fig.legend([handles[m] for m in labels], labels, loc="outside lower center", ncol=max(1, len(labels)), fontsize=9,
               frameon=False)


def _plot(ax, xs, ys, method, **kw):
    color, marker = STYLE[method]
    ax.plot(xs, ys, color=color, marker=marker, ms=6, lw=2, label=method,
            markeredgecolor="white", markeredgewidth=1, **kw)


def fig_vs(rows, x, y, fixed, fname, ylabel, xlabel, logy=True, extra=""):
    fig, axs = _axes(extra)
    for ax, case in zip(axs, CASES):
        sel = [r for r in rows if r["case"] == case and all(r[k] == v for k, v in fixed.items())]
        for m in STYLE:
            pts = sorted((r[x], r[y]) for r in sel if r["method"] == m)
            if pts:
                _plot(ax, *zip(*pts), m)
        ax.set_xscale("log", base=2 if x == "n" else 10)
        if x == "n":
            ns = sorted({r["n"] for r in sel})
            ax.set_xticks(ns, [str(v) for v in ns])
            ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axs[0].set_ylabel(ylabel, color=MUTED, fontsize=9)
    _legend(fig, axs)
    fig.savefig(RES / fname, dpi=130)
    plt.close(fig)


def fig_traces(traces, n, chi, fname):
    fig, axs = _axes(f"  (n={n}, χ={chi:.0e})")
    for ax, case in zip(axs, CASES):
        for m in STYLE:
            e = traces.get(f"{case}|{n}|{chi:.0e}|{m}")
            if e:
                e = e[:61]
                color, marker = STYLE[m]
                ax.semilogy(range(len(e)), e, color=color, lw=2, label=m, marker=marker, ms=4, markevery=5)
        ax.set_xlabel("PCG iteration", color=MUTED, fontsize=9)
        ax.set_ylim(1e-14, 2)
    axs[0].set_ylabel("‖u_k − u_h‖ / ‖u_h‖", color=MUTED, fontsize=9)
    _legend(fig, axs)
    fig.savefig(RES / fname, dpi=130)
    plt.close(fig)


def tables(rows):
    out = []
    for case in CASES:
        sel = [r for r in rows if r["case"] == case]
        methods = [m for m in STYLE if any(r["method"] == m for r in sel)]
        order = ["LGF", "GMG", "BoomerAMG"] + [m for m in methods if m.startswith("M2")]
        head = "".join(f"<th>{m}</th>" for m in order)
        body = []
        for n in sorted({r["n"] for r in sel}):
            for chi in sorted({r["chi"] for r in sel}):
                cells = []
                for m in order:
                    r = next((r for r in sel if r["n"] == n and r["chi"] == chi and r["method"] == m), None)
                    if r is None:
                        cells.append("<td>–</td>")
                        continue
                    it = f"{r['iterations']}" + ("" if r["converged"] == "True" else "✗")
                    cells.append(f"<td><b>{it}</b><br>{r['total_ms']:.0f} ms<br>{r['alg_error']:.1e}</td>")
                body.append(f"<tr><td>{n}</td><td>{chi:.0e}</td>{''.join(cells)}</tr>")
        out.append(f"<h3>{TITLES[case]}</h3><table><thead><tr><th>n</th><th>χ</th>{head}</tr></thead>"
                   f"<tbody>{''.join(body)}</tbody></table>")
    (RES / "tables.html").write_text("\n".join(out))


def main():
    rows, traces = load()
    nmax = max(r["n"] for r in rows)
    fig_vs(rows, "chi", "iterations", {"n": nmax}, "iters_vs_chi.png", "PCG iterations", "χ", extra=f"  (n={nmax})")
    fig_vs(rows, "n", "iterations", {"chi": 1e8}, "iters_vs_n.png", "PCG iterations", "n", extra="  (χ=1e8)")
    fig_vs(rows, "n", "total_ms", {"chi": 1e8}, "time_vs_n.png", "setup + solve [ms]", "n", extra="  (χ=1e8)")
    fig_vs(rows, "n", "ms_to_err_1e8", {"chi": 1e8}, "time_to_1e-8_vs_n.png", "setup + time to error 1e-8 [ms]", "n",
           extra="  (χ=1e8)")
    fig_vs(rows, "chi", "its_err_1e8", {"n": nmax}, "its_to_1e-8_vs_chi.png", "iterations to error 1e-8", "χ",
           extra=f"  (n={nmax})")
    fig_vs(rows, "chi", "total_error", {"n": nmax}, "error_vs_chi.png", "rel. L2 error vs exact u", "χ",
           extra=f"  (n={nmax})")
    n_tr = max(r["n"] for r in rows if r["chi"] == 1e8 and r["case"] == CASES[-1])
    fig_traces(traces, n_tr, 1e8, "error_traces.png")
    tables(rows)


if __name__ == "__main__":
    main()
