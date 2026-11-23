"""H50: evaluate the locked predictions (protocol_amfr.md) and write report tables / figure."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "report"
OUT.mkdir(exist_ok=True)
ABL = ("amfr", "amfr-sep", "amfr-noelim", "amfr-all")


def load(name):
    d = defaultdict(dict)
    for r in csv.DictReader(open(R / name)):
        r["its"], r["reason"], r["total_s"] = int(r["its"]), int(r["reason"]), float(r["total_s"])
        d[r["case"]][r["method"]] = r
    return d


A, B, A128, C = load("test_A.csv"), load("test_B.csv"), load("test_A128.csv"), load("test_C.csv")
gen = {**A, **B, **A128}


def ok(r):
    return r["reason"] > 0


def ratio(c, m="amfr", base="amg-3d"):
    return c[m]["total_s"] / c[base]["total_s"]


print("== P1 robustness")
conv = sum(ok(c["amfr"]) for c in gen.values())
it15 = sum(ok(c["amfr"]) and c["amfr"]["its"] <= 1.5 * c["amg-3d"]["its"] for c in gen.values())
print(f"converged {conv}/{len(gen)}; its <= 1.5x amg-3d {it15}/{len(gen)}")
print("== P2 speed")
rA = np.array([ratio(c) for c in A.values()] + [ratio(c) for c in A128.values()])
rB = np.array([ratio(c) for c in B.values()])
allr = np.concatenate([rA, rB])
print(f"T-A median {np.median(rA):.3f} (range {rA.min():.2f}-{rA.max():.2f}); T-B median {np.median(rB):.3f} "
      f"(range {rB.min():.2f}-{rB.max():.2f}); <= 1.10 in {(allr <= 1.10).sum()}/{len(allr)}")
print("== P3 rule quality")
q = [c["amfr"]["total_s"] <= 1.15 * min(c[m]["total_s"] for m in ABL if ok(c[m])) for c in gen.values()]
print(f"amfr within 1.15x of best ablation in {sum(q)}/{len(q)}")
print("== P4 vs amg-unk")
u = [c["amfr"]["total_s"] < c["amg-unk"]["total_s"] or not ok(c["amg-unk"]) for c in gen.values()]
print(f"amfr faster than amg-unk in {sum(u)}/{len(u)}")
print("== P5 real")
for m, r in C["real"].items():
    print(m, r["structure"], r["its"], r["reason"], f"{float(r['setup_s']):.1f} {r['total_s']:.1f}")
print("== P6 structure on T-A m=64")
el = {c: A[c]["amfr"]["structure"].startswith("e1") for c in A}
g1 = [el[c] for c in A if c.split(":")[3] == "1e-3"]
g2 = [el[c] for c in A if c.split(":")[3] != "1e-3"]
print(f"G=1e-3 eliminated {sum(g1)}/{len(g1)}; G>=3e-2 eliminated {sum(g2)}/{len(g2)}")

# ---- tables
def row_tex(case, c, label):
    best = min(ABL, key=lambda m: c[m]["total_s"] if ok(c[m]) else np.inf)
    f = lambda m: (f"{c[m]['its']} / {c[m]['total_s']:.2f}" if ok(c[m]) else "失败")
    return (f"{label} & {f('amg-3d')} & {f('amg-unk')} & \\texttt{{{c['amfr']['structure']}}} & {f('amfr')} & "
            f"{ratio(c):.2f} & {best.replace('amfr-', '')} \\\\")


with open(OUT / "table_A.tex", "w") as fh:
    for case, c in list(A.items()) + list(A128.items()):
        _, m, F, G = case.split(":")
        fh.write(row_tex(case, c, f"{m} & {F} & {G}") + "\n")
with open(OUT / "table_B.tex", "w") as fh:
    for case, c in B.items():
        _, m, ng, s, z, l, _ = case.split(":")
        fh.write(row_tex(case, c, f"{ng} & {s} & {z} & {l}") + "\n")

# ---- figure: time ratio vs amg-3d for amfr and ablations
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, data, title in ((ax[0], {**A, **A128}, "T-A gen3t (3T)"), (ax[1], B, "T-B multigroup + material")):
    keys = list(data)
    xs = np.arange(len(keys))
    for m, mk, col in (("amg-unk", "x", "0.6"), ("amfr-sep", "v", "C1"), ("amfr-noelim", "^", "C2"),
                       ("amfr-all", "s", "C4"), ("amfr", "o", "C0")):
        y = [data[k][m]["total_s"] / data[k]["amg-3d"]["total_s"] if ok(data[k][m]) else np.nan for k in keys]
        a.scatter(xs, y, marker=mk, s=28 if m == "amfr" else 14, color=col, label=m, zorder=3 if m == "amfr" else 2)
    a.axhline(1, color="k", lw=0.8)
    a.set_yscale("log")
    a.set_title(title)
    a.set_xlabel("case index")
ax[0].set_ylabel("total time / amg-3d")
ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "ratios.png", dpi=160)
