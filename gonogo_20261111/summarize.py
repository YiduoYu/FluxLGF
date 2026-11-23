"""Summarise results.jsonl by the decision rule of protocol.md (written after locking; reads results only).

usage: summarize.py [results.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
CASES = ("A_exp", "B_step", "C_exp+incl", "D_exp+rand")


def load(path):
    runs = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        runs[(d["case"], d["n"], d["chi"], d["method"])].append(d)
    return runs


def stat(rs, key):
    """Median of key over repeats; None if any repeat failed (no value / crash)."""
    vals = [r.get(key) for r in rs]
    if not vals or any(v is None for v in vals):
        return None
    return median(vals)


def best(runs, case, n, chi, fam, key):
    cand = []
    for (c, nn, x, m), rs in runs.items():
        if (c, nn, x) != (case, n, chi) or (fam == "fdlgf") != m.startswith("fdlgf"):
            continue
        v = stat(rs, key)
        if v is not None:
            cand.append((v, m))
    return min(cand) if cand else (None, None)


def table(runs, n, chi, key):
    print(f"\n### n = {n}, chi = {chi:.0e}, {key} (median of repeats, s)")
    methods = sorted({m for (_, nn, x, m) in runs if (nn, x) == (n, chi)})
    print(f"{'method':11s}" + "".join(f"{c:>22s}" for c in CASES))
    for m in methods:
        cells = []
        for c in CASES:
            rs = runs.get((c, n, chi, m))
            if not rs:
                cells.append(f"{'':>22s}")
                continue
            v, it = stat(rs, key), stat(rs, "k_err" if key == "t_err" else "k_res")
            spread = [r.get(key) for r in rs if r.get(key) is not None]
            rng = f"[{min(spread):.2f}-{max(spread):.2f}]" if spread else ""
            cells.append(f"{'FAIL':>22s}" if v is None else f"{v:7.3f} {rng:>10s} {int(it):3d}")
        print(f"{m:11s}" + "".join(cells))


def decide(runs, n=127, chi=1e8, key="t_err"):
    print(f"\n## Decision (n = {n}, chi = {chi:.0e}, {key})")
    wins = []
    for c in CASES:
        tf, mf = best(runs, c, n, chi, "fdlgf", key)
        tb, mb = best(runs, c, n, chi, "base", key)
        R = None if tf is None or tb is None else tb / tf
        wins.append(R is not None and R >= 1.5)
        print(f"{c:11s} FD-LGF {mf} {tf}  best baseline {mb} {tb}  R = {R if R is None else round(R, 2)}")
    ab, cd = wins[0] and wins[1], sum(wins)
    verdict = "GO" if cd >= 3 else ("REPOSITION" if ab else "NO-GO")
    print(f"R >= 1.5 on {cd}/4 cases -> {verdict}")
    return verdict


def extras(runs, n=127, chi=1e8):
    print(f"\n## Prediction 3: best BoomerAMG vs amg_old, total time t_res, n = {n}, chi = {chi:.0e}")
    for c in CASES:
        old = stat(runs.get((c, n, chi, "amg_old"), []), "t_res")
        b = min(((stat(rs, "t_res"), m) for (cc, nn, x, m), rs in runs.items()
                 if (cc, nn, x) == (c, n, chi) and m.startswith("amg") and stat(rs, "t_res") is not None), default=None)
        if old is None or b is None:
            print(f"{c:11s} incomplete")
            continue
        print(f"{c:11s} amg_old {old:.2f}  best {b[1]} {b[0]:.2f}  ratio {old / b[0]:.2f}")
    print(f"\n## Prediction 4: error at the residual stop 1e-10, n = {n}, chi = {chi:.0e}")
    for c in CASES:
        row = []
        for (cc, nn, x, m), rs in sorted(runs.items()):
            if (cc, nn, x) == (c, n, chi):
                e = stat(rs, "err_at_res")
                row.append(f"{m}={e:.1e}" if e is not None else f"{m}=FAIL")
        print(c, " ".join(row))
    print("\n## Setup share and memory, n = 127, chi = 1e8")
    for (c, nn, x, m), rs in sorted(runs.items()):
        if (nn, x) == (n, chi):
            s, t, mem = stat(rs, "setup"), stat(rs, "t_err"), stat(rs, "rss_peak_gb")
            if t:
                print(f"{c:11s} {m:11s} setup {s:6.2f}  t_err {t:6.2f}  peak RSS {mem:.2f} GB")
    loads = [r["load1"] for rs in runs.values() for r in rs if "load1" in r]
    crashes = [r for rs in runs.values() for r in rs if "crash" in r]
    print(f"\nruns {sum(len(v) for v in runs.values())}, crashes/timeouts {len(crashes)}, load1 median {median(loads):.1f} "
          f"range {min(loads):.1f}-{max(loads):.1f}")
    for r in crashes[:10]:
        print("  crash", r["case"], r["n"], r["chi"], r["method"], str(r["crash"])[:60])


if __name__ == "__main__":
    runs = load(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.jsonl")
    for n in (63, 127):
        for chi in (1e4, 1e8):
            table(runs, n, chi, "t_err")
    table(runs, 127, 1e8, "t_res")
    decide(runs)
    print("\n(secondary checks of the same rule)")
    decide(runs, 127, 1e4)
    decide(runs, 127, 1e8, "t_res")
    decide(runs, 63, 1e8)
    extras(runs)
