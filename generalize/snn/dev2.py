"""Exploratory dev study for H48 (dev seed 555 only): probe value vs difficulty, and CGD2 with forced deflation
(lam_thr = 1) for m_max in {12, 24}.  usage: python dev2.py OUT.csv n k"""
import csv
import sys
from time import perf_counter

import numpy as np

from cases_param import FAMILIES, make, sample
from cgd2 import solve_cgd2
from snn_defl import amg, dpcg

out, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(out, "a", newline="") as fh:
    wr = None
    for fam in FAMILIES:
        for t, (p, chi) in enumerate(sample(np.random.default_rng(555), k)):
            A, b, _ = make(fam, n, chi, p); mc = lambda m: make(fam, m, chi, p)[0]   # noqa: E731
            t0 = perf_counter(); P, _ = amg(A); _, its = dpcg(A, P, b); tb = perf_counter() - t0
            row = dict(fam=fam, n=n, inst=t, base_its=its, base_s=round(tb, 3))
            for mm in (12, 24):
                _, inf = solve_cgd2(A, b, n, mc, lam_thr=1.0, m_max=mm)
                row.update({f"its_m{mm}": inf["its"], f"ratio_m{mm}": round(inf["total"] / tb, 3), "lam_probe": round(inf["lam_probe"], 4)})
            _, inf = solve_cgd2(A, b, n, mc, lam_thr=0.0)
            row["ratio_skip"] = round(inf["total"] / tb, 3)
            if wr is None:
                wr = csv.DictWriter(fh, fieldnames=list(row))
                if fh.tell() == 0: wr.writeheader()
            wr.writerow(row); fh.flush(); print(row, flush=True)
