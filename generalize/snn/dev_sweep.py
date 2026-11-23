"""Exploratory dev sweep (dev seed 555, NOT the test seed): choose n_c, K, tau for CGD.  usage: python dev_sweep.py OUT.csv"""
import csv
import itertools
import sys
from time import perf_counter

import numpy as np

from cases_param import FAMILIES, make, sample
from cgd import solve_cgd
from snn_defl import amg, dpcg

DEV_SEED, N = 555, 511
with open(sys.argv[1], "w", newline="") as fh:
    wr = None
    for fam in FAMILIES:
        for t, (p, chi) in enumerate(sample(np.random.default_rng(DEV_SEED), 3)):
            A, b, _ = make(fam, N, chi, p)
            t0 = perf_counter(); P, _ = amg(A); _, its = dpcg(A, P, b); tb = perf_counter() - t0
            line = [f"{fam} #{t} base {its} {tb:.2f}s"]
            for nc, K, tau in itertools.product((127, 255), (40, 80), (0.3, 0.5)):
                _, inf = solve_cgd(A, b, N, lambda m: make(fam, m, chi, p)[0], nc=nc, K=K, tau=tau)
                row = dict(fam=fam, inst=t, base_its=its, base_s=round(tb, 3), nc=nc, K=K, tau=tau,
                           **{k: round(v, 4) if isinstance(v, float) else v for k, v in inf.items()}, ratio=round(inf["total"] / tb, 3))
                row.setdefault("lam1", "")
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row)); wr.writeheader()
                wr.writerow({k: row.get(k, "") for k in wr.fieldnames}); fh.flush()
                line.append(f"{nc}/{K}/{tau}: {inf['its']}it m{inf['m']} {row['ratio']:.2f}")
            print(" | ".join(line), flush=True)
