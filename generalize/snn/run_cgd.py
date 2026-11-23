"""H47 locked run (protocol_cgd.md): BoomerAMG-PCG vs coarse-grid deflation (CGD) on unseen instances.
usage: python run_cgd.py OUT.csv n k_inst family [family ...]"""
import csv
import sys
from time import perf_counter

import numpy as np

from cases_param import make, sample
from cgd import solve_cgd
from snn_defl import amg, dpcg

TEST_SEED = 2026
CFG = dict(K=40, tau=0.5, m_max=32, K_det=20, tau_det=0.06)


def n_coarse(n):
    return max(127, (n + 1) // 8 - 1)


if __name__ == "__main__":
    out, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(out, "a", newline="") as fh:
        wr = None
        for fam in sys.argv[4:]:
            for t, (p, chi) in enumerate(sample(np.random.default_rng(TEST_SEED), k)):
                A, b, _ = make(fam, n, chi, p)
                t0 = perf_counter(); P, ts = amg(A); x0, its0 = dpcg(A, P, b); tb = perf_counter() - t0
                r0 = np.linalg.norm(b - A @ x0) / np.linalg.norm(b)
                x1, inf = solve_cgd(A, b, n, lambda m: make(fam, m, chi, p)[0], nc=n_coarse(n), **CFG)
                r1 = np.linalg.norm(b - A @ x1) / np.linalg.norm(b)
                row = dict(fam=fam, n=n, inst=t, chi=round(chi, 1), p=" ".join(f"{v:.3f}" for v in p), nc=n_coarse(n),
                           base_its=its0, base_setup_s=round(ts, 3), base_total_s=round(tb, 3), base_res=f"{r0:.2e}",
                           cgd_its=inf["its"], m=inf["m"], lam1_det=round(inf["lam1_det"], 4),
                           cgd_amg_s=round(inf["t_amg"], 3), cgd_basis_s=round(inf["t_basis"], 3),
                           cgd_solve_s=round(inf["t_solve"], 3), cgd_total_s=round(inf["total"], 3), cgd_res=f"{r1:.2e}",
                           ratio=round(inf["total"] / tb, 3))
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row))
                    if fh.tell() == 0:
                        wr.writeheader()
                wr.writerow(row); fh.flush()
                print(f"{fam} n={n} #{t} base {its0} {tb:.2f}s | cgd {inf['its']} m{inf['m']} det {inf['lam1_det']:.3f} "
                      f"{inf['total']:.2f}s ratio {row['ratio']:.2f}", flush=True)
