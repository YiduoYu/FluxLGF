"""H48 locked run (protocol_cgd2.md): BoomerAMG-PCG vs adaptive coarse-grid deflation (CGD2), unseen instances.
usage: python run_cgd2.py OUT.csv n k_inst family [family ...]"""
import csv
import sys
from time import perf_counter

import numpy as np

from cases_param import make, sample
from cgd2 import solve_cgd2
from snn_defl import amg, dpcg

TEST_SEED = 3030
CFG = dict(k0=10, lam_thr=0.035, K=40, tau=0.5, m_max=24)

if __name__ == "__main__":
    out, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(out, "a", newline="") as fh:
        wr = None
        for fam in sys.argv[4:]:
            for t, (p, chi) in enumerate(sample(np.random.default_rng(TEST_SEED), k)):
                A, b, _ = make(fam, n, chi, p)
                t0 = perf_counter(); P, ts = amg(A); x0, its0 = dpcg(A, P, b); tb = perf_counter() - t0
                r0 = np.linalg.norm(b - A @ x0) / np.linalg.norm(b)
                del P
                x1, inf = solve_cgd2(A, b, n, lambda m: make(fam, m, chi, p)[0], **CFG)
                r1 = np.linalg.norm(b - A @ x1) / np.linalg.norm(b)
                row = dict(fam=fam, n=n, inst=t, chi=round(chi, 1), p=" ".join(f"{v:.3f}" for v in p),
                           base_its=its0, base_total_s=round(tb, 3), base_res=f"{r0:.2e}",
                           lam_probe=round(inf["lam_probe"], 4), deflated=int(inf["m"] > 0), m=inf["m"],
                           cgd_its=inf["its"], cgd_amg_s=round(inf["t_amg"], 3), cgd_basis_s=round(inf["t_basis"], 3),
                           cgd_solve_s=round(inf["t_solve"], 3), cgd_total_s=round(inf["total"], 3), cgd_res=f"{r1:.2e}",
                           ratio=round(inf["total"] / tb, 3))
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row))
                    if fh.tell() == 0:
                        wr.writeheader()
                wr.writerow(row); fh.flush()
                print(f"{fam} n={n} #{t} base {its0} {tb:.2f}s | probe {inf['lam_probe']:.3f} m{inf['m']} cgd {inf['its']} "
                      f"{inf['total']:.2f}s ratio {row['ratio']:.2f}", flush=True)
