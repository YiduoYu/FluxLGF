"""Evaluate SNN deflation on unseen instances.  usage: python evaluate.py family model.pt n K_test OUT.csv [oracle]
Methods (all BoomerAMG theta 0.25 agg 0, PCG, rtol 1e-8, x0 = 0 or the deflation initial guess, single thread):
  base       BoomerAMG-PCG
  snn        deflation with the SNN basis (online = AMG setup + net eval + A-orthonormalisation + solve)
  snn+1      as snn, basis polished once: W <- W - B A W  (M extra V-cycles, counted)
  coarse-lz  non-neural control: Lanczos (K=100) for B_c A_c on the (n-1)/2... grid n_c = 127, M Ritz vectors interpolated
  oracle     M slowest modes of B A on the target itself (Lanczos K=200; cost not counted, ceiling only)
"""
import csv
import sys
from time import perf_counter

import numpy as np
import torch

from cases_param import make, sample
from snn_defl import make_net, amg, basis, dpcg, interp, lanczos_A, orthonormalise

TEST_SEED, NC = 777, 127
import os
SKIP = os.environ.get('SKIP_SNN') == '1'


def run(fam, model, n, k_test, out, oracle):
    ck = torch.load(model, weights_only=False); M = ck["M"]
    net = make_net(ck.get("arch", "mlp"), M); net.load_state_dict(ck["state"]); net.eval()
    rows = []
    for t, (p, chi) in enumerate(sample(np.random.default_rng(TEST_SEED), k_test)):
        A, b, X = make(fam, n, chi, p)
        rec = dict(fam=fam, n=n, inst=t, chi=round(chi, 1), p=" ".join(f"{v:.3f}" for v in p), M=M)
        P, ts = amg(A)
        t0 = perf_counter(); _, its = dpcg(A, P, b); tb = perf_counter() - t0
        res = dict(base=(its, ts, 0.0, tb))
        if not SKIP:
            t0 = perf_counter(); W = basis(net, X, p); W, AW = orthonormalise(W, A); tsnn = perf_counter() - t0
            t0 = perf_counter(); _, its = dpcg(A, P, b, W, AW); res["snn"] = (its, ts, tsnn, perf_counter() - t0)
            t0 = perf_counter()
            W2 = W - np.column_stack([P(c) for c in AW.T]); W2, AW2 = orthonormalise(W2, A); tpol = perf_counter() - t0
            t0 = perf_counter(); _, its = dpcg(A, P, b, W2, AW2); res["snn+1"] = (its, ts, tsnn + tpol, perf_counter() - t0)
        t0 = perf_counter()
        Ac, _, _ = make(fam, NC, chi, p); Pc, _ = amg(Ac); U = lanczos_A(Ac, Pc, 100)[1][:, :M]
        W3, AW3 = orthonormalise(interp(U, NC, n), A); tc = perf_counter() - t0
        t0 = perf_counter(); _, its = dpcg(A, P, b, W3, AW3); res["coarse-lz"] = (its, ts, tc, perf_counter() - t0)
        if oracle:
            U = lanczos_A(A, P, 200)[1][:, :M]; W4, AW4 = orthonormalise(U, A)
            t0 = perf_counter(); _, its = dpcg(A, P, b, W4, AW4); res["oracle"] = (its, ts, float("nan"), perf_counter() - t0)
        tot0 = sum(res["base"][1:])
        for k, (its, a, s, so) in res.items():
            rows.append(dict(rec, method=k, its=its, amg_setup_s=round(a, 4), defl_setup_s=round(s, 4),
                             solve_s=round(so, 4), total_s=round(a + s + so, 4),
                             ratio=round((a + (0 if np.isnan(s) else s) + so) / tot0, 3)))
        print(f"{fam} n={n} #{t} chi={chi:9.0f} p={rec['p']}  " +
              "  ".join(f"{k} {v[0]}it/{rows[-len(res)+i]['ratio']:.2f}" for i, (k, v) in enumerate(res.items())), flush=True)
    with open(out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if fh.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    fam, model, n, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    run(fam, model, n, k, out, len(sys.argv) > 6 and sys.argv[6] == "oracle")
