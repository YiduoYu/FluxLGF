"""Training data for the supervised SNN: for S random instances of a family on the n_train grid, the M_T slowest modes
of B A (BoomerAMG, Lanczos K = 150 in the A inner product).  Only A, the V-cycle and coordinates are used.
usage: python gen_targets.py family OUT.npz [S] [n_train] [M_T] [seed]"""
import sys
from time import perf_counter

import numpy as np

from cases_param import make, sample
from snn_defl import amg, lanczos_A

if __name__ == "__main__":
    fam, out = sys.argv[1], sys.argv[2]
    S, n, MT, seed = (int(a) for a in (sys.argv[3:] + ["64", "255", "32", "0"][len(sys.argv) - 3:]))
    rng = np.random.default_rng(1000 + seed)
    P_, C_, U_, L_ = [], [], [], []
    t0 = perf_counter()
    for k, (p, chi) in enumerate(sample(rng, S)):
        A, _, X = make(fam, n, chi, p); P, _ = amg(A)
        lam, U = lanczos_A(A, P, 150, seed=k)
        U = U[:, :MT] / np.linalg.norm(U[:, :MT], axis=0)
        P_.append(p); C_.append(chi); U_.append(U.astype(np.float32)); L_.append(lam[:MT])
        if k % 8 == 0:
            print(f"{fam} {k}/{S} lam[:4] {np.round(lam[:4], 3)} {perf_counter()-t0:.0f}s", flush=True)
    np.savez(out, p=np.array(P_), chi=np.array(C_), U=np.array(U_), lam=np.array(L_), X=X, n=n,
             time_s=perf_counter() - t0)
