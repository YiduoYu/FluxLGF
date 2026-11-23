"""Exploratory: is the BoomerAMG slow subspace transferable?  Slow modes (Lanczos, A inner product) computed on a
source instance (grid n_s, contrast chi_s) are bilinearly interpolated to the target (n, chi) and used for deflation.
usage: python transfer.py family n chi"""
import sys

import numpy as np

from alow import interp
from ceiling_circ import lanczos_A
from explore2 import amg, make_case, pcg

fam, n, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])


def slow(nn, cc, K=200):
    A = make_case(fam, nn, cc).A.tocsr(); P, _ = amg(A, strong_threshold=0.25, agg_nl=0)
    return lanczos_A(A, P, nn * nn, K)[1]


c = make_case(fam, n, chi); A, b = c.A.tocsr(), c.f.ravel(); P, _ = amg(A, strong_threshold=0.25, agg_nl=0)
_, its0, *_ = pcg(A, P, b)
srcs = {"self": slow(n, chi), f"n={(n-1)//2}": interp(slow((n - 1) // 2, chi), (n - 1) // 2, n),
        f"n={(n-3)//4}": interp(slow((n - 3) // 4, chi), (n - 3) // 4, n),
        "chi/10": slow(n, chi / 10), "chi*10": slow(n, chi * 10)}
out = [f"{fam:6s} n={n} chi={chi:g} base {its0}"]
for name, U in srcs.items():
    out.append(name + ": " + "/".join(str(pcg(A, P, b, W=np.linalg.qr(U[:, :m])[0])[1]) for m in (8, 16, 32)))
print(" | ".join(out), flush=True)
