"""Exploratory: per family, base BoomerAMG-PCG vs deflation with (a) exact AMG slow modes (Lanczos, ceiling),
(b) m lowest eigenvectors of A (fine), (c) coarse-grid eigenvectors interpolated.  usage: python controls.py family n chi"""
import sys
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

from alow import interp
from ceiling_circ import lanczos_A
from explore2 import amg, make_case, pcg

fam, n, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
A, b = (lambda c: (c.A.tocsr(), c.f.ravel()))(make_case(fam, n, chi))
P, ts = amg(A, strong_threshold=0.25, agg_nl=0)
t0 = perf_counter(); _, its0, *_ = pcg(A, P, b); tb = perf_counter() - t0
lam, U, res = lanczos_A(A, P, n * n, 200)
mu, E = eigsh(A.tocsc(), k=64, sigma=0); E = E[:, np.argsort(mu)]
nc = (n - 1) // 2; muc, Ec = eigsh(make_case(fam, nc, chi).A.tocsc(), k=64, sigma=0); Ei = interp(Ec[:, np.argsort(muc)], nc, n)
row = [f"{fam:8s} n={n} chi={chi:g} base {its0:3d} ({ts+tb:.2f}s) #lam<0.1 {np.sum(lam<0.1)}"]
for m in (8, 16, 32, 64):
    its = []
    for W in (U[:, :m], E[:, :m], Ei[:, :m]):
        _, k, *_ = pcg(A, P, b, W=np.linalg.qr(W)[0]); its.append(k)
    row.append(f"m={m}: slow {its[0]} eigA {its[1]} coarse {its[2]}")
print(" | ".join(row), flush=True)
