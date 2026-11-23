"""Exploratory: deflated PCG with the m lowest eigenvectors of A (fine grid, exact), and with the lowest eigenvectors of
the coarse-grid operator (n_c = (n-1)/2) interpolated bilinearly -- the non-neural version of 'learn near-kernel on a
coarse grid, transfer'.   usage: python alow.py n chi"""
import sys
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import eigsh

from explore2 import amg, make_case, pcg


def interp(E, nc, n):
    tc = np.linspace(0, 1, nc + 2); tf = np.arange(1, n + 1) / (n + 1)
    Yf, Xf = np.meshgrid(tf, tf, indexing="ij"); pts = np.column_stack([Yf.ravel(), Xf.ravel()])
    out = []
    for i in range(E.shape[1]):
        g = np.zeros((nc + 2, nc + 2)); g[1:-1, 1:-1] = E[:, i].reshape(nc, nc)
        out.append(RegularGridInterpolator((tc, tc), g)(pts))
    return np.column_stack(out)


if __name__ == "__main__":
    n, chi = int(sys.argv[1]), float(sys.argv[2])
    c = make_case("circ", n, chi); A, b = c.A.tocsr(), c.f.ravel()
    P, ts = amg(A, strong_threshold=0.25, agg_nl=0)
    _, its0, *_ = pcg(A, P, b); print(f"n={n} base its {its0}")
    t0 = perf_counter(); mu, E = eigsh(A.tocsc(), k=64, sigma=0); E = E[:, np.argsort(mu)]; print(f"fine eigsh {perf_counter()-t0:.1f}s")
    nc = (n - 1) // 2; Ac = make_case("circ", nc, chi).A.tocsc()
    t0 = perf_counter(); muc, Ec = eigsh(Ac, k=64, sigma=0); Ec = Ec[:, np.argsort(muc)]; tc = perf_counter() - t0
    Ei = interp(Ec, nc, n); print(f"coarse eigsh (nc={nc}) {tc:.1f}s")
    for m in (8, 16, 32, 64):
        for name, W in (("fine-eig", E[:, :m]), ("coarse-eig-interp", Ei[:, :m])):
            W, _ = np.linalg.qr(W)
            t0 = perf_counter(); _, its, *_ = pcg(A, P, b, W=W); t = perf_counter() - t0
            print(f"  m={m:2d} {name:18s} its {its:3d} solve {t:.2f}s", flush=True)
