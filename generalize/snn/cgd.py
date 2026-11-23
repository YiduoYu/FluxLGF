"""Coarse-grid deflation (CGD) for BoomerAMG-PCG -- generic, no training.

Given the target matrix A (grid n) and the SAME discretization on a coarse grid n_c (we own the discretization):
  1. BoomerAMG on A_c, K-step Lanczos for B_c A_c (A_c inner product, full reorthogonalisation).
  2. Detector: a K_det-step coarse Lanczos; if its smallest Ritz value >= tau_det, BoomerAMG has no slow modes ->
     plain PCG (no deflation).  Otherwise keep the K-step Ritz vectors with value < tau (at most m_max).
  3. Bilinear interpolation (sparse tensor-product matrix) to the fine grid, A-orthonormalise, deflated PCG.
Online cost = fine AMG setup + coarse assembly/AMG/Lanczos + interpolation/orthonormalisation + solve.
"""
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from snn_defl import amg, dpcg, lanczos_A, orthonormalise


def interp_1d(nc, n):
    """Linear interpolation from nc interior coarse nodes (zero Dirichlet ends) to n interior fine nodes on [0, 1]."""
    xf = np.arange(1, n + 1) / (n + 1); hc = 1.0 / (nc + 1)
    j = np.floor(xf / hc).astype(int); w = xf / hc - j               # between coarse nodes j and j+1 (0 = boundary)
    rows, cols, vals = [], [], []
    for jj, ww in ((j, 1 - w), (j + 1, w)):
        ok = (jj >= 1) & (jj <= nc)
        rows.append(np.nonzero(ok)[0]); cols.append(jj[ok] - 1); vals.append(ww[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nc))


def interp_2d(nc, n):
    I = interp_1d(nc, n)
    return sp.kron(I, I, format="csr")                               # arrays are [y, x] raveled, same 1D map


def cgd_basis(A, n, make_coarse, nc, K, tau, m_max, K_det=20, tau_det=0.3):
    """Returns (W, AW, info); W is None if the detector says 'no slow modes'.
    Detector: K_det-step Lanczos on the coarse grid; skip if the smallest Ritz value >= tau_det.
    Otherwise K-step Lanczos (same start vector, so the first K_det steps are repeated -- cheap on the coarse grid) and
    keep the Ritz vectors with value < tau (at most m_max)."""
    t0 = perf_counter()
    Ac = make_coarse(nc); Pc, _ = amg(Ac)
    lam = lanczos_A(Ac, Pc, K_det)[0]
    info = dict(lam1_det=float(lam[0]), m=0)
    if lam[0] >= tau_det:
        info["t_basis"] = perf_counter() - t0
        return None, None, info
    lam, U = lanczos_A(Ac, Pc, K)
    m = int(min(m_max, np.sum(lam < tau)))
    W, AW = orthonormalise(interp_2d(nc, n) @ U[:, :m], A)
    info.update(lam1=float(lam[0]), m=m, t_basis=perf_counter() - t0)
    return W, AW, info


def solve_cgd(A, b, n, make_coarse, nc=127, K=60, tau=0.3, m_max=32, P=None, **kw):
    t0 = perf_counter()
    if P is None:
        P, _ = amg(A)
    t_amg = perf_counter() - t0
    W, AW, info = cgd_basis(A, n, make_coarse, nc, K, tau, m_max, **kw)
    t1 = perf_counter(); x, its = dpcg(A, P, b, W, AW); t_solve = perf_counter() - t1
    info.update(its=its, t_amg=t_amg, t_solve=t_solve, total=t_amg + info["t_basis"] + t_solve)
    return x, info
