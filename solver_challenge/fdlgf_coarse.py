"""FD-LGF with a coefficient-adapted coarse space (FD-LGF-C) for the 3T matrix (blocked [Te; Ti; Tr], m^3 cells each).

Same balanced (BNN / symmetric deflation) form as FD-LGF:
    P = Q + (I - Q A) M (I - A Q),    Q = Z (Z^T A Z)^-1 Z^T
  M  global part: Green-Jacobi LGF, M = S Gc S per temperature (S = diag(A)^-1/2, Gc = exact inverse of the
     constant-coefficient FV operator fitted to S A S, DST-II in x, y, z; O(N log N), O(N) memory)
     (Ladecký et al., arXiv:2508.02613).
  Q  replaces FD-LGF's exact local LU on features, which fails when features cover the whole grid: a coarse
     solve on piecewise-constant vectors over "coefficient regions" = connected sets of cells in the same box
     (box = b^3 cells) and the same decade of the local diffusion coefficient, per temperature (deflation vectors
     on high-coefficient regions: Vuik, Segal & Meijerink, JCP 1999; region-constant coarse spaces: Aksoylu,
     Graham, Klie & Scheichl, CVS 2008). Optionally "equal-temperature" vectors (one per region of Te) couple
     the three temperatures in the coarse space (the PCTL idea). Optional one damped-Jacobi smoothing of Z.
P is SPD (BNN); Q A Q = Q.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from run_sc import ConstantBackground


def regions(K, box, decade=1.0):
    """Labels of connected sets of cells sharing (box index, floor(log10 K / decade)); K shape (m, m, m)."""
    m = K.shape[0]
    c = np.arange(m) // box
    nb = -(-m // box)
    key = (c[:, None, None] * nb + c[None, :, None]) * nb + c[None, None, :]
    if decade:
        key = key * 1000 + np.floor(np.log10(K) / decade).astype(np.int64) + 500
    idx = np.arange(m**3).reshape(m, m, m)
    rows, cols = [], []
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        same = key[tuple(lo)] == key[tuple(hi)]
        rows.append(idx[tuple(lo)][same])
        cols.append(idx[tuple(hi)][same])
    r, cc = np.concatenate(rows), np.concatenate(cols)
    G = sp.coo_matrix((np.ones(len(r)), (r, cc)), shape=(m**3, m**3))
    return connected_components(G, directed=False)[1]


def local_K(B, m):
    """Cell diffusion coefficient proxy: mean |off-diagonal| of the temperature block (= K/h^2 for 7-point)."""
    off = abs(B - sp.diags(B.diagonal()))
    return np.maximum(np.asarray(off.sum(axis=1)).ravel() / 6.0, 1e-300).reshape(m, m, m)


class FDLGFCoarse:
    def __init__(self, A, m, *, box=8, decade=1.0, coupled=True, smooth=False, block=False, coarse=True):
        N = m**3
        self.A, self.N, self.m = A.tocsr(), N, m
        d = self.A.diagonal()
        if block:   # point-block scaling: S = blockdiag(L_c^-1), L_c = Cholesky of the 3x3 cell block
            perm = np.arange(3 * N).reshape(3, N).T.ravel()                         # interlaced index
            Ai = self.A[perm][:, perm]
            cells = np.stack([Ai[3 * np.arange(N) + i, 3 * np.arange(N) + j].A1 for i in range(3) for j in range(3)],
                             axis=1).reshape(N, 3, 3)
            Linv = np.linalg.inv(np.linalg.cholesky(cells))
            Si = sp.block_diag(list(Linv), format="csr")                            # interlaced L^-1
            inv = np.argsort(perm)
            self.S = Si[inv][:, inv].tocsr()                                         # blocked ordering
        else:
            self.S = sp.diags(1 / np.sqrt(d)).tocsr()
        self.ST = self.S.T.tocsr()
        As = (self.S @ self.A @ self.ST).tocsr()
        self.G = []
        cols, rows, ncol = [], [], 0
        for k in range(3):
            sl = slice(k * N, (k + 1) * N)
            B = As[sl, sl]
            off = -B.data[B.data < 0]
            Kc = float(np.exp(np.mean(np.log(off)))) / m**2
            r = max(float(np.median(B.diagonal())) - 6 * Kc * m**2, 0.0)
            self.G.append((sl, ConstantBackground(Kc, r, m)))
            lab = regions(local_K(self.A[sl, sl], m), box, decade)
            if coupled and k == 0:
                lab_e = lab
            rows.append(np.arange(N) + k * N)
            cols.append(lab + ncol)
            ncol += lab.max() + 1
        Z = sp.csr_matrix((np.ones(3 * N), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * N, ncol))
        if coupled:                                                  # equal-temperature vectors on Te's regions
            Zc = sp.csr_matrix((np.ones(3 * N), (np.arange(3 * N), np.tile(lab_e, 3))), shape=(3 * N, lab_e.max() + 1))
            Z = sp.hstack([Z, Zc]).tocsr()
        if not coarse:
            Z = Z[:, :0]
        if smooth:                                                   # one damped-Jacobi step (smoothed aggregation)
            Z = (Z - (2 / 3) * sp.diags(1 / d) @ (self.A @ Z)).tocsr()
        E = (Z.T @ self.A @ Z).tocsc()
        if Z.shape[1] == 0:
            self.Z, self.lu, self.coarse_size, self.coarse_nnz = Z, None, 0, 0
            return
        if coupled:                                                  # equal-temperature vectors are dependent
            E = E + 1e-12 * abs(E).max() * sp.identity(E.shape[0], format="csc")
        self.Z, self.lu = Z, splu(E)
        self.coarse_size, self.coarse_nnz = E.shape[0], E.nnz

    def M(self, r):
        y = np.empty_like(r)
        rs = self.S @ r
        for sl, G in self.G:
            y[sl] = G(rs[sl])
        return self.ST @ y

    def Q(self, r):
        if self.lu is None:
            return np.zeros_like(r)
        return self.Z @ self.lu.solve(self.Z.T @ r)

    def __call__(self, r):
        qr = self.Q(r)
        y = self.M(r - self.A @ qr)
        return qr + y - self.Q(self.A @ y)

    def apply(self, pc, x, y):                                       # petsc4py python PC
        y.setArray(self(x.getArray(readonly=True)))
