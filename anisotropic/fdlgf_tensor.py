"""FD-LGF for tensor coefficients: same balanced form P = Q + (I - Q A) G (I - A Q) as ../fdlgf.py, with a
grid-aligned anisotropic background K_b = diag(k1(x), k2(x)) whose flux operator (+ the y-mean of q) is inverted
exactly by DST-I along y and one tridiagonal solve along x per mode, O(n^2 log n).
"""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "CNN function")]

import numpy as np
from scipy.fft import dst, idst
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.ndimage import binary_dilation, label
from scipy.sparse.linalg import splu

from fdlgf import FDLGFPreconditioner


class AnisoBackground:
    """Exact inverse of -d/dx(k1 d/dx) - k2 d^2/dy^2 + r(x) (flux stencil, k1, k2 nodal along x with boundary)."""

    kind = "DST-y + tridiag-x (diag K)"

    def __init__(self, k1, k2, q, h):
        n = len(k1) - 2
        w = 0.5 * (k1[:-1] + k1[1:]) / h**2
        mu = (2 - 2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))) / h**2
        diag = (w[:-1] + w[1:] + np.mean(q, axis=0))[None, :] + mu[:, None] * k2[1:-1][None, :]
        off = np.tile(np.append(-w[1:-1], 0.0), n)[:-1]
        self.n, self.lu = n, dgttrf(off.copy(), diag.ravel(), off.copy())[:5]

    def __call__(self, f):
        g = dst(f, type=1, norm="ortho", axis=0)
        x = dgttrs(*self.lu, g.ravel())[0]
        return idst(x.reshape(self.n, self.n), type=1, norm="ortho", axis=0)


class FDLGFTensor(FDLGFPreconditioner):
    """FDLGFPreconditioner with a given background inverse G and feature mask (interior, n x n)."""

    def __init__(self, A, G, mask, *, w=1 / 16):
        n = mask.shape[0]
        self.n, self.A, self.G = n, A.tocsr(), G
        self.ring = max(1, round(w * (n + 1)))
        regions, nreg = label(binary_dilation(mask, iterations=self.ring)) if mask.any() else (mask, 0)
        flat = regions.ravel()
        self.solves = []
        for c in range(1, nreg + 1):
            ids = np.flatnonzero(flat == c)
            self.solves.append((ids, splu(self.A[ids][:, ids].tocsc())))
        self.feature_fraction = float(mask.mean())
        self.local_fraction = float(sum(len(ids) for ids, _ in self.solves) / n**2)
        self.lu_nnz = int(sum(lu.L.nnz + lu.U.nnz for _, lu in self.solves))
