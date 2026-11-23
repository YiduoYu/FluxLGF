"""FD-LGF (fictitious-domain lattice Green's function preconditioner, FDM background; called 方法二 in H30–H32):
balanced fictitious-domain LGF with an exact separable background inverse.

Problem: -div(a grad u) + q u = f on (0,1)^2, zero Dirichlet, 5-point flux stencil of axy_lgf.operators
(arrays indexed [j, i] = [y, x]; a_nodes includes the boundary, shape (n+2, n+2)).

Preconditioner (one application):
    P r = Q r + (I - Q A) G (I - A Q) r
  b = a1(x) a2(y)  separable background, fitted robustly to log a (features excluded);
  G                exact inverse of the flux operator with coefficient b (+ a separable part of q):
                   DST-I along y + tridiagonal solves along x when a2 is constant (O(n^2 log n)),
                   otherwise fast diagonalisation (two 1D generalised eigenproblems, Lynch-Rice-Thomas);
  Q                exact sparse-LU solves on the connected components of the feature mask
                   |log a - log b| > log 2, dilated by a ring of physical width w.
P is SPD when G and the local solves are (Q = sum_k R_k^T A_k^-1 R_k).
"""

from time import perf_counter

import numpy as np
from scipy.fft import dst, idst
from scipy.linalg import eigh
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.ndimage import binary_dilation, label
from scipy.sparse.linalg import splu

from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion, reaction_field
from axy_lgf.solvers import _pcg, _validate_problem


# ---------------------------------------------------------------- background / feature split
def split_coefficient(a_nodes, ratio=2.0, sweeps=3):
    """log a ~ g1(x) + g2(y) by alternating medians over the non-feature nodes.

    Returns (a1 along x, a2 along y, mask), boundary nodes included; mask marks |log a - log b| > log ratio.
    Valid when features cover less than half of every grid row and column; otherwise pass split=(a1, a2, mask).
    """
    L = np.log(a_nodes)
    keep = np.ones(L.shape, bool)
    g2 = np.zeros(L.shape[0])
    for _ in range(sweeps):
        D = np.where(keep, L - g2[:, None], np.nan)
        g1 = np.nanmedian(D, axis=0)
        D = np.where(keep, L - g1[None, :], np.nan)
        g2 = np.nanmedian(D, axis=1)
        keep = np.abs(L - g1[None, :] - g2[:, None]) <= np.log(ratio)
    return np.exp(g1), np.exp(g2), ~keep


def _flux_tridiag(a1, h):
    """Interior 1D flux matrix (dense) for nodal coefficient a1 of length n+2."""
    w = 0.5 * (a1[:-1] + a1[1:]) / h**2
    return np.diag(w[:-1] + w[1:]) - np.diag(w[1:-1], 1) - np.diag(w[1:-1], -1)


class DSTTridiagBackground:
    """Exact inverse for b = a1(x), q replaced by its y-mean r(x): DST-I in y, one tridiagonal system in x per mode."""

    kind = "DST-y + tridiag-x"

    def __init__(self, a1, q, h):
        n = len(a1) - 2
        d1, w = a1[1:-1], 0.5 * (a1[:-1] + a1[1:]) / h**2
        r = np.mean(q, axis=0)
        mu = (2 - 2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))) / h**2
        diag = (w[:-1] + w[1:] + r)[None, :] + mu[:, None] * d1[None, :]
        off = np.tile(np.append(-w[1:-1], 0.0), n)[:-1]  # block-diagonal: zero coupling between modes
        self.n, self.lu = n, dgttrf(off.copy(), diag.ravel(), off.copy())[:5]

    def __call__(self, f):
        g = dst(f, type=1, norm="ortho", axis=0)
        x = dgttrs(*self.lu, g.ravel())[0]
        return idst(x.reshape(self.n, self.n), type=1, norm="ortho", axis=0)


class FDMBackground:
    """Exact inverse of D2 U (Tx + R) + Ty U D1 (b = a1(x) a2(y), R = diag(mean_j q/a2)) by fast diagonalisation."""

    kind = "fast diagonalisation"

    def __init__(self, a1, a2, q, h):
        d1, d2 = a1[1:-1], a2[1:-1]
        tx = _flux_tridiag(a1, h) + np.diag(np.mean(q / d2[:, None], axis=0))
        lam, self.V = eigh(tx, np.diag(d1))
        mu, self.W = eigh(_flux_tridiag(a2, h), np.diag(d2))
        self.den = mu[:, None] + lam[None, :]

    def __call__(self, f):
        return self.W @ ((self.W.T @ f @ self.V) / self.den) @ self.V.T


class TransposedBackground:
    """b = a2(y) only: the DST-x + tridiagonal-y solver, i.e. DSTTridiagBackground on transposed arrays."""

    kind = "DST-x + tridiag-y"

    def __init__(self, a2, q, h):
        self.inner = DSTTridiagBackground(a2, np.ascontiguousarray(q.T), h)

    def __call__(self, f):
        return self.inner(f.T).T


def background_inverse(a1, a2, q, h):
    if np.ptp(np.log(a2)) < 1e-12:
        return DSTTridiagBackground(a1 * a2[0], q, h)
    if np.ptp(np.log(a1)) < 1e-12:
        return TransposedBackground(a2 * a1[0], q, h)
    return FDMBackground(a1, a2, q, h)


# ---------------------------------------------------------------- preconditioner
class FDLGFPreconditioner:
    """P = Q + (I - Q A) G (I - A Q); acts on n x n arrays (as axy_lgf's _pcg expects)."""

    def __init__(self, A, a_nodes, q, h, *, w=1 / 16, split=None):
        n = a_nodes.shape[0] - 2
        a1, a2, mask = split_coefficient(a_nodes) if split is None else split
        self.n, self.A = n, A.tocsr()
        self.G = background_inverse(a1, a2, q, h)
        mask = mask[1:-1, 1:-1]
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

    def _Q(self, r):
        z = np.zeros_like(r)
        for ids, lu in self.solves:
            z[ids] = lu.solve(r[ids])
        return z

    def __call__(self, r):
        n = self.n
        r = r.ravel()
        if not self.solves:
            return self.G(r.reshape(n, n))
        qr = self._Q(r)
        y = self.G((r - self.A @ qr).reshape(n, n)).ravel()
        return (qr + y - self._Q(self.A @ y)).reshape(n, n)


def solve_fdlgf_pcg(a, f, grid, *, q=0.0, a_nodes=None, tol=1e-10, maxiter=500, u0=None, w=1 / 16, split=None):
    """PCG with FD-LGF, same loop / stopping rule as solve_axy_pcg and solve_multigrid_pcg.
    a_nodes (with boundary) is required: the background is fitted on it. split = (a1, a2, mask) overrides the fit."""
    started = perf_counter()
    _validate_problem(a, f, grid, maxiter=maxiter, u0=u0)
    qq = np.ascontiguousarray(reaction_field(q, grid))
    A = assemble_reaction_diffusion(a, grid, qq, a_nodes=a_nodes)
    P = FDLGFPreconditioner(A, a_nodes, qq, grid.h, w=w, split=split)
    setup_time = perf_counter() - started
    action = lambda u: apply_reaction_diffusion(u, a, grid, qq, a_nodes=a_nodes)
    result = _pcg(action, P, f, grid, tol=tol, maxiter=maxiter, u0=u0, a0=1.0)
    result.setup_time = setup_time
    result.preconditioner = P
    return result
