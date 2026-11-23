"""3D preconditioners: traditional LGF, GMG, hypre BoomerAMG and 方法二. All act on (n, n, n) arrays.

Same definitions as the 2D baselines, with the textbook 3D parameters:
  LGF       (a0(-Delta_h) + sigma I)^-1 by 3D DST-I, a0 = mean a, sigma = mean q   (Xia, eqs. (14)-(15))
  GMG       trilinear P, P^T, Galerkin P^T A P, V(2,2) damped Jacobi, omega = (12/7) / rowsum(D^-1|A|)
            (= 6/7 on the 7-point Laplacian, the 3D optimum: mu = 5/7), Cholesky on 3^3
  BoomerAMG PETSc defaults except strong threshold 0.5 (hypre's recommendation for 3D; 0.25 is the 2D value)
  方法二    P = Q + (I - QA) G (I - AQ); G = exact inverse for b = a1(x) by DST-I in (y, z) + tridiagonal solves in x;
            Q = sparse LU on the feature mask dilated by a ring of width w.
"""

import numpy as np
from scipy.fft import dst, dstn, idst, idstn
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.ndimage import binary_dilation, label
from scipy.sparse import coo_matrix, kron
from scipy.sparse.linalg import splu


def _lam1d(n, h):
    return (2.0 / h**2) * (1.0 - np.cos(np.arange(1, n + 1) * np.pi / (n + 1)))


class LGF3:
    def __init__(self, n, h, a0, sigma):
        l = _lam1d(n, h)
        self.den = a0 * (l[:, None, None] + l[None, :, None] + l[None, None, :]) + sigma

    def __call__(self, r):
        return idstn(dstn(r, type=1, norm="ortho") / self.den, type=1, norm="ortho")


def _interp1d(n):
    nc = (n - 1) // 2
    j = np.arange(nc)
    rows = np.concatenate([2 * j, 2 * j + 1, 2 * j + 2])
    return coo_matrix((np.repeat([0.5, 1.0, 0.5], nc), (rows, np.tile(j, 3))), shape=(n, nc)).tocsr()


class GMG3:
    def __init__(self, A, n, omega=12 / 7):
        self.mats, self.P, self.smooth = [A.tocsr()], [], []
        while n > 3:
            a = self.mats[-1]
            d = a.diagonal()
            bound = float(np.max(np.asarray(abs(a).sum(axis=1)).ravel() / d))
            self.smooth.append((omega / bound) / d)
            p1 = _interp1d(n)
            p = kron(kron(p1, p1), p1, format="csr")
            self.P.append(p)
            self.mats.append((p.T @ a @ p).tocsr())
            n = (n - 1) // 2
        self.coarse = cho_factor(self.mats[-1].toarray(), lower=True)

    def _cycle(self, lev, b):
        if lev == len(self.mats) - 1:
            return cho_solve(self.coarse, b)
        a, p, s = self.mats[lev], self.P[lev], self.smooth[lev]
        u = np.zeros_like(b)
        for _ in range(2):
            u += s * (b - a @ u)
        u += p @ self._cycle(lev + 1, p.T @ (b - a @ u))
        for _ in range(2):
            u += s * (b - a @ u)
        return u

    def __call__(self, r):
        return self._cycle(0, r.ravel()).reshape(r.shape)


class BoomerAMG3:
    def __init__(self, A, strong_threshold=0.5):
        from petsc4py import PETSc
        A = A.tocsr()
        self.mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix("bamg3d_")
        PETSc.Options()["bamg3d_pc_hypre_boomeramg_strong_threshold"] = strong_threshold
        self.pc.setOperators(self.mat)
        self.pc.setType("hypre")
        self.pc.setHYPREType("boomeramg")
        self.pc.setFromOptions()
        self.pc.setUp()
        self.x, self.y = self.mat.createVecs()

    def __call__(self, r):
        self.x.setArray(r.ravel())
        self.pc.apply(self.x, self.y)
        return self.y.getArray().copy().reshape(r.shape)


class DSTTridiag3:
    """Exact inverse for b = a1(x) (y, z homogeneous), q replaced by its (y, z)-mean r(x)."""

    def __init__(self, a1, q, h):
        n = len(a1) - 2
        d1, w = a1[1:-1], 0.5 * (a1[:-1] + a1[1:]) / h**2
        r = q.mean(axis=(0, 1))
        l = _lam1d(n, h)
        mu = (l[:, None] + l[None, :]).ravel()                      # (z, y) modes
        diag = (w[:-1] + w[1:] + r)[None, :] + mu[:, None] * d1[None, :]
        off = np.tile(np.append(-w[1:-1], 0.0), n * n)[:-1]
        self.n, self.lu = n, dgttrf(off.copy(), diag.ravel(), off.copy())[:5]

    def __call__(self, f):
        g = dstn(f, type=1, norm="ortho", axes=(0, 1))
        x = dgttrs(*self.lu, g.ravel())[0].reshape(f.shape)
        return idstn(x, type=1, norm="ortho", axes=(0, 1))


class Method2_3D:
    def __init__(self, A, a1, mask, q, h, w):
        n = len(a1) - 2
        self.A, self.n = A.tocsr(), n
        self.G = DSTTridiag3(a1, q, h)
        self.ring = max(1, round(w * (n + 1)))
        regions, nreg = label(binary_dilation(mask, iterations=self.ring)) if mask.any() else (mask, 0)
        flat = regions.ravel()
        self.solves = []
        for c in range(1, nreg + 1):
            ids = np.flatnonzero(flat == c)
            self.solves.append((ids, splu(self.A[ids][:, ids].tocsc())))
        self.local_fraction = float(sum(len(i) for i, _ in self.solves) / n**3)
        self.lu_nnz = int(sum(lu.L.nnz + lu.U.nnz for _, lu in self.solves))

    def _Q(self, r):
        z = np.zeros_like(r)
        for ids, lu in self.solves:
            z[ids] = lu.solve(r[ids])
        return z

    def __call__(self, r):
        shape = r.shape
        if not self.solves:
            return self.G(r)
        r = r.ravel()
        qr = self._Q(r)
        y = self.G((r - self.A @ qr).reshape(shape)).ravel()
        return (qr + y - self._Q(self.A @ y)).reshape(shape)
