"""FD-LGF for the real SolverChallenge26_01 matrix (H40): separable background + exact solve on the feature.

The real matrix is, outside a corner ball, three constant-coefficient FV operators
    A_bg,k = kappa_k (T x I x I + I x T x I + I x I x T) + r_k,   T = Neumann 1D graph Laplacian (+ beta_k/kappa_k at the
far end for T1's Robin faces), plus a constant one-way coupling -e12 from T2 into the T1 rows (e12 = 963 is ~99 % of T1's
reaction; the other couplings are < 2e-11 of the diagonal there and are left to A - A_bg). Each A_bg,k^-1 is applied
exactly by the fast diagonalization method (FDM, the separable generalised LGF of H16): T = V diag(lam) V^T, three
128 x 128 matrix products per direction; the block lower-triangular background is solved T2, T3, then T1.
Feature F = cells where a row of A differs from A_bg by more than `tol` x the row's largest entry, dilated by `ring`; Q = R_F^T A_FF^-1 R_F
with A_FF (all three fields on F) factorised by MUMPS. Balanced form P = Q + (I - QA) G (I - AQ), G = blockdiag A_bg,k^-1.
"""

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.ndimage import binary_dilation


def fit_background(A, m):
    """Per block: kappa (median face coefficient), r (median of diag - sum|within-block offdiag|), beta (far-face
    Robin); and e12 (median T1-T2 coupling)."""
    N = m**3
    d = A.diagonal()
    out = []
    for k in range(3):
        s = slice(k * N, (k + 1) * N)
        rowabs = np.asarray(abs(A[s, s]).sum(axis=1)).ravel()
        kappa = float(np.median(-A.diagonal(1)[s.start:s.start + N - 1]))
        react = (2 * d[s] - rowabs).reshape(m, m, m)
        r = float(np.median(react))
        beta = float(np.median(react[:, m // 2, -1]) - r)            # x = m-1 face, away from the corner ball
        out.append((kappa, r, beta if abs(beta) > 1e-6 * r else 0.0))
    return out, float(np.median(-A.diagonal(N)[:N]))


def background_matrix(m, kappa, r, beta):
    """Sparse A_bg for one block (lexicographic, x fastest)."""
    T = sp.diags([-np.ones(m - 1), np.r_[1, 2 * np.ones(m - 2), 1], -np.ones(m - 1)], [-1, 0, 1]).tolil()
    T[m - 1, m - 1] += beta / kappa
    T = T.tocsr()
    I = sp.identity(m, format="csr")
    L = sp.kron(sp.kron(I, I), T) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(T, I), I)
    return (kappa * L + r * sp.identity(m**3)).tocsr(), T


class FDM:
    """Exact inverse of kappa (T(+)T(+)T) + r by the fast diagonalization method."""

    def __init__(self, T, kappa, r, m):
        lam, self.V = np.linalg.eigh(T.toarray())
        self.den = kappa * (lam[:, None, None] + lam[None, :, None] + lam[None, None, :]) + r
        self.m = m

    def _apply(self, X, W):                                        # X[z, y, x] -> multiply W^T along every axis
        X = X @ W                                                  # x
        X = np.einsum("zyx,yj->zjx", X, W, optimize=True)          # y
        return np.einsum("zyx,zj->jyx", X, W, optimize=True)       # z

    def __call__(self, f):
        g = self._apply(f.reshape((self.m,) * 3), self.V) / self.den
        return self._apply(g, self.V.T).ravel()


class FDLGFReal:
    def __init__(self, A, m, ring=4, tol=1e-6):
        N = m**3
        self.A, self.m, self.N = A, m, N
        self.bg, self.e12 = fit_background(A, m)
        mats, self.G = [], []
        for kappa, r, beta in self.bg:
            Abg, T = background_matrix(m, kappa, r, beta)
            mats.append(Abg)
            self.G.append(FDM(T, kappa, r, m))
        Abg = sp.bmat([[mats[0], -self.e12 * sp.identity(N), None], [None, mats[1], None], [None, None, mats[2]]])
        Dabs = abs(A - Abg.tocsr())
        scale = abs(A).max(axis=1).toarray().ravel()
        bad = np.asarray(Dabs.max(axis=1).todense()).ravel() > tol * scale
        cells = bad.reshape(3, N).any(axis=0).reshape(m, m, m)
        self.core_fraction = float(cells.mean())
        if ring:
            cells = binary_dilation(cells, iterations=ring)
        self.cells = cells
        self.fraction = float(cells.mean())
        idx = np.flatnonzero(cells.ravel())
        self.F = np.concatenate([idx + k * N for k in range(3)])
        AFF = A[self.F][:, self.F].tocsr()
        self.nF = AFF.shape[0]
        M = PETSc.Mat().createAIJ(size=AFF.shape, csr=(AFF.indptr, AFF.indices, AFF.data))
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp.setOperators(M)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.ksp.setUp()
        self._b, self._x = M.createVecs()

    def Q(self, r):
        out = np.zeros_like(r)
        self._b.setArray(r[self.F])
        self.ksp.solve(self._b, self._x)
        out[self.F] = self._x.getArray()
        return out

    def Gmul(self, r):
        N = self.N
        x2 = self.G[1](r[N:2 * N])
        return np.concatenate([self.G[0](r[:N] + self.e12 * x2), x2, self.G[2](r[2 * N:])])

    def __call__(self, r):
        q = self.Q(r)
        z = self.Gmul(r - self.A @ q)
        return q + z - self.Q(self.A @ z)

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))
