"""FD-LGF-3T: FD-LGF extended to coupled multi-temperature, high-contrast, non-separable problems on a structured grid.

Matrix: blocked [T1; T2; T3], each block a 7-point FV operator on m^3 cells (C order), same-cell couplings between
blocks (the 3T radiation-diffusion matrix of SolverChallenge26_01). SPD assumed (CG).

Balanced (BNN) form of FD-LGF, P = Q + (I - Q A) M (I - A Q), with both ingredients changed:
  M  (omega > 0: wrapped in block-Jacobi smoothing, see M)
     point-block Green-Jacobi LGF:  M = L^-T (Gc_1 ⊕ Gc_2 ⊕ Gc_3) L^-1, L = blockdiag over cells of the Cholesky
     factor of the cell's 3x3 block (diagonals + exchange couplings), Gc_k = exact inverse (DST-II in x, y, z) of the
     constant-coefficient FV operator fitted to block k of L^-1 A L^-T. The 3x3 scaling removes the stiff local
     exchange coupling exactly; the scalar-Jacobi version (Ladecký et al. 2025) leaves one near-null mode per cell.
  Q  coarse correction on piecewise-constant vectors: one vector per temperature per box of b^3 cells, optionally
     split by decade of the local diffusion coefficient (region-constant deflation, Vuik et al. 1999; Aksoylu et
     al. 2008). It replaces FD-LGF's exact local LU on features, which covers the whole grid here.
Coarse matrix E = Z^T A Z: sparse LU (`coarse="lu"`) or, when it is itself a structured 3-field matrix (decade = 0),
the same preconditioner recursively (`coarse="ml"`, LU below `lu_size`).
"""

from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import splu


class ConstantGreen:
    """Exact inverse of K(-Δ_h) + r on m^3 cells (FV, Dirichlet at faces, h = 1/m): DST-II in each direction."""

    def __init__(self, K, r, m):
        lam = (2 - 2 * np.cos(np.arange(1, m + 1) * np.pi / m)) * m**2
        self.den = K * (lam[:, None, None] + lam[None, :, None] + lam[None, None, :]) + r
        self.m = m

    def __call__(self, f):
        g = dstn(f.reshape((self.m,) * 3), type=2, norm="ortho")
        return idstn(g / self.den, type=2, norm="ortho").ravel()


def cell_blocks(A, N):
    """3x3 cell blocks (N, 3, 3) of a blocked 3-field matrix (couplings on the offset ±N, ±2N diagonals)."""
    B = np.zeros((N, 3, 3))
    for i in range(3):
        for j in range(3):
            B[:, i, j] = A.diagonal((j - i) * N)[min(i, j) * N:min(i, j) * N + N] if i != j else \
                A.diagonal()[i * N:(i + 1) * N]
    return B


def chol3_inv(B):
    """Inverse of the Cholesky factor of SPD 3x3 blocks, B shape (N, 3, 3); returns (3, 3, N), lower triangular."""
    b = lambda i, j: B[:, i, j]
    l00 = np.sqrt(b(0, 0))
    l10, l20 = b(1, 0) / l00, b(2, 0) / l00
    l11 = np.sqrt(b(1, 1) - l10**2)
    l21 = (b(2, 1) - l20 * l10) / l11
    l22 = np.sqrt(b(2, 2) - l20**2 - l21**2)
    L = np.zeros((3, 3, B.shape[0]))
    L[0, 0], L[1, 1], L[2, 2] = 1 / l00, 1 / l11, 1 / l22
    L[1, 0] = -l10 / (l00 * l11)
    L[2, 1] = -l21 / (l11 * l22)
    L[2, 0] = (l10 * l21 - l11 * l20) / (l00 * l11 * l22)
    return L


def inv3(B):
    """Inverse of general 3x3 blocks B (N, 3, 3) by the adjugate; returns (3, 3, N)."""
    b = lambda i, j: B[:, i, j]
    C = np.empty((3, 3, B.shape[0]))
    for i in range(3):
        for j in range(3):
            r, c = [k for k in range(3) if k != j], [k for k in range(3) if k != i]
            C[i, j] = (-1) ** (i + j) * (b(r[0], c[0]) * b(r[1], c[1]) - b(r[0], c[1]) * b(r[1], c[0]))
    det = b(0, 0) * C[0, 0] + b(0, 1) * C[1, 0] + b(0, 2) * C[2, 0]
    return C / det


def block_scaling(A, N):
    """S = blockdiag_c(L_c^-1) in blocked ordering, as a sparse matrix (L_c = Cholesky of the 3x3 cell block)."""
    Linv = np.linalg.inv(np.linalg.cholesky(cell_blocks(A, N)))
    c = np.arange(N)
    rows = np.concatenate([c + i * N for i in range(3) for j in range(3)])
    cols = np.concatenate([c + j * N for i in range(3) for j in range(3)])
    vals = np.concatenate([Linv[:, i, j] for i in range(3) for j in range(3)])
    S = sp.csr_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N))
    S.eliminate_zeros()
    return S


def box_aggregates(m, box):
    """Box index (C order on the (m/box)^3 coarse grid) of every cell."""
    c = np.arange(m) // box
    mc = -(-m // box)
    return ((c[:, None, None] * mc + c[None, :, None]) * mc + c[None, None, :]).ravel(), mc


class FDLGF3T:
    """green=False replaces the LGF part M0 by the exact 3x3 block-Jacobi inverse (no DST: the H37 finding that the
    constant-coefficient Green function does not help on this class). dtype=np.float32 runs the preconditioner
    (matvecs, scaling, DST) in single precision; the coarsest LU stays in double."""

    def __init__(self, A, m, *, box=4, coarse="ml", lu_size=20000, level=0, omega=0.0, cycles=1, green=True,
                 dtype=np.float64, form="bnn", nu=1):
        t0 = perf_counter()
        N = m**3
        A = A.tocsr()
        self.N, self.m, self.level, self.omega, self.dtype, self.form, self.nu = N, m, level, omega, dtype, form, nu
        self.A = A.astype(dtype)
        B = cell_blocks(A, N)
        if green:                                   # needs the symmetric scaling S = blockdiag(L_c^-1)
            self.L = L = chol3_inv(B).astype(dtype)                                  # (3, 3, N), lower triangular
            self.D = sum(L[k, :, None] * L[k, None, :] for k in range(3))            # D^-1 = L^-T L^-1, (3, 3, N)
        else:                                       # block Jacobi only: plain 3x3 inverse, also for nonsymmetric A
            self.D = inv3(B).astype(dtype)
        self.G = []
        if green:
            for k in range(3):
                # off-diagonal row sums of block k of S A S^T without forming it: sum_a L_ka (Aoff_aa @ L_ka)
                rs = np.zeros(N)
                for a in range(k + 1):
                    sl = slice(a * N, (a + 1) * N)
                    Baa = A[sl, sl]
                    Boff = Baa - sp.diags(Baa.diagonal())
                    rs -= L[k, a] * (Boff @ L[k, a])
                Kc = max(float(np.median(rs)) / (6 * m**2), 1e-300)
                r = max(1.0 - 6 * Kc * m**2, 0.0)                                    # scaled diagonal = 1
                self.G.append((slice(k * N, (k + 1) * N), ConstantGreen(Kc, r, m)))
        agg, mc = box_aggregates(m, box)
        Nc = mc**3
        self.agg = np.concatenate([agg + k * Nc for k in range(3)])
        self.nc = 3 * Nc
        self.Z = Z = sp.csr_matrix((np.ones(3 * N, dtype), (np.arange(3 * N), self.agg)), shape=(3 * N, 3 * Nc))
        self.ZT = Z.T.tocsr()
        E = (self.ZT @ A @ Z).tocsr()
        self.setup_s = perf_counter() - t0
        kw = dict(box=box, coarse="ml", lu_size=lu_size, level=level + 1, omega=omega, cycles=cycles, green=green,
                  dtype=dtype, form=form, nu=nu)
        if coarse == "ml" and 3 * Nc > lu_size and mc % box == 0:
            inner = FDLGF3T(E, mc, **kw)
            self.coarse = inner if cycles == 1 else self._cycled(inner, E.astype(dtype), cycles)
            self.coarse.sizes = inner.sizes
        else:
            lu = splu(E.tocsc())
            self.coarse = lambda r: lu.solve(r.astype(np.float64)).astype(dtype)
            self.coarse_lu_nnz = lu.L.nnz + lu.U.nnz
        self.sizes = [3 * N] + getattr(self.coarse, "sizes", [3 * Nc])

    @staticmethod
    def _cycled(P, E, cycles):
        """Coarse solve by `cycles` stationary steps with P (W-cycle-like); the result 2P - PEP (cycles = 2) is
        symmetric."""
        def solve(r):
            x = P(r)
            for _ in range(cycles - 1):
                x = x + P(r - E @ x)
            return x
        return solve

    def _S(self, x):
        """S x, S = blockdiag(L_c^-1) (lower triangular), on blocked vectors."""
        L, x = self.L, x.reshape(3, -1)
        return np.concatenate([L[0, 0] * x[0], L[1, 0] * x[0] + L[1, 1] * x[1],
                               L[2, 0] * x[0] + L[2, 1] * x[1] + L[2, 2] * x[2]])

    def _ST(self, y):
        L, y = self.L, y.reshape(3, -1)
        return np.concatenate([L[0, 0] * y[0] + L[1, 0] * y[1] + L[2, 0] * y[2],
                               L[1, 1] * y[1] + L[2, 1] * y[2], L[2, 2] * y[2]])

    def Dinv(self, x):
        """Exact inverse of the 3x3 cell blocks."""
        D, x = self.D, x.reshape(3, -1)
        return np.concatenate([D[i, 0] * x[0] + D[i, 1] * x[1] + D[i, 2] * x[2] for i in range(3)])

    def M0(self, r):
        if not self.G:
            return self.Dinv(r)
        rs = self._S(r)
        y = np.empty_like(rs)
        for sl, G in self.G:
            y[sl] = G(rs[sl])
        return self._ST(y)

    def M(self, r):
        """LGF, or with omega > 0 the symmetric composition (I - MA) = (I - wD^-1 A)(I - M0 A)(I - wD^-1 A),
        D^-1 the exact inverse of the 3x3 cell blocks (block-Jacobi smoothing at coefficient jumps)."""
        if not self.omega:
            return self.M0(r)
        y = self.omega * self.Dinv(r)
        y = y + self.M0(r - self.A @ y)
        return y + self.omega * self.Dinv(r - self.A @ y)

    def Q(self, r):
        return self.Z @ self.coarse(self.ZT @ r)

    def __call__(self, r):
        r = np.asarray(r, dtype=self.dtype)
        if self.form == "vcycle":
            # symmetric V(nu, nu) cycle: block-Jacobi smoothing, Galerkin coarse correction (2 nu fine matvecs)
            y = self.omega * self.Dinv(r)
            for _ in range(self.nu - 1):
                y = y + self.omega * self.Dinv(r - self.A @ y)
            y = y + self.Q(r - self.A @ y)
            for _ in range(self.nu):
                y = y + self.omega * self.Dinv(r - self.A @ y)
            return y
        qr = self.Q(r)
        y = self.M(r - self.A @ qr)
        return qr + y - self.Q(self.A @ y)

    def apply(self, pc, x, y):                                       # petsc4py python PC
        y.setArray(self(x.getArray(readonly=True)).astype(np.float64))
