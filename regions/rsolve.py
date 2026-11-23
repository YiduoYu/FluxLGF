"""Scalar FD-LGF for one block (H52): fitted constant-coefficient background inverted exactly by the fast
diagonalization method (FDM; DCT when there is no Robin term) + exact (MUMPS LU) solve on the feature cells where the
block differs from the background, combined in the balanced form P = Q + (I - QM) G (I - MQ).
Unlike H40's fdlgf-real (one LU over all three fields of the 6.4 % feature, 403k unknowns, 42 s), this is used per
scalar block after electron elimination (S3 = ion Schur complement, A11 = radiation)."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from fdlgf_real import FDM, background_matrix  # noqa: E402


def fit(M, m):
    """kappa = median face coefficient, r = median reaction, beta = Robin at the x = m-1 face (0 if negligible)."""
    d = M.diagonal()
    kappa = float(np.median(-M.diagonal(1)))
    react = (2 * d - np.asarray(abs(M).sum(axis=1)).ravel()).reshape(m, m, m)
    r = float(np.median(react))
    beta = float(np.median(react[:, m // 2, -1]) - r)
    return kappa, r, (beta if abs(beta) > 1e-6 * abs(r) else 0.0)


class ScalarFDLGF:
    def __init__(self, M, m, ring=2, tol=1e-6):
        M = M.tocsr()
        self.M, self.m = M, m
        self.kappa, self.r, self.beta = fit(M, m)
        Mbg, T = background_matrix(m, self.kappa, self.r, self.beta)
        self.G = FDM(T, self.kappa, self.r, m)
        scale = abs(M).max(axis=1).toarray().ravel()
        bad = np.asarray(abs(M - Mbg).max(axis=1).todense()).ravel() > tol * scale
        cells = bad.reshape(m, m, m)
        self.core_fraction = float(cells.mean())
        if ring:
            cells = binary_dilation(cells, iterations=ring)
        self.F = np.flatnonzero(cells.ravel())
        C = np.ones(m**3, bool); C[self.F] = False
        self.C = np.flatnonzero(C)
        self.MCF = M[self.C][:, self.F].tocsr()
        self.MFC = M[self.F][:, self.C].tocsr()
        self._factor(M[self.F][:, self.F].tocsr())

    BLR = 0.0        # H56: MUMPS block low-rank threshold CNTL(7) for the feature LU (0: off, full rank as in H53-H55)
    ORDERING = 0     # H56: MUMPS ICNTL(7) ordering (0: PETSc/MUMPS default as in H53-H55; 5 = METIS)
    CHOLESKY = False # H52's CHOLMOD on symmetric feature blocks, for same-batch comparison with LU (audit item 3)

    def _factor(self, MFF):
        P = PETSc.Mat().createAIJ(size=MFF.shape, csr=(MFF.indptr, MFF.indices, MFF.data))
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        self.ksp.setOperators(P)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        # MUMPS LU (H53). CHOLMOD Cholesky is 30 % cheaper but less accurate on the near-singular core of SC26_01:
        # 53 vs 16 outer its to 1e-8 (H52 post-lock).
        if self.CHOLESKY and (MFF != MFF.T).nnz == 0:
            P.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            pc.setType("cholesky"); pc.setFactorSolverType("cholmod")
        else:
            pc.setType("lu"); pc.setFactorSolverType("mumps")
        if self.BLR or self.ORDERING:
            pc.setFactorSetUpSolverType()
            Fm = pc.getFactorMatrix()
            if self.BLR:
                Fm.setMumpsIcntl(35, 1); Fm.setMumpsCntl(7, self.BLR)
            if self.ORDERING:
                Fm.setMumpsIcntl(7, self.ORDERING)
        self.ksp.setUp()
        self._b, self._x = P.createVecs()
        self._keep = P

    def lu(self, rF):
        self._b.setArray(rF)
        self.ksp.solve(self._b, self._x)
        return self._x.getArray().copy()

    def __call__(self, r):
        """Balanced form P = Q + (I - QM) G (I - MQ), rewritten with the identities (r - M Q r)_F = 0 and
        Q M z = z_F + M_FF^-1 M_FC z_C, so that no product with the feature rows M_FF is formed (those rows carry the
        1e17 / near-cancelling entries of the core and made the computed P nonlinear at the 1e-8 level)."""
        qF = self.lu(r[self.F])
        w = np.zeros_like(r)
        w[self.C] = r[self.C] - self.MCF @ qF
        z = self.G(w)
        out = np.empty_like(r)
        out[self.C] = z[self.C]
        out[self.F] = qF - self.lu(self.MFC @ z[self.C])
        return out


def faces(M, m):
    """(n, 6) face coefficients of every cell (+x, -x, +y, -y, +z, -z); nan where there is no neighbour."""
    n = m**3
    f = np.full((n, 6), np.nan)
    idx = np.arange(n)
    for k, s in enumerate((1, m, m * m)):
        v = np.zeros(n)
        v[:n - s] = -M.diagonal(s)
        coord = (idx // s) % m
        plus, minus = coord < m - 1, coord > 0
        f[plus, 2 * k] = v[plus]
        f[minus, 2 * k + 1] = v[idx[minus] - s]
    return f


def make_T(m, kappa, beta):
    T = sp.diags([-np.ones(m - 1), np.r_[1, 2 * np.ones(m - 2), 1], -np.ones(m - 1)], [-1, 0, 1]).tolil()
    T[m - 1, m - 1] += beta / kappa
    return T.tocsr()


class MultiFDLGF(ScalarFDLGF):
    """H54: several constant-coefficient backgrounds instead of one. Regions = 6-connected components (>= m^3/1000
    cells) of cells whose face coefficients agree to 1 + 1e-6; each region k gets its own constants (median face
    kappa_k, median reaction r_k, far-face Robin beta_k) and its own FDM inverse G_k on the whole cube. A cell is
    background if its row matches its region's constant-coefficient row to 1e-6 of the row's largest entry; all other
    cells (interfaces between regions, irregular cells) plus a ring form the feature F (exact LU). Background:
    z_C = sum_k mask_k G_k(mask_k w). Same cancellation-free balanced form as ScalarFDLGF."""

    MIN_REGION = None   # H56: minimum region size in cells (None: max(512, m^3/1000) as in H54/H55)
    ADMIT = "open"  # which regions get a background inverse: all | big (>= 2 % of cells) | local (largest + r >= 6 kappa)
    #                 | open (largest + regions that are not insulated islands: median boundary-face coefficient >= 0.1 kappa_k)
    #                 | box  (all regions; each region's FDM on its own bounding box, Neumann at the box faces)

    def __init__(self, M, m, ring=2, tol=1e-6):
        from scipy import ndimage
        M = M.tocsr()
        self.M, self.m = M, m
        n = m**3
        d = M.diagonal()
        rowmax = abs(M).max(axis=1).toarray().ravel()
        f = faces(M, m)
        nf = np.sum(~np.isnan(f), axis=1)
        with np.errstate(invalid="ignore"):
            homog = np.nanmax(f, axis=1) <= (1 + tol) * np.nanmin(f, axis=1)
        lab, nlab = ndimage.label(homog.reshape(m, m, m))
        lab = lab.ravel()
        size = np.bincount(lab, minlength=nlab + 1)
        z, y, x = np.unravel_index(np.arange(n), (m, m, m))
        nfar = (x == m - 1).astype(int) + (y == m - 1) + (z == m - 1)
        react = 2 * d - np.asarray(abs(M).sum(axis=1)).ravel()
        match = np.zeros(n, bool)
        self.regions = []
        big = [k for k in np.flatnonzero(size >= (self.MIN_REGION or max(512, n // 1000))) if k != 0]
        largest = max(big, key=lambda k: size[k]) if big else -1
        for k in big:
            cells = lab == k
            kappa = float(np.nanmedian(f[cells]))
            r = float(np.median(react[cells & (nfar == 0)])) if np.any(cells & (nfar == 0)) else float(np.median(react[cells]))
            one = cells & (nfar == 1)
            beta = float(np.median(react[one]) - r) if one.any() else 0.0
            if abs(beta) <= 1e-6 * abs(r):
                beta = 0.0
            if self.ADMIT == "big" and k != largest and size[k] < 0.02 * n:
                continue
            if self.ADMIT == "local" and k != largest and r < 6 * kappa:
                continue
            if self.ADMIT == "open" and k != largest:
                inside = cells.reshape(m, m, m)
                ring1 = binary_dilation(inside) & ~inside             # cells just outside the region
                bnd = np.nanmin(f[ring1.ravel()], axis=1)              # weakest face of each cell around the region
                if np.median(bnd) < 0.1 * kappa:                        # insulated island (Neumann-like): leave to F
                    continue
            expect = nf * kappa + r + beta * nfar
            ok = cells & (np.nanmax(np.abs(f - kappa), axis=1) <= tol * rowmax) & (np.abs(d - expect) <= tol * rowmax)
            match |= ok
            if self.ADMIT == "box":
                G = FDMBox(cells.reshape(m, m, m), kappa, r, beta, m)
            elif self.ADMIT == "dbox" and k != largest:            # H55: Dirichlet on the interior box faces
                G = FDMBox(cells.reshape(m, m, m), kappa, r, beta, m, dirichlet=True)
            else:
                G = FDM(make_T(m, kappa, beta), kappa, r, m)
            self.regions.append(dict(kappa=kappa, r=r, beta=beta, mask=ok, G=G, largest=(k == largest)))
        cells = ~match.reshape(m, m, m)
        self.core_fraction = float(cells.mean())
        if ring:
            cells = binary_dilation(cells, iterations=ring)
        Fmask = cells.ravel()
        for reg in self.regions:
            reg["mask"] = reg["mask"] & ~Fmask
        self.F = np.flatnonzero(Fmask)
        self.C = np.flatnonzero(~Fmask)
        self.MCF = M[self.C][:, self.F].tocsr()
        self.MFC = M[self.F][:, self.C].tocsr()
        self._factor(M[self.F][:, self.F].tocsr())
        self.G = self._G

    def _G(self, w):
        z = np.zeros_like(w)
        for reg in self.regions:
            mk = reg["mask"]
            z[mk] = reg["G"](np.where(mk, w, 0.0))[mk]
        return z


class FDMBox:
    """Exact inverse of the constant-coefficient operator kappa (T_z (+) T_y (+) T_x) + r on the bounding box of a region
    (Neumann at the box faces, Robin beta where the box reaches the far face m-1), applied to vectors on the m^3 grid
    (zero outside the box). dirichlet=True (H55): Dirichlet (ghost value 0) at box faces inside the cube instead of
    Neumann, so the constant mode of an insulated island is not near-singular (r/kappa ~ 1e-14) in the fast inverse."""

    def __init__(self, region, kappa, r, beta, m, dirichlet=False):
        idx = np.nonzero(region)
        self.sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
        self.V, lam = [], []
        for s_ in self.sl:
            n_ = s_.stop - s_.start
            if n_ == 1:                                          # both end faces of the single cell (audit item 6)
                low = 1.0 if dirichlet and s_.start > 0 else 0.0
                high = beta / kappa if s_.stop == m else 1.0 if dirichlet else 0.0
                l_, V_ = np.array([low + high]), np.ones((1, 1))
            else:
                T = make_T(n_, kappa, beta if s_.stop == m else 0.0).toarray()
                if s_.stop != m:
                    T[-1, -1] = 2.0 if dirichlet else 1.0
                if dirichlet and s_.start > 0:
                    T[0, 0] = 2.0
                l_, V_ = np.linalg.eigh(T)
            lam.append(l_); self.V.append(V_)
        self.den = kappa * (lam[0][:, None, None] + lam[1][None, :, None] + lam[2][None, None, :]) + r
        self.m = m

    def __call__(self, f):
        X = f.reshape((self.m,) * 3)[self.sl]
        Vz, Vy, Vx = self.V
        g = np.einsum("zyx,za,yb,xc->abc", X, Vz, Vy, Vx, optimize=True) / self.den
        out = np.zeros((self.m,) * 3)
        out[self.sl] = np.einsum("abc,za,yb,xc->zyx", g, Vz, Vy, Vx, optimize=True)
        return out.ravel()


def dd_matvec(M, v):
    """M v from the stored entries of M, with every product (two_prod) and the row accumulation (two_sum) in
    double-double and one final rounding: exact for the stored matrix even where 1e17 entries cancel to a
    reaction-size result."""
    from ir_fp128 import two_prod, two_sum
    Mc = (M.tocsr() @ sp.diags((v != 0).astype(float))).tocsr()
    Mc.eliminate_zeros()
    out = np.zeros(Mc.shape[0])
    cnt = np.diff(Mc.indptr)
    nz = np.flatnonzero(cnt)                                     # only rows that touch the support of v
    if nz.size == 0:
        return out
    rows = np.repeat(np.arange(nz.size), cnt[nz])
    pos = np.arange(Mc.nnz) - np.repeat(Mc.indptr[nz], cnt[nz])
    X, Y = np.zeros((nz.size, int(cnt.max()))), np.zeros((nz.size, int(cnt.max())))
    X[rows, pos], Y[rows, pos] = Mc.data, v[Mc.indices]
    s, e = np.zeros(nz.size), np.zeros(nz.size)
    for j in range(X.shape[1]):
        p, pe = two_prod(X[:, j], Y[:, j])
        s, t = two_sum(s, p)
        e += t + pe
    out[nz] = s + e
    return out


class DeflatedFDLGF(MultiFDLGF):
    """H55: multi-region background with an island coarse space. Every region except the largest gets a Dirichlet-box
    fast inverse (no island test) AND its indicator v_k joins the exact part of the balanced form: W = [I_F, v_1 .. v_K],
    Q = W (W^T M W)^-1 W^T, P = Q + (I - QM) G (I - MQ). The constant mode of an insulated island, which a region-only
    fast inverse misjudges ~100x (H54), is then fixed by the Galerkin solve. W^T M W = [[M_FF, (Mv)_F], [(M^T v)_F^T, D]]
    is sparse (the border is nonzero only next to the island) and is factored by one MUMPS LU, so pivoting treats the
    near-singular island mode as the full-core LU of bt-fd did. M v_k, M^T v_k and D = v^T M v are formed once with
    exact (double-double) sums.
    Why the Dirichlet box: with the whole-cube (Neumann) FDM of H54 an island's constant mode is near-singular
    (r/kappa ~ 1e-14), z = G r is ~1e10 too large along v and (I - QM) removes it by cancellation (linearity defect
    7.6e-4 on S3). With Dirichlet at the box faces G underestimates that mode and the coarse solve adds it back
    (defect ~1e-16)."""

    LIN = False      # coarse space per region: indicator + x, y, z (False: indicator only)
    FORM = "bal"     # one: P = Q + (I - QM) G | bal: P = Q + (I - QM) G (I - MQ)
    ADMIT = "dbox"   # every region; non-largest ones with a Dirichlet-box fast inverse (FDMBox(dirichlet=True))

    def _factor(self, MFF):
        from math import fsum
        m = self.m
        V = []
        for reg in self.regions:
            mk = reg["mask"]
            if reg["largest"] or not mk.any():
                continue
            V.append(mk.astype(float))
            if self.LIN:                                        # + x, y, z on the region (centred, unit radius)
                for c in np.unravel_index(np.flatnonzero(mk), (m, m, m)):
                    c = c - c.mean()
                    v = np.zeros(m**3); v[mk] = c / max(np.abs(c).max(), 1.0)
                    V.append(v)
        self.K = K = len(V)
        F, C = self.F, self.C
        U = [dd_matvec(self.M, v) for v in V]                   # M v_l
        Mt = self.M.T.tocsr()
        Ut = [dd_matvec(Mt, v) for v in V]                      # M^T v_k
        sup = [np.flatnonzero(v) for v in V]
        D = np.array([[fsum(V[k][sup[k]] * U[l][sup[k]]) for l in range(K)] for k in range(K)]).reshape(K, K)
        col = lambda vs, idx: sp.csr_matrix(np.vstack([v[idx] for v in vs]) if K else (0, len(idx)))  # noqa: E731
        self.UC, self.UtC = col(U, C), col(Ut, C)                # (M v_l)_C, (M^T v_k)_C
        self.VC = col(V, C)
        aug = sp.bmat([[MFF, col(U, F).T], [col(Ut, F), sp.csr_matrix(D)]]).tocsr() if K else MFF
        self.D = D
        super()._factor(aug)

    def __call__(self, r):
        """FORM "bal": P = Q + (I - QM) G (I - MQ); "one": P = Q + (I - QM) G (error operator (I - QM)(I - GM), same
        spectrum). Both are linear to ~1e-16 with the Dirichlet-box G; "bal" takes ~1 fewer iteration (m = 64)."""
        F, C, nF = self.F, self.C, len(self.F)
        rC = r[C]
        y0 = self.lu(np.r_[r[F], self.VC @ rC])                             # Q r = W y0
        w = np.zeros_like(r)
        w[C] = rC if self.FORM == "one" else rC - self.MCF @ y0[:nF] - self.UC.T @ y0[nF:]   # "bal": (I - MQ) r
        zC = self.G(w)[C]                                                   # z = G w on C, z_F := 0
        y = self.lu(np.r_[self.MFC @ zC, self.UtC @ zC])                     # Q M z = W y
        out = np.empty_like(r)
        out[F] = y0[:nF] - y[:nF]
        out[C] = zC + self.VC.T @ (y0[nF:] - y[nF:])
        return out
