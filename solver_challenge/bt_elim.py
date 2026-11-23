"""Electron-elimination block-triangular preconditioner for the real 3T matrix (H41, protocol_bt.md).

Blocks [T1 radiation; T2 electron; T3 ion]. Dropping A21 (T1 -> T2 feedback, ~1e-12) and the T2 off-diagonals (~2e-8):
    S3 x3 = r3 - A32 D22^-1 r2,  S3 = A33 - A32 D22^-1 A23,  x2 = D22^-1 (r2 - A23 x3),  A11 x1 = r1 - A12 x2.
A23 = A32 are diagonal, so S3 keeps the 7-point pattern of A33; its diagonal is formed as
(a33 - |a23|) + |a23| (d22 - |a23|) / d22, which avoids the cancellation a33 ~ |a23| in the shell.
S3^-1 and A11^-1: one BoomerAMG V-cycle (exact=False) or CG + BoomerAMG to rtol 1e-10 (exact=True).
fdm=True (H42, bt-g): A11^-1 by V + G + V, G = FDM inverse of the fitted constant-coefficient radiation background
(fdlgf_real), which removes the smooth outside-the-ball modes the V-cycle leaves.
H43 generalisations: electron_amg=True (bt-v2) solves the electron block by one BoomerAMG V-cycle on A22 instead of the
diagonal (for strong electron conduction); sweeps=2 (bt-v2x2) applies the preconditioner twice as a stationary
iteration, z = P r; z += P (r - A z), which brings in the dropped radiation feedback A21 through the residual.
"""
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

AMG3D = (("coarsen_type", "HMIS"), ("interp_type", "ext+i"), ("P_max", 4), ("strong_threshold", 0.5), ("agg_nl", 1))


def amg_defaults(prefix):
    """Write AMG3D as defaults under this options prefix: a value already in the database (caller, PETSC_OPTIONS) wins."""
    opts = PETSc.Options()
    for k, v in AMG3D:
        key = prefix + "pc_hypre_boomeramg_" + k
        if not opts.hasName(key):
            opts[key] = v


def schur_ion(A33, a23, a32, d22):
    """S3 = A33 - diag(a32 a23 / d22) with the cancellation-free diagonal (needs a23 = a32 <= 0)."""
    assert np.array_equal(a23, a32) and (a23 <= 0).all()
    c = -a23
    a33 = A33.diagonal()
    S3 = A33.copy().tocsr()
    S3.setdiag((a33 - c) + c * (d22 - c) / d22)
    return S3


def scalar_solver(M, prefix, exact):
    P = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(P)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    amg_defaults(prefix)
    if exact:
        ksp.setType("cg")
        ksp.setTolerances(rtol=1e-10, atol=0.0, max_it=200)
    else:
        ksp.setType("preonly")
    ksp.setFromOptions()
    ksp.setUp()
    b, x = P.createVecs()
    return ksp, b, x


class BTElim:
    def __init__(self, A, exact=False, fdm=False, electron_amg=False, sweeps=1):
        N = A.shape[0] // 3
        self.N = N
        s = [slice(k * N, (k + 1) * N) for k in range(3)]
        self.A12 = A[s[0], s[1]].tocsr()
        self.d22 = A[s[1], s[1]].diagonal()
        self.a23 = A[s[1], s[2]].diagonal()
        self.a32 = A[s[2], s[1]].diagonal()
        for blk in (A[s[1], s[2]], A[s[2], s[1]]):
            assert blk.nnz == np.count_nonzero(blk.diagonal())          # T2-T3 coupling is diagonal
        S3 = schur_ion(A[s[2], s[2]].tocsr(), self.a23, self.a32, self.d22)
        self.s3 = scalar_solver(S3, "bt_s3_", exact)
        self.A11 = A[s[0], s[0]].tocsr()
        self.a11 = scalar_solver(self.A11, "bt_a11_", exact)
        self.inner_its = [0, 0]
        self.a22 = scalar_solver(A[s[1], s[1]].tocsr(), "bt_a22_", exact) if electron_amg else None
        self.A = A if sweeps > 1 else None
        self.sweeps = sweeps
        self.G = None
        if fdm:
            from fdlgf_real import FDM, background_matrix, fit_background
            m = round(N ** (1 / 3))
            kappa, r, beta = fit_background(A, m)[0][0]
            self.G = FDM(background_matrix(m, kappa, r, beta)[1], kappa, r, m)

    def _solve(self, which, rhs):
        ksp, b, x = self.s3 if which == 0 else self.a11
        b.setArray(rhs)
        ksp.solve(b, x)
        self.inner_its[which] += max(ksp.getIterationNumber(), 1)
        return x.getArray().copy()

    def _a11(self, rhs):
        if self.G is None:
            return self._solve(1, rhs)
        z = self._solve(1, rhs)                                      # V + G + V
        z = z + self.G(rhs - self.A11 @ z)
        return z + self._solve(1, rhs - self.A11 @ z)

    def _once(self, r):
        N = self.N
        r1, r2, r3 = r[:N], r[N:2 * N], r[2 * N:]
        x3 = self._solve(0, r3 - self.a32 * (r2 / self.d22))
        if self.a22 is None:
            x2 = (r2 - self.a23 * x3) / self.d22
        else:
            ksp, b, x = self.a22
            b.setArray(r2 - self.a23 * x3)
            ksp.solve(b, x)
            x2 = x.getArray().copy()
        x1 = self._a11(r1 - self.A12 @ x2)
        return np.concatenate([x1, x2, x3])

    def __call__(self, r):
        z = self._once(r)
        for _ in range(self.sweeps - 1):
            z = z + self._once(r - self.A @ z)
        return z

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))


class BTJoint:
    """H44: electron and ion solved jointly. Drops only A21 (radiation feedback, harmless when the loop gain is < 1, H43):
        [A22 A23; A32 A33] [x2; x3] = [r2; r3]    one BoomerAMG V-cycle on the 2-field block
        A11 x1 = r1 - A12 x2                        one BoomerAMG V-cycle
    nodal=False: the 2-field block as one scalar matrix (field-blocked ordering, amg-3d settings).
    nodal=True : interleaved ordering, block size 2 (num_functions = 2) and nodal coarsening (hypre nodal = 1)."""

    def __init__(self, A, nodal=False):
        N = A.shape[0] // 3
        self.N = N
        self.A12 = A[:N, N:2 * N].tocsr()
        B = A[N:, N:].tocsr()
        self.perm = None
        if nodal:
            self.perm = np.arange(2 * N).reshape(2, N).T.ravel()       # interleaved: (Te_c, Ti_c) per cell
            B = B[self.perm][:, self.perm].tocsr()
            B.sort_indices()
        P = PETSc.Mat().createAIJ(size=B.shape, csr=(B.indptr, B.indices, B.data), bsize=2 if nodal else 1)
        prefix = "bt_jn_" if nodal else "bt_j_"
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(P)
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("boomeramg")
        amg_defaults(prefix)
        if nodal:
            PETSc.Options()[prefix + "pc_hypre_boomeramg_nodal_coarsen"] = 1
        ksp.setType("preonly")
        ksp.setFromOptions()
        ksp.setUp()
        self.b23, self.x23 = P.createVecs()
        self.ksp23 = ksp
        self.a11 = scalar_solver(A[:N, :N].tocsr(), "bt_j11_", False)

    def __call__(self, r):
        N = self.N
        r23 = r[N:]
        self.b23.setArray(r23 if self.perm is None else r23[self.perm])
        self.ksp23.solve(self.b23, self.x23)
        y = self.x23.getArray()
        x23 = np.empty(2 * N)
        if self.perm is None:
            x23[:] = y
        else:
            x23[self.perm] = y
        ksp, b, x = self.a11
        b.setArray(r[:N] - self.A12 @ x23[:N])
        ksp.solve(b, x)
        return np.concatenate([x.getArray(), x23])

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))
