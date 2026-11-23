"""bt-fd (H52): electron elimination block-triangular preconditioner (bt-v structure, H41) with the two scalar blocks
solved by one application of the scalar FD-LGF (rsolve.ScalarFDLGF) instead of one BoomerAMG V-cycle. No AMG anywhere:
setup = two sparse LU factorisations of the feature cells (~5 % of each block) + 128 x 128 eigendecompositions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from bt_elim import schur_ion  # noqa: E402
from rsolve import DeflatedFDLGF, MultiFDLGF, ScalarFDLGF  # noqa: E402


class BTFD:
    def __init__(self, A, ring=2, tol=1e-6, multi=False):
        N = A.shape[0] // 3
        m = round(N ** (1 / 3))
        self.N = N
        s = [slice(k * N, (k + 1) * N) for k in range(3)]
        self.A12 = A[s[0], s[1]].tocsr()
        self.d22 = A[s[1], s[1]].diagonal()
        self.a23 = A[s[1], s[2]].diagonal()
        self.a32 = A[s[2], s[1]].diagonal()
        S = DeflatedFDLGF if multi == "defl" else MultiFDLGF if multi else ScalarFDLGF
        self.p3 = S(schur_ion(A[s[2], s[2]].tocsr(), self.a23, self.a32, self.d22), m, ring, tol)
        self.p1 = S(A[s[0], s[0]].tocsr(), m, ring, tol)

    def __call__(self, r):
        N = self.N
        r1, r2, r3 = r[:N], r[N:2 * N], r[2 * N:]
        x3 = self.p3(r3 - self.a32 * (r2 / self.d22))
        x2 = (r2 - self.a23 * x3) / self.d22
        x1 = self.p1(r1 - self.A12 @ x2)
        return np.concatenate([x1, x2, x3])

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))
