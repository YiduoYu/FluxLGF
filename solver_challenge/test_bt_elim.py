"""Unit check for bt_elim.BTElim: on a 3T system that has exactly the assumed structure (A21 = 0, A22 diagonal,
A23 = A32 diagonal), the bt-ex preconditioner is the exact inverse; with A21 != 0 it is not."""
import numpy as np
import scipy.sparse as sp

from bt_elim import BTElim

rng = np.random.default_rng(0)
m = 10; N = m**3
T = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
I = sp.identity(m)
L = (sp.kron(sp.kron(I, I), T) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(T, I), I)).tocsr()
k1, k3 = rng.uniform(1, 1e3, N), rng.uniform(1e-3, 1e6, N)
A11 = (sp.diags(np.sqrt(k1)) @ L @ sp.diags(np.sqrt(k1)) + sp.diags(rng.uniform(1, 2, N))).tocsr()
A33 = (sp.diags(np.sqrt(k3)) @ L @ sp.diags(np.sqrt(k3))).tocsr()
c = rng.uniform(1e-2, 1e6, N)                                   # electron-ion exchange
A33 = A33 + sp.diags(c + rng.uniform(1e-3, 1, N))
d22 = c + rng.uniform(1e-3, 1, N)
A12 = sp.diags(-rng.uniform(1, 1e3, N))
for a21_scale, label in ((0.0, "exact structure"), (1e-1, "A21 != 0")):
    A21 = sp.diags(-a21_scale * rng.uniform(0, 1, N))
    A = sp.bmat([[A11, A12, None], [A21, sp.diags(d22), sp.diags(-c)], [None, sp.diags(-c), A33]]).tocsr()
    P = BTElim(A, exact=True)
    z = rng.standard_normal(3 * N)
    err = np.linalg.norm(P(A @ z) - z) / np.linalg.norm(z)
    print(f"{label}: ||P(Az) - z|| / ||z|| = {err:.2e}")

# H43 variants: two sweeps shrink the A21 error to about its square; electron_amg is exact when A22 is not diagonal
A21 = sp.diags(-1e-1 * rng.uniform(0, 1, N))
A = sp.bmat([[A11, A12, None], [A21, sp.diags(d22), sp.diags(-c)], [None, sp.diags(-c), A33]]).tocsr()
z = rng.standard_normal(3 * N)
for sweeps in (1, 2):
    P = BTElim(A, exact=True, sweeps=sweeps)
    print(f"A21 != 0, sweeps={sweeps}: {np.linalg.norm(P(A @ z) - z) / np.linalg.norm(z):.2e}")
A22 = (sp.diags(d22) + 1e2 * L).tocsr()                          # strong electron diffusion
A = sp.bmat([[A11, A12, None], [None, A22, sp.diags(-c)], [None, sp.diags(-c), A33]]).tocsr()
for amg in (False, True):
    P = BTElim(A, exact=True, electron_amg=amg, sweeps=2 if amg else 1)
    print(f"A22 non-diagonal, electron_amg={amg}: {np.linalg.norm(P(A @ z) - z) / np.linalg.norm(z):.2e}")

# H44 joint electron-ion variants: with A21 = 0 the only error is the V-cycles; check the permutation round trip and that
# one application reduces the error on a G = 1 type system (electron diffusion comparable to its reaction)
from bt_elim import BTJoint
A22 = (sp.diags(d22) + sp.diags(d22 / 6) @ L).tocsr()
A = sp.bmat([[A11, A12, None], [None, A22, sp.diags(-c)], [None, sp.diags(-c), A33]]).tocsr()
for nodal in (False, True):
    P = BTJoint(A, nodal=nodal)
    e = rng.standard_normal(3 * N)
    norms = [np.linalg.norm(e)]
    for k in range(5):
        e = e - P(A @ e); norms.append(np.linalg.norm(e))
    print(f"BTJoint nodal={nodal}: error after 5 stationary steps {norms[-1] / norms[0]:.2e}")
