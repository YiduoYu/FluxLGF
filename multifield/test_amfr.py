"""Unit checks for amfr.py: (1) on gen3t base AMFR builds bt-v's structure and reproduces BTElim's output;
(2) with tau_d = -1 on G = 1 it builds bt-j's structure and reproduces BTJoint; (3) exact elimination algebra:
with a diagonal field and exact group solves, AMFR is an exact inverse when no coupling is dropped."""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, "../solver_challenge")
from gen3t import generate
from bt_elim import BTElim, BTJoint
import amfr

rng = np.random.default_rng(1)
A, b = generate(16, 2e-11, 2e-4)
P = amfr.AMFR(A, 3)
r = rng.standard_normal(A.shape[0])
print("base structure", P.structure(), "rel diff vs BTElim", np.linalg.norm(P(r) - BTElim(A)(r)) / np.linalg.norm(BTElim(A)(r)))
A, b = generate(16, 2e-11, 1.0)
P = amfr.AMFR(A, 3)
print("G=1 structure", P.structure(), "rel diff vs BTJoint", np.linalg.norm(P(r) - BTJoint(A)(r)) / np.linalg.norm(BTJoint(A)(r)))
# exactness: replace V-cycles by LU
amfr.vcycle = lambda M: (lambda f: (lambda rr: f(rr)))(spla.factorized(M.tocsc()))
for F, G in ((2e-11, 2e-4), (1.0, 2e-4), (1.0, 1.0)):
    A, b = generate(10, F, G)
    # zero electron diffusion so elimination is exact
    N = A.shape[0] // 3
    A = A.tolil(); 
    A22 = A[N:2*N, N:2*N].tocsr(); d = A22.diagonal()
    A[N:2*N, N:2*N] = sp.diags(d)
    A = A.tocsr()
    P = amfr.AMFR(A, 3, tau_g=0.0, phi=0.0)          # everything joined -> exact
    print(F, G, P.structure(), "exact residual", np.linalg.norm(A @ P(r[:A.shape[0]]) - r[:A.shape[0]]) / np.linalg.norm(r[:A.shape[0]]))
    z = spla.spsolve(A.tocsc(), r[:A.shape[0]])
    print("   direct LU residual", np.linalg.norm(A @ z - r[:A.shape[0]]) / np.linalg.norm(r[:A.shape[0]]),
          " AMFR vs LU solution diff", np.linalg.norm(P(r[:A.shape[0]]) - z) / np.linalg.norm(z))
# well-scaled random test: 3 fields, field 1 diagonal
n = 200
L = sp.diags([-1, 2.5, -1], [-1, 0, 1], shape=(n, n))
A = sp.bmat([[L, sp.diags(-0.3 * np.ones(n)), None], [sp.diags(-0.4 * np.ones(n)), sp.diags(3 * np.ones(n)), sp.diags(-0.5 * np.ones(n))],
             [None, sp.diags(-0.6 * np.ones(n)), 2 * L]]).tocsr()
P = amfr.AMFR(A, 3, tau_g=0.0, phi=0.0)
rr = rng.standard_normal(3 * n)
print("well-scaled exactness", np.linalg.norm(A @ P(rr) - rr) / np.linalg.norm(rr))
