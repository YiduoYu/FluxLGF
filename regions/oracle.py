"""H52 oracle (exploratory): scalar FD-LGF per block on SC26_01 after electron elimination.
Per block (S3, A11): feature size, LU setup time, PCG iterations to 1e-10 with P = ScalarFDLGF (and with plain FDM
background for reference). usage: python oracle.py [ring] [tol]"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from bt_elim import schur_ion  # noqa: E402
from rsolve import ScalarFDLGF  # noqa: E402

X = Path(__file__).resolve().parent.parent / "SolverChallenge26-tenproblems" / "extracted"
ring = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
A = sp.load_npz(X / "A.npz").tocsr()
b = np.load(X / "b.npy")
N = A.shape[0] // 3
m = round(N ** (1 / 3))
s = [slice(k * N, (k + 1) * N) for k in range(3)]
d22 = A[s[1], s[1]].diagonal()
a23 = A[s[1], s[2]].diagonal(); a32 = A[s[2], s[1]].diagonal()
blocks = {"S3": schur_ion(A[s[2], s[2]].tocsr(), a23, a32, d22), "A11": A[s[0], s[0]].tocsr()}
rhs = {"S3": b[s[2]] - a32 * b[s[1]] / d22, "A11": b[s[0]]}
for name, M in blocks.items():
    print(f"== {name}: symmetric {abs(M - M.T).max() / abs(M).max():.1e}", flush=True)
    t = perf_counter()
    P = ScalarFDLGF(M, m, ring=ring, tol=tol)
    t_setup = perf_counter() - t
    print(f"   background kappa {P.kappa:.3e} r {P.r:.3e} beta {P.beta:.3e}; feature {P.core_fraction:.2%} -> "
          f"{len(P.F):,} cells with ring {ring}; setup {t_setup:.1f} s", flush=True)
    t = perf_counter(); P(rhs[name]); print(f"   one apply {perf_counter() - t:.2f} s", flush=True)
    for label, prec in (("FD-LGF", P), ("background only", P.G)):
        its = []
        D = np.sqrt(abs(M.diagonal()))
        t = perf_counter()
        x, info = spla.cg(M, rhs[name], rtol=1e-10, maxiter=300, M=spla.LinearOperator(M.shape, prec, dtype=float),
                          callback=lambda xk: its.append(1))
        rel = np.linalg.norm(rhs[name] - M @ x) / np.linalg.norm(rhs[name])
        relD = np.linalg.norm((rhs[name] - M @ x) / D) / np.linalg.norm(rhs[name] / D)
        print(f"   PCG + {label:15s}: {len(its)} its, info {info}, true rel res {rel:.1e}, D-scaled {relD:.1e}, "
              f"{perf_counter() - t:.1f} s", flush=True)
