"""Why does bt-fd stall between 1e-4 and 1e-8 on SC26_01? (exploratory)
(1) linearity defect ||P(x+y) - P(x) - P(y)|| / ||P(x) + P(y)|| of the block solvers (FD-LGF vs V-cycle);
(2) outer GMRES residual history; (3) which field the stalled residual lives in."""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from bt_elim import BTElim
from btfd import BTFD
X = Path(__file__).resolve().parent.parent / "SolverChallenge26-tenproblems" / "extracted"
A = sp.load_npz(X / "A.npz").tocsr(); b = np.load(X / "b.npy")
N = A.shape[0] // 3
rng = np.random.default_rng(0)
P = BTFD(A, ring=2); V = BTElim(A)
for nm, f in (("S3 fdlgf", P.p3), ("A11 fdlgf", P.p1), ("S3 vcycle", lambda r: V._solve(0, r)), ("A11 vcycle", lambda r: V._solve(1, r))):
    x, y = rng.standard_normal(N), rng.standard_normal(N)
    px, py, pxy = f(x), f(y), f(x + y)
    print(f"{nm:11s} linearity defect {np.linalg.norm(pxy - px - py) / np.linalg.norm(px + py):.1e}", flush=True)
x, y = rng.standard_normal(3 * N), rng.standard_normal(3 * N)
print(f"bt-fd full linearity defect {np.linalg.norm(P(x + y) - P(x) - P(y)) / np.linalg.norm(P(x) + P(y)):.1e}")
M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
ksp = PETSc.KSP().create(); ksp.setOperators(M); ksp.setType("gmres"); ksp.setGMRESRestart(50)
ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
pc = ksp.getPC(); pc.setType("python"); pc.setPythonContext(P)
hist = []
ksp.setMonitor(lambda k, it, rn: hist.append(rn))
ksp.setTolerances(rtol=1e-8, max_it=40); ksp.setUp()
xv, rhs = M.createVecs(); rhs.setArray(b); ksp.solve(rhs, xv)
nb = np.linalg.norm(b)
print("history:", " ".join(f"{h / nb:.1e}" for h in hist))
xs = xv.getArray(); r = b - A @ xs; d = A.diagonal()
for k, nm in enumerate(("T1", "T2", "T3")):
    s = slice(k * N, (k + 1) * N)
    print(f"{nm}: ||r||/||b|| {np.linalg.norm(r[s]) / nb:.1e}, D-scaled share {np.linalg.norm(r[s] / d[s]):.1e}, ||b_k|| {np.linalg.norm(b[s]):.1e}")
