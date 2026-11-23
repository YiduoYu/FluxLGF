"""H55 diagnosis: GMRES(50) residual history of bt-fd / bt-fdd variants on one case (default real)."""
import sys
from pathlib import Path
import numpy as np
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real, rsolve
from btfd import BTFD
case = sys.argv[1]
A, b = run_real.load(case)
M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
for v in sys.argv[2].split(","):
    if v == "fd":
        P = BTFD(A)
    else:                                   # defl[-one]
        rsolve.DeflatedFDLGF.FORM = "one" if v.endswith("one") else "bal"
        P = BTFD(A, multi="defl")
    ksp = PETSc.KSP().create(); ksp.setOperators(M); ksp.setType("gmres"); ksp.setGMRESRestart(200)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC(); pc.setType("python"); pc.setPythonContext(P)
    ksp.setTolerances(rtol=1e-9, atol=0, max_it=80); ksp.setConvergenceHistory(); ksp.setUp()
    x, rhs = M.createVecs(); rhs.setArray(b); ksp.solve(rhs, x)
    h = ksp.getConvergenceHistory() / np.linalg.norm(b)
    print(f"{case} {v:9s} (restart 200): " + " ".join(f"{t:.0e}" for t in h), flush=True)
