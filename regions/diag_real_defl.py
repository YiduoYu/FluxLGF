"""H55 diagnosis: which block makes bt-fdd stall at 1e-8 on the real matrix? Swap one block back to ScalarFDLGF."""
import sys
from pathlib import Path
import numpy as np
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real, rsolve
from btfd import BTFD
A, b = run_real.load("real")
M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
base = BTFD(A, multi="defl"); ref = BTFD(A)
for nm, (p3, p1) in {"S3 defl, A11 scalar": (base.p3, ref.p1), "S3 scalar, A11 defl": (ref.p3, base.p1)}.items():
    P = BTFD.__new__(BTFD); P.__dict__.update(base.__dict__); P.p3, P.p1 = p3, p1
    ksp = PETSc.KSP().create(); ksp.setOperators(M); ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC(); pc.setType("python"); pc.setPythonContext(P)
    ksp.setTolerances(rtol=1e-8, atol=0, max_it=200); ksp.setUp()
    x, rhs = M.createVecs(); rhs.setArray(b); ksp.solve(rhs, x)
    print(f"{nm}: {ksp.getIterationNumber()} its", flush=True)
for nm, p in (("S3", base.p3), ("A11", base.p1)):
    print(nm, [(f"{r['mask'].sum()} cells", f"k={r['kappa']:.3e}", f"r={r['r']:.3e}", r["largest"]) for r in p.regions], "D", np.diag(p.D))
