"""H52 post-lock exploration: is the 1e-8 tail on the real matrix (53 its, locked run) caused by the CHOLMOD switch
made after the 16-it exploratory run? Same bt-fd with MUMPS LU vs CHOLMOD; linearity defect of each S3 solver."""
import sys
from pathlib import Path
import numpy as np
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsolve, run_real
from btfd import BTFD

orig = PETSc.PC.setFactorSolverType
A, b = run_real.load("real")
for factor in ("mumps", "cholmod"):
    def setup_hook(self, pkg, _f=factor):
        if _f == "mumps":
            self.setType("lu"); return orig(self, "mumps")
        return orig(self, pkg)
    PETSc.PC.setFactorSolverType = setup_hook
    P = BTFD(A, ring=2)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(P.N), rng.standard_normal(P.N)
    d = np.linalg.norm(P.p3(x + y) - P.p3(x) - P.p3(y)) / np.linalg.norm(P.p3(x) + P.p3(y))
    print(f"{factor}: S3 linearity defect {d:.1e}", flush=True)
    for tol in (1e-4, 1e-6, 1e-8):
        row = run_real.run(A, b, "bt-fd:2", tol)
        print(f"{factor} tol {tol:g}: its {row['its']} total {row['total_s']:.1f} s (setup {row['setup_s']:.1f})", flush=True)
PETSc.PC.setFactorSolverType = orig
