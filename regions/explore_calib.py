"""H56 exploration: bt-fd / bt-fdd with the frequency-calibrated background (calib.py) vs without."""
import sys
from pathlib import Path
import numpy as np
from petsc4py import PETSc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
from calib import calibrate
from time import perf_counter


def gmres(A, b, P, tol):
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create(); ksp.setOperators(M); ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC(); pc.setType("python"); pc.setPythonContext(P); ksp.setTolerances(rtol=tol, atol=0, max_it=300)
    ksp.setUp(); x, r = M.createVecs(); r.setArray(b); t = perf_counter(); ksp.solve(r, x)
    return ksp.getIterationNumber(), perf_counter() - t


def largest_G(p):
    if hasattr(p, "regions"):
        return next(r["G"] for r in p.regions if r["largest"])
    return p.G


for case in sys.argv[2:]:
    A, b = run_real.load(case)
    for mode in sys.argv[1].split(","):
        base, cal = mode.split("+") if "+" in mode else (mode, None)
        t = perf_counter(); P = BTFD(A, multi="defl" if base == "fdd" else False)
        info = ""
        if cal:
            for nm, p in (("S3", P.p3), ("A11", P.p1)):
                rho, _ = calibrate(p, p.M, largest_G(p), rounds=int(cal))
                info += f" {nm} rho " + " ".join(f"{r:.2f}" for r in rho)
        ts = perf_counter() - t
        res = [gmres(A, b, P, tol) for tol in (1e-4, 1e-8)]
        print(f"{case} {mode:7s} setup {ts:.1f} | " + " | ".join(f"{tol:g}: {i} its {s:.1f} s" for tol, (i, s) in zip((1e-4, 1e-8), res)) + " |" + info, flush=True)
