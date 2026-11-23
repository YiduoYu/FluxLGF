"""Shared solve driver: same accuracy test for every solver.

Refinement to the double-double original residual ||b - Ax|| / ||b|| < 1e-8, inner rtol 1e-6, x0 = 0, right preconditioning,
restart 50, single thread. Time is CPU seconds of setup + solve. The accuracy check itself (the double-double residual) is excluded and reported as dd: it is a
Python row loop that dominates on graphs with huge rows (6 s per call on the scale-free graph).
"""
import resource
import sys
from pathlib import Path
from time import process_time

import numpy as np
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
from ir_fp128 import DDResidual, two_sum  # noqa: E402

MAX_ITS = 200
MAX_STEPS = 6

# BoomerAMG configurations. "fast" is the project's standing 3D setting; the rest give it a fair chance on each class.
_B = dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=1)
AMG = {
    "fast.25": dict(_B, strong_threshold=0.25),
    "fast.5": dict(_B, strong_threshold=0.5),
    "fast.7": dict(_B, strong_threshold=0.7),
    "fast.9": dict(_B, strong_threshold=0.9),
    "noagg.25": dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=0, strong_threshold=0.25),
    "noagg.5": dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=0, strong_threshold=0.5),
    "falgout.25": dict(coarsen_type="Falgout", interp_type="classical", strong_threshold=0.25),
    "falgout.5": dict(coarsen_type="Falgout", interp_type="classical", strong_threshold=0.5),
}
NODAL = {
    "nodal.5": dict(_B, strong_threshold=0.5, nodal_coarsen=4, vec_interp_variant=2),
    "nodal.25": dict(_B, strong_threshold=0.25, nodal_coarsen=4, vec_interp_variant=2),
}
_count = [0]


def run(A, b, pc_kind, amg_cfg=None, P=None, bs=1, max_its=MAX_ITS, max_steps=MAX_STEPS):
    """pc_kind 'amg' or 'py'. Returns ok, its, rel, cpu, setup_cpu; P's setup is timed by the caller."""
    res = DDResidual(A, b)
    nb = np.linalg.norm(b)
    c0 = process_time()
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    if bs > 1:
        M.setBlockSize(bs)
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setTolerances(rtol=1e-6, atol=0.0, max_it=max_its)
    pc = ksp.getPC()
    if pc_kind == "amg":
        ksp.setType("gmres"); ksp.setGMRESRestart(50)
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        _count[0] += 1
        pre = f"a{_count[0]}_"
        ksp.setOptionsPrefix(pre)
        o = PETSc.Options()
        for k, v in amg_cfg.items():
            o[pre + "pc_hypre_boomeramg_" + k] = v
        ksp.setFromOptions()
    else:
        ksp.setType("fgmres"); ksp.setGMRESRestart(50)
        pc.setType("python"); pc.setPythonContext(P)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setUp()
    setup = process_time() - c0
    dv, rv = M.createVecs()
    xh, xl, r = np.zeros_like(b), np.zeros_like(b), b.copy()
    its, rel = [], np.inf
    dd = 0.0
    for _ in range(max_steps):
        rv.setArray(r); ksp.solve(rv, dv)
        its.append(ksp.getIterationNumber())
        d = dv.getArray()
        if not np.all(np.isfinite(d)):
            rel = np.inf; break
        xh, e = two_sum(xh, d); xl += e
        xh, xl = two_sum(xh, xl)
        c1 = process_time()
        r = res(xh, xl); rel = np.linalg.norm(r) / nb
        dd += process_time() - c1
        if rel < 1e-8 or not np.isfinite(rel):
            break
    return dict(ok=bool(rel < 1e-8), its=sum(its), rel=float(rel), cpu=process_time() - c0 - dd, dd=dd, setup=setup, rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
