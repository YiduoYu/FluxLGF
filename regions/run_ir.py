"""H52 under the OFFICIAL criterion ||b - Ax||_2 / ||b||_2 (real SC26_01): mixed-precision iterative refinement,
x and r = b - A x in double-double (solver_challenge/ir_fp128.py), each correction A d = r by GMRES(50) to inner_rtol
with amg-3d, bt-v, bt-j, bt-fd, bt-fd-chol (CHOLMOD feature solves) or bt-fdd (H55); suffix -t01 on the bt-fd kinds
uses matching tolerance 0.1 instead of 1e-6. Records the time (setup included, dd
residuals included) at which the dd residual first drops below 1e-4 and 1e-8; prep_s is the common cost before that
clock starts (DDResidual tables). Also writes, per method, STEM_METHOD.npz (final xh, xl) and provenance files.
AMG parameters can be overridden with PETSC_OPTIONS="-ir_amg_pc_hypre_boomeramg_strong_threshold 0.7".
usage: python run_ir.py OUT.csv inner_rtol METHOD [METHOD ...]"""
import csv
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solver_challenge"))
from bt_elim import BTElim, BTJoint, amg_defaults  # noqa: E402
from btfd import BTFD  # noqa: E402
from ir_fp128 import DDResidual, two_sum  # noqa: E402
from provenance import feature_stats, save  # noqa: E402
from rsolve import ScalarFDLGF  # noqa: E402

X = HERE.parent / "SolverChallenge26-tenproblems" / "extracted"
KINDS = ("amg-3d", "bt-v", "bt-j", "bt-fd", "bt-fd-t01", "bt-fd-chol", "bt-fd-chol-t01", "bt-fdd")


def inner(A, rtol, kind):
    if kind not in KINDS:
        raise ValueError(f"unknown method {kind!r}, expected one of {KINDS}")
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if kind == "amg-3d":
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        ksp.setOptionsPrefix("ir_amg_")
        amg_defaults("ir_amg_")
    else:
        pc.setType("python")
        if kind == "bt-v":
            pc.setPythonContext(BTElim(A))
        elif kind == "bt-j":
            pc.setPythonContext(BTJoint(A))
        else:
            ScalarFDLGF.CHOLESKY = kind.startswith("bt-fd-chol")
            pc.setPythonContext(BTFD(A, ring=2, tol=0.1 if kind.endswith("-t01") else 1e-6, multi="defl" if kind == "bt-fdd" else False))
    ksp.setTolerances(rtol=rtol, atol=0.0, max_it=500)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp, M.createVecs()


def solve_ir(A, b, res, kind, rtol, stem, prep_s=0.0):
    """One method: returns the CSV rows (one per target) and writes STEM_KIND.npz + provenance."""
    nb = np.linalg.norm(b)
    d = A.diagonal()
    t0 = perf_counter()
    ksp, (dv, rv) = inner(A, rtol, kind)
    t_setup = perf_counter() - t0
    xh = np.zeros_like(b); xl = np.zeros_like(b); r = b.copy()
    hit, its, reasons = {}, [], []
    t_solve = t_res = 0.0
    for step in range(1, 11):
        t1 = perf_counter()
        rv.setArray(r); ksp.solve(rv, dv)
        t2 = perf_counter()
        its.append(ksp.getIterationNumber()); reasons.append(int(ksp.getConvergedReason()))
        xh, t = two_sum(xh, dv.getArray()); xl += t
        xh, xl = two_sum(xh, xl)
        r = res(xh, xl)
        rel = np.linalg.norm(r) / nb
        t_solve += t2 - t1; t_res += perf_counter() - t2
        for target in (1e-4, 1e-8):
            if rel < target and target not in hit:
                hit[target] = (perf_counter() - t0, step, sum(its), t_solve, t_res)
        print(f"{kind} step {step}: inner its {its[-1]} reason {reasons[-1]} dd residual {rel:.2e} t {perf_counter() - t0:.1f} s", flush=True)
        if rel < 1e-8:
            break
    t_all = perf_counter() - t0
    scaled = np.linalg.norm(r / d) / np.linalg.norm(b / d)
    rows = []
    for target in (1e-4, 1e-8):
        tt, st, it, ts, tr = hit.get(target, (float("nan"), -1, -1, float("nan"), float("nan")))
        rows.append(dict(method=kind, inner_rtol=rtol, target=target, total_s=tt, steps=st, inner_its=it, setup_s=t_setup,
                         solve_s=ts, resid_s=tr, prep_s=prep_s, final_rel=rel, final_scaled=scaled,
                         inner_reasons=" ".join(map(str, reasons)), full_run_s=t_all,
                         maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9))
        print(f"{kind}: ||b-Ax||/||b|| < {target:g} after {st} steps / {it} inner its, {tt:.1f} s (setup {t_setup:.1f}, prep {prep_s:.1f})", flush=True)
    np.savez(f"{stem}_{kind}.npz", xh=xh, xl=xl)
    P = ksp.getPC().getPythonContext() if ksp.getPC().getType() == "python" else None
    save(f"{stem}_{kind}", ksp, A, b, features=feature_stats(P) if isinstance(P, BTFD) else None)
    return rows


if __name__ == "__main__":
    out, rtol = sys.argv[1], float(sys.argv[2])
    A = sp.load_npz(X / "A.npz").tocsr(); b = np.load(X / "b.npy")
    t0 = perf_counter()
    res = DDResidual(A, b)
    prep_s = perf_counter() - t0
    rows = []
    for kind in sys.argv[3:]:
        rows += solve_ir(A, b, res, kind, rtol, str(Path(out).with_suffix("")), prep_s)
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
