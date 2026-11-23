"""Quick look: FluxMG vs BoomerAMG on one matrix (PETSc Krylov for both, right preconditioning, x0 = 0, rtol 1e-8, max 500).
Timings here are numpy prototype timings, useful only for iteration counts and rough magnitudes.
usage: try_one.py CASE [CASE ...]     CASE: suite name | g3t:m:F:G | gen:m:n:p:aspect:width:delta:seed | mg:m:ng:sbar:zeta:lam:seed | poisson:m | aniso:m:a"""
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
import flux_mg  # noqa: E402
from bt_elim import amg_defaults  # noqa: E402


def lap7(faces, r, shape):
    """Assemble a 7-point matrix from face arrays (kz, ky, kx) and row excess r."""
    nz, ny, nx = shape
    idx = np.arange(nz * ny * nx).reshape(shape)
    I, J, V = [], [], []
    d = r.ravel().copy()
    for a, k in enumerate(faces):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[a], hi[a] = slice(None, -1), slice(1, None)
        p, q, v = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel(), k.ravel()
        I += [p, q]; J += [q, p]; V += [-v, -v]
        np.add.at(d, p, v); np.add.at(d, q, v)
    I.append(idx.ravel()); J.append(idx.ravel()); V.append(d)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(d.size, d.size))


def load(case):
    """Returns A, b, dims, fields."""
    p = case.split(":")
    if p[0] == "real":
        X = ROOT / "SolverChallenge26-tenproblems" / "extracted"
        return sp.load_npz(X / "A.npz").tocsr(), np.load(X / "b.npy"), (128, 128, 128), 3
    if p[0] == "g3t":
        from gen3t import generate
        A, b = generate(int(p[1]), float(p[2]), float(p[3]))
        m = int(p[1]); return A.tocsr(), np.asarray(b), (m, m, m), 3
    if p[0] == "gen":
        from gen_gen import generate
        A, b = generate(int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), int(p[7]))
        m = int(p[1]); return A.tocsr(), np.asarray(b), (m, m, m), 3
    if p[0] == "mg":
        from mgrd import generate
        A, b = generate(int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), int(p[6]))
        m = int(p[1]); return A.tocsr(), np.asarray(b), (m, m, m), int(p[2]) + 1
    if p[0] in ("poisson", "aniso"):
        m = int(p[1]); a = float(p[2]) if p[0] == "aniso" else 1.0
        shape = (m, m, m)
        faces = [a * np.ones((m - 1, m, m)), np.ones((m, m - 1, m)), np.ones((m, m, m - 1))]
        r = np.zeros(shape); r[0], r[-1] = 1.0, 1.0
        A = lap7(faces, r, shape)
        b = np.random.default_rng(0).standard_normal(m**3)
        return A, b, shape, 1
    from suite import load as sl
    A, b = sl(case)
    n = round(A.shape[0] ** (1 / 3))
    return A.tocsr(), np.asarray(b), (n, n, n), 1


def solve(A, b, kind, P=None, theta=None):
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setTolerances(rtol=1e-8, atol=0.0, max_it=500)
    pc = ksp.getPC()
    t0 = perf_counter()
    if kind == "amg":
        ksp.setType("gmres"); ksp.setGMRESRestart(50)
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        pre = f"t{theta:g}_".replace(".", "p")
        ksp.setOptionsPrefix(pre); amg_defaults(pre)
        PETSc.Options()[pre + "pc_hypre_boomeramg_strong_threshold"] = theta
        ksp.setFromOptions()
    else:
        ksp.setType("fgmres"); ksp.setGMRESRestart(50)
        pc.setType("python"); pc.setPythonContext(P)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setUp()
    t1 = perf_counter()
    x, rhs = M.createVecs(); rhs.setArray(b)
    ksp.solve(rhs, x)
    t2 = perf_counter()
    r = b - A @ x.getArray()
    d = A.diagonal()
    return ksp.getIterationNumber(), int(ksp.getConvergedReason()), t1 - t0, t2 - t1, np.linalg.norm(r / d) / np.linalg.norm(b / d)


class FluxOp:
    """PETSc python Mat: A in flux form on level 0 of a FluxMG hierarchy."""

    def __init__(self, P):
        self.P = P

    def mult(self, mat, x, y):
        y.setArray(self.P.op(x.getArray(readonly=True)))


def solve_ir(A, b, kind, P=None, theta=None, flux_op=False, max_it=500, max_steps=12):
    """Refinement in double-double to the original residual 1e-8; inner rtol 1e-6, x0 = 0. Returns dict."""
    from ir_fp128 import DDResidual, two_sum
    res = DDResidual(A, b)
    nb = np.linalg.norm(b)
    t0 = perf_counter()
    c0 = process_time()
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    op = M
    if flux_op:
        op = PETSc.Mat().createPython(A.shape, FluxOp(P)); op.setUp()
    ksp = PETSc.KSP().create()
    ksp.setOperators(op, M)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setTolerances(rtol=1e-6, atol=0.0, max_it=max_it)
    pc = ksp.getPC()
    if kind == "amg":
        ksp.setType("gmres"); ksp.setGMRESRestart(50)
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        pre = f"ir{theta:g}_".replace(".", "p")
        ksp.setOptionsPrefix(pre); amg_defaults(pre)
        PETSc.Options()[pre + "pc_hypre_boomeramg_strong_threshold"] = theta
        ksp.setFromOptions()
    else:
        ksp.setType("fgmres"); ksp.setGMRESRestart(50)
        pc.setType("python"); pc.setPythonContext(P)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setUp()
    setup = perf_counter() - t0
    dv, rv = M.createVecs()
    xh, xl, r = np.zeros_like(b), np.zeros_like(b), b.copy()
    its, reasons, rel = [], [], np.inf
    for _ in range(max_steps):
        rv.setArray(r); ksp.solve(rv, dv)
        its.append(ksp.getIterationNumber()); reasons.append(int(ksp.getConvergedReason()))
        xh, e = two_sum(xh, dv.getArray()); xl += e
        xh, xl = two_sum(xh, xl)
        r = res(xh, xl); rel = np.linalg.norm(r) / nb
        if rel < 1e-8 or not np.isfinite(rel):
            break
    return dict(ok=bool(rel < 1e-8), its=sum(its), each=its, reasons=reasons, rel=rel, setup=setup, total=perf_counter() - t0, cpu=process_time() - c0)


if __name__ == "__main__":
    for case in sys.argv[1:]:
        A, b, dims, F = load(case)
        q = solve_ir(A, b, "amg", theta=0.25)
        print(f"{case:30s} n={A.shape[0]:8d} F={F:2d} amg.25 ok {q['ok']} its {q['its']:3d} {q['each']} rel {q['rel']:.1e} setup {q['setup']:.2f} total {q['total']:.2f}", flush=True)
        t0 = perf_counter()
        P = flux_mg.FluxMG(A, dims, F)
        tb = perf_counter() - t0
        q = solve_ir(A, b, "flux", P=P, flux_op=True)
        print(f"{'':30s} {'':14s} flux K={P.K} ok {q['ok']} its {q['its']:3d} {q['each']} reasons {sorted(set(q['reasons']))} rel {q['rel']:.1e} build {tb:.2f} total {q['total']:.2f}", flush=True)
