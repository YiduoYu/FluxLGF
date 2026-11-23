"""H50 runner: monolithic BoomerAMG baselines vs AMFR and its ablations on multi-field matrices.
GMRES(50), right preconditioning, unpreconditioned residual, rtol 1e-8, x0 = 0, max 500 its, single thread, sequential.
Cases:  g3t:m:F:G[:s1,s2,s3]        gen3t.py (own 3T discretization), optional symmetric field scaling diag(s) A diag(s)
        mg:m:ng:sbar:zeta:lam:seed   mgrd.py (multigroup radiation diffusion + material temperature)
        real                         SolverChallenge26_01 (H40 criterion: solver-reported residual)
Methods: amg-3d (BoomerAMG, H40 settings, field-blocked scalar matrix), amg-unk (same settings, interleaved order with
block size k -> hypre systems AMG, unknown approach), amfr, amfr-sep (no grouping, tau_g = inf), amfr-noelim (tau_d = -1),
amfr-all (all remaining fields in one group), amfr2 (H51: spectral-radius joining).
usage: python run_mf.py OUT.csv CASE [CASE ...]     (methods from env RUN_METHODS)
"""
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solver_challenge"))
import amfr  # noqa: E402
from amfr import AMFR, amg_defaults  # noqa: E402

METHODS = tuple(os.environ.get("RUN_METHODS", "amg-3d,amg-unk,amfr,amfr-sep,amfr-noelim").split(","))


def build(case):
    p = case.split(":")
    if p[0] == "g3t":
        from gen3t import generate
        A, b = generate(int(p[1]), float(p[2]), float(p[3]))
        if len(p) > 4:
            s = np.repeat(np.array(list(map(float, p[4].split(",")))), A.shape[0] // 3)
            A = (sp.diags(s) @ A @ sp.diags(s)).tocsr()
            b = s * b
        return A, b, 3
    if p[0] == "mg":
        from mgrd import generate
        ng = int(p[2])
        A, b = generate(int(p[1]), ng, float(p[3]), float(p[4]), float(p[5]), int(p[6]))
        return A, b, ng + 1
    if p[0] == "real":
        import load_real
        d = HERE.parent / "SolverChallenge26-tenproblems" / "extracted"
        A = sp.load_npz(d / "A.npz").tocsr()
        b = np.load(d / "b.npy")
        return A, b, 3
    raise ValueError(case)


def run(A, b, k, name, tol=1e-8, maxit=500):
    t0 = perf_counter()
    perm = None
    structure = ""
    if name == "amg-unk":
        N = A.shape[0] // k
        perm = np.arange(k * N).reshape(k, N).T.ravel()
        A = A[perm][:, perm].tocsr()
        A.sort_indices()
        b = b[perm]
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data), bsize=k if name == "amg-unk" else 1)
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if name in ("amg-3d", "amg-unk"):
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        prefix = "g_" + name.replace("-", "") + "_"
        ksp.setOptionsPrefix(prefix)
        amg_defaults(prefix)
    else:
        kw = {"amfr": {}, "amfr-sep": dict(tau_g=np.inf), "amfr-noelim": dict(tau_d=-1.0),
              "amfr-all": dict(tau_g=-1.0, phi=0.0), "amfr2": dict(join="spectral")}[name]
        P = AMFR(A, k, **kw)
        structure = P.structure()
        pc.setType("python"); pc.setPythonContext(P)
    ksp.setTolerances(rtol=tol, atol=0.0, max_it=maxit)
    ksp.setFromOptions()
    ksp.setUp()
    x, rhs = M.createVecs()
    rhs.setArray(b)
    t1 = perf_counter()
    ksp.solve(rhs, x)
    t2 = perf_counter()
    r = b - A @ x.getArray()
    d = A.diagonal()
    return dict(method=name, structure=structure, its=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                reported=ksp.getResidualNorm() / np.linalg.norm(b), true=np.linalg.norm(r) / np.linalg.norm(b),
                scaled=np.linalg.norm(r / d) / np.linalg.norm(b / d), setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0)


if __name__ == "__main__":
    out = sys.argv[1]
    with open(out, "w", newline="") as fh:
        wr = None
        for case in sys.argv[2:]:
            A, b, k = build(case)
            for name in METHODS:
                row = dict(case=case, k=k, n=A.shape[0], **run(A, b, k, name))
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row)); wr.writeheader()
                wr.writerow(row); fh.flush()
                print(f"{case:28s} {name:11s} {row['structure']:14s} its {row['its']:3d} reason {row['reason']:3d} "
                      f"true {row['true']:.1e} setup {row['setup_s']:.2f} total {row['total_s']:.2f} s", flush=True)
