"""H52 runs on the real SC26_01 matrix (H40 criterion: GMRES(50) residual estimate, rtol 1e-4 and 1e-8, x0 = 0,
right preconditioning, single thread). Methods: amg-3d, bt-v (H41), bt-fd[:ring] (H52).
The solver-reported residual is a diagnostic only: equal rtol does NOT mean equal accuracy between methods (audit
2026-11-05, item 1). Accuracy comparisons use run_ir.py (double-double original residual).
usage: python run_real.py OUT.csv METHOD [METHOD ...]"""
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solver_challenge"))
from bt_elim import BTElim, amg_defaults  # noqa: E402
from btfd import BTFD  # noqa: E402
from provenance import save  # noqa: E402

X = HERE.parent / "SolverChallenge26-tenproblems" / "extracted"
FD_KINDS = ("bt-fd", "bt-fdm", "bt-fdd")


def load(case):
    if case == "real":
        return sp.load_npz(X / "A.npz").tocsr(), np.load(X / "b.npy")
    if case.startswith("geo:"):                        # geo:m:r_core:r_shell:cap:F:G
        from gen_geo import generate as gg
        _, m, r1, r2, cap, F, G = case.split(":")
        return gg(int(m), float(r1), float(r2), float(cap), float(F), float(G))
    if case.startswith("gen:"):                        # gen:m:n:p:aspect:width:delta:seed (H56 generalization)
        from gen_gen import generate as gn
        _, m, n, p, asp, w, dl, sd = case.split(":")
        return gn(int(m), int(n), float(p), float(asp), float(w), float(dl), int(sd))
    from gen3t import generate                        # g3t:m:F:G
    _, m, F, G = case.split(":")
    return generate(int(m), float(F), float(G))


def run(A, b, name, tol, stem=None):
    if name not in ("amg-3d", "bt-v") and name.split(":")[0] not in FD_KINDS:
        raise ValueError(f"unknown method {name!r}")
    t0 = perf_counter()
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    ksp.setType("gmres"); ksp.setGMRESRestart(50)
    ksp.setPCSide(PETSc.PC.Side.RIGHT); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if name == "amg-3d":
        pc.setType("hypre"); pc.setHYPREType("boomeramg")
        prefix = f"r_amg{tol:g}_".replace("-", "m").replace(".", "p")
        ksp.setOptionsPrefix(prefix)
        amg_defaults(prefix)
    elif name == "bt-v":
        pc.setType("python"); pc.setPythonContext(BTElim(A))
    else:                                              # bt-fd[:ring], bt-fdm[:ring] (H54: multi-region background), bt-fdd[:ring] (H55: + island coarse space)
        ring = int(name.split(":")[1]) if ":" in name else 2
        pc.setType("python"); pc.setPythonContext(BTFD(A, ring=ring, multi="defl" if name.startswith("bt-fdd") else name.startswith("bt-fdm")))
    ksp.setTolerances(rtol=tol, atol=0.0, max_it=500)
    ksp.setFromOptions()
    ksp.setUp()
    x, rhs = M.createVecs()
    rhs.setArray(b)
    t1 = perf_counter()
    ksp.solve(rhs, x)
    t2 = perf_counter()
    r = b - A @ x.getArray()
    d = A.diagonal()
    if stem:
        save(stem, ksp, A, b)
    return dict(method=name, tol=tol, its=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                reported=ksp.getResidualNorm() / np.linalg.norm(b), true=np.linalg.norm(r) / np.linalg.norm(b),
                scaled=np.linalg.norm(r / d) / np.linalg.norm(b / d), setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0)


if __name__ == "__main__":
    out, case = sys.argv[1], sys.argv[2]
    A, b = load(case)
    with open(out, "w", newline="") as fh:
        wr = None
        for name in sys.argv[3:]:
            for tol in (1e-4, 1e-8):
                row = dict(case=case, **run(A, b, name, tol, f"{Path(out).with_suffix('')}_{name}_{tol:g}"))
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row)); wr.writeheader()
                wr.writerow(row); fh.flush()
                print(f"{case} {name:8s} tol {tol:g}: its {row['its']:3d} reason {row['reason']:2d} scaled {row['scaled']:.1e} "
                      f"setup {row['setup_s']:.1f} solve {row['solve_s']:.1f} total {row['total_s']:.1f} s", flush=True)
