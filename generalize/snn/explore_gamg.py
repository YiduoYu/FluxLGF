"""Exploratory (dev seed 555): PETSc GAMG (smoothed aggregation) with near-null space = [1, coarse slow modes] and
PyAMG SA (evolution strength) as PCG preconditioners, vs BoomerAMG.  usage: python explore_gamg.py family n"""
import itertools
import sys
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from cases_param import make, sample
from cgd import interp_2d
from snn_defl import amg, dpcg, lanczos_A

_c = itertools.count()


def gamg(A, V=None, thr=0.0, extra=None):
    prefix = f"g{next(_c)}_"
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    vecs = []
    if V is not None:
        Q, _ = np.linalg.qr(np.column_stack([np.ones(A.shape[0]), V]))
        for i in range(Q.shape[1]):
            v = M.createVecLeft(); v.setArray(np.ascontiguousarray(Q[:, i])); vecs.append(v)
        M.setNearNullSpace(PETSc.NullSpace().create(constant=False, vectors=vecs))
    pc = PETSc.PC().create(); pc.setOptionsPrefix(prefix); pc.setOperators(M); pc.setType("gamg")
    o = PETSc.Options()
    for k, v in dict(pc_gamg_type="agg", pc_gamg_threshold=thr, **(extra or {})).items():
        o[prefix + k] = v
    pc.setFromOptions()
    t0 = perf_counter(); pc.setUp(); ts = perf_counter() - t0
    xv, bv = M.createVecs()

    def apply(r):
        bv.setArray(np.ascontiguousarray(r)); pc.apply(bv, xv)
        return xv.getArray().copy()
    apply.keep = (pc, M, xv, bv, vecs)
    return apply, ts


def pyamg_sa(A, V=None, strength="evolution"):
    import pyamg
    B = np.ones((A.shape[0], 1)) if V is None else np.linalg.qr(np.column_stack([np.ones(A.shape[0]), V]))[0]
    t0 = perf_counter(); ml = pyamg.smoothed_aggregation_solver(A, B=B, strength=strength, max_coarse=500); ts = perf_counter() - t0
    Mp = ml.aspreconditioner(cycle="V")
    return (lambda r: Mp @ r), ts


def timed(A, b, mk):
    P, ts = mk(); t0 = perf_counter(); _, its = dpcg(A, P, b, maxit=1000); return its, ts, perf_counter() - t0


if __name__ == "__main__":
    fam, n = sys.argv[1], int(sys.argv[2])
    for t, (p, chi) in enumerate(sample(np.random.default_rng(555), 2)):
        A, b, _ = make(fam, n, chi, p)
        nc = (n + 1) // 4 - 1; Ac = make(fam, nc, chi, p)[0]; Pc, _ = amg(Ac)
        t0 = perf_counter(); U = interp_2d(nc, n) @ lanczos_A(Ac, Pc, 40)[1][:, :8]; tU = perf_counter() - t0
        res = {"boomer": timed(A, b, lambda: amg(A))}
        for thr in (0.0, 0.05):
            for k in (0, 2, 4):
                res[f"gamg t{thr} k{k}"] = timed(A, b, lambda: gamg(A, U[:, :k] if k else None, thr))
        for k in (0, 2):
            res[f"pyamg-evo k{k}"] = timed(A, b, lambda: pyamg_sa(A, U[:, :k] if k else None))
        print(f"{fam} n={n} #{t} (modes {tU:.2f}s) " + " | ".join(f"{k}: {v[0]}it {v[1]:.2f}+{v[2]:.2f}s" for k, v in res.items()), flush=True)
