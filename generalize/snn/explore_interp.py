"""Exploratory (dev seed 555): BoomerAMG with interpolation vectors (near-null space) = coarse-grid slow modes,
interpolated.  usage: python explore_interp.py family n"""
import itertools
import sys
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from cases_param import make, sample
from cgd import interp_2d
from snn_defl import AMG_OPTS, amg, dpcg, lanczos_A

_c = itertools.count()


def amg_vec(A, V, variant, qmax=0, smooth=False, extra=None):
    prefix = f"v{next(_c)}_"
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    vecs = []
    for i in range(V.shape[1]):
        v = M.createVecLeft(); v.setArray(V[:, i] / np.linalg.norm(V[:, i])); vecs.append(v)
    M.setNearNullSpace(PETSc.NullSpace().create(constant=False, vectors=vecs))
    pc = PETSc.PC().create(); pc.setOptionsPrefix(prefix); pc.setOperators(M)
    pc.setType("hypre"); pc.setHYPREType("boomeramg")
    o = PETSc.Options()
    for k, v in dict(AMG_OPTS, vec_interp_variant=variant, vec_interp_qmax=qmax, vec_interp_smooth=smooth, **(extra or {})).items():
        o[prefix + "pc_hypre_boomeramg_" + k] = v
    pc.setFromOptions()
    t0 = perf_counter(); pc.setUp(); ts = perf_counter() - t0
    xv, bv = M.createVecs()

    def apply(r):
        bv.setArray(np.ascontiguousarray(r)); pc.apply(bv, xv)
        return xv.getArray().copy()
    apply.keep = (pc, M, xv, bv, vecs)
    return apply, ts


if __name__ == "__main__":
    fam, n = sys.argv[1], int(sys.argv[2])
    for t, (p, chi) in enumerate(sample(np.random.default_rng(555), 2)):
        A, b, _ = make(fam, n, chi, p)
        t0 = perf_counter(); P, ts = amg(A); _, its = dpcg(A, P, b); tb = perf_counter() - t0
        nc = (n + 1) // 4 - 1; Ac = make(fam, nc, chi, p)[0]; Pc, _ = amg(Ac)
        U = interp_2d(nc, n) @ lanczos_A(Ac, Pc, 40)[1][:, :8]
        out = [f"{fam} n={n} #{t} base {its} ({ts:.2f}+{tb-ts:.2f}s)"]
        for k, var in itertools.product((1, 2, 4), (1, 2, 3)):
            try:
                Pv, tsv = amg_vec(A, U[:, :k], var)
                t0 = perf_counter(); _, itv = dpcg(A, Pv, b, maxit=500); tv = perf_counter() - t0
                out.append(f"k{k}v{var}: {itv}it ({tsv:.2f}+{tv:.2f}s)")
            except Exception as e:  # noqa: BLE001
                out.append(f"k{k}v{var}: fail {str(e)[:40]}")
        print(" | ".join(out), flush=True)
