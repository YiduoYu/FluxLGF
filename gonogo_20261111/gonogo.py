"""Go/no-go for the FD-LGF paper: FD-LGF against the strongest available baselines at equal error, 3D chi^x series.

See protocol.md (written and locked before any full-run result).

usage (hypre_env, single thread):
  gonogo.py --prep N...                       reference u_h for case A (FD-LGF PCG to round-off, checked by BoomerAMG)
  gonogo.py --one CASE N CHI METHOD           one run in this process, prints one JSON line
  gonogo.py --run OUT REPEATS N... [--chi C...]  every (case, n, chi, method) REPEATS times, shuffled, one process each
"""
import ctypes as ct
import json
import os
import random
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "three_d"))

import numpy as np  # noqa: E402

from cases3d import CASES, make_case  # noqa: E402
from solvers3d import Method2_3D  # noqa: E402

ERR_TOL, RES_TOL, MAXITER, TIMEOUT = 1e-8, 1e-10, 500, 900
REF = HERE / "ref"
STRUCT = ROOT / "fluxmg" / "structured" / "libhypre_struct.dylib"

# BoomerAMG settings (PETSc option suffixes). amg_old is the setting of reports 01 / 29.
AMG = {
    "amg_old":    dict(strong_threshold=0.5),
    "amg_fal25":  dict(strong_threshold=0.25),
    "amg_hmis":   dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, strong_threshold=0.5),
    "amg_hmis25": dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, strong_threshold=0.25),
    "amg_pmis":   dict(coarsen_type="PMIS", interp_type="ext+i", P_max=4, strong_threshold=0.5),
    "amg_agg":    dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, strong_threshold=0.5, agg_nl=1),
}
# hypre struct settings: relax 1 weighted Jacobi, 3 symmetric red-black; rap 0 Galerkin, 1 non-Galerkin.
PFMG = {
    "pfmg_j":    dict(relax=1, rap=0, npre=2, npost=2, skip=1),   # best of the 24-setting tune (fluxmg/structured/tune.txt)
    "pfmg_rb":   dict(relax=3, rap=0, npre=1, npost=1, skip=1),   # best symmetric red-black of that tune
    "pfmg_rb22": dict(relax=3, rap=0, npre=2, npost=2, skip=0),   # fewest iterations on high-contrast cases in that tune
}
RINGS = {"fdlgf_w32": 1 / 32, "fdlgf_w64": 1 / 64}


def methods(case):
    fd = list(RINGS) if case in ("C_exp+incl", "D_exp+rand") else ["fdlgf"]
    return fd + list(AMG) + list(PFMG) + ["smg"]


def family(m):
    return "fdlgf" if m.startswith("fdlgf") else m.split("_")[0]


# ---------------------------------------------------------------- preconditioners (flat vectors in, flat out)
class Boomer:
    def __init__(self, A, opts, tag):
        from petsc4py import PETSc
        self.mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix(tag + "_")
        for k, v in opts.items():
            PETSc.Options()[f"{tag}_pc_hypre_boomeramg_{k}"] = v
        self.pc.setOperators(self.mat)
        self.pc.setType("hypre")
        self.pc.setHYPREType("boomeramg")
        self.pc.setFromOptions()
        self.pc.setUp()
        self.x, self.y = self.mat.createVecs()

    def __call__(self, r):
        self.x.setArray(r)
        self.pc.apply(self.x, self.y)
        return self.y.getArray().copy()


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        from petsc4py import PETSc  # noqa: F401  (MPI up first)
        PD = ct.POINTER(ct.c_double)
        _LIB = ct.CDLL(str(STRUCT))
        _LIB.struct_new.restype = ct.c_void_p
        _LIB.struct_new.argtypes = [ct.c_int] * 4 + [PD] + [ct.c_int] * 6 + [ct.c_double]
        _LIB.struct_apply.argtypes = [ct.c_void_p, PD, PD]
    return _LIB


def _p(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_double))


def stencil7(A, n):
    """(N, 7): centre, -x, +x, -y, +y, -z, +z; flat index (k n + j) n + i, x fastest = hypre's (x, y, z)."""
    N = n**3
    B = A.tocoo()
    pos = np.full(2 * N + 1, -1)
    pos[N + np.array([0, -1, 1, -n, n, -n * n, n * n])] = np.arange(7)
    e = pos[B.col - B.row + N]
    assert (e >= 0).all(), "not a 7-point matrix"
    st = np.zeros((N, 7))
    st[B.row, e] = B.data
    return st


class Struct:
    def __init__(self, A, n, smg=False, relax=3, rap=0, npre=1, npost=1, skip=0):
        self.st = np.ascontiguousarray(stencil7(A, n))
        self.h = lib().struct_new(int(smg), n, n, n, _p(self.st), relax, rap, npre, npost, skip, 1, 0.0)

    def __call__(self, r):
        z = np.empty(r.size)
        lib().struct_apply(self.h, _p(np.ascontiguousarray(r)), _p(z))
        return z


class FDLGF:
    def __init__(self, c, w):
        self.M = Method2_3D(c.A, c.bg_a1, c.mask, c.q, c.grid.h, w)
        self.shape = (c.n,) * 3

    def __call__(self, r):
        return self.M(r.reshape(self.shape)).ravel()


def build(c, m):
    if m in AMG:
        return Boomer(c.A, AMG[m], m)
    if m in PFMG:
        return Struct(c.A, c.n, **PFMG[m])
    if m == "smg":
        return Struct(c.A, c.n, smg=True, npre=1, npost=1)
    return FDLGF(c, RINGS.get(m, 1 / 32))


# ---------------------------------------------------------------- PCG with both stopping points recorded
def pcg_two_stops(A, P, f, u_ref, t_start):
    """Plain PCG, x0 = 0. Runs until rel L2 error <= ERR_TOL and true rel residual <= RES_TOL (or MAXITER).
    Returns the first iteration (and wall time since t_start) at which each criterion holds. The error norm is computed every
    iteration for every method (same overhead for all)."""
    fn, un = np.linalg.norm(f), np.linalg.norm(u_ref)
    u = np.zeros_like(f)
    r = f.copy()
    z = P(r)
    p = z.copy()
    rz = float(r @ z)
    out = dict(k_err=None, t_err=None, k_res=None, t_res=None, err_at_res=None, res_at_err=None, maxerr_at_err=None, breakdown=False)
    k = 0
    while k < MAXITER:
        Ap = A @ p
        pAp = float(p @ Ap)
        if not np.isfinite(pAp) or pAp <= 0:
            out["breakdown"] = True
            break
        alpha = rz / pAp
        u += alpha * p
        r -= alpha * Ap
        k += 1
        if out["k_err"] is None:
            e = np.linalg.norm(u - u_ref) / un
            if e <= ERR_TOL:
                out.update(k_err=k, t_err=perf_counter() - t_start, res_at_err=float(np.linalg.norm(f - A @ u) / fn),
                           maxerr_at_err=float(np.max(np.abs(u - u_ref)) / np.max(np.abs(u_ref))))
        if out["k_res"] is None and np.linalg.norm(r) / fn <= RES_TOL:
            rt = f - A @ u
            if np.linalg.norm(rt) / fn <= RES_TOL:
                out.update(k_res=k, t_res=perf_counter() - t_start, err_at_res=float(np.linalg.norm(u - u_ref) / un))
            r = rt   # restart from the true residual, as ops3d.pcg does
            z = P(r)
            p = z.copy()
            rz = float(r @ z)
            if out["k_err"] is not None and out["k_res"] is not None:
                break
            continue
        if out["k_err"] is not None and out["k_res"] is not None:
            break
        z = P(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    out["iters_run"] = k
    out["final_err"] = float(np.linalg.norm(u - u_ref) / un)
    return out


def reference(c):
    if c.u_h is not None:
        return c.u_h.ravel()
    return np.load(REF / f"A_exp_{c.n}_{c.chi:.0e}.npy")


def one(case, n, chi, m):
    c = make_case(case, n, chi)
    c.A = c.A.tocsr()
    f = c.f.ravel().copy()
    u_ref = reference(c)
    lib()   # petsc4py / MPI / hypre library load, untimed and done for every method
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load = os.getloadavg()[0]
    t0, c0 = perf_counter(), process_time()
    P = build(c, m)
    setup = perf_counter() - t0
    s = pcg_two_stops(c.A, P, f, u_ref, t0)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    extra = {}
    if isinstance(P, FDLGF):
        extra = dict(local_fraction=P.M.local_fraction, lu_nnz=P.M.lu_nnz)
    return dict(case=case, n=n, chi=chi, method=m, family=family(m), setup=setup, cpu_total=process_time() - c0,
                rss_before_gb=rss0 / 1e9, rss_peak_gb=rss1 / 1e9, load1=load, **s, **extra)


# ---------------------------------------------------------------- reference solutions for case A
def prep(sizes, chis):
    REF.mkdir(exist_ok=True)
    for n in sizes:
        for chi in chis:
            c = make_case("A_exp", n, chi)
            A = c.A.tocsr()
            f = c.f.ravel()
            P = FDLGF(c, 1 / 32)
            u = np.zeros_like(f)
            # FD-LGF PCG to round-off
            r = f.copy(); z = P(r); p = z.copy(); rz = r @ z
            for _ in range(60):
                Ap = A @ p; a = rz / (p @ Ap); u += a * p; r -= a * Ap
                if np.linalg.norm(r) / np.linalg.norm(f) < 1e-15:
                    break
                z = P(r); rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
            # independent check: BoomerAMG PCG to 1e-14
            B = Boomer(A, AMG["amg_hmis"], "chk")
            v = np.zeros_like(f); r = f.copy(); z = B(r); p = z.copy(); rz = r @ z
            for _ in range(200):
                Ap = A @ p; a = rz / (p @ Ap); v += a * p; r -= a * Ap
                if np.linalg.norm(r) / np.linalg.norm(f) < 1e-14:
                    break
                z = B(r); rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
            diff = np.linalg.norm(u - v) / np.linalg.norm(u)
            res = np.linalg.norm(f - A @ u) / np.linalg.norm(f)
            print(f"ref A n={n} chi={chi:.0e}: residual {res:.1e}, |u_fd - u_amg|/|u| = {diff:.1e}", flush=True)
            assert diff < 1e-10, "reference check failed"
            np.save(REF / f"A_exp_{n}_{chi:.0e}.npy", u)


def run(out, repeats, sizes, chis):
    jobs = [(case, n, chi, m) for n in sizes for chi in chis for case in CASES for m in methods(case)] * repeats
    random.Random(20261111).shuffle(jobs)
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for i, (case, n, chi, m) in enumerate(jobs):
        cmd = [sys.executable, "-u", __file__, "--one", case, str(n), repr(chi), m]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
            line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else json.dumps(
                dict(case=case, n=n, chi=chi, method=m, family=family(m), crash=p.returncode, err=p.stderr[-400:]))
        except subprocess.TimeoutExpired:
            line = json.dumps(dict(case=case, n=n, chi=chi, method=m, family=family(m), crash=f"timeout {TIMEOUT} s"))
        with open(out, "a") as fh:
            fh.write(line + "\n")
        print(i + 1, len(jobs), line[:160], flush=True)


def _floats(args):
    return [float(a) for a in args]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--prep":
        chi_at = a.index("--chi") if "--chi" in a else len(a)
        prep([int(x) for x in a[1:chi_at]], _floats(a[chi_at + 1:]) or [1e4, 1e8])
    elif a[0] == "--one":
        print(json.dumps(one(a[1], int(a[2]), float(a[3]), a[4])))
    elif a[0] == "--run":
        chi_at = a.index("--chi") if "--chi" in a else len(a)
        run(a[1], int(a[2]), [int(x) for x in a[3:chi_at]], _floats(a[chi_at + 1:]) or [1e4, 1e8])
