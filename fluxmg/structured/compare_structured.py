"""FluxMG against hypre's structured multigrid: is its lead over BoomerAMG only "structured against unstructured"?

Written before any result was seen.

Methods, each one V-cycle per preconditioner apply:
  flux     FluxMG (the code in ../flux_mg.py)
  pfmg     hypre PFMG, the best setting of --tune (tune.txt); fields go through FluxMG's own Reduction, grouping probe included
  pfmg_rb  hypre PFMG, the best symmetric red-black setting of --tune, far fewer iterations on the high-contrast cases; same reduction
  smg      hypre SMG, same reduction
  amg      BoomerAMG theta 0.25, the yardstick of round 1
A group of several fields has no structured solver here, so pfmg, pfmg_rb and smg use one BoomerAMG V-cycle (theta 0.25) on it. Those runs
are hybrids and carry joint = true in the results.

Same criterion for all: double-double original residual 1e-8 by refinement, inner rtol 1e-6, x0 = 0, at most 200 inner iterations
per step and 6 steps, single thread, CPU seconds including setup. Every run is its own process; runs are shuffled.

Fairness runs in favour of hypre: pfmg, pfmg_rb and smg get the same FGMRES, the cancellation-free flux operator and the reduction of FluxMG.
Only PFMG settings are tuned, on the suite only, never on held-out cases.
Not tested: BoxMG and AGMG (not available here), SysPFMG (see the change below).

Predictions:
  P1  pfmg is faster than flux on poisson:64.
  P2  flux is faster than pfmg on each of lognorm3, tubes, insul, perc3.
  P3  smg needs no more iterations than pfmg on the aligned anisotropy cases, and takes longer than flux on most cases.
  P4  void, see below. It said: syspfmg on the 3T systems is slower than pfmg with reduction.
  P5  flux is faster than the best of pfmg, pfmg_rb and smg on at least 70 % of suite and held-out cases.
Decision: if P5 fails, FluxMG's lead over BoomerAMG is mostly the structured grid and must be reported that way.

Change after writing this, before any comparison result: SysPFMG was planned as a method and for joint groups. With any cross-field
stencil entry it fails or crashes, also in a stand-alone C program (syspfmg_check.c), so it was dropped and P4 and P5 were edited
accordingly. BoomerAMG on joint groups replaces it, which makes the opponent stronger on those cases.
Second change, after tuning and before any comparison result: pfmg_rb added, because one setting cannot serve both the 3T and the
high-contrast cases, and P1-P3 read "pfmg" as the better of the two. A pair (case, method) that failed once is not repeated.

usage:
  compare_structured.py --tune                 PFMG settings on tuning cases, one process, prints geometric means
  compare_structured.py --one CASE METHOD      one run, prints JSON
  compare_structured.py --run OUT [REPEATS]    all cases and methods, shuffled, appended to OUT as JSON lines; real always 3 times
"""
import ctypes as ct
import json
import random
import resource
import subprocess
import sys
from pathlib import Path
from time import process_time

import numpy as np

HERE = Path(__file__).resolve().parent
FLUX = HERE.parent
ROOT = FLUX.parent
sys.path[:0] = [str(FLUX), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
import flux_mg  # noqa: E402
from eval_fluxmg import CASES, HOLDOUT, MAX_ITS, MAX_STEPS  # noqa: E402
from eval_grouping import CASES as GROUPING, get  # noqa: E402
from try_one import solve_ir  # noqa: E402

PY = sys.executable
ENV = Path(PY).parent.parent
PD = ct.POINTER(ct.c_double)
# From --tune (tune.txt). relax: 0 Jacobi, 1 weighted Jacobi, 2 red-black, 3 symmetric red-black. rap: 0 Galerkin, 1 non-Galerkin.
PFMG = dict(relax=1, rap=0, npre=2, npost=2, skip=1)
PFMG_RB = dict(relax=3, rap=0, npre=1, npost=1, skip=1)
SMG = dict(npre=1, npost=1)
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        so, src = HERE / "libhypre_struct.dylib", HERE / "hypre_struct.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["clang", "-O3", "-shared", "-fPIC", str(src), f"-I{ENV}/include", f"-L{ENV}/lib", "-lHYPRE", "-lmpi",
                            f"-Wl,-rpath,{ENV}/lib", "-o", str(so)], check=True)
        from petsc4py import PETSc  # noqa: F401  (MPI up first)
        _LIB = ct.CDLL(str(so))
        _LIB.struct_new.restype = ct.c_void_p
        _LIB.struct_new.argtypes = [ct.c_int] * 4 + [PD] + [ct.c_int] * 6 + [ct.c_double]
        _LIB.struct_apply.argtypes = [ct.c_void_p, PD, PD]
    return _LIB


def _p(a):
    return a.ctypes.data_as(PD)


def stencil7(B, dims):
    """The 7 diagonals of a 7-point matrix as (N, 7): centre, -x, +x, -y, +y, -z, +z."""
    nz, ny, nx = dims
    N = nz * ny * nx
    B = B.tocoo()
    pos = np.full(2 * N + 1, -1)
    pos[N + np.array([0, -1, 1, -nx, nx, -nx * ny, nx * ny])] = np.arange(7)
    e = pos[B.col - B.row + N]
    assert (e >= 0).all(), "not a 7-point matrix"
    st = np.zeros((N, 7))
    st[B.row, e] = B.data
    return st


class Mono:
    """One V-cycle of PFMG or SMG on a one-field matrix."""

    def __init__(self, M, dims, smg=False, relax=2, rap=0, npre=1, npost=1, skip=0, maxit=1, tol=0.0):
        self.st = np.ascontiguousarray(stencil7(M, dims))
        self.h = lib().struct_new(int(smg), *dims, _p(self.st), relax, rap, npre, npost, skip, maxit, tol)

    def __call__(self, r):
        z = np.empty(r.size)
        lib().struct_apply(self.h, _p(np.ascontiguousarray(r)), _p(z))
        return z


def boomer(M, name):
    """One BoomerAMG V-cycle at the settings of the yardstick, theta 0.25, for a group of several fields."""
    import amfr
    from petsc4py import PETSc
    P = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOptionsPrefix(name)
    ksp.setOperators(P)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    amfr.amg_defaults(name)
    PETSc.Options()[name + "pc_hypre_boomeramg_strong_threshold"] = 0.25
    ksp.setType("preonly")
    ksp.setFromOptions()
    ksp.setUp()
    b, x = P.createVecs()

    def solve(r):
        b.setArray(r)
        ksp.solve(b, x)
        return x.getArray().copy()
    solve.keep = (ksp, P, b, x)
    return solve


class Hypre:
    """Preconditioner object with the interface solve_ir expects of FluxMG."""

    def __init__(self, A, dims, F, method, pfmg=None):
        pfmg = pfmg or (PFMG_RB if method == "pfmg_rb" else PFMG)
        N = int(np.prod(dims))
        self.op = flux_mg.FluxOp(*flux_mg.parse_matrix(A, dims, F))
        self.joint = 0

        def solver(M):
            M = M.tocsr()
            if M.nnz == M.shape[0] and np.count_nonzero(M.diagonal()) == M.shape[0]:
                d = M.diagonal().copy()
                return lambda r: r / d
            if M.shape[0] > N:
                self.joint += 1
                return boomer(M, f"joint{self.joint}_")
            return Mono(M, dims, smg=True, **SMG) if method == "smg" else Mono(M, dims, **pfmg)
        self.P = solver(A) if F == 1 else flux_mg.Reduction(A, dims, F, solver)

    def __call__(self, r):
        return self.P(r)

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))


def one(case, method, **kw):
    A, b, dims, F = get(case)
    t0 = process_time()
    if method == "amg":
        q = solve_ir(A, b, "amg", theta=0.25, max_it=MAX_ITS, max_steps=MAX_STEPS)
        build, t = q["setup"], q["cpu"]
    else:
        P = flux_mg.FluxMG(A, dims, F) if method == "flux" else Hypre(A, dims, F, method, **kw)
        build = process_time() - t0
        q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=MAX_ITS, max_steps=MAX_STEPS)
        t = build + q["cpu"]
    return dict(case=case, method=method, ok=bool(q["ok"]), its=q["its"], each=q["each"], rel=float(q["rel"]), time=t, build=build,
                joint=method in ("pfmg", "pfmg_rb", "smg") and P.joint > 0,
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)


def tune():
    """Geometric mean of the time, failures counted as 8 x the slowest finished, on suite cases only."""
    mono = ["poisson:64", "aniso:64:100", "lognorm3", "tubes", "insul", "perc3"]
    multi = ["g3t:64:2e-11:2e-4", "gen:64:10:2:1:0:2e-2:1", "gen:64:1:2:1:4:2e-2:1", "mg:32:4:1:1:0:7", "mg:32:8:1e2:1e-2:1:7"]
    settings = [dict(relax=r, rap=a, npre=n, npost=n, skip=s) for r in (1, 2, 3) for a in (0, 1) for n in (1, 2) for s in (0, 1)]
    show = []
    for kw in settings:
        ts = []
        for c in mono + multi:
            q = one(c, "pfmg", pfmg=kw)
            ts.append((q["time"] if q["ok"] else 1e3, q["its"]))
        show.append((float(np.exp(np.mean(np.log([t for t, _ in ts])))), kw, [i for _, i in ts]))
        print(f"pfmg {show[-1][0]:7.3f} {kw} its {show[-1][2]}", flush=True)
    print("best pfmg", min(show, key=lambda s: s[0])[:2])


def run(out, repeats):
    cases = [c for c, _ in CASES] + [c for c, _ in HOLDOUT] + list(GROUPING)
    methods = ("flux", "pfmg", "pfmg_rb", "smg", "amg")
    jobs = [(c, m) for c in cases for m in methods] * repeats + [("real", m) for m in methods] * 3
    random.Random(2026).shuffle(jobs)
    failed = set()
    for n, (c, m) in enumerate(jobs):
        if (c, m) in failed:
            continue
        try:
            p = subprocess.run([PY, "-u", __file__, "--one", c, m], capture_output=True, text=True, timeout=900)
            line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else json.dumps(dict(case=c, method=m, ok=False, crash=p.returncode, err=p.stderr[-300:]))
        except subprocess.TimeoutExpired:
            line = json.dumps(dict(case=c, method=m, ok=False, crash="timeout 900 s"))
        if not json.loads(line)["ok"]:
            failed.add((c, m))
        with open(out, "a") as f:
            f.write(line + "\n")
        print(n + 1, len(jobs), line[:120], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--tune":
        tune()
    elif sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], sys.argv[3])))
    else:
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
