"""Solvers for the 3T matrix (surrogate or the real SolverChallenge26_01 matrix), PETSc/hypre, single process.

  /opt/anaconda3/envs/hypre_env/bin/python run_sc.py OUT m [solver ...]       (surrogate, m^3 cells per temperature)
  /opt/anaconda3/envs/hypre_env/bin/python run_sc.py OUT MATRIX[,RHS] [solver ...]   (real matrix, see load_real.py)
A nonsymmetric matrix switches every CG to right-preconditioned GMRES(50) (still the true residual norm).
Stopping: ||b - A x||_2 / ||b||_2 < 1e-4 and < 1e-8 (the challenge's two constraints; unpreconditioned norm, x0 = 0).
Time = setup + solve (matrix assembly / loading excluded), peak RSS reported.
SC_CRITERION=reported (H40, real matrix): stop on the KSP's own residual estimate instead (||b - Ax|| has a float64
floor of ~1e-2 ||b|| on the real matrix); the true and diag-scaled residuals are reported alongside.

Solvers
  amg        CG + BoomerAMG on the whole matrix (PETSc defaults)
  amg-nodal  CG + BoomerAMG systems AMG (interlaced ordering, 3 functions, nodal coarsening)
  split-mult GMRES + fieldsplit by temperature, multiplicative (block Gauss-Seidel), BoomerAMG per block
  split-sym  CG + fieldsplit symmetric multiplicative (block SGS, SPD), BoomerAMG per block  [PCTL-like]
  amg-3d     CG + BoomerAMG with the usual 3D settings (HMIS, ext+i, P_max 4, θ = 0.5, 1 level aggressive)
  fdlgf3t    CG + FD-LGF-3T (fdlgf3t.py): point-block Green-Jacobi LGF + block-Jacobi smoothing (ω = 0.6) in the
             balanced form with box-constant coarse spaces, multilevel (b = 4, W-cycle, LU below 5000); -lu: one
             coarse level solved by sparse LU
  fdlgf3t-fast  H37 tuned (box 2, ω = 0.5, V-cycle); bnn3t-fast: the same with the LGF part replaced by the
             exact 3x3 block-Jacobi inverse (no DST)
  agg3t-w    H37b: the same pieces as a plain symmetric V(1,1) cycle with W-cycle coarse levels (box 2, ω = 0.8):
             3x3 block-Jacobi smoothing + box-aggregation Galerkin coarse spaces, no LGF
  pctl       CG + physical-variable coarsening two-level preconditioner (class PCTL)
  fdlgf-GJ   CG + Green-Jacobi scaled background: D^-1/2 Gc D^-1/2
  fdlgf-real H40 (real matrix): FDM background (fitted constant blocks + T1 Robin + e12 coupling) and MUMPS LU on the
             corner-ball feature (row deviation > 1e-6, ring 4), balanced form (fdlgf_real.py)
  bt-v/bt-ex H41 (real matrix): electron-elimination block-triangular preconditioner (bt_elim.py); S3 and A11 by one
             BoomerAMG V-cycle (bt-v, GMRES) or CG + BoomerAMG to 1e-10 (bt-ex, FGMRES); bt-g (H42): A11 by
             V-cycle + FDM background inverse + V-cycle; bt-j / bt-jn (H44): electron and ion jointly by one
             BoomerAMG V-cycle on the 2-field block (scalar / nodal), then the radiation block
  fdlgf-G    CG + FD-LGF background only: per block exact inverse of the constant-coefficient FV operator
             (DST-II in x, y, z) with block-mean K and reaction; no local part (mask fraction reported)
"""

import csv
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from petsc4py import PETSc
from scipy.fft import dstn, idstn

from surrogate3t import build

HERE = Path(__file__).resolve().parent
MAXIT = 2000
SYMMETRIC = True
CRITERION = os.environ.get("SC_CRITERION", "true")   # "true": restart until ||b - Ax|| meets tol; "reported": H40


def petsc_mat(A):
    A = A.tocsr()
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


class ConstantBackground:
    """Exact inverse of -K Δ_h + r on m^3 cells (FV, Dirichlet at faces): DST-II in each direction."""

    def __init__(self, K, r, m):
        lam = (2 - 2 * np.cos(np.arange(1, m + 1) * np.pi / m)) * m**2
        self.den = K * (lam[:, None, None] + lam[None, :, None] + lam[None, None, :]) + r
        self.m = m

    def __call__(self, f):
        g = dstn(f.reshape((self.m,) * 3), type=2, norm="ortho")
        return idstn(g / self.den, type=2, norm="ortho").ravel()


class FDLGFBackgroundPC:
    """Block-diagonal over temperatures, each block the constant-coefficient background inverse (geometric means)."""

    def __init__(self, A, m):
        N = m**3
        d = A.diagonal()
        self.parts = []
        for k in range(3):
            s = slice(k * N, (k + 1) * N)
            B = A[s, s]
            off = -B.data[B.data < 0]                                   # face transmissibilities K_f / h^2
            K = float(np.exp(np.mean(np.log(off)))) / m**2
            lap_diag = 6 * K * m**2
            r = float(np.exp(np.mean(np.log(np.maximum(d[s] - lap_diag, 1e-300)))))
            self.parts.append((s, ConstantBackground(K, r, m)))
            logK = np.log(off * 1.0 / m**2)
            self.mask_fraction = getattr(self, "mask_fraction", []) + [float(np.mean(np.abs(logK - np.log(K)) > np.log(2)))]

    def apply(self, pc, x, y):
        xa, ya = x.getArray(readonly=True), y.getArray()
        for s, G in self.parts:
            ya[s] = G(xa[s])


class GreenJacobiPC(FDLGFBackgroundPC):
    """Green-Jacobi variant (H16/H17 idea): P = D^-1/2 Gc D^-1/2, Gc the constant-coefficient inverse fitted to the
    unit-diagonal matrix D^-1/2 A D^-1/2 (Liouville-type scaling for smooth K)."""

    def __init__(self, A, m):
        from scipy.sparse import diags
        self.s = 1 / np.sqrt(A.diagonal())
        super().__init__(diags(self.s) @ A @ diags(self.s), m)

    def apply(self, pc, x, y):
        xa, ya = x.getArray(readonly=True) * self.s, y.getArray()
        for s, G in self.parts:
            ya[s] = G(xa[s]) * self.s[s]


class PCTL:
    """Physical-variable coarsening two-level preconditioner (Xu, Mo & An; Yue et al. CiCP 2022), symmetric form:
    smoother S = block diagonal over temperatures, one BoomerAMG V-cycle per block; coarse space = equal temperatures
    in each cell, R = [I I I], A_c = R A R^T (scalar 7-point: the exchange terms cancel), one BoomerAMG V-cycle.
    x = S r;  x += R^T A_c^-1 R (r - A x);  x += S (r - A x)."""

    def __init__(self, A, m):
        N = m**3
        self.A, self.N = A.tocsr(), N
        self.amg = [self._amg(A[k * N:(k + 1) * N, k * N:(k + 1) * N]) for k in range(3)]
        Ac = sum(A[i * N:(i + 1) * N, j * N:(j + 1) * N] for i in range(3) for j in range(3))
        self.amg_c = self._amg(Ac)

    @staticmethod
    def _amg(B):
        pc = PETSc.PC().create()
        pc.setOperators(petsc_mat(B))
        pc.setType("hypre")
        pc.setHYPREType("boomeramg")
        pc.setUp()
        x, y = pc.getOperators()[0].createVecs()
        def apply(r):
            x.setArray(r)
            pc.apply(x, y)
            return y.getArray().copy()
        return apply

    def _S(self, r):
        N = self.N
        return np.concatenate([self.amg[k](r[k * N:(k + 1) * N]) for k in range(3)])

    def apply(self, pc, x, y):
        r = x.getArray(readonly=True)
        N = self.N
        u = self._S(r)
        rc = r - self.A @ u
        u += np.tile(self.amg_c(rc[:N] + rc[N:2 * N] + rc[2 * N:]), 3)
        u += self._S(r - self.A @ u)
        y.setArray(u)


def ksp_for(name, A, m):
    N = m**3
    ksp = PETSc.KSP().create()
    opts = PETSc.Options()
    pre = f"{name.replace('-', '_')}_"
    ksp.setOptionsPrefix(pre)
    info = {}
    if name == "amg-nodal":
        perm = np.arange(3 * N).reshape(3, N).T.ravel()                 # interlaced: (e, i, r) per cell
        A = A[perm][:, perm]
        info["perm"] = perm
    M = petsc_mat(A)
    if name == "amg-nodal":
        M.setBlockSize(3)
    ksp.setOperators(M)
    pc = ksp.getPC()
    ksp.setType("cg")
    if name.endswith("-ds"):                                        # KSP-level diagonal scaling D^-1/2 A D^-1/2
        opts[pre + "ksp_diagonal_scale"] = 1
        opts[pre + "ksp_diagonal_scale_fix"] = 1
        name = name[:-3]
    if name in ("amg", "amg-nodal", "amg-3d"):
        pc.setType("hypre")
        pc.setHYPREType("boomeramg")
        if name == "amg-3d":                                            # usual 3D settings (HMIS, ext+i, θ = 0.5)
            from bt_elim import amg_defaults
            amg_defaults(pre)
        if name == "amg-nodal":
            opts[pre + "pc_hypre_boomeramg_nodal_coarsen"] = 1
            opts[pre + "pc_hypre_boomeramg_vec_interp_variant"] = 1
    elif name.startswith("split"):
        pc.setType("fieldsplit")
        iss = [PETSc.IS().createStride(N, k * N, 1) for k in range(3)]
        pc.setFieldSplitIS(*[(str(k), iss[k]) for k in range(3)])
        if name == "split-mult":
            ksp.setType("gmres")
            opts[pre + "pc_fieldsplit_type"] = "multiplicative"
        else:
            opts[pre + "pc_fieldsplit_type"] = "symmetric_multiplicative"
        for k in range(3):
            opts[f"{pre}fieldsplit_{k}_ksp_type"] = "preonly"
            opts[f"{pre}fieldsplit_{k}_pc_type"] = "hypre"
            opts[f"{pre}fieldsplit_{k}_pc_hypre_type"] = "boomeramg"
    elif name.startswith(("fdlgf3t", "bnn3t", "agg3t")):                     # improved FD-LGF (fdlgf3t.py)
        from fdlgf3t import FDLGF3T
        pc.setType("python")
        cfg = {"fdlgf3t": dict(box=4, omega=0.6, cycles=2),                # H36 locked configuration
               "fdlgf3t-lu": dict(box=4, omega=0.6, cycles=2, lu_size=10**9),
               "fdlgf3t-fast": dict(box=2, omega=0.5, cycles=1),           # H37 tuned: 2^3 boxes, V-cycle
               "bnn3t-fast": dict(box=2, omega=0.5, cycles=1, green=False),   # same without the LGF / DST
               "agg3t-w": dict(form="vcycle", box=2, omega=0.8, nu=1, cycles=2, green=False)}[name]   # H37b
        pc.setPythonContext(FDLGF3T(A, m, coarse="ml", **{"lu_size": 5000, **cfg}))
    elif name == "pctl":
        pc.setType("python")
        pc.setPythonContext(PCTL(A, m))
    elif name == "fdlgf-real":                                          # H40, real matrix only
        from fdlgf_real import FDLGFReal
        pc.setType("python")
        ctx = FDLGFReal(A, m, ring=4, tol=1e-6)
        pc.setPythonContext(ctx)
        info["mask_fraction"] = ctx.fraction
    elif name in ("bt-j", "bt-jn"):                                    # H44, real matrix only
        from bt_elim import BTJoint
        pc.setType("python")
        pc.setPythonContext(BTJoint(A, nodal=name == "bt-jn"))
    elif name in ("bt-v", "bt-ex", "bt-g"):                            # H41/H42, real matrix only
        from bt_elim import BTElim
        pc.setType("python")
        pc.setPythonContext(BTElim(A, exact=name == "bt-ex", fdm=name == "bt-g"))
        if name == "bt-ex":
            ksp.setType("fgmres")
    elif name in ("fdlgf-G", "fdlgf-GJ"):
        pc.setType("python")
        ctx = (FDLGFBackgroundPC if name == "fdlgf-G" else GreenJacobiPC)(A, m)
        pc.setPythonContext(ctx)
        info["mask_fraction"] = ctx.mask_fraction
    if not SYMMETRIC and ksp.getType() == "cg":
        ksp.setType("gmres")
    if ksp.getType() in ("gmres", "fgmres"):
        ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setGMRESRestart(50)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setFromOptions()
    return ksp, M, info


def solve(name, A, b, m, tol):
    t0 = perf_counter()
    ksp, M, info = ksp_for(name, A, m)
    ksp.setTolerances(rtol=tol, atol=0.0, max_it=MAXIT)
    bb = b[info["perm"]] if "perm" in info else b
    x, rhs = M.createVecs()
    rhs.setArray(bb)
    ksp.setUp()
    t1 = perf_counter()
    # the KSP's residual estimate can drift from the true residual (badly scaled rows): restart from the current x
    # until the true ||b - Ax|| / ||b|| meets tol (at most 10 restarts / MAXIT iterations in total); timed
    r = M.createVecLeft()
    its, restarts = 0, 0
    if CRITERION == "reported":   # H40: the solver's own (unpreconditioned) residual estimate decides, no restarts
        ksp.solve(rhs, x)
        its = ksp.getIterationNumber()
        M.mult(x, r)
        rel = float(np.linalg.norm(bb - r.getArray()) / np.linalg.norm(bb))
        t2 = perf_counter()
        d = M.getDiagonal().getArray()
        return dict(solver=name, tol=tol, iterations=its, restarts=0, reason=int(ksp.getConvergedReason()),
                    reported_rel_res=ksp.getResidualNorm() / np.linalg.norm(bb), true_rel_res=rel,
                    scaled_rel_res=float(np.linalg.norm((bb - r.getArray()) / d) / np.linalg.norm(bb / d)),
                    setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0, mask_fraction=info.get("mask_fraction"),
                    peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    while True:   # PETSc's default test is ||r_est|| < rtol ||b||; each restart asks 10x more of the estimate
        ksp.setTolerances(rtol=tol * 0.1**restarts, atol=0.0, max_it=MAXIT - its)
        ksp.solve(rhs, x)
        its += ksp.getIterationNumber()
        M.mult(x, r)
        rel = float(np.linalg.norm(bb - r.getArray()) / np.linalg.norm(bb))
        if rel < tol or ksp.getConvergedReason() < 0 or its >= MAXIT or restarts == 10:
            break
        restarts += 1
        ksp.setInitialGuessNonzero(True)
    t2 = perf_counter()
    return dict(solver=name, tol=tol, iterations=its, restarts=restarts, reason=int(ksp.getConvergedReason()),
                true_rel_res=rel, setup_s=t1 - t0, solve_s=t2 - t1, total_s=t2 - t0,
                mask_fraction=info.get("mask_fraction"),
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)


def main(out, m, solvers):
    global SYMMETRIC
    if m.split("@")[0].isdigit():                    # "m", "m@dt", "m@dt@scaled|emis" (nonsymmetric)
        m, dt, scaled = (m.split("@") + ["", ""])[:3]
        m = int(m)
        A, b = build(m, float(dt) if dt else 1e-3, scaled=scaled == "scaled", emission=scaled == "emis")
        SYMMETRIC = abs(A - A.T).max() <= 1e-12 * abs(A).max()
    else:
        import load_real
        path, _, rhs = m.partition(",")
        A = load_real.load(path)
        A, b = load_real.to_blocked(A, load_real.load_rhs(rhs or None, A))
        m = round((A.shape[0] // 3) ** (1 / 3))
        SYMMETRIC = abs(A - A.T).max() <= 1e-12 * abs(A).max()
    print(f"m={m} order={A.shape[0]:,} nnz={A.nnz:,}", flush=True)
    rows = []
    for name in solvers:
        for tol in (1e-4, 1e-8):
            row = dict(m=m, case=out, **solve(name, A, b, m, tol))
            rows.append(row)
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            with (HERE / "results" / f"{out}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["amg", "amg-nodal", "split-sym", "split-mult", "fdlgf-G"])
