"""H45 ceiling diagnostics: how much could each family of generic add-ons gain over BoomerAMG, at best?

  D0  baseline      BoomerAMG (H40 amg-3d settings) right-preconditioned GMRES(50), rtol 1e-8, x0 = 0
  D1  deflation     oracle slow modes of M^-1 A (harmonic Ritz from 100 Arnoldi steps of A M^-1), k in {4,16,48};
                    z = M r; z += Q (r - A z), Q = U (U^T A U)^-1 U^T.  Ceiling for SNN / DeepONet / learned coarse spaces
  D2  initial guess x0 = x* (1 + eps eta), eta smooth random field, eps in {1e-1, 1e-2}. Ceiling for neural initial guesses
  D3  AMG settings  36-point grid (theta, agg_nl, P_max, relax). Ceiling for learned AMG parameter selection
All with the same numpy GMRES (MGS, restart 50), so iteration counts are comparable.  single thread.
usage: python diag.py OUT.csv case [case ...]
"""
import csv
import itertools
import sys
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from suite import CASES, load

BASE = dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, strong_threshold=0.5, agg_nl=1,
            relax_type_all="symmetric-SOR/Jacobi")
_count = itertools.count()


def amg(A, **over):
    o = dict(BASE, **over)
    prefix = f"d{next(_count)}_"
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    pc = PETSc.PC().create()
    pc.setOptionsPrefix(prefix)
    pc.setOperators(M)
    pc.setType("hypre"); pc.setHYPREType("boomeramg")
    opts = PETSc.Options()
    for k, v in o.items():
        opts[prefix + "pc_hypre_boomeramg_" + k] = v
    pc.setFromOptions()
    t0 = perf_counter(); pc.setUp(); setup = perf_counter() - t0
    xv, bv = M.createVecs()

    def apply(r):
        bv.setArray(r); pc.apply(bv, xv)
        return xv.getArray().copy()
    apply.keep = (pc, M, xv, bv)
    return apply, setup


def gmres(A, P, b, x0=None, tol=1e-8, m=50, maxit=500):
    """Right-preconditioned restarted GMRES(m); stops on the GMRES residual estimate <= tol ||b|| (solver-reported
    criterion, as chosen for H40: the 3T matrices have a float64 floor of the true residual)."""
    nb = np.linalg.norm(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    its = 0
    while its < maxit:
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta <= tol * nb:
            return x, its
        V = [r / beta]; Z = []
        H = np.zeros((m + 1, m)); g = np.zeros(m + 1); g[0] = beta
        cs, sn = np.zeros(m), np.zeros(m)
        j = 0
        for j in range(m):
            z = P(V[j]); Z.append(z)
            w = A @ z
            for _ in range(2):                                   # MGS with one re-orthogonalisation pass
                for i in range(j + 1):
                    h = V[i] @ w; H[i, j] += h; w -= h * V[i]
            H[j + 1, j] = np.linalg.norm(w); V.append(w / H[j + 1, j])
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]; H[i, j] = t
            d = np.hypot(H[j, j], H[j + 1, j]); cs[j], sn[j] = H[j, j] / d, H[j + 1, j] / d
            H[j, j] = d; H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]; g[j] *= cs[j]
            its += 1
            if abs(g[j + 1]) <= tol * nb or its >= maxit:
                break
        y = np.linalg.solve(np.triu(H[:j + 1, :j + 1]), g[:j + 1])
        x = x + np.column_stack(Z) @ y
        if abs(g[j + 1]) <= tol * nb:                            # solver-reported criterion (as PETSc, H40)
            return x, its
    return x, its


def slow_modes(A, P, n, K=100, seed=0):
    """Harmonic Ritz vectors of A M^-1 (smallest |theta|), mapped to u = M^-1 y: approximate slow modes of M^-1 A."""
    v = np.random.default_rng(seed).standard_normal(n); V = [v / np.linalg.norm(v)]
    H = np.zeros((K + 1, K))
    for j in range(K):
        w = A @ P(V[j])
        for _ in range(2):
            for i in range(j + 1):
                h = V[i] @ w; H[i, j] += h; w -= h * V[i]
        H[j + 1, j] = np.linalg.norm(w); V.append(w / H[j + 1, j])
    Hk = H[:K, :K]
    ek = np.zeros(K); ek[-1] = 1
    T = Hk + H[K, K - 1] ** 2 * np.linalg.solve(Hk.T, ek)[:, None] * ek[None, :]
    th, G = np.linalg.eig(T)
    order = np.argsort(np.abs(th))
    Vk = np.column_stack(V[:K])
    return th[order], Vk, G[:, order]


def deflation(A, P, Vk, G, k):
    cols = []
    for i in range(k):
        y = Vk @ G[:, i]
        cols += [y.real] + ([y.imag] if np.abs(y.imag).max() > 1e-12 * np.abs(y.real).max() else [])
    Y = np.column_stack(cols[:k])
    U = np.column_stack([P(Y[:, i]) for i in range(Y.shape[1])])
    U, _ = np.linalg.qr(U)
    AU = A @ U
    E = np.linalg.inv(U.T @ AU)

    def Pd(r):
        z = P(r)
        return z + U @ (E @ (U.T @ (r - A @ z)))
    return Pd


def smooth_field(shape, seed, corr=0.2):
    rng = np.random.default_rng(seed)
    k = [np.fft.fftfreq(s) * s for s in shape]
    K2 = sum(np.meshgrid(*[ki**2 for ki in k], indexing="ij"))
    eta = np.real(np.fft.ifftn(np.exp(-0.25 * (np.pi * corr) ** 2 * K2) * np.fft.fftn(rng.standard_normal(shape))))
    return (eta / np.abs(eta).max()).ravel()


def shape_of(name, n):
    if name == "circ":
        s = int(round(np.sqrt(n))); return (s, s)
    if name.startswith("3t"):
        s = int(round((n // 3) ** (1 / 3))); return (3, s, s, s)
    s = int(round(n ** (1 / 3))); return (s, s, s)


def run_case(name, wr):
    A, b = load(name)
    n = A.shape[0]

    def rec(diag, setting, its, t_setup, t_solve, extra=""):
        row = dict(case=name, n=n, diag=diag, setting=setting, its=its, setup_s=round(t_setup, 3),
                   solve_s=round(t_solve, 3), total_s=round(t_setup + t_solve, 3), extra=extra)
        wr.writerow(row)
        print(f"{name:9s} {diag:3s} {setting:40s} its {its:4d} total {t_setup + t_solve:7.2f} s {extra}", flush=True)

    P, ts = amg(A)
    t0 = perf_counter(); x, its = gmres(A, P, b); t = perf_counter() - t0
    rec("D0", "base", its, ts, t)
    xs, _ = gmres(A, P, b, tol=1e-12, maxit=1500)
    # D1
    th, Vk, G = slow_modes(A, P, n)
    for k in (4, 16, 48):
        Pd = deflation(A, P, Vk, G, k)
        t0 = perf_counter(); _, its = gmres(A, Pd, b); t = perf_counter() - t0
        rec("D1", f"k={k}", its, ts, t, extra=f"|theta_k|={abs(th[k - 1]):.2e} |theta_1|={abs(th[0]):.2e}")
    # D2
    eta = smooth_field(shape_of(name, n), seed=1)
    for eps in (1e-1, 1e-2):
        x0 = xs * (1 + eps * eta)
        t0 = perf_counter(); _, its = gmres(A, P, b, x0=x0); t = perf_counter() - t0
        r0 = np.linalg.norm(b - A @ x0) / np.linalg.norm(b)
        rec("D2", f"eps={eps:g}", its, ts, t, extra=f"r0={r0:.1e}")
    # D3
    for th_, agg, pm, rel in itertools.product((0.25, 0.5, 0.7), (0, 1, 2), (2, 4),
                                               ("symmetric-SOR/Jacobi", "l1scaled-SOR/Jacobi")):
        try:
            Pp, tsp = amg(A, strong_threshold=th_, agg_nl=agg, P_max=pm, relax_type_all=rel)
            t0 = perf_counter(); _, its = gmres(A, Pp, b); t = perf_counter() - t0
        except Exception as e:  # noqa: BLE001  (a failing setting is a data point, not a crash)
            its, tsp, t = -1, 0.0, 0.0
            print("fail", e)
        rec("D3", f"theta={th_} agg={agg} Pmax={pm} relax={rel[:6]}", its, tsp, t)


if __name__ == "__main__":
    out, names = sys.argv[1], sys.argv[2:] or CASES
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["case", "n", "diag", "setting", "its", "setup_s", "solve_s", "total_s", "extra"])
        wr.writeheader()
        for name in names:
            run_case(name, wr); fh.flush()
