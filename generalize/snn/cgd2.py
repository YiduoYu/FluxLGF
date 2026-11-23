"""H48 adaptive coarse-grid deflation (CGD2) for BoomerAMG-PCG.

1. BoomerAMG on A; plain PCG for k0 iterations (these iterations are kept, never wasted).
2. Fine-grid probe: smallest Ritz value lam_k0 of B A from the PCG Lanczos tridiagonal (k0 steps).  It sees the real
   difficulty on the target grid (the H47 detector only saw the coarse grid).
3. lam_k0 >= lam_thr: continue the same PCG to convergence.
   lam_k0 <  lam_thr: coarse rediscretization at n_c = (n+1)/4 - 1, BoomerAMG, K-step Lanczos for B_c A_c, Ritz vectors
   with value < tau (at most m_max), sparse bilinear interpolation, A-orthonormalise, then deflated PCG restarted from
   the current iterate (x <- x + W W^T r).
"""
from time import perf_counter

import numpy as np

from cgd import interp_2d
from snn_defl import amg, lanczos_A, orthonormalise


def tri_min(al, be):
    k = len(al); T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1 / al[j] + (be[j - 1] / al[j - 1] if j else 0)
        if j + 1 < k:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(be[j]) / al[j]
    return float(np.linalg.eigvalsh(T)[0])


def pcg_run(A, P, b, x, tol, maxit, W=None, AW=None, k_stop=None, state=None):
    """(Deflated) PCG from x.  Stops at convergence, maxit, or after k_stop iterations (then returns the state to
    continue).  Returns x, its, converged, alphas, betas, state."""
    nb = np.linalg.norm(b)
    Pd = (lambda r: P(r)) if W is None else (lambda r: (lambda z: z - W @ (AW.T @ z))(P(r)))
    if state is None:
        r = b - A @ x
        if W is not None:
            x = x + W @ (W.T @ r); r = b - A @ x
        z = Pd(r); p = z.copy(); rz = r @ z
    else:
        r, p, rz = state
    al, be = [], []
    for k in range(maxit):
        q = A @ p; a = rz / (p @ q); x = x + a * p; r = r - a * q; al.append(a)
        if np.linalg.norm(r) <= tol * nb:
            return x, k + 1, True, al, be, None
        z = Pd(r); rz2 = r @ z; bb = rz2 / rz; be.append(bb); rz = rz2; p = z + bb * p
        if k_stop is not None and k + 1 >= k_stop:
            return x, k + 1, False, al, be, (r, p, rz)
    return x, maxit, False, al, be, None


def solve_cgd2(A, b, n, make_coarse, k0=10, lam_thr=0.02, K=40, tau=0.5, m_max=24, tol=1e-8, maxit=3000):
    t0 = perf_counter()
    P, _ = amg(A); t_amg = perf_counter() - t0
    t1 = perf_counter()
    x, k1, conv, al, be, st = pcg_run(A, P, b, np.zeros_like(b), tol, maxit, k_stop=k0)
    info = dict(t_amg=t_amg, k0=k1, m=0, lam_probe=float("nan"), t_basis=0.0)
    if conv:
        info.update(its=k1, t_solve=perf_counter() - t1)
    else:
        lam = tri_min(al, be[:len(al) - 1]); info["lam_probe"] = lam
        if lam >= lam_thr:
            x, k2, conv, *_ = pcg_run(A, P, b, x, tol, maxit - k1, state=st)
            info.update(its=k1 + k2)
        else:
            tb = perf_counter()
            nc = (n + 1) // 4 - 1
            Ac = make_coarse(nc); Pc, _ = amg(Ac)
            lc, U = lanczos_A(Ac, Pc, K)
            m = int(max(1, min(m_max, np.sum(lc < tau))))
            W, AW = orthonormalise(interp_2d(nc, n) @ U[:, :m], A)
            info.update(m=m, t_basis=perf_counter() - tb)
            x, k2, conv, *_ = pcg_run(A, P, b, x, tol, maxit - k1, W=W, AW=AW)
            info.update(its=k1 + k2)
        info["t_solve"] = perf_counter() - t1 - info["t_basis"]
    info.update(converged=conv, total=perf_counter() - t0)
    return x, info
