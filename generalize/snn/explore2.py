"""Exploratory: circ, (1) best BoomerAMG config for PCG, (2) Ritz vectors of M^-1 A from the PCG Lanczos process
(full storage), (3) oracle deflated PCG ceiling, (4) are slow modes radial (fraction of the mode explained by its
ring average)?   usage: python explore2.py n chi"""
import sys
from time import perf_counter

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, ".."); sys.path.insert(0, "../../anisotropic")
from diag import amg  # noqa: E402
from cases_aniso import make_case  # noqa: E402


def pcg(A, P, b, tol=1e-8, maxit=3000, W=None, keep=False):
    """(Deflated) PCG.  W: deflation basis (n x k) -> init-CG + projected preconditioner (Saad et al. 2000 form)."""
    nb = np.linalg.norm(b)
    if W is not None:
        AW = A @ W; E = np.linalg.cholesky(W.T @ AW)
        cs = lambda v: sla.cho_solve((E, True), v)                    # noqa: E731
        x = W @ cs(W.T @ b)
    else:
        x = np.zeros_like(b)
    r = b - A @ x
    def Pd(r):
        z = P(r)
        return z if W is None else z - W @ cs(AW.T @ z)
    z = Pd(r); p = z.copy(); rz = r @ z
    al, be, Z = [], [], []
    for k in range(maxit):
        if keep: Z.append(z / np.sqrt(rz))
        q = A @ p; a = rz / (p @ q); x += a * p; r -= a * q; al.append(a)
        if np.linalg.norm(r) <= tol * nb:
            return x, k + 1, al, be, Z
        z = Pd(r); rz2 = r @ z; bb = rz2 / rz; be.append(bb); rz = rz2; p = z + bb * p
    return x, maxit, al, be, Z


def tri(al, be):
    k = len(al); T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1 / al[j] + (be[j - 1] / al[j - 1] if j else 0)
        if j + 1 < k: T[j, j + 1] = T[j + 1, j] = np.sqrt(be[j]) / al[j]
    return T


def ring_fraction(v, n, nb=None):
    t = np.arange(1, n + 1) / (n + 1); X, Y = np.meshgrid(t, t, indexing="xy")
    R = np.hypot(X - .5, Y - .5).ravel(); nb = nb or n // 2
    idx = np.minimum((R / R.max() * nb).astype(int), nb - 1)
    s = np.bincount(idx, v, nb) / np.maximum(np.bincount(idx, None, nb), 1)
    return np.linalg.norm(s[idx]) ** 2 / np.linalg.norm(v) ** 2


if __name__ == "__main__":
    n, chi = int(sys.argv[1]), float(sys.argv[2])
    c = make_case("circ", n, chi); A, b = c.A.tocsr(), c.f.ravel()
    best = None
    for th in (0.25, 0.5):
        for agg in (0, 1):
            P, ts = amg(A, strong_threshold=th, agg_nl=agg)
            t0 = perf_counter(); _, its, *_ = pcg(A, P, b); t = perf_counter() - t0
            print(f"n={n} theta={th} agg={agg}: PCG its {its} setup {ts:.2f} solve {t:.2f} total {ts+t:.2f}", flush=True)
            if best is None or ts + t < best[0]: best = (ts + t, th, agg, P, its)
    _, th, agg, P, its0 = best
    print(f"best config theta={th} agg={agg}")
    x, its, al, be, Z = pcg(A, P, b, keep=True)
    T = tri(al[:-1], be); lam, Y = np.linalg.eigh(T)
    Zm = np.column_stack(Z[:len(lam)])
    for k in (8, 16, 32, 64, 128):
        if k > len(lam): break
        W = Zm @ Y[:, :k]; W, _ = np.linalg.qr(W)
        t0 = perf_counter(); _, itsd, *_ = pcg(A, P, b, W=W); t = perf_counter() - t0
        rf = np.mean([ring_fraction(W[:, i], n) for i in range(min(k, 16))])
        print(f"  oracle deflation k={k:3d}: its {itsd:4d} (base {its0})  solve {t:.2f}s  lam_k {lam[k-1]:.3e}  "
              f"ring-fraction(first16) {rf:.3f}", flush=True)
    print("ring fraction of slow modes 1..8:", [round(ring_fraction(Zm @ Y[:, i], n), 3) for i in range(8)])
    rng = np.random.default_rng(0); print("ring fraction of random vector:", round(ring_fraction(rng.standard_normal(n*n), n), 4))
