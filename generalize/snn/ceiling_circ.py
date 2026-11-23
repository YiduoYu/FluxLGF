"""Exploratory: accurate slow modes of M^-1 A (BoomerAMG, best config theta 0.25 agg 0) by Lanczos in the A inner
product with full reorthogonalisation and a random start, then the oracle deflated-PCG ceiling.
usage: python ceiling_circ.py n chi K"""
import sys
from time import perf_counter

import numpy as np

from explore2 import amg, make_case, pcg, ring_fraction


def lanczos_A(A, P, n, K, seed=0):
    v = np.random.default_rng(seed).standard_normal(n)
    Av = A @ v; s = np.sqrt(v @ Av); v /= s; Av /= s
    V = np.empty((n, K + 1), order="F"); AV = np.empty((n, K + 1), order="F"); V[:, 0] = v; AV[:, 0] = Av
    T = np.zeros((K + 1, K))
    for j in range(K):
        w = P(np.ascontiguousarray(AV[:, j])); Aw = A @ w
        for _ in range(2):
            h = AV[:, :j + 1].T @ w; w -= V[:, :j + 1] @ h; Aw = A @ w; T[:j + 1, j] += h
        b = np.sqrt(w @ Aw); T[j + 1, j] = b; V[:, j + 1] = w / b; AV[:, j + 1] = Aw / b
    Tk = 0.5 * (T[:K] + T[:K].T)
    lam, Y = np.linalg.eigh(Tk)
    res = np.abs(T[K, K - 1] * Y[-1, :])        # ||(M^-1 A) u - lam u||_A for A-normalised Ritz vectors
    return lam, V[:, :K] @ Y, res


if __name__ == "__main__":
    n, chi, K = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    c = make_case("circ", n, chi); A, b = c.A.tocsr(), c.f.ravel()
    P, ts = amg(A, strong_threshold=0.25, agg_nl=0)
    t0 = perf_counter(); _, its0, *_ = pcg(A, P, b); t_base = perf_counter() - t0
    print(f"n={n} chi={chi:g} base PCG its {its0} setup {ts:.2f} solve {t_base:.2f}", flush=True)
    t0 = perf_counter(); lam, U, res = lanczos_A(A, P, n * n, K); print(f"Lanczos K={K} {perf_counter()-t0:.1f}s")
    print("lam[:10]", np.array2string(lam[:10], precision=4))
    for th in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  #eig < {th}: {np.sum(lam < th)}  (converged res<1e-2*lam: {np.sum((lam < th) & (res < 1e-2 * lam))})")
    for k in (8, 16, 32, 64, 128, 192):
        if k > K // 2: break
        W, _ = np.linalg.qr(U[:, :k])
        t0 = perf_counter(); _, its, *_ = pcg(A, P, b, W=W); t = perf_counter() - t0
        print(f"  deflation k={k:3d}: its {its:4d} (base {its0}) solve {t:.2f}s (base {t_base:.2f})  lam_k {lam[k-1]:.3e} "
              f"res_k/lam_k {res[k-1]/lam[k-1]:.1e} ring {np.mean([ring_fraction(U[:, i], n) for i in range(k)]):.2f}",
              flush=True)
    np.save(f"results/slowmodes_n{n}_chi{chi:g}.npy", U[:, :min(K // 2, 128)].astype(np.float32))
    np.save(f"results/slowlam_n{n}_chi{chi:g}.npy", lam)
