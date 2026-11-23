"""Exploratory (not locked): circ anisotropy, BoomerAMG-PCG iterations and Ritz spectrum of M^-1 A from the PCG
Lanczos tridiagonal.  usage: python explore_circ.py n chi theta"""
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, "..")
from diag import amg  # noqa: E402
from suite import load  # noqa: E402


def pcg(A, P, b, tol=1e-8, maxit=2000):
    x = np.zeros_like(b); r = b.copy(); z = P(r); p = z.copy(); rz = r @ z; nb = np.linalg.norm(b)
    al, be = [], []
    for k in range(maxit):
        q = A @ p; a = rz / (p @ q); x += a * p; r -= a * q
        al.append(a)
        if np.linalg.norm(r) <= tol * nb:
            return x, k + 1, al, be
        z = P(r); rz2 = r @ z; bb = rz2 / rz; be.append(bb); rz = rz2; p = z + bb * p
    return x, maxit, al, be


def ritz(al, be):
    k = len(al); T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1 / al[j] + (be[j - 1] / al[j - 1] if j else 0)
        if j + 1 < k:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(be[j]) / al[j]
    return np.linalg.eigvalsh(T)


if __name__ == "__main__":
    n, chi, th = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    sys.path.insert(0, "../../anisotropic")
    from cases_aniso import make_case
    c = make_case("circ", n, chi); A, b = c.A.tocsr(), c.f.ravel()
    P, ts = amg(A, strong_threshold=th)
    t0 = perf_counter(); x, its, al, be = pcg(A, P, b); t = perf_counter() - t0
    ev = ritz(al, be)
    print(f"n={n} chi={chi:g} theta={th} PCG its {its} setup {ts:.2f}s solve {t:.2f}s  "
          f"ritz min {ev[0]:.2e} #<0.01 {np.sum(ev<0.01)} #<0.1 {np.sum(ev<0.1)} #<0.3 {np.sum(ev<0.3)} max {ev[-1]:.2f}")
    print("smallest 12:", np.array2string(ev[:12], precision=3))
