"""H52 feasibility (no solve): split the scalar blocks of SC26_01 (ion Schur S3 after electron elimination, radiation A11)
into regions algebraically and measure (1) how strongly the regions are coupled, (2) how close each region is to a
constant-coefficient operator, (3) the cost of a 128^3 DCT.
Regions: drop couplings with strength |a_ij| / sqrt(a_ii a_jj) < delta; connected components of what is left.
Leak of a region C (constant mode): sum_{i in C, j not in C} |a_ij| / sum_{i in C} (a_ii - sum_{j in C, j != i} |a_ij|).
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.fft import dctn, idctn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from bt_elim import schur_ion  # noqa: E402

X = Path(__file__).resolve().parent.parent / "SolverChallenge26-tenproblems" / "extracted"


def strength(M):
    M = M.tocoo()
    d = M.tocsr().diagonal()
    off = M.row != M.col
    r, c, v = M.row[off], M.col[off], M.data[off]
    return r, c, np.abs(v) / np.sqrt(np.abs(d[r] * d[c]))


def regions(M, delta):
    r, c, s = strength(M)
    keep = s >= delta
    n = M.shape[0]
    G = sp.csr_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(n, n))
    return connected_components(G, directed=False)


def describe(name, M, m, deltas):
    print(f"\n=== {name}: n={M.shape[0]:,}")
    r, c, s = strength(M)
    h, e = np.histogram(np.log10(np.maximum(s, 1e-300)), bins=[-300, -15, -12, -9, -6, -4, -2, -1, 0, 1])
    print("strength histogram (log10 edges):", dict(zip([f"{a:g}..{b:g}" for a, b in zip(e[:-1], e[1:])], h)))
    M = M.tocsr()
    d = M.diagonal()
    for delta in deltas:
        ncomp, lab = regions(M, delta)
        size = np.bincount(lab)
        big = np.argsort(size)[::-1]
        single = int((size == 1).sum())
        print(f"delta {delta:g}: {ncomp:,} components; singletons {single:,}; largest sizes {size[big[:5]].tolist()}")
        Mc = M.tocoo()
        same = lab[Mc.row] == lab[Mc.col]
        offd = Mc.row != Mc.col
        within = np.bincount(Mc.row[same & offd], weights=np.abs(Mc.data[same & offd]), minlength=M.shape[0])
        cross = np.bincount(Mc.row[~same], weights=np.abs(Mc.data[~same]), minlength=M.shape[0])
        rho = d - within
        for k in big[:4]:
            if size[k] < 2:
                break
            idx = lab == k
            leak = cross[idx].sum() / rho[idx].sum()
            react = (rho[idx] - cross[idx]).sum() / d[idx].sum()
            # homogeneity: face coefficients inside the component
            fs = np.abs(Mc.data[same & offd & idx[Mc.row]])
            med = np.median(fs)
            dev = float(np.mean((fs > 2 * med) | (fs < med / 2)))
            z, y, x = np.unravel_index(np.nonzero(idx)[0], (m, m, m))
            rr = np.sqrt(x**2 + y**2 + z**2)
            print(f"   comp size {size[k]:>9,}  radius {rr.min():5.1f}-{rr.max():5.1f}  leak {leak:.2e}  "
                  f"reaction/diag {react:.1e}  face median {med:.2e}, off by >2x {dev:.1%}")
        if single:
            sidx = size[lab] == 1
            print(f"   singletons: strength-to-rest max {cross[sidx].max() / d[sidx].min():.1e}, "
                  f"sum|offdiag|/diag max {np.max((cross[sidx] + within[sidx]) / d[sidx]):.1e}")


if __name__ == "__main__":
    A = sp.load_npz(X / "A.npz").tocsr()
    N = A.shape[0] // 3
    m = round(N ** (1 / 3))
    s = [slice(k * N, (k + 1) * N) for k in range(3)]
    d22 = A[s[1], s[1]].diagonal()
    a23 = A[s[1], s[2]].diagonal(); a32 = A[s[2], s[1]].diagonal()
    S3 = schur_ion(A[s[2], s[2]].tocsr(), a23, a32, d22)
    describe("S3 (ion Schur)", S3, m, (1e-12, 1e-6, 1e-3))
    describe("A11 (radiation)", A[s[0], s[0]].tocsr(), m, (1e-6, 1e-3, 3e-2))
    f = np.random.default_rng(0).standard_normal((m, m, m))
    for w in (1, 8):
        t = perf_counter()
        for _ in range(5):
            g = idctn(dctn(f, type=2, norm="ortho", workers=w) / 1.0, type=2, norm="ortho", workers=w)
        print(f"DCT-II + inverse on {m}^3, workers {w}: {(perf_counter() - t) / 5:.3f} s;  A@x: ", end="")
        xv = np.random.default_rng(1).standard_normal(3 * N)
        t = perf_counter(); A @ xv; print(f"{perf_counter() - t:.3f} s")
