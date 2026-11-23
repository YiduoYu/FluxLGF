"""Floor test: replace the same-sign lump by a nonnegative edge measure with the same second moments on smooth test vectors.

For row i and test vectors t^r, t^s the energy density is m_rs(i) = sum_j (-a_ij) (t^r_i - t^r_j)(t^s_i - t^s_j) (carre du champ). Find w_ij >= 0 on
distance-1 or distance-2 neighbours with the same moments (Caratheodory-Tchakaloff, solved by NNLS). Then measure the outer iterations with the exact
inverse of the fitted M-matrix: the floor no hierarchy can beat.
usage: gamma_fit.py k sweeps dist lambda CASE...
"""
import sys
from pathlib import Path

import numpy as np
import pyamg.relaxation.relaxation as R
import scipy.sparse as sp
import scipy.sparse.linalg as sl
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from problems import load  # noqa: E402


def test_vectors(A, k, sweeps, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    T = rng.standard_normal((n, k))
    for r in range(k):
        x = T[:, r].copy()
        for _ in range(sweeps):
            R.gauss_seidel(A, x, np.zeros(n), iterations=1, sweep="symmetric")
        T[:, r] = x / np.abs(x).max()
    return T


def lump(A):
    C = A.tocoo()
    pos = (C.row != C.col) & (C.data > 0)
    rs = np.bincount(C.row[pos], C.data[pos], A.shape[0])
    return (sp.csr_matrix((C.data[~pos], (C.row[~pos], C.col[~pos])), shape=A.shape) + sp.diags(rs)).tocsr()


LAM = 100.0


def fit(A, T, dist):
    A = A.tocsr(); n = A.shape[0]; k = T.shape[1]
    G = abs(A).tocsr()
    if dist == 2:
        G = (G @ G).tocsr()
    iu = np.triu_indices(k)
    I, J, V = [], [], []
    d = A.diagonal()
    rs = np.asarray(A.sum(1)).ravel()
    bad = 0
    for i in range(n):
        s, e = A.indptr[i], A.indptr[i + 1]
        ja, va = A.indices[s:e], A.data[s:e]
        off = ja != i
        ja, va = ja[off], va[off]
        if (va < 0).all():
            I.append(np.full(ja.size, i)); J.append(ja); V.append(-va)
            continue
        cand = G.indices[G.indptr[i]:G.indptr[i + 1]]
        cand = cand[cand != i]
        Dc = T[cand] - T[i]
        Phi = (Dc[:, iu[0]] * Dc[:, iu[1]]).T            # moments x candidates
        Da = T[ja] - T[i]
        m = ((Da[:, iu[0]] * Da[:, iu[1]]) * (-va)[:, None]).sum(0)
        # stay near the negative part of the row, and only then match the moments: rough modes keep their energy
        w0 = np.zeros(cand.size)
        pos = {c: p for p, c in enumerate(cand)}
        for c, v in zip(ja, va):
            if v < 0 and c in pos:
                w0[pos[c]] = -v
        scale = max(w0.max(), np.abs(va).max())
        sc = np.abs(m).max() + np.abs(Phi).max() * scale + 1e-300
        # scaled unknown w/scale: identity block keeps w near w0 with weight 1; moment block has weight LAM
        Aug = np.vstack([np.sqrt(LAM) * Phi * scale / sc, np.eye(cand.size)])
        rhs = np.concatenate([np.sqrt(LAM) * m / sc, w0 / scale])
        try:
            w, _ = nnls(Aug, rhs, maxiter=20 * cand.size)
        except RuntimeError:
            bad += 1; w = w0 / scale
        w = w * scale
        # keep the size of the row: total weight must not vanish
        nz = w > 1e-14 * max(w.max(), 1e-300) if w.max() > 0 else w > 0
        if not nz.any():
            bad += 1; I.append(np.full(ja.size, i)); J.append(ja); V.append(np.maximum(-va, 0))
            continue
        I.append(np.full(int(nz.sum()), i)); J.append(cand[nz]); V.append(w[nz])
    W = sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=A.shape)
    W = (W + W.T) * 0.5
    M = sp.diags(np.asarray(W.sum(1)).ravel() + rs) - W
    return M.tocsr(), bad


def outer(A, b, M):
    lu = sl.splu(M.tocsc())
    c = [0]
    _, info = sl.gmres(A, b, M=sl.LinearOperator(A.shape, lu.solve), rtol=1e-6, restart=50, maxiter=8, callback=lambda r: c.__setitem__(0, c[0] + 1), callback_type="pr_norm")
    return c[0] if info == 0 else f">{c[0]}"


if __name__ == "__main__":
    k, sweeps, dist = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    LAM = float(sys.argv[4])
    for cs in sys.argv[5:]:
        A, b, _ = load(cs)
        A = A.tocsr()
        L = lump(A)
        T = test_vectors(A, k, sweeps)
        M, bad = fit(A, T, dist)
        print(f"{cs:24s} k={k} sw={sweeps} d={dist} lam={LAM:g} | lump floor {outer(A, b, L)} | fit floor {outer(A, b, M)} nnz/row {M.nnz / A.shape[0]:.1f} (A {A.nnz / A.shape[0]:.1f}) dmin {M.diagonal().min():.1e} bad {bad}", flush=True)
