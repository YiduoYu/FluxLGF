"""Problem classes where BoomerAMG is known or measured to be weak (round 3). Written before any SSR tuning on them.

Sources of the choice: AIR papers (lAIR, constrained lAIR): advection-tuned AMG degrades when diffusion dominates and classical AMG
degrades when advection dominates, so one configuration struggles when both regimes share a domain. Multigroup radiation papers
(Xiangtan group): block preconditioners beat monolithic AMG. Markov chain / directed graph AMG (Miller thesis, NS-LAMG): nonsymmetric
singular-like M-matrices need special treatment. Graph Laplacians with skewed degrees (LAMG).

Families
  cdmix:n:eps_lo:seed     -div(eps grad u) + w.grad u on (0,1)^2, upwind, eps = 1 in random blobs and eps_lo elsewhere, recirculating wind
  cd3m:n:eps              3D swirl, eps per cell switches between 1 and eps along a smooth random field
  markov:kind:n:alpha:seed  I - alpha P^T, P the random-walk matrix of a directed graph (kind ba or ws), dangling nodes teleport
  dlap:kind:n:seed        directed graph Laplacian D_out - W^T plus 1e-4 D_out, nonsymmetric M-matrix
  mg:...                  multigroup radiation (problems.py family, new parameters)
usage: from problems_weak import load, WEAK, WEAK_HOLD
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import problems  # noqa: E402

CACHE = HERE / "cache"


def _upwind_faces(eps_face, w, h):
    """flux coefficients for one axis: diffusion eps/h^2 plus first-order upwind."""
    lo = np.maximum(w, 0) / h       # inflow from the lower neighbour
    up = np.maximum(-w, 0) / h      # inflow from the upper neighbour
    return lo, up


def _field2(n, seed, scale):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((n, n)), sigma=n * scale, mode="wrap")
    return f / f.std()


def cdmix(n, eps_lo, seed):
    h = 1 / (n + 1)
    x = (np.arange(n) + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    wx, wy = 2 * Y * (1 - X**2), -2 * X * (1 - Y**2)
    eps = np.where(_field2(n, seed, 0.06) > 0.6, 1.0, eps_lo)
    idx = np.arange(n * n).reshape(n, n)
    I, J, V = [], [], []
    d = np.zeros(n * n)
    for axis, w in ((0, wx), (1, wy)):
        a = idx.take(range(0, n - 1), axis=axis).ravel(); b = idx.take(range(1, n), axis=axis).ravel()
        e = eps.ravel()
        ef = 2 * e[a] * e[b] / (e[a] + e[b]) / h**2                       # harmonic face coefficient
        wf = 0.5 * (w.ravel()[a] + w.ravel()[b])
        lo = np.maximum(wf, 0) / h; up = np.maximum(-wf, 0) / h
        # row b receives from a (lower), row a receives from b (upper)
        I += [b, a]; J += [a, b]; V += [-(ef + lo), -(ef + up)]
        np.add.at(d, b, ef + lo); np.add.at(d, a, ef + up)
    # Dirichlet walls: missing neighbours add their face coefficient to the diagonal
    e = eps
    for axis in (0, 1):
        for side in (0, n - 1):
            s = [slice(None)] * 2; s[axis] = side
            d[idx[tuple(s)].ravel()] += 2 * e[tuple(s)].ravel() / h**2
    I.append(idx.ravel()); J.append(idx.ravel()); V.append(d)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n * n, n * n))


def cd3m(n, eps_lo, seed=4):
    h = 1 / (n + 1)
    x = (np.arange(n) + 1) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ws = (np.sin(np.pi * X) * np.cos(np.pi * Y), -np.cos(np.pi * X) * np.sin(np.pi * Y), 0.5 * np.ones_like(X))
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((n, n, n)), sigma=n * 0.08, mode="wrap"); f /= f.std()
    eps = np.where(f > 0.6, 1.0, eps_lo).ravel()
    idx = np.arange(n**3).reshape(n, n, n)
    I, J, V = [], [], []
    d = np.zeros(n**3)
    for axis, w in enumerate(ws):
        a = idx.take(range(0, n - 1), axis=axis).ravel(); b = idx.take(range(1, n), axis=axis).ravel()
        ef = 2 * eps[a] * eps[b] / (eps[a] + eps[b]) / h**2
        wf = 0.5 * (w.ravel()[a] + w.ravel()[b])
        lo = np.maximum(wf, 0) / h; up = np.maximum(-wf, 0) / h
        I += [b, a]; J += [a, b]; V += [-(ef + lo), -(ef + up)]
        np.add.at(d, b, ef + lo); np.add.at(d, a, ef + up)
        for side in (0, n - 1):
            s = [slice(None)] * 3; s[axis] = side
            d[idx[tuple(s)].ravel()] += 2 * eps[idx[tuple(s)].ravel()] / h**2
    I.append(idx.ravel()); J.append(idx.ravel()); V.append(d)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n**3, n**3))


def _digraph(kind, n, seed):
    """directed weighted edge list (i -> j)."""
    rng = np.random.default_rng(seed)
    L = problems.graph(kind, n, seed).tocoo()
    m = L.row != L.col
    i, j = L.row[m], L.col[m]
    keep = i < j
    i, j = i[keep], j[keep]
    w = rng.lognormal(0, 1, i.size)
    r = rng.random(i.size)
    # 40 % one way, 40 % the other way, 20 % both ways with different weights
    fi = np.concatenate([i[r < 0.4], j[(r >= 0.4) & (r < 0.8)], i[r >= 0.8], j[r >= 0.8]])
    fj = np.concatenate([j[r < 0.4], i[(r >= 0.4) & (r < 0.8)], j[r >= 0.8], i[r >= 0.8]])
    fw = np.concatenate([w[r < 0.4], w[(r >= 0.4) & (r < 0.8)], w[r >= 0.8], rng.lognormal(0, 1, int((r >= 0.8).sum()))])
    N = L.shape[0]
    return sp.csr_matrix((fw, (fi, fj)), shape=(N, N))


def markov(kind, n, alpha, seed):
    W = _digraph(kind, n, seed)
    N = W.shape[0]
    out = np.asarray(W.sum(1)).ravel()
    dang = out == 0
    out[dang] = 1.0
    P = sp.diags(1 / out) @ W                                            # row stochastic on non-dangling rows
    A = sp.identity(N, format="csr") - alpha * P.T.tocsr()
    return A.tocsr()


def dlap(kind, n, seed):
    W = _digraph(kind, n, seed)
    out = np.asarray(W.sum(1)).ravel()
    out[out == 0] = 1.0
    return (sp.diags(out * (1 + 1e-4)) - W.T).tocsr()


def _make(case):
    p = case.split(":")
    fam = p[0]
    if fam == "cdmix":
        A = cdmix(int(p[1]), float(p[2]), int(p[3]))
    elif fam == "cd3m":
        A = cd3m(int(p[1]), float(p[2]), int(p[3]))
    elif fam == "markov":
        A = markov(p[1], int(p[2]), float(p[3]), int(p[4]))
    elif fam == "dlap":
        A = dlap(p[1], int(p[2]), int(p[3]))
    else:
        return problems._make(case)
    A = A.tocsr(); A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, {"bs": 1}


def load(case):
    if case.split(":")[0] not in ("cdmix", "cd3m", "markov", "dlap"):
        return problems.load(case)
    CACHE.mkdir(exist_ok=True)
    key = case.replace(":", "_")
    fa, fb = CACHE / f"{key}.npz", CACHE / f"{key}.npy"
    if fa.exists():
        return sp.load_npz(fa).tocsr(), np.load(fb), {"bs": 1}
    A, b, info = _make(case)
    sp.save_npz(fa, A); np.save(fb, b)
    return A, b, info


# Chosen before any SSR result on them. Holdout: new seeds, sizes and parameters, never used while changing SSR.
WEAK = (
    ("cdmix:512:1e-5:1", "mixed regime 2D, eps 1 / 1e-5"),
    ("cdmix:512:1e-3:2", "mixed regime 2D, eps 1 / 1e-3"),
    ("cd3m:64:1e-4:4", "mixed regime 3D swirl"),
    ("cd3d:64:1e-5", "swirl 3D, eps 1e-5"),
    ("markov:ba:200000:0.99:1", "random walk, scale free, alpha 0.99"),
    ("markov:ws:200000:0.999:1", "random walk, small world, alpha 0.999"),
    ("dlap:ba:200000:1", "directed Laplacian, scale free"),
    ("dlap:ws:200000:1", "directed Laplacian, small world"),
    ("mg:32:16:1e2:1e-2:1:3", "multigroup, 16 groups"),
    ("mg:32:8:1e3:1e-3:0:4", "multigroup, thick, no conduction"),
    ("graph:ba:300000:3", "scale-free Laplacian, larger"),
)
WEAK_HOLD = (
    ("cdmix:700:1e-4:7", "mixed regime 2D, new"),
    ("cd3m:72:1e-5:9", "mixed regime 3D, new"),
    ("markov:ba:300000:0.995:5", "random walk, scale free, new"),
    ("dlap:ws:300000:6", "directed Laplacian, small world, new"),
    ("mg:32:12:3e2:3e-3:1:11", "multigroup, 12 groups, new"),
    ("graph:ba:250000:8", "scale-free Laplacian, new seed"),
)
# Second holdout, fixed on 2026-11-08 after the excess-update bug was found on WEAK_HOLD (which is no longer clean). Never used for changes.
WEAK_HOLD2 = (
    ("cdmix:600:1e-5:13", "mixed regime 2D, holdout 2"),
    ("cd3m:60:1e-3:21", "mixed regime 3D, holdout 2"),
    ("markov:ws:250000:0.9999:17", "random walk, small world, alpha 0.9999"),
    ("markov:rgg:200000:0.99:3", "random walk, geometric graph (unseen kind)"),
    ("dlap:ba:300000:19", "directed Laplacian, scale free, holdout 2"),
    ("dlap:rgg:200000:23", "directed Laplacian, geometric graph (unseen kind)"),
    ("mg:32:24:1e2:1e-2:1:29", "multigroup, 24 groups"),
    ("mg:24:8:3e3:3e-3:0:31", "multigroup, thick, no conduction, holdout 2"),
    ("graph:ba:400000:37", "scale-free Laplacian, 400k"),
)

if __name__ == "__main__":
    import time
    for c, label in WEAK + (WEAK_HOLD if "--hold" in sys.argv else ()):
        t = time.time()
        A, b, _ = load(c)
        sym = abs(A - A.T).max() <= 1e-10 * abs(A).max()
        d = A.diagonal()
        print(f"{c:28s} n={A.shape[0]:8d} nnz={A.nnz:9d} sym={sym!s:5s} mindiag={d.min():.2e} {time.time() - t:5.1f}s  {label}", flush=True)
