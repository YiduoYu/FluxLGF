"""Generalization suite: sparse matrices with no grid information handed to the solver.

Every case returns (A csr, b, info). info["bs"] is used only to give BoomerAMG its nodal option in the tuned baseline;
the new solver never sees it. Matrices are cached under cache/ as they take seconds to generate.
Names: <family>:<params>. See DEV and HOLDOUT.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "fluxmg"), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
CACHE = HERE / "cache"


# ---- finite elements on tetrahedra ----------------------------------------------------------------------------------
def box_tets(nodes, jitter=0.0, seed=0, stretch=None):
    """Kuhn split of a box into 6 tets per cell; nodes = (nx, ny, nz). Returns points, tets."""
    nx, ny, nz = nodes
    g = [np.linspace(0, 1, m) for m in nodes]
    if stretch is not None:
        g[2] = np.linspace(0, 1, nz) * stretch
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nx * ny * nz).reshape(nodes)
    if jitter:
        rng = np.random.default_rng(seed)
        inner = (X > 0) & (X < 1) & (Y > 0) & (Y < 1) & (Z > 0) & (Z < g[2][-1])
        d = np.array([1 / (nx - 1), 1 / (ny - 1), g[2][1] - g[2][0]])
        pts[inner.ravel()] += jitter * d * rng.uniform(-1, 1, (int(inner.sum()), 3))
    c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[:-1, 1:, :-1], idx[1:, 1:, :-1], idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[:-1, 1:, 1:], idx[1:, 1:, 1:]]
    paths = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    return pts, np.vstack([np.column_stack([c[a].ravel() for a in p]) for p in paths])


def _grads(pts, tets):
    v = pts[tets]
    J = v[:, 1:] - v[:, 0][:, None]                              # rows: edge vectors
    det = np.linalg.det(J)
    invJ = np.linalg.inv(J)                                      # columns: gradients of basis 1..3
    G = np.empty((len(tets), 4, 3))
    G[:, 1:] = np.transpose(invJ, (0, 2, 1))
    G[:, 0] = -G[:, 1:].sum(axis=1)
    return G, np.abs(det) / 6


def p1_diffusion(pts, tets, K=None, coef=None):
    """P1 stiffness with tensor K (3x3) and per-tet scalar coef."""
    G, vol = _grads(pts, tets)
    GK = G if K is None else G @ K
    w = vol if coef is None else vol * coef
    Ke = np.einsum("eik,ejk,e->eij", GK, G, w)
    I = np.repeat(tets, 4, axis=1).ravel()
    J = np.tile(tets, (1, 4)).ravel()
    return sp.csr_matrix((Ke.ravel(), (I, J)), shape=(len(pts), len(pts)))


def p1_elasticity(pts, tets, nu, coef=None):
    """Isotropic linear elasticity, P1 vector elements, dofs interlaced xyz."""
    G, vol = _grads(pts, tets)
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    w = vol if coef is None else vol * coef
    # K[ia,jb] = lam Gia Gjb + mu (Gib Gja + delta_ab Gi.Gj)
    dot = np.einsum("eik,ejk->eij", G, G)
    Ke = lam * np.einsum("eia,ejb->eiajb", G, G) + mu * np.einsum("eib,eja->eiajb", G, G)
    for a in range(3):
        Ke[:, :, a, :, a] += mu * dot
    Ke = Ke * w[:, None, None, None, None]
    dof = (3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 12)
    Ke = Ke.reshape(len(tets), 12, 12)
    I = np.repeat(dof, 12, axis=1).ravel()
    J = np.tile(dof, (1, 12)).ravel()
    n = 3 * len(pts)
    return sp.csr_matrix((Ke.ravel(), (I, J)), shape=(n, n))


def rot3(a, b, c):
    def R(ax, t):
        C, S = np.cos(t), np.sin(t)
        i, j = (ax + 1) % 3, (ax + 2) % 3
        M = np.eye(3); M[i, i] = C; M[j, j] = C; M[i, j] = -S; M[j, i] = S
        return M
    return R(0, a) @ R(1, b) @ R(2, c)


def keep(A, mask):
    """Dirichlet elimination: keep unknowns where mask is true."""
    ix = np.flatnonzero(mask)
    A = A.tocsr()[ix][:, ix]
    A.sum_duplicates(); A.eliminate_zeros()
    return A.tocsr()


def boundary_mask(pts):
    lo, hi = pts.min(0), pts.max(0)
    return np.any(np.isclose(pts, lo) | np.isclose(pts, hi), axis=1)


# ---- random mesh ----------------------------------------------------------------------------------------------------
def delaunay(n, seed):
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    m = int(round(n ** (1 / 3)))
    # uniform points plus a clustered patch, so mesh size varies by about 10 and element quality is poor
    a = rng.random((int(0.8 * n), 3))
    c = 0.3 + 0.05 * rng.standard_normal((int(0.2 * n), 3))
    c = np.clip(c, 0.01, 0.99)
    g = np.linspace(0, 1, max(m // 3, 6))
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    box = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    box = box[boundary_mask(box)]
    pts = np.vstack([box, a, c])
    tri = Delaunay(pts)
    t = tri.simplices
    # drop slivers of zero volume
    v = pts[t]
    vol = np.abs(np.linalg.det(v[:, 1:] - v[:, 0][:, None])) / 6
    t = t[vol > 1e-12]
    return pts, t


# ---- finite differences ---------------------------------------------------------------------------------------------
def lap1(n):
    return sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")


def upwind1(v, h):
    """Upwind first derivative v * d/dx for velocity field applied row-wise; returns (lower, upper) coefficient diagonals."""
    return np.maximum(v, 0) / h, np.maximum(-v, 0) / h


def cd2d(n, eps, wind):
    """-eps Lap u + w.grad u, upwind, on (0,1)^2, Dirichlet. n interior points per axis."""
    h = 1 / (n + 1)
    x = (np.arange(n) + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    if wind == "rec":
        wx, wy = 2 * Y * (1 - X**2), -2 * X * (1 - Y**2)        # recirculating, the standard double-glazing wind
    else:
        wx, wy = np.cos(0.7) * np.ones_like(X), np.sin(0.7) * np.ones_like(X)
    idx = np.arange(n * n).reshape(n, n)
    I, J, V = [], [], []
    d = np.full(n * n, 4 * eps / h**2)
    for axis, w in ((0, wx), (1, wy)):
        lo, up = upwind1(w, h)
        d += (lo + up).ravel()
        a = idx.take(range(0, n - 1), axis=axis).ravel(); b = idx.take(range(1, n), axis=axis).ravel()
        lo_b = lo.take(range(1, n), axis=axis).ravel(); up_a = up.take(range(0, n - 1), axis=axis).ravel()
        I += [b, a]; J += [a, b]; V += [-(eps / h**2 + lo_b), -(eps / h**2 + up_a)]
    I.append(idx.ravel()); J.append(idx.ravel()); V.append(d)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n * n, n * n))


def cd3d(n, eps):
    h = 1 / (n + 1)
    x = (np.arange(n) + 1) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ws = (np.sin(np.pi * X) * np.cos(np.pi * Y) , -np.cos(np.pi * X) * np.sin(np.pi * Y), 0.5 * np.ones_like(X))   # cellular swirl plus lift
    idx = np.arange(n**3).reshape(n, n, n)
    I, J, V = [], [], []
    d = np.full(n**3, 6 * eps / h**2)
    for axis, w in enumerate(ws):
        lo, up = upwind1(w, h)
        d += (lo + up).ravel()
        a = idx.take(range(0, n - 1), axis=axis).ravel(); b = idx.take(range(1, n), axis=axis).ravel()
        lo_b = lo.take(range(1, n), axis=axis).ravel(); up_a = up.take(range(0, n - 1), axis=axis).ravel()
        I += [b, a]; J += [a, b]; V += [-(eps / h**2 + lo_b), -(eps / h**2 + up_a)]
    I.append(idx.ravel()); J.append(idx.ravel()); V.append(d)
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n**3, n**3))


def fourth2d(n):
    """Fourth-order 13-point Laplacian: positive off-diagonals, not an M-matrix."""
    D = sp.diags([1 / 12 * np.ones(n - 2), -16 / 12 * np.ones(n - 1), 30 / 12 * np.ones(n), -16 / 12 * np.ones(n - 1), 1 / 12 * np.ones(n - 2)], [-2, -1, 0, 1, 2], format="csr")
    I = sp.identity(n, format="csr")
    return (sp.kron(D, I) + sp.kron(I, D)).tocsr()


def helmholtz2d(n, kh):
    """5-point Laplacian minus (kh)^2 I: indefinite, kh = k h."""
    L = lap1(n)
    I = sp.identity(n, format="csr")
    return (sp.kron(L, I) + sp.kron(I, L) - kh**2 * sp.identity(n * n)).tocsr()


# ---- graphs ---------------------------------------------------------------------------------------------------------
def laplacian_of(i, j, w, n, shift):
    W = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    W = W + W.T
    W.setdiag(0); W.eliminate_zeros()
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0                                          # isolated vertices
    return (sp.diags(deg * (1 + shift)) - W).tocsr()


def graph(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "ws":                                             # small world on a ring lattice in 3D
        m = int(round(n ** (1 / 3))); n = m**3
        idx = np.arange(n).reshape(m, m, m)
        I, J = [], []
        for ax in range(3):
            I.append(idx.take(range(m - 1), axis=ax).ravel()); J.append(idx.take(range(1, m), axis=ax).ravel())
        i, j = np.concatenate(I), np.concatenate(J)
        s = rng.random(i.size) < 0.05                            # 5 % of the edges rewired at random
        j = j.copy(); j[s] = rng.integers(0, n, int(s.sum()))
        w = np.exp(rng.normal(0, 1, i.size))
    elif kind == "ba":                                           # preferential attachment, scale free
        m0 = 4
        targets = list(range(m0))
        rep = list(range(m0))
        I, J = [], []
        rep_arr = np.empty(2 * m0 * n, dtype=np.int64); rep_arr[:m0] = np.arange(m0); nr = m0
        for v in range(m0, n):
            t = np.unique(rep_arr[rng.integers(0, nr, m0)])
            I.append(np.full(t.size, v)); J.append(t)
            k = t.size
            rep_arr[nr:nr + k] = t; rep_arr[nr + k:nr + 2 * k] = v; nr += 2 * k
        i, j = np.concatenate(I), np.concatenate(J)
        w = np.ones(i.size)
    elif kind == "rgg":                                          # random geometric graph in 3D
        from scipy.spatial import cKDTree
        p = rng.random((n, 3))
        r = (12 / (4 / 3 * np.pi * n)) ** (1 / 3)
        pairs = cKDTree(p).query_pairs(r, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        w = np.ones(i.size)
    else:
        raise ValueError(kind)
    return laplacian_of(i, j, w, n, 1e-6)


# ---- case table -----------------------------------------------------------------------------------------------------
def _make(case):
    p = case.split(":")
    fam = p[0]
    info = {"bs": 1}
    if fam == "poisson3d":
        m = int(p[1]); L = lap1(m); I = sp.identity(m, format="csr")
        A = (sp.kron(sp.kron(L, I), I) + sp.kron(sp.kron(I, L), I) + sp.kron(sp.kron(I, I), L)).tocsr()
    elif fam == "rot2d":
        from pyamg.gallery import diffusion_stencil_2d, stencil_grid
        m, eps, deg = int(p[1]), float(p[2]), float(p[3])
        A = stencil_grid(diffusion_stencil_2d(epsilon=eps, theta=np.deg2rad(deg), type="FE"), (m, m), format="csr")
    elif fam == "rot3d":
        m, e1, e2 = int(p[1]), float(p[2]), float(p[3])
        Q = rot3(0.7, 0.5, 0.3); K = Q @ np.diag([1, e1, e2]) @ Q.T
        pts, t = box_tets((m, m, m)); A = keep(p1_diffusion(pts, t, K=K), ~boundary_mask(pts))
    elif fam == "fe3d":                                          # Delaunay, optional two-material jump
        n, chi, seed = int(p[1]), float(p[2]), int(p[3])
        pts, t = delaunay(n, seed)
        coef = None
        if chi != 1:
            c = pts[t].mean(1); coef = np.where(np.sin(9 * c[:, 0]) * np.sin(9 * c[:, 1]) * np.sin(9 * c[:, 2]) > 0.2, chi, 1.0)
        A = keep(p1_diffusion(pts, t, coef=coef), ~boundary_mask(pts))
    elif fam == "stretch":                                       # isotropic PDE, mesh aspect ratio from a stretched z axis
        m, ratio = int(p[1]), float(p[2])
        pts, t = box_tets((m, m, m), jitter=0.2, stretch=1 / ratio)
        A = keep(p1_diffusion(pts, t), ~boundary_mask(pts))
    elif fam == "cd2d":
        A = cd2d(int(p[1]), float(p[2]), p[3])
    elif fam == "cd3d":
        A = cd3d(int(p[1]), float(p[2]))
    elif fam == "elast":                                         # cantilever, clamped at x = 0, jittered mesh
        m, nu, chi = int(p[1]), float(p[2]), float(p[3])
        pts, t = box_tets((m, m, m), jitter=0.2, seed=3)
        coef = None
        if chi != 1:
            c = pts[t].mean(1); coef = np.where(np.sin(7 * c[:, 0]) * np.sin(7 * c[:, 1]) * np.sin(7 * c[:, 2]) > 0.3, chi, 1.0)
        K = p1_elasticity(pts, t, nu, coef=coef)
        free = np.repeat(pts[:, 0] > 1e-12, 3)
        A = keep(K, free); info["bs"] = 3
    elif fam == "graph":
        A = graph(p[1], int(p[2]), int(p[3]))
    elif fam == "fourth":
        A = fourth2d(int(p[1]))
    elif fam == "helm":
        A = helmholtz2d(int(p[1]), float(p[2]))
    else:                                                        # anything the FluxMG loader knows
        from try_one import load as fl
        A, b, dims, F = fl(case)
        info["bs"] = 1
        return A.tocsr(), np.asarray(b), info
    A = A.tocsr(); A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, info


def load(case):
    CACHE.mkdir(exist_ok=True)
    key = case.replace(":", "_")
    fa, fb = CACHE / f"{key}.npz", CACHE / f"{key}.npy"
    if fa.exists():
        info = {"bs": 3 if case.startswith("elast") else 1}
        return sp.load_npz(fa).tocsr(), np.load(fb), info
    A, b, info = _make(case)
    if A.shape[0] > 0:
        sp.save_npz(fa, A); np.save(fb, b)
    return A, b, info


# (case, class). No knobs by class in the new solver.
DEV = (
    ("poisson3d:64", "3D Poisson 7-pt"),
    ("aniso:64:100", "aligned anisotropy"),
    ("rot2d:512:1e-3:30", "rotated anisotropy 2D, 9-pt FE"),
    ("rot3d:48:1e-2:1e-4", "rotated anisotropy 3D, P1"),
    ("fe3d:150000:1:1", "unstructured P1, graded Delaunay"),
    ("fe3d:150000:1e6:2", "unstructured P1, jump 1e6"),
    ("stretch:48:100", "stretched mesh, aspect 100"),
    ("lognorm3", "lognormal contrast 1e6"),
    ("tubes", "conducting tubes"),
    ("perc3", "percolating contrast"),
    ("fourth:512", "4th-order 13-pt, not M-matrix"),
    ("cd2d:512:1e-3:rec", "recirculating flow 2D, eps 1e-3"),
    ("cd2d:512:1e-5:rec", "recirculating flow 2D, eps 1e-5"),
    ("cd3d:64:1e-3", "swirl flow 3D, eps 1e-3"),
    ("elast:41:0.3:1", "elasticity nu 0.3"),
    ("elast:41:0.45:1", "elasticity nu 0.45"),
    ("elast:33:0.3:1e4", "elasticity, stiffness jump 1e4"),
    ("graph:ws:200000:1", "small-world graph Laplacian"),
    ("graph:ba:200000:1", "scale-free graph Laplacian"),
    ("graph:rgg:200000:1", "random geometric graph"),
    ("g3t:64:2e-11:2e-4", "3T, SC26_01-like"),
    ("gen:64:3:2:1:0:2e-2:1", "3T, 3 capsules"),
    ("mg:32:8:1e2:1e-2:1:7", "multigroup, AMG hard"),
)
# Never looked at while choosing changes. New seeds, parameters and sizes.
HOLDOUT = (
    ("poisson3d:100", "3D Poisson, larger"),
    ("rot2d:700:1e-4:60", "rotated anisotropy, new angle"),
    ("rot3d:60:1e-3:1e-3", "rotated anisotropy 3D, new"),
    ("fe3d:300000:1e4:5", "unstructured jump, larger"),
    ("stretch:60:1000", "stretched mesh, aspect 1000"),
    ("cd2d:700:1e-4:rec", "recirculating, new eps"),
    ("cd2d:512:1e-3:uni", "uniform oblique flow"),
    ("cd3d:80:1e-4", "swirl 3D, new eps"),
    ("elast:51:0.4:1", "elasticity, larger"),
    ("graph:ws:300000:2", "small-world, new seed"),
    ("graph:rgg:300000:2", "geometric graph, new seed"),
    ("gen:64:10:2:1:0:2e-2:5", "3T, 10 capsules"),
    ("mg:32:4:1e2:1:1:8", "multigroup, new"),
    ("insul", "insulating inclusions"),
)
# Reported but outside the score: both solvers are expected to struggle.
STRETCH = (
    ("helm:512:0.2", "indefinite Helmholtz, kh 0.2"),
)


if __name__ == "__main__":
    import time
    for c, label in DEV + HOLDOUT + STRETCH:
        if "--dev" in sys.argv and (c, label) not in DEV:
            continue
        t = time.time()
        A, b, info = load(c)
        d = A.diagonal()
        asym = abs(A - A.T).sum() / abs(A).sum()
        print(f"{c:28s} n={A.shape[0]:8d} nnz/row={A.nnz / A.shape[0]:6.1f} asym={asym:.1e} dmin={d.min():.1e} dmax={d.max():.1e} {time.time() - t:.1f}s", flush=True)
