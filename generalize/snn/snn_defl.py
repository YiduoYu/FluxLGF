"""SNN deflation for BoomerAMG-PCG (H46).

Network (subspace layer as in Xu & Sheng, arXiv 2404.08223): Phi(x, y; p) = bubble(x, y) * Linear(tanh MLP([x, y, p])),
M outputs = M basis functions.  The SNN 'least-squares output layer' is replaced by the Galerkin coarse solve of
deflated PCG, so the network only has to provide the subspace.

Loss (preconditioner-aware Ritz loss, uses only A, the BoomerAMG V-cycle B and the coordinates):
    L = mean of the M lowest Ritz values of the pencil (H, G), G = Q^T A Q, H = (A Q)^T B (A Q), Q = qr(Phi)
= mean generalised Ritz value of B A (A inner product) on span Phi; it is minimised by the M slowest modes of AMG.
"""
import itertools
from time import perf_counter

import numpy as np
import scipy.linalg as sla
import torch
from petsc4py import PETSc

torch.set_default_dtype(torch.float64)
AMG_OPTS = dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, strong_threshold=0.25, agg_nl=0,
                relax_type_all="symmetric-SOR/Jacobi")
_count = itertools.count()


def amg(A):
    prefix = f"s{next(_count)}_"
    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    pc = PETSc.PC().create(); pc.setOptionsPrefix(prefix); pc.setOperators(M)
    pc.setType("hypre"); pc.setHYPREType("boomeramg")
    opts = PETSc.Options()
    for k, v in AMG_OPTS.items():
        opts[prefix + "pc_hypre_boomeramg_" + k] = v
    pc.setFromOptions()
    t0 = perf_counter(); pc.setUp(); setup = perf_counter() - t0
    xv, bv = M.createVecs()

    def apply(r):
        bv.setArray(np.ascontiguousarray(r)); pc.apply(bv, xv)
        return xv.getArray().copy()
    apply.keep = (pc, M, xv, bv)
    return apply, setup


class SNN(torch.nn.Module):
    def __init__(self, n_par=3, width=128, depth=4, M=32):
        super().__init__()
        layers, d = [], 2 + n_par
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        self.body = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(width, M)

    def forward(self, X, p):
        z = torch.cat([2 * X - 1, (2 * p - 1).expand(X.shape[0], -1)], 1)
        bub = 16 * X[:, :1] * (1 - X[:, :1]) * X[:, 1:] * (1 - X[:, 1:])
        return bub * self.out(self.body(z))


class FFSNN(SNN):
    """SNN with random Fourier features of (x, y) (scale sigma) concatenated to [x, y, p]: removes the spectral bias."""
    def __init__(self, n_par=3, width=128, depth=4, M=32, n_ff=64, sigma=8.0, seed=0):
        super().__init__(n_par + 2 * n_ff, width, depth, M)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Bf", sigma * torch.randn(2, n_ff, generator=g))

    def forward(self, X, p):
        z = 2 * np.pi * X @ self.Bf
        Xf = torch.cat([torch.sin(z), torch.cos(z)], 1)
        return super().forward(X, torch.cat([p.expand(X.shape[0], -1), 0.5 * (Xf + 1)], 1))


class FCSNN(torch.nn.Module):
    """Field-coordinate SNN: Phi_k = bubble * w(x;p) * cos(k pi s(x;p)), k = 0..M-1, with a learned scalar coordinate s
    in (0,1) and a learned envelope w > 0 (one tanh MLP, 2 outputs).  Inductive bias: the near-kernel of a strongly
    anisotropic operator consists of functions of one transverse coordinate; s and w are learned from the data."""
    def __init__(self, n_par=3, width=128, depth=4, M=32):
        super().__init__()
        self.core = SNN(n_par, width, depth, 2)
        self.M = M

    def forward(self, X, p):
        z = torch.cat([2 * X - 1, (2 * p - 1).expand(X.shape[0], -1)], 1)
        o = self.core.out(self.core.body(z))
        s, w = torch.sigmoid(o[:, :1]), torch.nn.functional.softplus(o[:, 1:]) + 1e-3
        bub = 16 * X[:, :1] * (1 - X[:, :1]) * X[:, 1:] * (1 - X[:, 1:])
        return bub * w * torch.cos(np.pi * s * torch.arange(self.M, dtype=X.dtype)[None, :])


def make_net(arch, M):
    return {"mlp": SNN, "ff": FFSNN, "fc": FCSNN}[arch](M=M)


class _B(torch.autograd.Function):
    """Y = B X for the (symmetric) AMG V-cycle; backward is B again."""
    @staticmethod
    def forward(ctx, X, P):
        ctx.P = P
        return torch.from_numpy(np.column_stack([P(c) for c in X.detach().numpy().T]))

    @staticmethod
    def backward(ctx, G):
        return torch.from_numpy(np.column_stack([ctx.P(c) for c in G.detach().numpy().T])), None


def torch_sparse(A):
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, A.shape).coalesce()


def pencil(Phi, At, P):
    """Euclidean QR of the basis, then the exact Ritz pencil (H, G) of B A in the A inner product on span Phi:
    G = Q^T A Q, H = (A Q)^T B (A Q).  A unit vector has A-energy >= lambda_min(A) > 0, so no near-degenerate direction
    can fake a small Ritz value (A-normalising the columns instead puts the slow, low-energy combinations exactly in
    the ill-conditioned directions)."""
    Q, _ = torch.linalg.qr(Phi)
    AQ = torch.sparse.mm(At, Q)
    G = Q.T @ AQ; H = AQ.T @ _B.apply(AQ, P)
    return 0.5 * (H + H.T), 0.5 * (G + G.T)


def ritz_values(Phi, At, P):
    H, G = pencil(Phi, At, P)
    L = torch.linalg.cholesky(G)
    Li = torch.linalg.solve_triangular(L, torch.eye(G.shape[0]), upper=False)
    return torch.linalg.eigvalsh(Li @ H @ Li.T)


def ritz_loss(Phi, At, P, M=None):
    """Mean of the M lowest regularised Ritz values of B A on span Phi (M = all columns by default)."""
    mu = ritz_values(Phi, At, P)
    return mu[:M or len(mu)].mean()


def coarse_coefficients(net, fam_make, p, M, nc=127):
    """'feat' variant: the SNN least-squares step done on a coarse grid.  Returns C (F x M) such that Phi C spans the M
    slowest Ritz vectors of B_c A_c within the F network features."""
    import scipy.linalg as sl
    A, _, X = fam_make(nc, p); P, _ = amg(A)
    with torch.no_grad():
        Phi = net(torch.from_numpy(X), torch.as_tensor(p)[None, :])
        Q, R = torch.linalg.qr(Phi)
        H, G = pencil(Phi, torch_sparse(A), P)
    mu, V = sl.eigh(H.numpy(), G.numpy())
    return sl.solve_triangular(R.numpy(), V[:, :M])          # Phi C = Q V


def basis(net, X, p, chunk=200_000):
    with torch.no_grad():
        pt = torch.as_tensor(p)[None, :]
        return np.vstack([net(torch.from_numpy(X[i:i + chunk]), pt).numpy() for i in range(0, len(X), chunk)])


def orthonormalise(W, A, drop=1e-10):
    """A-orthonormal basis of span W via the M x M Gram matrix (no QR of the tall matrix); returns W, A W."""
    AW = A @ W
    G = 0.5 * (W.T @ AW + AW.T @ W)
    s, V = np.linalg.eigh(G); keep = s > drop * s.max()
    T = V[:, keep] / np.sqrt(s[keep])
    return W @ T, AW @ T


def dpcg(A, P, b, W=None, AW=None, tol=1e-8, maxit=3000):
    """PCG (W None) or deflated PCG with A-orthonormal W (W^T A W = I): x0 = W W^T b, z = (I - W (AW)^T) B r."""
    nb = np.linalg.norm(b)
    x = np.zeros_like(b) if W is None else W @ (W.T @ b)
    r = b - A @ x

    def Pd(r):
        z = P(r)
        return z if W is None else z - W @ (AW.T @ z)
    z = Pd(r); p = z.copy(); rz = r @ z
    for k in range(maxit):
        q = A @ p; a = rz / (p @ q); x += a * p; r -= a * q
        if np.linalg.norm(r) <= tol * nb:
            return x, k + 1
        z = Pd(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
    return x, maxit


def lanczos_A(A, P, K, seed=0):
    """K-step Lanczos for B A in the A inner product (full reorthogonalisation, A w recomputed); Ritz vectors ascending."""
    n = A.shape[0]
    v = np.random.default_rng(seed).standard_normal(n); Av = A @ v; s = np.sqrt(v @ Av)
    V = np.empty((n, K + 1), order="F"); AV = np.empty((n, K + 1), order="F"); V[:, 0] = v / s; AV[:, 0] = Av / s
    T = np.zeros((K + 1, K))
    for j in range(K):
        w = P(AV[:, j]); Aw = A @ w
        for _ in range(2):
            h = AV[:, :j + 1].T @ w; w -= V[:, :j + 1] @ h; Aw = A @ w; T[:j + 1, j] += h
        bt = np.sqrt(w @ Aw); T[j + 1, j] = bt; V[:, j + 1] = w / bt; AV[:, j + 1] = Aw / bt
    lam, Y = np.linalg.eigh(0.5 * (T[:K] + T[:K].T))
    return lam, V[:, :K] @ Y


def interp(U, nc, n):
    """Bilinear interpolation of nodal fields on the nc x nc interior grid (zero boundary) to the n x n interior grid."""
    from scipy.interpolate import RegularGridInterpolator
    tc = np.linspace(0, 1, nc + 2); tf = np.arange(1, n + 1) / (n + 1)
    Yf, Xf = np.meshgrid(tf, tf, indexing="ij"); pts = np.column_stack([Yf.ravel(), Xf.ravel()])
    out = np.empty((n * n, U.shape[1]))
    for i in range(U.shape[1]):
        g = np.zeros((nc + 2, nc + 2)); g[1:-1, 1:-1] = U[:, i].reshape(nc, nc)
        out[:, i] = RegularGridInterpolator((tc, tc), g)(pts)
    return out
