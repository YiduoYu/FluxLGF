"""3D counterpart of axy_lgf's operator and PCG: -div(a grad u) + q u on (0,1)^3, zero Dirichlet.

7-point flux stencil with arithmetic face values a_{i+1/2} = (a_i + a_{i+1}) / 2 (same convention as the 2D code and
Xia's eq. (5)), boundary coefficient samples included through a_nodes of shape (n+2)^3. Arrays are indexed
[k, j, i] = [z, y, x]; flat vectors use C order, i.e. index (k n + j) n + i.
"""

from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix


class Grid3:
    def __init__(self, n):
        self.n, self.h = n, 1.0 / (n + 1)
        self.nodes = np.arange(n + 2) * self.h      # boundary included

    def mesh(self):
        """Interior coordinates (Z, Y, X), each (n, n, n)."""
        x = self.nodes[1:-1]
        return np.meshgrid(x, x, x, indexing="ij")

    def mesh_nodes(self):
        return np.meshgrid(self.nodes, self.nodes, self.nodes, indexing="ij")


def faces(a_nodes):
    """Face diffusivities: ax (n,n,n+1), ay (n,n+1,n), az (n+1,n,n)."""
    c = a_nodes
    ax = 0.5 * (c[1:-1, 1:-1, :-1] + c[1:-1, 1:-1, 1:])
    ay = 0.5 * (c[1:-1, :-1, 1:-1] + c[1:-1, 1:, 1:-1])
    az = 0.5 * (c[:-1, 1:-1, 1:-1] + c[1:, 1:-1, 1:-1])
    return ax, ay, az


def apply_op(u, a_nodes, q, h):
    ax, ay, az = faces(a_nodes)
    p = np.pad(u, 1)
    c = p[1:-1, 1:-1, 1:-1]
    fx = ax[:, :, 1:] * (p[1:-1, 1:-1, 2:] - c) - ax[:, :, :-1] * (c - p[1:-1, 1:-1, :-2])
    fy = ay[:, 1:, :] * (p[1:-1, 2:, 1:-1] - c) - ay[:, :-1, :] * (c - p[1:-1, :-2, 1:-1])
    fz = az[1:, :, :] * (p[2:, 1:-1, 1:-1] - c) - az[:-1, :, :] * (c - p[:-2, 1:-1, 1:-1])
    return -(fx + fy + fz) / h**2 + q * u


def assemble(a_nodes, q, h):
    n = a_nodes.shape[0] - 2
    ax, ay, az = faces(a_nodes)
    idx = np.arange(n**3).reshape(n, n, n)
    diag = (ax[:, :, :-1] + ax[:, :, 1:] + ay[:, :-1, :] + ay[:, 1:, :] + az[:-1] + az[1:]) / h**2 + q
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    for w, lo, hi in ((ax[:, :, 1:-1], idx[:, :, :-1], idx[:, :, 1:]),
                      (ay[:, 1:-1, :], idx[:, :-1, :], idx[:, 1:, :]),
                      (az[1:-1], idx[:-1], idx[1:])):
        v = -w.ravel() / h**2
        rows += [lo.ravel(), hi.ravel()]
        cols += [hi.ravel(), lo.ravel()]
        vals += [v, v]
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n**3,) * 2).tocsr()


def rel_l2(u, ref):
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


class Result:
    pass


def pcg(action, P, f, *, tol, maxiter):
    """Same recurrence and stopping rule as axy_lgf.solvers._pcg (relative residual, recomputed on convergence)."""
    started = perf_counter()
    u = np.zeros_like(f)
    r = f.copy()
    z = P(r)
    p = z.copy()
    rz = float(np.sum(r * z))
    fn = float(np.linalg.norm(f))
    converged, it = False, 0
    while it < maxiter and not converged:
        Ap = action(p)
        alpha = rz / float(np.sum(p * Ap))
        u = u + alpha * p
        r = r - alpha * Ap
        it += 1
        if np.linalg.norm(r) / fn <= tol:
            r = f - action(u)
            if np.linalg.norm(r) / fn <= tol:
                converged = True
                break
            z = P(r)
            p = z.copy()
            rz = float(np.sum(r * z))
            continue
        z = P(r)
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    res = Result()
    res.u, res.iterations = u, it
    res.converged = bool(np.linalg.norm(f - action(u)) / fn <= tol)
    res.solve_time = perf_counter() - started
    return res
