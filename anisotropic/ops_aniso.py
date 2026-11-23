"""Tensor-coefficient operator -div(K grad u) + q u on (0,1)^2, zero Dirichlet; arrays indexed [j, i] = [y, x].

Energy form (K11, K12, K22 nodal arrays with boundary, shape (n+2, n+2)):
    E(u) = sum_xfaces K11_f (dx u)^2 + sum_yfaces K22_f (dy u)^2 + 2 sum_nodes K12 (cx u)(cy u) + sum q u^2,
face values = arithmetic mean of the two nodal values (as in the scalar 5-point flux stencil), cx, cy = central
differences. A = (1/2) Hessian of E is the standard symmetric 9-point stencil; with K = a I it is exactly the scalar
flux stencil of axy_lgf. Since (cx u)^2 <= mean of the two adjacent (dx u)^2 and K12^2 <= K11 K22 nodewise,
Cauchy-Schwarz gives E(u) >= sum q u^2, so A is SPD for q > 0. A is not an M-matrix when K12 != 0.
"""

import numpy as np
from scipy.sparse import diags, identity, kron


def tensor(theta, lam1, lam2):
    """K = lam1 e e^T + lam2 (I - e e^T), e = (cos theta, sin theta). Returns (K11, K12, K22)."""
    c, s = np.cos(theta), np.sin(theta)
    return lam1 * c * c + lam2 * s * s, (lam1 - lam2) * c * s, lam1 * s * s + lam2 * c * c


def assemble(K11, K12, K22, q, h):
    n = K11.shape[0] - 2
    I = identity(n, format="csr")
    d = diags([np.ones(n), -np.ones(n)], [0, -1], shape=(n + 1, n)) / h       # n nodes -> n+1 faces
    c = diags([np.ones(n - 1), -np.ones(n - 1)], [1, -1]) / (2 * h)            # central difference
    Dx, Dy, Cx, Cy = kron(I, d), kron(d, I), kron(I, c), kron(c, I)
    w11 = 0.5 * (K11[1:-1, :-1] + K11[1:-1, 1:]).ravel()
    w22 = 0.5 * (K22[:-1, 1:-1] + K22[1:, 1:-1]).ravel()
    w12 = diags(K12[1:-1, 1:-1].ravel())
    A = Dx.T @ diags(w11) @ Dx + Dy.T @ diags(w22) @ Dy + Cx.T @ w12 @ Cy + Cy.T @ w12 @ Cx + diags(q.ravel())
    return A.tocsr()
