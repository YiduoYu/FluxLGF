"""Surrogate for SolverChallenge26_01 (the real matrix is not public): one backward-Euler / Picard step of the 3D
three-temperature (3T) energy equations on a 128^3 cell-centred grid, finite-volume 7-point stencil.

    c_e (Te - Te0)/dt - div(K_e grad Te) + w_ei (Te - Ti) + w_er (Te - Tr) = 0
    c_i (Ti - Ti0)/dt - div(K_i grad Ti) - w_ei (Te - Ti)                  = 0
    c_r (Tr - Tr0)/dt - div(K_r grad Tr) - w_er (Te - Tr)                  = 0

Unknowns are ordered by temperature, [Te; Ti; Tr] (cells in C order [z, y, x] inside each block), so
A = [[De + W_ei + W_er, -W_ei, -W_er], [-W_ei, Di + W_ei, 0], [-W_er, 0, Dr + W_er]], symmetric positive definite.
Sparsity is exactly that of the challenge: 3 x (7 N - 6 m^2) + 4 N = 52,133,888 nonzeros for m = 128 (N = m^3).

Coefficients follow the usual ICF test models (Spitzer-type electron/ion conduction, opacity-limited radiation
diffusion, Te-dependent exchange; e.g. Yue et al., CiCP 32 (2022); Shu et al., AAMM 13 (2021)):
    K_e = A_e Te^2.5,  K_i = A_i Ti^2.5,  K_r = A_r Tr^3 / rho^2 (thin gas: large, dense shell: small),
    w_ei = A_ei rho^2 Te^-1.5,  w_er = A_er rho^2 Te^-0.5,  c_* = rho * const.
State: spherical capsule (gas r < 0.3, dense shell 0.3-0.38, ablator gas outside) with a radiation wave driven from
the face x = 0 (Tr = 1 there, Dirichlet), Te, Ti behind it. Face transmissibilities = harmonic means; Dirichlet
faces add 2K/h^2 to the diagonal (Tr's boundary value 1 at x = 0 goes into b).
"""

import numpy as np
from scipy.sparse import coo_matrix


def state(m):
    h = 1.0 / m
    c = (np.arange(m) + 0.5) * h
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    rho = np.where(r < 0.3, 0.05, np.where(r < 0.38, 20.0, 0.5))
    front = 0.35 + 0.08 * np.sin(6 * np.pi * Y) * np.sin(4 * np.pi * Z)             # wrinkled wave front
    wave = 1.0 / (1.0 + np.exp((X - front) / 0.02))
    Tr = 1e-3 + wave * (1 - 0.6 * (r < 0.38) * (r >= 0.3))
    Te = 1e-3 + 0.9 * wave + 0.3 * np.exp(-(r / 0.08) ** 2)                            # + hot spot
    Ti = 1e-3 + 0.5 * wave + 0.6 * np.exp(-(r / 0.08) ** 2)
    return h, rho, Te, Ti, Tr


def _diffusion(K, h, diag_extra, dirichlet_lo_x=None):
    """FV 7-point matrix (COO parts) of -div(K grad) on m^3 cells, zero Dirichlet except optional value at x = 0."""
    m = K.shape[0]
    idx = np.arange(m**3).reshape(m, m, m)
    rows, cols, vals = [], [], []
    diag = diag_extra.copy()
    rhs = np.zeros_like(K)
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        t = 2 * K[lo] * K[hi] / (K[lo] + K[hi]) / h**2                                      # harmonic mean
        rows += [idx[lo].ravel(), idx[hi].ravel()]
        cols += [idx[hi].ravel(), idx[lo].ravel()]
        vals += [-t.ravel(), -t.ravel()]
        diag[lo] += t
        diag[hi] += t
        for side in (0, -1):                                                             # boundary faces
            s = [slice(None)] * 3
            s[ax] = side
            s = tuple(s)
            diag[s] += 2 * K[s] / h**2
            if ax == 2 and side == 0 and dirichlet_lo_x is not None:
                rhs[s] += 2 * K[s] / h**2 * dirichlet_lo_x
    return rows, cols, vals, diag, rhs


def build(m=128, dt=1e-3, scaled=False, emission=False):
    """emission=True linearises the electron-radiation exchange as emission a(Te^4 - Tr^4) instead of w(Te - Tr):
    w_er (g_e Te - g_r Tr), g_e = 2Te^3/(Te^3 + Tr^3), g_r = 2Tr^3/(Te^3 + Tr^3), so the Te-Tr coupling is
    nonsymmetric (the usual source of nonsymmetry in 3T matrices; still a column-diagonally-dominant Z-matrix).
    scaled=True divides every equation by its c/dt (each row by the cell's heat capacity per time step), as codes
    that normalise the energy equations do: the matrix becomes nonsymmetric (same sparsity, still a Z-matrix)."""
    h, rho, Te, Ti, Tr = state(m)
    N = m**3
    Ke, Ki, Kr = 1.0 * Te**2.5, 0.05 * Ti**2.5, 1.0 * Tr**3 / rho**2
    wei, wer = 50.0 * rho**2 * Te**-1.5, 5.0 * rho**2 * Te**-0.5
    ce, ci, cr = 1.0 * rho, 1.0 * rho, 0.2 * Tr**3
    ge = gr = np.ones_like(Te)
    if emission:
        ge, gr = 2 * Te**3 / (Te**3 + Tr**3), 2 * Tr**3 / (Te**3 + Tr**3)
    blocks = [(Ke, ce / dt + wei + wer * ge, Te, ce / dt, None), (Ki, ci / dt + wei, Ti, ci / dt, None),
              (Kr, cr / dt + wer * gr, Tr, cr / dt, 1.0)]
    R, C, V, b = [], [], [], []
    for k, (K, extra, T0, cdt, bc) in enumerate(blocks):
        rows, cols, vals, diag, rhs = _diffusion(K, h, extra, bc)
        R += [r + k * N for r in rows] + [np.arange(N) + k * N]
        C += [c + k * N for c in cols] + [np.arange(N) + k * N]
        V += vals + [diag.ravel()]
        b.append((cdt * T0 + rhs).ravel())
    i = np.arange(N)
    for (p, q), wpq, wqp in (((0, 1), wei, wei), ((0, 2), wer * gr, wer * ge)):         # exchange couplings
        R += [i + p * N, i + q * N]
        C += [i + q * N, i + p * N]
        V += [-wpq.ravel(), -wqp.ravel()]
    A = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(3 * N, 3 * N)).tocsr()
    A.sum_duplicates()
    b = np.concatenate(b)
    if scaled:
        from scipy.sparse import diags
        w = 1.0 / np.concatenate([ce.ravel(), ci.ravel(), cr.ravel()]) * dt
        A, b = (diags(w) @ A).tocsr(), w * b
    return A, b


if __name__ == "__main__":
    import sys
    from time import perf_counter

    m = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t = perf_counter()
    A, b = build(m)
    print(f"m={m} order={A.shape[0]:,} nnz={A.nnz:,} build {perf_counter() - t:.1f}s "
          f"symmetric={abs(A - A.T).max() < 1e-12 * abs(A).max()}")
    d = A.diagonal()
    for k, name in enumerate("e i r".split()):
        s = slice(k * m**3, (k + 1) * m**3)
        print(f"  block {name}: diag range {d[s].min():.2e} .. {d[s].max():.2e}")
