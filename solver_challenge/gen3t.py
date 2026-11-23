"""Our own 3T discretization (H43): linearised three-temperature energy equations on a 1/8 capsule, backward Euler with
frozen coefficients, cell-centred finite volumes, 7-point stencil, m^3 cells, unit spacing (coefficients per face).

    radiation E:  c1 E - div(k1 grad E) + e E - e T_e                 = b1    (a12 = -e)
    electron  Te: (c2 + w + f) Te - div(k2 grad Te) - f E - w T_i     = b2    (a21 = -f, a23 = -w)
    ion       Ti: (c3 + w) Ti - div(k3 grad Ti) - w T_e               = b3    (a32 = -w)

c = heat capacity / dt, e = emission coupling into the radiation row (c sigma 4 a T^3), f = absorption coupling into the
electron row (c sigma), w = electron-ion exchange. Symmetry planes x, y, z = 0 (Neumann); radiation far faces Robin
(beta per face). Three layers as in the real SolverChallenge26_01 matrix (core r < 0.336 m, shell r < 0.42 m, outside),
with the per-layer values of that matrix (report 06) as the base regime. Two knobs:
    feedback  F: f = F (c2 + w) per cell           (real matrix: |a21| / d22 ~ 2e-11)
    e-diffusion G: k2 = G (c2 + w + f) / 6 per cell (real matrix: 6 k2 / d22 ~ 2e-4 outside)
usage: from gen3t import generate; A, b = generate(m, F, G)
"""
import numpy as np
import scipy.sparse as sp

# per layer: core, shell, outside  (values of the real matrix, report 06 table)
K1 = (1e10, 1.4e7, 6.08e9)       # radiation face coefficient
C1 = (10.0, 1e4, 10.0)           # radiation c1 (rest of the radiation reaction)
E12 = (963.0, 1.8e6, 963.0)      # emission coupling e = -a12
C2 = (496.0, 3e4, 48.0)          # electron heat capacity / dt
W = (1e-13, 4.5e6, 1e-11)        # electron-ion exchange
K3 = (3e16, 7e-4, 3e11)          # ion face coefficient
C3 = (512.0, 3e4, 190.0)         # ion heat capacity / dt
BETA = 1.92e6                    # radiation Robin coefficient per far face


def layers(m):
    z, y, x = np.indices((m, m, m))
    r = np.sqrt(x**2 + y**2 + z**2) * 128.0 / m
    return np.where(r < 43, 0, np.where(r < 54, 1, 2)).ravel()


def diffusion(m, k):
    """7-point FV operator with harmonic-mean face coefficients, Neumann on all faces (lexicographic, x fastest)."""
    k = k.reshape(m, m, m)
    n = m**3
    diag = np.zeros(n)
    rows, cols, vals = [], [], []
    idx = np.arange(n).reshape(m, m, m)
    for axis, stride in ((2, 1), (1, m), (0, m * m)):
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[axis] = slice(0, m - 1); hi[axis] = slice(1, m)
        ka, kb = k[tuple(lo)], k[tuple(hi)]
        face = (2 * ka * kb / (ka + kb)).ravel()
        i, j = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()
        rows += [i, j]; cols += [j, i]; vals += [-face, -face]
        np.add.at(diag, i, face); np.add.at(diag, j, face)
    off = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return off + sp.diags(diag)


def generate(m, F, G, seed=0):
    L = layers(m)
    n = m**3
    pick = lambda t: np.asarray(t, dtype=float)[L]
    k1, c1, e, c2, w, k3, c3 = map(pick, (K1, C1, E12, C2, W, K3, C3))
    f = F * (c2 + w)
    k2 = G * (c2 + w + f) / 6.0
    robin = np.zeros((m, m, m))
    robin[:, :, -1] += BETA; robin[:, -1, :] += BETA; robin[-1, :, :] += BETA
    A11 = diffusion(m, k1) + sp.diags(c1 + e + robin.ravel())
    A22 = diffusion(m, k2) + sp.diags(c2 + w + f)
    A33 = diffusion(m, k3) + sp.diags(c3 + w)
    A = sp.bmat([[A11, sp.diags(-e), None],
                 [sp.diags(-f), A22, sp.diags(-w)],
                 [None, sp.diags(-w), A33]]).tocsr()
    A.sort_indices()
    # right-hand side: heat-capacity terms times a smooth positive "old" state of the real solution's magnitude
    z, y, x = np.indices((m, m, m)).reshape(3, -1) / m
    rng = np.random.default_rng(seed)
    old = 3e-4 * (1 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)) * (1 + 0.01 * rng.standard_normal(n))
    b = np.concatenate([c1 * old * 0.5, c2 * old, c3 * old])
    return A, b


def coefficient_diagnostics(A):
    """Coefficient-level regime numbers (no solve): median |a21| / d22 and median sum|offdiag A22| / d22."""
    N = A.shape[0] // 3
    A22 = A[N:2 * N, N:2 * N]
    d22 = A22.diagonal()
    a21 = A[N:2 * N, :N].diagonal()
    off = np.asarray(abs(A22).sum(axis=1)).ravel() - abs(d22)
    return float(np.median(abs(a21) / d22)), float(np.median(off / d22))
