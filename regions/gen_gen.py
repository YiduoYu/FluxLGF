"""Generalization generator (H56): gen3t's 3T equations and per-layer coefficient tables (core / shell / outside) on
geometries that bt-fd / bt-fdd were not designed around.

case "gen:m:n:p:aspect:width:delta:seed"
  n       number of capsules (core + insulating shell). n = 1: centred at the origin, core radius 43, shell 54 (in units
          of the 128-cell grid, as gen3t). n > 1: random centres in the cube, core radii 6..20, shell/core ratio 1.15..1.6.
  p       norm of the capsule shape: 2 sphere, 1 octahedron, large (e.g. 8) ~ box.
  aspect  z semi-axis / x semi-axis (1: round; 0.33 oblate, 3 prolate).
  width   transition width in cells: coefficients are interpolated log-linearly between the layers across `width`
          cells around each interface (0: sharp, as gen3t).
  delta   smooth relative variation of the face coefficients k1, k3 everywhere (k *= 1 + delta * smooth field in
          [-1, 1]), a stand-in for temperature-dependent conductivities. 0: piecewise constant.
  seed    placement / radii of random capsules and the right-hand side noise.
phase, rhs_seed (H56 time sequences): shift of the smooth variation field and right-hand side seed per time step
(geometry fixed, coefficients drift, as in successive steps of a 3T simulation with k(T)).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from gen3t import BETA, C1, C2, C3, E12, K1, K3, W, diffusion  # noqa: E402


def level(m, n, p, aspect, width, seed):
    """Continuous layer index s in [0, 2] per cell (0 core, 1 shell, 2 outside); nearest capsule wins (min)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((m, m, m)) * 128.0 / m                     # as gen3t.layers (128-grid units)
    if n == 1:
        caps = [((0.0, 0.0, 0.0), 43.0, 54.0 / 43.0)]
    else:
        caps = [(tuple(rng.uniform(0, 128, 3)), rng.uniform(6, 20), rng.uniform(1.15, 1.6)) for _ in range(n)]
    s = np.full((m, m, m), 2.0)
    for (cx, cy, cz), rc, ratio in caps:
        ax = np.abs(x - cx) / rc
        ay = np.abs(y - cy) / rc
        az = np.abs(z - cz) / (rc * aspect)
        d = (ax**p + ay**p + az**p) ** (1.0 / p)                     # 1 on the core boundary, ratio on the shell boundary
        if width > 0:
            w = width * 128.0 / m / rc                               # width in units of d
            si = np.clip((d - 1) / w + 0.5, 0, 1) + np.clip((d - ratio) / w + 0.5, 0, 1)
        else:
            si = (d >= 1).astype(float) + (d >= ratio)
        s = np.minimum(s, si)
    return s.ravel()


def per_cell(table, s):
    """Log-linear interpolation of a per-layer table (core, shell, outside) at fractional layer index s."""
    lt = np.log(np.asarray(table, dtype=float))
    i = np.minimum(np.floor(s).astype(int), 1)
    t = s - i
    return np.exp((1 - t) * lt[i] + t * lt[i + 1])


def generate(m, n, p, aspect, width, delta, seed, F=2e-11, G=2e-4, phase=0.0, rhs_seed=None):
    s = level(m, n, p, aspect, width, seed)
    N = m**3
    k1, c1, e, c2, w, k3, c3 = (per_cell(t, s) for t in (K1, C1, E12, C2, W, K3, C3))
    if delta:
        z, y, x = np.indices((m, m, m)).reshape(3, -1) / m
        field = np.sin(2.3 * x + 0.7 + phase) * np.cos(1.9 * y + 0.3 + 0.6 * phase) * np.sin(2.9 * z + 1.1 + 0.3 * phase)
        k1, k3 = k1 * (1 + delta * field), k3 * (1 + delta * field)
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
    z, y, x = np.indices((m, m, m)).reshape(3, -1) / m
    rng = np.random.default_rng(seed if rhs_seed is None else rhs_seed)
    old = 3e-4 * (1 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)) * (1 + 0.01 * rng.standard_normal(N))
    b = np.concatenate([c1 * old * 0.5, c2 * old, c3 * old])
    return A, b
