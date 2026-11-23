"""Parametric anisotropic families (our own 9-point discretization, ../../anisotropic/ops_aniso.py), K = chi e e^T +
(I - e e^T), q = 1 + 0.2 cos 2pi x cos 2pi y, f = A u*, u* = sin(pi x) sin(pi y) (as cases_aniso).  p in [0,1]^3 is a
normalised geometry parameter; chi is sampled separately and is NOT given to the network.

  circ   e tangent to circles around c = (0.3 + 0.4 p0, 0.3 + 0.4 p1)                      (p2 unused)
  disk   K = I except a disk at c = (0.3 + 0.4 p0, 0.3 + 0.4 p1), radius 0.15 + 0.1 p2, e at angle 30 deg
  band   K = I except a strip of width 0.04 at angle pi p0 through (0.5, 0.5 + 0.3 (p1 - 0.5))  (p2 unused)
  rot    e at constant angle pi p0 everywhere                                                  (AMG-easy control)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anisotropic"))
from ops_aniso import assemble, tensor  # noqa: E402

FAMILIES = ("circ", "disk", "band", "rot")


def field(fam, X, Y, chi, p):
    one = np.ones_like(X)
    if fam == "circ":
        cx, cy = 0.3 + 0.4 * p[0], 0.3 + 0.4 * p[1]
        return tensor(np.arctan2(Y - cy, X - cx) + np.pi / 2, chi * one, one)
    if fam == "disk":
        cx, cy, r = 0.3 + 0.4 * p[0], 0.3 + 0.4 * p[1], 0.15 + 0.1 * p[2]
        return tensor(np.pi / 6, np.where((X - cx) ** 2 + (Y - cy) ** 2 < r * r, chi, 1.0), one)
    if fam == "band":
        th, oy = np.pi * p[0], 0.5 + 0.3 * (p[1] - 0.5)
        inside = np.abs((X - 0.5) * np.sin(th) - (Y - oy) * np.cos(th)) < 0.02
        return tensor(th, np.where(inside, chi, 1.0), one)
    if fam == "rot":
        return tensor(np.pi * p[0], chi * one, one)
    raise ValueError(fam)


def make(fam, n, chi, p):
    t = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(t, t, indexing="xy")
    K11, K12, K22 = field(fam, X, Y, chi, p)
    x, y = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    q = 1 + 0.2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    A = assemble(K11, K12, K22, q, 1.0 / (n + 1))
    b = A @ (np.sin(np.pi * x) * np.sin(np.pi * y)).ravel()
    return A.tocsr(), b, np.column_stack([x.ravel(), y.ravel()])


def sample(rng, k):
    """k parameter draws: p ~ U[0,1]^3, chi log-uniform in [1e3, 1e5]."""
    return [(rng.random(3), 10 ** rng.uniform(3, 5)) for _ in range(k)]
