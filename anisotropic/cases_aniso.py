"""Anisotropic tensor families K = χ e e^T + (I - e e^T) (eigenvalues χ along e, 1 across), so the anisotropy
ratio is χ. Discrete MMS as in ../high_contrast: u* = sin(πx) sin(πy), f = A_h u*, q = 1 + 0.2 cos 2πx cos 2πy.

  aligned  e = (1, 0) everywhere (grid-aligned control; FD-LGF background is exact)
  rot30    e at 30° everywhere (constant, non-grid-aligned)
  rot45    e at 45° everywhere
  circ     e = circular field around (1/2, 1/2) (magnetised-plasma type, e.g. arXiv:2301.13351)
  disk     isotropic K = I, except a disk r < 0.2 at the centre with e at 30°
  band     isotropic K = I, except a strip of width 0.04 through the centre along 30° with e along the strip

Each case carries the FD-LGF data: background diag(k1(x), k2(x)) (exact inverse by DST-y + tridiag-x) and the feature
mask where K differs from the background. aligned: background = K, no mask. rot30, rot45, circ: background = the
grid-aligned part diag(mean K11, mean K22), no mask (the anisotropy is global). disk, band: background I, mask = the
anisotropic region.
"""

import numpy as np

from ops_aniso import assemble, tensor

FAMILIES = ("aligned", "rot30", "rot45", "circ", "disk", "band")
TH = np.pi / 6


class Case:
    pass


def field(family, X, Y, chi):
    ones = np.ones_like(X)
    if family == "aligned":
        return tensor(0.0, chi * ones, ones)
    if family == "rot30":
        return tensor(TH, chi * ones, ones)
    if family == "rot45":
        return tensor(np.pi / 4, chi * ones, ones)
    if family == "circ":
        return tensor(np.arctan2(Y - 0.5, X - 0.5) + np.pi / 2, chi * ones, ones)
    if family == "disk":
        inside = (X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.2**2
        return tensor(TH, np.where(inside, chi, 1.0), ones)
    if family == "band":
        inside = np.abs((X - 0.5) * np.sin(TH) - (Y - 0.5) * np.cos(TH)) < 0.02
        return tensor(TH, np.where(inside, chi, 1.0), ones)
    raise ValueError(family)


def make_case(family, n, chi):
    c = Case()
    c.name, c.n, c.chi, c.h = family, n, chi, 1.0 / (n + 1)
    t = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(t, t, indexing="xy")
    c.K = K11, K12, K22 = field(family, X, Y, chi)
    x = X[1:-1, 1:-1]
    y = Y[1:-1, 1:-1]
    c.q = 1 + 0.2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    c.A = assemble(K11, K12, K22, c.q, c.h)
    c.u_h = np.sin(np.pi * x) * np.sin(np.pi * y)
    c.f = (c.A @ c.u_h.ravel()).reshape(n, n)
    if family in ("disk", "band"):
        c.bg = (np.ones(n + 2), np.ones(n + 2))
        c.mask = (np.abs(K12) > 0)[1:-1, 1:-1] | (np.abs(K11 - 1) > 1e-12)[1:-1, 1:-1]
    else:
        c.bg = (np.full(n + 2, K11[1:-1, 1:-1].mean()), np.full(n + 2, K22[1:-1, 1:-1].mean()))
        c.mask = np.zeros((n, n), bool)
    c.feature_fraction = float(c.mask.mean())
    return c
