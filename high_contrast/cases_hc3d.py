"""3D high-contrast families (unit cube), a = χ^g with g in [0, 1]. Discrete MMS like the 2D families.

  sheets   thin high-a plates x ∈ {0.25, 0.5, 0.75} ± 0.01 (a = a1(x): FD-LGF background is exact)
  tubes    lattice of 27 straight high-a tubes (radius 0.02) along x, y and z through the cube, background 1
  insul    conducting background (g = 1) cut by insulating plates (g = 0, width 0.02) at x, y, z = 1/2 with a
           0.1 × 0.1 hole in the middle of each plate: 8 octants joined only through the holes
  perc3    16³ random two-phase voxels, p = 1/2 (seed 5)
  lognorm3 isotropic Gaussian random field, correlation length 0.15 (seed 6)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "three_d"))

import numpy as np

from ops3d import Grid3, apply_op, assemble

FAMILIES3 = ("sheets", "tubes", "insul", "perc3", "lognorm3")


def _grf3(n, corr, seed):
    m = 2 * (n + 2)
    k = np.fft.fftfreq(m, d=1.0 / m)
    K2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    spec = np.exp(-0.5 * (np.pi * corr) ** 2 * K2)
    noise = np.random.default_rng(seed).standard_normal((m, m, m))
    g = np.real(np.fft.ifftn(np.sqrt(spec) * np.fft.fftn(noise)))[: n + 2, : n + 2, : n + 2]
    return (g - g.min()) / (g.max() - g.min())


def g3(family, Z, Y, X, n):
    if family == "sheets":
        return np.any([np.abs(X - c) < 0.01 for c in (0.25, 0.5, 0.75)], axis=0).astype(float)
    if family == "tubes":
        g = np.zeros_like(X)
        for c1 in (0.25, 0.5, 0.75):
            for c2 in (0.25, 0.5, 0.75):
                g[(Y - c1) ** 2 + (Z - c2) ** 2 < 0.02**2] = 1      # along x
                g[(X - c1) ** 2 + (Z - c2) ** 2 < 0.02**2] = 1      # along y
                g[(X - c1) ** 2 + (Y - c2) ** 2 < 0.02**2] = 1      # along z
        return g
    if family == "insul":
        g = np.ones_like(X)
        for P, U, V in ((X, Y, Z), (Y, X, Z), (Z, X, Y)):
            plate = (np.abs(P - 0.5) < 0.01) & ~((np.abs(U - 0.5) < 0.05) & (np.abs(V - 0.5) < 0.05))
            g[plate] = 0
        return g
    if family == "perc3":
        cells = np.random.default_rng(5).random((16, 16, 16)) < 0.5
        idx = lambda T: np.floor(16 * T).astype(int).clip(0, 15)
        return cells[idx(Z), idx(Y), idx(X)].astype(float)
    if family == "lognorm3":
        return _grf3(n, 0.15, 6)
    raise ValueError(family)


class Case3:
    pass


def make_case3(family, n, chi):
    c = Case3()
    c.name, c.n, c.chi = family, n, chi
    c.grid = g = Grid3(n)
    Zn, Yn, Xn = g.mesh_nodes()
    gg = g3(family, Zn, Yn, Xn, n)
    c.a_nodes = chi**gg
    Z, Y, X = g.mesh()
    c.q = 1 + 0.2 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
    c.u_exact = c.u_h = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    c.f = apply_op(c.u_exact, c.a_nodes, c.q, g.h)
    c.A = assemble(c.a_nodes, c.q, g.h)
    # FD-LGF 3D supports a background a1(x): exact for sheets; constant background (low phase) otherwise,
    # conducting background for insul
    if family == "sheets":
        c.bg_a1, mask = c.a_nodes[0, 0, :].copy(), np.zeros_like(gg, bool)
    elif family == "insul":
        c.bg_a1, mask = chi * np.ones(n + 2), gg < 0.5
    else:
        c.bg_a1 = np.ones(n + 2)
        mask = np.abs(np.log(c.a_nodes)) > np.log(2.0)
    c.mask = mask[1:-1, 1:-1, 1:-1]
    c.feature_fraction = float(c.mask.mean())
    return c
