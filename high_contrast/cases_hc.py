"""High-contrast coefficient families that are not functions of x alone (2D, unit square).

Every family is a = χ^g with g in [0, 1] (g = 0 and 1 attained), so max a / min a = χ exactly, as in the professor's
χ^x. Discrete MMS: u* = sin(πx) sin(πy), f = A_h u*, q = 1 + 0.2 cos 2πx cos 2πy.
Each case also carries the FD-LGF split (a1(x), a2(y), feature mask), known from the construction.

  sep2     g = (x + H(y-1/2)) / 2                     separable in x and y (χ^x layers × a step in y)
  chan     g = y/2 on the background, g = 1 on 6 thin channels (3 horizontal, 3 vertical, width 0.02, gaps)
  checker  4×4 checkerboard of g ∈ {0, 1}             Kellogg-type cross points, not separable
  lognorm  g = normalised Gaussian random field, correlation length 0.1 (isotropic)
  layered  g = normalised anisotropic Gaussian field, correlation lengths 0.5 (x) and 0.03 (y): SPE10-like layers
  diag     g = H(sin(2π·8·(x+y)/√2)), 16 diagonal stripes: non-grid-aligned layering
  walls    g = 1 everywhere except low walls (g = 0, width 0.02) of a 4×4 maze with one gap per wall segment
  perc     32×32 random two-phase checkerboard, each cell g ∈ {0, 1} with p = 1/2 (percolation threshold), seed 3
  diagfine g = H(sin(2π·64·(x+y)/√2)): 128 thin diagonal stripes (effective non-grid-aligned anisotropy)
  rough    g = normalised isotropic Gaussian field with short correlation length 0.02
Random fields are generated once on a 1025×1025 node grid and sampled, so the same field is used for every n.
"""

import numpy as np

from axy_lgf.grid import SquareGrid
from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion

FAMILIES = ("sep2", "chan", "checker", "lognorm", "layered", "diag", "walls", "perc", "diagfine", "rough")
NFINE = 1024


def _grf(lx, ly, seed):
    """Gaussian random field with Gaussian covariance on (NFINE+1)^2 nodes of [0,1]^2, normalised to [0,1]."""
    m = 2 * NFINE
    k = np.fft.fftfreq(m, d=1.0 / m)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    spec = np.exp(-0.5 * ((np.pi * lx * KX) ** 2 + (np.pi * ly * KY) ** 2))
    noise = np.random.default_rng(seed).standard_normal((m, m))
    g = np.real(np.fft.ifft2(np.sqrt(spec) * np.fft.fft2(noise)))[: NFINE + 1, : NFINE + 1]
    return (g - g.min()) / (g.max() - g.min())


_CACHE = {}


def _field(name):
    if name not in _CACHE:
        _CACHE[name] = {"lognorm": lambda: _grf(0.1, 0.1, 1), "layered": lambda: _grf(0.5, 0.03, 2),
                        "rough": lambda: _grf(0.02, 0.02, 4)}[name]()
    return _CACHE[name]


def g_field(family, X, Y, n):
    """g on the (n+2)^2 node grid (boundary included); X, Y node coordinates ([j, i] = [y, x])."""
    if family == "sep2":
        return (X + (Y > 0.5)) / 2
    if family == "chan":
        g = Y / 2
        for c in (0.3, 0.5, 0.7):
            g = np.where((np.abs(Y - c) < 0.01) & (np.abs(X - 0.5) < 0.45), 1.0, g)
            g = np.where((np.abs(X - c) < 0.01) & (np.abs(Y - 0.5) < 0.45), 1.0, g)
        return g
    if family == "checker":
        return ((np.floor(4 * X).clip(0, 3) + np.floor(4 * Y).clip(0, 3)) % 2).astype(float)
    if family in ("lognorm", "layered", "rough"):
        stride = NFINE // (n + 1)
        return _field(family)[::stride, ::stride]
    if family == "diag":
        return (np.sin(2 * np.pi * 8 * (X + Y) / np.sqrt(2)) > 0).astype(float)
    if family == "perc":
        cells = np.random.default_rng(3).random((32, 32)) < 0.5
        return cells[np.floor(32 * Y).astype(int).clip(0, 31), np.floor(32 * X).astype(int).clip(0, 31)].astype(float)
    if family == "diagfine":
        return (np.sin(2 * np.pi * 64 * (X + Y) / np.sqrt(2)) > 0).astype(float)
    if family == "walls":
        g = np.ones_like(X)
        for c in (0.25, 0.5, 0.75):
            for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
                gap = lambda t: np.abs(t - (lo + hi) / 2) < 0.03
                g = np.where((np.abs(Y - c) < 0.01) & (X >= lo) & (X <= hi) & ~gap(X), 0.0, g)
                g = np.where((np.abs(X - c) < 0.01) & (Y >= lo) & (Y <= hi) & ~gap(Y), 0.0, g)
        return g
    raise ValueError(family)


class Case:
    pass


def _robust_separable(L, keep, sweeps=3):
    g2 = np.zeros(L.shape[0])
    for _ in range(sweeps):
        g1 = np.nanmedian(np.where(keep, L - g2[:, None], np.nan), axis=0)
        g2 = np.nanmedian(np.where(keep, L - g1[None, :], np.nan), axis=1)
    return g1, g2


def make_case(family, n, chi):
    c = Case()
    c.name, c.n, c.chi = family, n, chi
    c.grid = SquareGrid(n, 0.0, 1.0)
    nodes = np.arange(n + 2) * c.grid.h
    X, Y = np.meshgrid(nodes, nodes, indexing="xy")
    g = g_field(family, X, Y, n)
    c.a_nodes = chi**g
    c.a = np.ascontiguousarray(c.a_nodes[1:-1, 1:-1])
    x, y = c.grid.mesh()
    c.q = 1 + 0.2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    c.u_exact = c.u_h = np.sin(np.pi * x) * np.sin(np.pi * y)
    c.f = apply_reaction_diffusion(c.u_exact, c.a, c.grid, c.q, a_nodes=c.a_nodes)
    c.A = assemble_reaction_diffusion(c.a, c.grid, c.q, a_nodes=c.a_nodes).tocsr()
    # FD-LGF split: background b = a1(x) a2(y), features = nodes where a/b is outside [1/2, 2]
    one = np.ones(n + 2)
    if family == "sep2":
        a1, a2 = chi ** (nodes / 2), chi ** ((nodes > 0.5) / 2)
    elif family == "chan":
        a1, a2 = one, chi ** (nodes / 2)
    elif family in ("checker", "diag", "perc", "diagfine"):
        a1, a2 = one, one                       # low phase as background, high phase = features
    elif family == "walls":
        a1, a2 = chi * one, one                 # conducting background, insulating walls = features
    else:                                       # random fields: robust separable fit of log a
        g1, g2 = _robust_separable(np.log(c.a_nodes), np.ones_like(X, bool))
        a1, a2 = np.exp(g1), np.exp(g2)
    mask = np.abs(np.log(c.a_nodes) - np.log(np.outer(a2, a1))) > np.log(2.0)
    c.split = (a1, a2, mask)
    c.feature_fraction = float(mask[1:-1, 1:-1].mean())
    return c
