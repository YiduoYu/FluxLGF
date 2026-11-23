"""Test problems for the 方法二 benchmark (A-C identical to H30; D added here).

A  a = chi^x (professor's case), continuous MMS u = exact PDE solution -> total error is meaningful.
B  step layer a = chi^{H(x-1/2)}, discrete MMS (f = A_h u*, so u_h = u*).
C  chi^x layers x nine discs r = 0.08 with factor 1e4, discrete MMS.
D  chi^x layers x 20 random discs (r in [0.03, 0.07], factor 1e4 or 1e-4, seed 0), discrete MMS.
q = 1 + 0.2 cos(2 pi x) cos(2 pi y) for B-D; A uses the q of make_reaction_problem.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from axy_lgf.grid import SquareGrid
from axy_lgf.operators import apply_reaction_diffusion, assemble_reaction_diffusion, reaction_field
from axy_lgf.reaction_manufactured import make_reaction_problem

CASES = ("A_exp", "B_step", "C_exp+incl", "D_exp+rand")
NINE = [(cx, cy, 0.08, 1e4) for cx in (0.25, 0.5, 0.75) for cy in (0.25, 0.5, 0.75)]


def random_discs(count=20, seed=0):
    rng = np.random.default_rng(seed)
    discs = []
    while len(discs) < count:
        r = rng.uniform(0.03, 0.07)
        cx, cy = rng.uniform(r + 0.03, 1 - r - 0.03, 2)
        if all(np.hypot(cx - x, cy - y) > r + s + 0.02 for x, y, s, _ in discs):
            discs.append((cx, cy, r, 1e4 if rng.random() < 0.5 else 1e-4))
    return discs


class Case:
    pass


def make_case(case, n, chi):
    c = Case()
    c.name, c.n, c.chi = case, n, chi
    if case == "A_exp":
        p = make_reaction_problem(n, "continuous", coefficient="exponential", chi=chi, a_scale=1.0)
        c.grid, c.a, c.a_nodes, c.f, c.u_exact = p.grid, p.a, p.a_nodes, p.f, p.u_exact
        c.q = np.ascontiguousarray(reaction_field(p.q, p.grid))
        c.bg_a1, mask = c.a_nodes[0].copy(), np.zeros((n + 2, n + 2), bool)
    else:
        c.grid = SquareGrid(n, 0.0, 1.0)
        nodes = np.arange(n + 2) * c.grid.h
        X, Y = np.meshgrid(nodes, nodes, indexing="xy")
        c.bg_a1 = chi ** ((nodes > 0.5) * 1.0 if case == "B_step" else nodes)
        factor = np.ones_like(X)
        discs = {"C_exp+incl": NINE, "D_exp+rand": random_discs()}.get(case, [])
        for cx, cy, r, fac in discs:
            factor[(X - cx) ** 2 + (Y - cy) ** 2 < r**2] = fac
        mask = factor != 1.0
        c.a_nodes = np.broadcast_to(c.bg_a1, X.shape) * factor
        c.a = np.ascontiguousarray(c.a_nodes[1:-1, 1:-1])
        x, y = c.grid.mesh()
        c.q = 1 + 0.2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
        c.u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        c.f = apply_reaction_diffusion(c.u_exact, c.a, c.grid, c.q, a_nodes=c.a_nodes)
    c.mask = mask[1:-1, 1:-1]
    c.split = (c.bg_a1, np.ones(n + 2), mask)  # known background / features, passed to 方法二
    c.A = assemble_reaction_diffusion(c.a, c.grid, c.q, a_nodes=c.a_nodes).tocsr()
    c.u_h = c.u_exact if case != "A_exp" else spsolve(c.A.tocsc(), c.f.ravel()).reshape(n, n)
    return c
