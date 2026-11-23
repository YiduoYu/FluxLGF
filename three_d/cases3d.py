"""3D test problems on (0,1)^3, analogues of the 2D cases A-D.

A  a = chi^x, u = sin(pi x) sin(pi y) sin(pi z), q = 1 + 0.2 cos(2 pi x) cos(2 pi y) cos(2 pi z), analytic f
   f = (3 pi^2 a + q) u - pi ln(chi) a cos(pi x) sin(pi y) sin(pi z)  -> total error vs the PDE solution.
B  step layer a = chi^{H(x-1/2)}, discrete MMS (f = A_h u*).
C  chi^x x 27 balls r = 0.08 centred on {1/4, 1/2, 3/4}^3, factor 1e4, discrete MMS.
D  chi^x x 30 random balls (r in [0.04, 0.08], factor 1e4 or 1e-4, seed 0), discrete MMS.
"""

import numpy as np

from ops3d import Grid3, apply_op, assemble

CASES = ("A_exp", "B_step", "C_exp+incl", "D_exp+rand")
GRID27 = [(cx, cy, cz, 0.08, 1e4) for cx in (0.25, 0.5, 0.75) for cy in (0.25, 0.5, 0.75) for cz in (0.25, 0.5, 0.75)]


def random_balls(count=30, seed=0):
    rng = np.random.default_rng(seed)
    balls = []
    while len(balls) < count:
        r = rng.uniform(0.04, 0.08)
        c = rng.uniform(r + 0.03, 1 - r - 0.03, 3)
        if all(np.linalg.norm(c - b[:3]) > r + b[3] + 0.02 for b in balls):
            balls.append((*c, r, 1e4 if rng.random() < 0.5 else 1e-4))
    return balls


class Case3:
    pass


def make_case(case, n, chi):
    c = Case3()
    c.name, c.n, c.chi = case, n, chi
    g = c.grid = Grid3(n)
    Zn, Yn, Xn = g.mesh_nodes()
    c.bg_a1 = chi ** ((g.nodes > 0.5) * 1.0 if case == "B_step" else g.nodes)
    factor = np.ones_like(Xn)
    balls = {"C_exp+incl": GRID27, "D_exp+rand": random_balls()}.get(case, [])
    for cx, cy, cz, r, fac in balls:
        factor[(Xn - cx) ** 2 + (Yn - cy) ** 2 + (Zn - cz) ** 2 < r**2] = fac
    c.a_nodes = np.broadcast_to(c.bg_a1, Xn.shape) * factor
    c.mask = (factor != 1.0)[1:-1, 1:-1, 1:-1]
    Z, Y, X = g.mesh()
    pi = np.pi
    c.q = 1 + 0.2 * np.cos(2 * pi * X) * np.cos(2 * pi * Y) * np.cos(2 * pi * Z)
    c.u_exact = np.sin(pi * X) * np.sin(pi * Y) * np.sin(pi * Z)
    if case == "A_exp":
        a = chi**X
        c.f = (3 * pi**2 * a + c.q) * c.u_exact - pi * np.log(chi) * a * np.cos(pi * X) * np.sin(pi * Y) * np.sin(pi * Z)
    else:
        c.f = apply_op(c.u_exact, c.a_nodes, c.q, g.h)
    c.A = assemble(c.a_nodes, c.q, g.h)
    c.u_h = None if case == "A_exp" else c.u_exact   # A: set by the benchmark (reference solve)
    return c
