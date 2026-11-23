"""H45 generalisation suite: our own discretizations, several PDE classes, none tuned to SC26_01.

  lognorm3 / tubes / insul / perc3   3D scalar diffusion, contrast 1e6, n = 64 (262k unknowns)
  circ                               2D field-aligned anisotropy (ratio 1e4), n = 511 (261k)
  3t-base / 3t-hard / 3t-feed        3T radiation diffusion (gen3t.py), m = 64 (786k): SC26_01-like, G = 1, F = 1 G = 1e-2
usage: from suite import CASES, load; A, b = load(name)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("high_contrast", "three_d", "anisotropic", "solver_challenge"):
    sys.path.insert(0, str(ROOT / d))

import numpy as np  # noqa: E402

CASES = ("lognorm3", "tubes", "insul", "perc3", "circ", "3t-base", "3t-hard", "3t-feed")


def load(name, size=None):
    if name in ("lognorm3", "tubes", "insul", "perc3"):
        from cases_hc3d import make_case3
        c = make_case3(name, size or 64, 1e6)
        return c.A.tocsr(), c.f.ravel()
    if name == "circ":
        from cases_aniso import make_case
        c = make_case("circ", size or 511, 1e4)
        return c.A.tocsr(), c.f.ravel()
    from gen3t import generate
    F, G = {"3t-base": (2e-11, 2e-4), "3t-hard": (2e-11, 1.0), "3t-feed": (1.0, 1e-2)}[name]
    A, b = generate(size or 64, F, G)
    return A.tocsr(), np.asarray(b)
