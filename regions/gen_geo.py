"""gen3t (own 3T discretization, H43) with other capsule geometries and heat-capacity (time-step) scalings (H52 tests).
generate(m, r_core, r_shell, cap, F, G): radii in units of the 128-cell grid (base 43, 54); cap multiplies the heat
capacity / dt terms C1, C2, C3 (cap 10 ~ dt / 10)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
import gen3t  # noqa: E402

BASE = (gen3t.C1, gen3t.C2, gen3t.C3, gen3t.layers)


def generate(m, r_core, r_shell, cap, F, G):
    def layers(mm):
        z, y, x = np.indices((mm, mm, mm))
        r = np.sqrt(x**2 + y**2 + z**2) * 128.0 / mm
        return np.where(r < r_core, 0, np.where(r < r_shell, 1, 2)).ravel()
    C1, C2, C3, lay = BASE
    gen3t.layers = layers
    gen3t.C1, gen3t.C2, gen3t.C3 = (tuple(cap * c for c in C) for C in (C1, C2, C3))
    try:
        return gen3t.generate(m, F, G)
    finally:
        gen3t.C1, gen3t.C2, gen3t.C3, gen3t.layers = C1, C2, C3, lay
