"""Ablation of FD-LGF-3T on the surrogate (iterations to ‖r‖/‖b‖ < 1e-8, scipy CG, x0 = 0): from the original FD-LGF
background to the full method, one ingredient at a time.

  /opt/anaconda3/envs/hypre_env/bin/python ablation_fdlgf3t.py m [m ...]      -> results/ablation.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from fdlgf3t import FDLGF3T
from fdlgf_coarse import FDLGFCoarse
from run_sc import FDLGFBackgroundPC
from surrogate3t import build

HERE = Path(__file__).resolve().parent


class _Vec:
    def __init__(self, a):
        self.a = a

    def getArray(self, readonly=False):
        return self.a


def background_only(A, m):
    ctx = FDLGFBackgroundPC(A, m)
    def apply(r):
        out = _Vec(np.empty_like(r))
        ctx.apply(None, _Vec(r), out)
        return out.a
    return apply


VARIANTS = [
    ("FD-LGF background G (block means)", lambda A, m: background_only(A, m)),
    ("+ scalar Jacobi scaling (Green-Jacobi)", lambda A, m: FDLGFCoarse(A, m, coarse=False).M),
    ("+ region coarse space (box 4, decades)", lambda A, m: FDLGFCoarse(A, m, box=4, decade=1.0, coupled=True)),
    ("3x3 point-block scaling, no coarse", lambda A, m: FDLGFCoarse(A, m, block=True, coarse=False).M),
    ("+ box coarse space (b = 4, LU)", lambda A, m: FDLGF3T(A, m, box=4, coarse="lu", lu_size=10**9)),
    ("+ block-Jacobi smoothing (LU coarse)", lambda A, m: FDLGF3T(A, m, box=4, coarse="lu", lu_size=10**9, omega=0.6)),
    ("FD-LGF-3T: multilevel W-cycle", lambda A, m: FDLGF3T(A, m, box=4, coarse="ml", lu_size=5000, omega=0.6, cycles=2)),
]


def main(sizes):
    rows = []
    for m in sizes:
        A, b = build(m)
        for name, make in VARIANTS:
            P = make(A, m)
            it = [0]
            x, info = cg(A, b, rtol=1e-8, maxiter=3000, M=LinearOperator(A.shape, P),
                         callback=lambda xk: it.__setitem__(0, it[0] + 1))
            res = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
            rows.append(dict(m=m, variant=name, iterations=it[0], converged=res < 1e-8, rel_res=res))
            print(rows[-1], flush=True)
            with (HERE / "results" / "ablation.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [16, 32, 64])
