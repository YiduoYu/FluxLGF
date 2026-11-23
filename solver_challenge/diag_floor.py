"""Attainable-accuracy check for the real matrix: how small can ||b - A x|| / ||b|| be in float64?
For a vector x, the rounding error of computing A x is about eps * (|A| |x|) row by row. We (1) take x = a test
solution of the right size and check || b - A x || / ||b|| where b = A x is formed in extended precision, and (2) report
eps ||(|A| |x|)|| / ||b|| as the floor estimate, per temperature block.   usage: python diag_floor.py A.npz b.npy"""
import sys

import numpy as np
import scipy.sparse as sp

A = sp.load_npz(sys.argv[1]).tocsr()
b = np.load(sys.argv[2])
N = A.shape[0] // 3
absA = abs(A)
eps = np.finfo(float).eps
nb = np.linalg.norm(b)
# block-wise right-hand-side norms
print("||b|| =", f"{nb:.3e}", " per block:", [f"{np.linalg.norm(b[k*N:(k+1)*N]):.3e}" for k in range(3)])
# scale of the solution: x ~ b / (diag - |offdiag| within the block) is the smooth-mode estimate; use a range of guesses
d = A.diagonal()
for name, x in (("x = 1", np.ones(A.shape[0])),
                ("x = b / rowsum(A)", b / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1e-300))):
    ax = absA @ np.abs(x)
    floor = eps * np.linalg.norm(ax) / nb
    blk = [eps * np.linalg.norm(ax[k*N:(k+1)*N]) / nb for k in range(3)]
    # actual rounding: b' = A x in long double vs float64
    xl = x.astype(np.longdouble)
    Ac = A.tocoo()
    bl = np.zeros(A.shape[0], dtype=np.longdouble)
    np.add.at(bl, Ac.row, Ac.data.astype(np.longdouble) * xl[Ac.col])
    r = (bl - (A @ x).astype(np.longdouble)).astype(float)
    print(f"{name}: |x| range {np.abs(x).min():.2e}..{np.abs(x).max():.2e}; eps*|| |A||x| ||/||b|| = {floor:.2e} "
          f"(blocks {', '.join(f'{v:.1e}' for v in blk)}); actual float64 error of A x / ||b|| = {np.linalg.norm(r)/nb:.2e}")
