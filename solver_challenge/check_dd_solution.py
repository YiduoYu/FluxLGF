"""Independent check of a double-double solution written by the C solver (SC26_P01_DUMP=prefix):
||b - A (hi + lo)|| / ||b|| in double-double (ir_fp128.DDResidual), and the same for hi alone (float64 solution).
usage: python check_dd_solution.py A.npz b.npy prefix"""
import sys

import numpy as np
import scipy.sparse as sp

from ir_fp128 import DDResidual

A = sp.load_npz(sys.argv[1]).tocsr()
b = np.load(sys.argv[2])
hi = np.fromfile(sys.argv[3] + "_hi.bin")
lo = np.fromfile(sys.argv[3] + "_lo.bin")
res = DDResidual(A, b)
nb = np.linalg.norm(b)
r = res(hi, lo)
N = len(b) // 3
print(f"double-double solution: ||b-Ax||/||b|| = {np.linalg.norm(r) / nb:.3e}  per block "
      + " ".join(f"{np.linalg.norm(r[k * N:(k + 1) * N]) / nb:.1e}" for k in range(3)))
print(f"hi only (float64 x):    ||b-Ax||/||b|| = {np.linalg.norm(res(hi, 0 * lo)) / nb:.3e}")
