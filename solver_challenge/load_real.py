"""Load the real SolverChallenge26_01 matrix (and right-hand side) and check what kind of matrix it is.

  /opt/anaconda3/envs/hypre_env/bin/python load_real.py MATRIX [RHS]
MATRIX: .mtx / .mtx.gz (Matrix Market), PETSc binary (.bin / .dat), or .npz (scipy). RHS: .mtx / .txt / .npy; if absent
b = A @ ones. Prints MD5 (challenge: c0d0a33e199a7746b0d9ba62396a0e79), order/nnz, symmetry, sign pattern
(M-matrix?), and whether the three temperatures are stored blocked ([Te; Ti; Tr], couplings at offset ±N) or
interlaced (offset ±1, ±2). to_blocked() returns the matrix in the blocked order run_sc.py uses.
"""

import hashlib
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

MD5 = "c0d0a33e199a7746b0d9ba62396a0e79"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    if path.endswith(".npz"):
        return sp.load_npz(path).tocsr()
    if path.endswith((".bin", ".dat", ".petsc")):
        from petsc4py import PETSc
        viewer = PETSc.Viewer().createBinary(path, "r")
        M = PETSc.Mat().load(viewer)
        indptr, indices, data = M.getValuesCSR()
        return sp.csr_matrix((data, indices, indptr), shape=M.getSize())
    try:
        import fast_matrix_market as fmm
        return fmm.mmread(path).tocsr()
    except ImportError:
        return scipy.io.mmread(path).tocsr()


def load_rhs(path, A):
    if path is None:
        return A @ np.ones(A.shape[0])
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".mtx", ".mtx.gz")):
        return np.asarray(scipy.io.mmread(path).todense()).ravel()
    return np.loadtxt(path).ravel()


def ordering(A):
    N = A.shape[0] // 3
    coo = A.tocoo()
    off = np.abs(coo.row.astype(np.int64) - coo.col)
    blocked = int(np.sum(off == N))
    interlaced = int(np.sum(off == 1) + np.sum(off == 2))
    return "blocked" if blocked > 0.5 * N else "interlaced" if interlaced > N else "unknown"


def to_blocked(A, b):
    if ordering(A) != "interlaced":
        return A, b
    N = A.shape[0] // 3
    perm = np.arange(3 * N).reshape(N, 3).T.ravel()
    return A[perm][:, perm].tocsr(), b[perm]


def describe(A):
    N = A.shape[0] // 3
    d = A.diagonal()
    offd = A - sp.diags(d)
    sym = abs(A - A.T).max() / abs(A).max()
    print(f"order {A.shape[0]:,}  nnz {A.nnz:,}  rel. asymmetry {sym:.2e}  ordering {ordering(A)}")
    print(f"diag > 0: {np.all(d > 0)}   positive off-diagonals: {int((offd.data > 0).sum()):,}")
    Ab, _ = to_blocked(A, np.zeros(A.shape[0]))
    db = Ab.diagonal()
    for k, name in enumerate(("T1", "T2", "T3")):
        s = slice(k * N, (k + 1) * N)
        B = Ab[s, s]
        rowsum = np.asarray(abs(B).sum(axis=1)).ravel() - np.abs(db[s])
        print(f"  block {name}: diag {db[s].min():.2e}..{db[s].max():.2e}, "
              f"min diag dominance {np.min(np.abs(db[s]) / np.maximum(rowsum, 1e-300)):.3f}")


if __name__ == "__main__":
    path = sys.argv[1]
    digest = md5(path)
    print(f"MD5 {digest} ({'matches' if digest == MD5 else 'DOES NOT match'} the challenge)")
    A = load(path)
    describe(A)
