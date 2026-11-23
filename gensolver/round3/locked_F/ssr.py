"""Python front of SSR (ssr.c). Input: a CSR matrix only. Levers are the module constants; eval reads them unchanged across all classes."""
import ctypes as C
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# levers
NU = 1            # smoothing sweeps pre and post on smoothed levels
EVERY = 2         # smooth every EVERY-th level (reduction halves the unknowns, so 2 levels are about one classic coarsening)
NS = 1            # samples per clique vertex
PIV = 1e-2        # pivot acceptance |a_kk| >= PIV sum |a_kj|
THETA = 0.25      # edges below THETA max do not block F (0: strict independent set)
DEG_PRIO = 1      # eliminate low degree first
COARSE = 300      # direct solve at or below this size
TOPK = 0          # 1: deterministic control, largest partner instead of a random one
SWEEPS_T = 0      # sweeps for the near-kernel vector; 0: constant, as in approximate Cholesky
SWEEPS_Y = 3      # sweeps for the left near-kernel vector, only when the transposed hierarchy is used (TRANS)
SWEEPS_SYM = 0    # 1: also relax the near-kernel vector on symmetric matrices
FLOOR = 1e-3      # smallest |t_i| / max |t|
GUARD = 0.0       # diagonal floor as a fraction of the exact Schur diagonal
COUPLE = 1        # 1: transport-plan sampling (exact row and column sums); 0: independent samples
DROP = 0.005      # coarse entries below DROP sqrt(d_i d_j) (DROPMIN: DROP min(d_i, d_j)) go to the diagonal, keeping the row sum
POS_TAU = 0.01    # LUMP 2: exact signed fill below POS_TAU sqrt(d_i d_j) is dropped to the diagonal
BETA = 1.0        # LUMP 2: fraction of same-sign entries kept on the finest level
KEEP = 4          # LUMP 3: entries per pivot kept as first-order factors
FFROW = 0         # LUMP 3: weak F-F edges go to the diagonal by row sum (1) or by absolute value (0)
LUMP = 1          # lump same-sign off-diagonals to the diagonal before elimination; 4: cluster pivots on symmetric non-M-matrices
BLOCK = 8         # LUMP 4: largest cluster
TAUF = 0.0        # LUMP 4: per-cluster update entries below TAUF sqrt(d_i d_j) are dropped
FILT = 0.0        # LUMP 4: same test on the assembled coarse matrix
DMODE = 0         # LUMP 4: compensation of a dropped entry, 0 scaled diagonal, 1 sign aware, 2 reroute through a neighbour
MAXCX = 60.0      # LUMP 4: stop when operator complexity exceeds this
NTV = 0           # LUMP 4: test vectors kept exact by the truncated rows, constant first, then relaxed
TVSW = 4          # LUMP 4: symmetric sweeps for the relaxed test vectors
TVR = 0           # LUMP 4: symmetric sweeps on the test vectors at each coarse level
RHO = 2.0         # DMODE 2: weak negative edges are rerouted through a common neighbour with this factor
SFILT = 0.0       # symmetric scalar path: reroute weak coarse entries (keeps the constant and definiteness)
MISDYN = 0        # 1: dynamic minimum-degree independent set
NFILT = 0.0       # nonsymmetric path: reroute weak coarse entries a->b through a->c->b (row and column sums kept)
DROPMIN = 1       # 1: DROP test against the smaller of the two diagonals (keeps interface couplings of high-contrast problems)
LOCALORD = 64     # > 0: F selection in random order within blocks of LOCALORD consecutive rows (cache locality)
FFMAX = 0.1       # > 0: F rows whose lumped weak F-F weight exceeds FFMAX |a_kk| go back to C
EXFIX = 1         # 1: excess update also for F rows with no C neighbours (0 reproduces the round-3 bug)
TRANS = 2         # 0: never, 1: always, 2: auto (rows with negative sums while columns are nonnegative): build on A^T, apply the transposed cycle
SMODE = 3         # 0: pre and post smoothing, 1: pre only, 2: post only, 3: post only on symmetric M-matrices
STR2 = 1          # 1: an edge is strong when it is strong for either of its rows
DOM = 50.0         # > 0: a level made only of rows with |a_kk| >= DOM sum |a_kj| when they are at least 5 %
SEED = 1

import os
_lib = C.CDLL(str(HERE / os.environ.get("SSR_LIB", "libssr.dylib")))
_d = C.POINTER(C.c_double)
_i = C.POINTER(C.c_int)
_lib.ssr_setup.restype = C.c_void_p
_lib.ssr_setup.argtypes = [C.c_int, _i, _i, _d, _d, _d, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double, C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_int]
_lib.ssr_apply.argtypes = [C.c_void_p, _d, _d]
_lib.ssr_apply_T.argtypes = [C.c_void_p, _d, _d]
_lib.ssr_stats.argtypes = [C.c_void_p, _d]
_lib.ssr_free.argtypes = [C.c_void_p]
_lib.ssr_level_info.argtypes = [C.c_void_p, C.c_int, _i]


def _dp(x):
    return x.ctypes.data_as(_d)


def near_kernel(A, sweeps):
    """Symmetric Gauss-Seidel sweeps on A t = 0 from the constant vector, without noise: noise survives in weakly coupled regions."""
    n = A.shape[0]
    t = np.ones(n)
    if sweeps:
        import pyamg.relaxation.relaxation as R
        for _ in range(sweeps):
            R.gauss_seidel(A, t, np.zeros(n), iterations=1, sweep="symmetric")
        t /= np.abs(t).max()
        small = np.abs(t) < FLOOR
        t[small] = FLOOR * np.where(t[small] < 0, -1.0, 1.0)
    return t


def _column_oriented(A):
    """True when many rows have clearly negative sums but almost no column does (Markov generators, directed Laplacians)."""
    d = np.abs(A.diagonal()) + 1e-300
    rs = np.asarray(A.sum(1)).ravel() / d
    cs = np.asarray(A.sum(0)).ravel() / d
    return (rs < -1e-8).mean() > 0.01 and (cs < -1e-8).mean() < 0.1 * (rs < -1e-8).mean()


class SSR:
    """PETSc python-PC compatible: apply(pc, x, y). Also callable on numpy arrays."""

    def __init__(self, A, **kw):
        g = globals()
        p = {k: kw.get(k, g[k]) for k in ("NU", "EVERY", "NS", "PIV", "THETA", "DEG_PRIO", "COARSE", "TOPK", "SWEEPS_T", "SWEEPS_Y", "SWEEPS_SYM", "SEED", "GUARD", "LUMP", "COUPLE", "DROP", "POS_TAU", "BETA", "KEEP", "FFROW", "BLOCK", "TAUF", "FILT", "DMODE", "MAXCX", "NTV", "TVSW", "TVR", "RHO", "SFILT", "MISDYN", "DOM", "NFILT", "STR2", "TRANS", "SMODE", "EXFIX", "FFMAX", "LOCALORD", "DROPMIN")}
        _lib.ssr_config(C.c_double(p["GUARD"]), C.c_int(p["LUMP"]), C.c_int(p["COUPLE"]), C.c_double(p["DROP"]), C.c_double(p["POS_TAU"]), C.c_double(p["BETA"]), C.c_int(p["KEEP"]), C.c_int(p["FFROW"]))
        _lib.ssr_config2(C.c_int(p["BLOCK"]), C.c_double(p["TAUF"]), C.c_double(p["FILT"]), C.c_int(p["DMODE"]), C.c_double(p["MAXCX"]))
        _lib.ssr_config3(C.c_int(p["NTV"]), C.c_int(p["TVSW"]), C.c_int(p["TVR"]))
        _lib.ssr_config4(C.c_double(p["RHO"]))
        _lib.ssr_config5(C.c_double(p["SFILT"]), C.c_int(p["MISDYN"]))
        _lib.ssr_config6(C.c_double(p["DOM"]))
        _lib.ssr_config7(C.c_double(p["NFILT"]))
        _lib.ssr_config8(C.c_int(p["STR2"]))
        _lib.ssr_config9(C.c_int(p["SMODE"]))
        _lib.ssr_config10(C.c_int(p["EXFIX"]))
        _lib.ssr_config11(C.c_double(p["FFMAX"]))
        _lib.ssr_config12(C.c_int(p["LOCALORD"]))
        _lib.ssr_config13(C.c_int(p["DROPMIN"]))
        A = A.tocsr()
        A.sort_indices()
        self.trans = False
        if p["TRANS"] == 1 or (p["TRANS"] == 2 and _column_oriented(A)):
            A = A.T.tocsr(); A.sort_indices(); self.trans = True
        self.ia, self.ja, self.a = A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.astype(np.float64)
        n = A.shape[0]
        self.n = n
        # symmetric matrices keep the constant vector; two-sided scaling (the discrete Eulerian scaling) is for nonsymmetric ones
        self.symmetric = abs(A - A.T).max() <= 1e-10 * abs(A).max()
        sy = p["SWEEPS_Y"] if self.trans else 0     # the left vector differs from the constant mainly for column-oriented matrices
        if self.symmetric and not p["SWEEPS_SYM"] or (not self.symmetric and not p["SWEEPS_T"] and not sy):
            t = y = np.ones(n)
        else:
            t = near_kernel(A, p["SWEEPS_T"])
            y = t if self.symmetric else near_kernel(A.T.tocsr(), sy or p["SWEEPS_T"])
        self.h = _lib.ssr_setup(n, self.ia.ctypes.data_as(_i), self.ja.ctypes.data_as(_i), _dp(self.a), _dp(t), _dp(y), p["NU"], p["EVERY"], p["NS"], p["PIV"], p["THETA"], p["DEG_PRIO"], p["COARSE"], p["TOPK"], p["SEED"], 0)
        s = np.zeros(4)
        _lib.ssr_stats(self.h, _dp(s))
        self.levels, self.complexity, self.coarse_n, self.transfer = int(s[0]), s[1], int(s[2]), s[3]
        if self.coarse_n < 0:
            raise RuntimeError(f"coarsest level too large: {-self.coarse_n}")
        self.z = np.empty(n)

    def __call__(self, r):
        r = np.ascontiguousarray(r, dtype=float)
        (_lib.ssr_apply_T if self.trans else _lib.ssr_apply)(self.h, _dp(r), _dp(self.z))
        return self.z.copy()

    def apply(self, pc, x, y):
        (_lib.ssr_apply_T if self.trans else _lib.ssr_apply)(self.h, _dp(x.getArray(readonly=True).astype(float)), _dp(self.z))
        y.setArray(self.z)

    def __del__(self):
        if getattr(self, "h", None):
            _lib.ssr_free(self.h)
