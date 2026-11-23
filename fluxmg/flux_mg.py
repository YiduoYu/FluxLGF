"""FluxMG: AMG-free structured multigrid in flux form, for 7-point cell-coupled diffusion-reaction systems on a tensor grid.

Operator, per field f and cell i (no diagonal is stored, so the 1e17 entries of a core never cancel):
    (A x)_{f,i} = sum_g C_{fg,i} x_{g,i} + sum_{faces} k_f (x_{f,i} - x_{f,nb})
C holds the row excess (exact row sum of the diagonal block) and the local couplings between fields.
Hierarchy: 2x2x2 boxes, piecewise-constant transfers. Galerkin coarsening only adds numbers: coarse faces are sums of fine
faces across the box boundary, coarse C is the sum of fine C, and the operator stays 7-point on every level.
Cycle: block red-black Gauss-Seidel smoothing, V, W or K (Notay, two GCR steps per coarse solve).
"""
import ctypes as ct
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

# Tunable levers
CYCLE = "K"          # V | W | K
NU_PRE = 1           # red-black sweeps before the coarse correction
NU_POST = 1          # black-red sweeps after it
ALPHA = 1.0          # scale of coarse faces (0.5 is the rediscretization value for a uniform field)
K_THRESH = 0.25      # K-cycle: skip the second GCR step if the residual dropped below this factor
COARSEST = 32        # solve directly when a level has at most this many cells
REACTION = "cluster" # sum: coarse C is the plain sum | cluster: cells weakly tied to the box's strongest cluster count in series
THETA = 0.01         # a face is strong if k >= THETA * (largest face inside the box)
DEFLATE = True       # one constant vector per diffusion-dominated region, per field
TAU = 10.0           # a face joins two cells of a region if k >= TAU * max(reaction of the two cells)
RHO = 0.01           # ... and k >= RHO * (strongest face at either cell), so a weak layer separates regions
MIN_REGION = 8       # smaller regions get no vector
MAX_REGIONS = 400    # if more are found, the smallest get no vector
ML_DEFLATE = True    # also correct with region vectors after each coarse solve on levels below the finest
ML_MIN_REGION = 2    # smallest region kept on those levels
REDUCE = True        # multi-field matrices: automatic reduction (AMFR rules) with one Core cycle per group
TAU_D, TAU_G, PHI, Q = 2e-2, 0.7, 1e-3, 0.95   # the AMFR elimination and pairwise grouping thresholds (multifield/amfr.py, H50)
PROBE_BOXES = 8      # grouping probe: boxes per axis of the aggregated system
PROBE_TAU = 0.7      # join all coupled fields if the block-triangular solve contracts slower than this on the aggregated system
ADAPT_ROUNDS = 0     # rounds of adaptive splitting of regions at setup (0: none)
ADAPT_CYCLES = 8     # cycles on a random vector per round
ADAPT_JUMP = 0.1     # cut a face if the slow error jumps there by more than this fraction of its local size
LINE = True          # adaptive zebra line smoothing on lines holding an anisotropic cell
ANISO = 30.0         # a cell is anisotropic along an axis if k there > ANISO * (other faces + reaction)
LINE_RUN = 1         # a line is smoothed only if it holds this many consecutive anisotropic cells
SEMI = True          # do not coarsen an axis whose typical face is much weaker than the strongest axis
SEMI_RATIO = 0.25    # axis is coarsened if median k >= SEMI_RATIO * largest median k
INTERP = "const"     # const: piecewise constant | linear: operator-dependent linear along coarsened axes (series resistors)
SINK = 0.0           # linear prolongation: a cell's reaction times SINK pulls its weight back to its own box
RESTRICT = "sum"     # sum: add up each box | transpose: transpose of the linear prolongation (INTERP = linear only)
COARSE_FACE = "sum"  # sum: ALPHA * sum of crossing faces | scaled: sum / coarsening ratio of the axis (rediscretization)
                     # | series: half + crossing + half face in series, summed across (a lower bound: cuts channels)

# The 12 faces inside a 2x2x2 box, cell id = 4 bz + 2 by + bx
_EDGES = [(c, c + 1, 2) for c in (0, 2, 4, 6)] + [(c, c + 2, 1) for c in (0, 1, 4, 5)] + [(c, c + 4, 0) for c in (0, 1, 2, 3)]


FAST = True          # run the cycle in C (kernel.c); False keeps the numpy version as the reference

_HERE = Path(__file__).resolve().parent
_PD = ct.POINTER(ct.c_double)


class _CLevel(ct.Structure):
    _fields_ = [(n, ct.c_int) for n in ("F", "nz", "ny", "nx", "n", "rz", "ry", "rx")] + \
               [(n, _PD) for n in ("kz", "ky", "kx", "C", "Dinv", "scale")] + \
               [("nlines", ct.c_int), ("line_axis", ct.POINTER(ct.c_int)), ("line_field", ct.POINTER(ct.c_int)),
                ("line_mask", ct.POINTER(ct.POINTER(ct.c_ubyte))), ("line_seg", ct.POINTER(ct.POINTER(ct.c_ubyte)))] + \
               [(f"w{i}", _PD) for i in range(1, 11)] + \
               [("rK", ct.c_int), ("rnkeep", ct.c_int), ("rkeep", ct.POINTER(ct.c_long)), ("rlab", ct.POINTER(ct.c_long)),
                ("rwt", _PD), ("rea", _PD), ("red", _PD), ("rcoef", _PD)] + \
               [("np", ct.c_int), ("ptrans", ct.c_int), ("pax", ct.c_int * 3), ("pown", _PD * 3), ("pnb", _PD * 3),
                ("pb1", _PD), ("pb2", _PD)]


class _CHier(ct.Structure):
    _fields_ = [("nlev", ct.c_int), ("cycle", ct.c_int), ("nu_pre", ct.c_int), ("nu_post", ct.c_int), ("k_thresh", ct.c_double),
                ("lev", ct.POINTER(_CLevel)), ("bn", ct.c_int), ("ba", _PD), ("bd", _PD), ("K", ct.c_int), ("nkeep", ct.c_int),
                ("keep", ct.POINTER(ct.c_long)), ("lab", ct.POINTER(ct.c_long)), ("ea", _PD), ("ed", _PD), ("coef", _PD), ("wt", _PD),
                ("q", _PD), ("r1", _PD), ("z", _PD), ("r2", _PD), ("setup_ops", ct.c_long)]


_LIB = None


def _kernel():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = _HERE / "libfluxmg.dylib"
    src = _HERE / "kernel.c"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["clang", "-O3", "-march=native", "-shared", "-fPIC", str(src), "-o", str(lib)], check=True)
    _LIB = ct.CDLL(str(lib))
    return _LIB


def _p(a):
    return a.ctypes.data_as(_PD)


def _series(a, b):
    """Conductance of a and b in series; 0 if either is 0, the other if one is inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 / a + 1.0 / b
        return np.where(s > 0, 1.0 / s, 0.0) * ((a > 0) & (b > 0))


def _box_sum(v, ax, box):
    """Add up v along axis ax over the boxes given by box (index of each point's box)."""
    cs = list(v.shape)
    cs[ax] = int(box[-1]) + 1
    out = np.zeros(cs)
    sl = [np.arange(s) for s in v.shape]
    sl[ax] = box
    np.add.at(out, np.ix_(*sl), v)
    return out


def _interp_weights(k, n, sink=None):
    """k (..., n-1): faces along the last axis. Weight of a fine cell's own box and of the next box on its far side, from the
    resistor chain: half of the box's inner face to its own coarse point, one face plus half the next box's inner face to the other."""
    shp = k.shape[:-1] + (n,)
    gown, gnb = np.zeros(shp), np.zeros(shp)
    i = np.arange(n)
    ev, od = i[(i % 2 == 0) & (i + 1 < n)], i[i % 2 == 1]
    gown[..., ev] = 2.0 * k[..., ev]
    gown[..., od] = 2.0 * k[..., od - 1]
    e2 = ev[ev >= 2]
    gnb[..., e2] = _series(k[..., e2 - 1], 2.0 * k[..., e2 - 2])
    o2 = od[od + 2 < n]
    gnb[..., o2] = _series(k[..., o2], 2.0 * k[..., o2 + 1])
    o1 = od[od + 2 == n]                                                     # next box is a single cell: its point is its centre
    gnb[..., o1] = k[..., o1]
    if sink is not None:                                                     # reaction ties a cell to its own box (piecewise constant)
        gown += sink
    if n % 2:                                                                # a single-cell box: its own point
        gown[..., n - 1], gnb[..., n - 1] = 1.0, 0.0
    tot = gown + gnb
    own = np.where(tot > 0, gown / np.where(tot > 0, tot, 1.0), 1.0)
    return own, np.where(tot > 0, gnb / np.where(tot > 0, tot, 1.0), 0.0)


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def row_excess(B):
    """Row sums of a CSR matrix in double-double (entries of 1e17 cancelling to 1e2 keep their digits)."""
    n = B.shape[0]
    s, e = np.zeros(n), np.zeros(n)
    B = B.tocsr()
    cnt = np.diff(B.indptr)
    for j in range(int(cnt.max())):
        rows = np.flatnonzero(cnt > j)
        s[rows], t = two_sum(s[rows], B.data[B.indptr[rows] + j])
        e[rows] += t
    return s + e


class GTH:
    """Dense LU of a row diagonally dominant M-matrix from its off-diagonal magnitudes and row excess (GTH-like elimination,
    Alfa, Xue, Ye 2002): pivots are sums, nothing is subtracted, so 1e17 entries next to a tiny excess lose no digits."""

    def __init__(self, off, excess):
        a = off.copy()
        np.fill_diagonal(a, 0.0)
        v = np.maximum(excess, 0.0).copy()
        n = a.shape[0]
        self.n, self.a, self.d = n, a, np.zeros(n)
        for k in range(n):
            d = v[k] + a[k, k + 1:].sum()
            self.d[k] = d
            if d <= 0:
                continue
            l = a[k + 1:, k] / d
            a[k + 1:, k] = l
            a[k + 1:, k + 1:] += np.outer(l, a[k, k + 1:])
            np.fill_diagonal(a[k + 1:, k + 1:], 0.0)
            v[k + 1:] += l * v[k]

    def solve(self, b):
        x = np.asarray(b, dtype=float).copy()
        n, a, d = self.n, self.a, self.d
        for k in range(n):                                               # L has positive multipliers
            x[k + 1:] += a[k + 1:, k] * x[k]
        for k in range(n - 1, -1, -1):
            if d[k] <= 0:
                x[k] = 0.0
                continue
            x[k] = (x[k] + a[k, k + 1:] @ x[k + 1:]) / d[k]
        return x


class Level:
    """faces[a]: array (F, nz, ny, nx) with one less entry along axis a+1; C: (nz, ny, nx, F, F)."""

    def __init__(self, faces, C):
        self.faces, self.C = faces, C
        self.lines = []
        self.F, self.shape = C.shape[3], C.shape[:3]
        F = self.F
        diag = np.zeros((F,) + self.shape)
        for a, k in enumerate(faces):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[a + 1], hi[a + 1] = slice(None, -1), slice(1, None)
            diag[tuple(lo)] += k
            diag[tuple(hi)] += k
        self.sumk = diag
        D = C.copy()
        idx = np.arange(F)
        D[..., idx, idx] += np.moveaxis(diag, 0, -1)
        self.Dinv = np.linalg.inv(D)
        self.scale = 1.0 / np.moveaxis(D[..., idx, idx], -1, 0)         # for scaled inner products
        z, y, x = np.indices(self.shape)
        self.color = ((z + y + x) % 2 == 0, (z + y + x) % 2 == 1)
        self.ratio = self.choose_ratio()
        if LINE:
            self.setup_lines()

    def choose_ratio(self):
        """Coarsening factor per axis."""
        n = self.shape
        if not SEMI:
            return tuple(2 if m > 1 else 1 for m in n)
        med = [float(np.median(k)) if k.size else 0.0 for k in self.faces]
        top = max(med)
        ratio = tuple(2 if n[a] > 1 and (top == 0 or med[a] >= SEMI_RATIO * top) else 1 for a in range(3))
        return ratio if ratio != (1, 1, 1) else tuple(2 if m > 1 else 1 for m in n)

    def setup_lines(self):
        """For each field and axis: (axis, field, masks of the two zebra colours), for lines with an anisotropic cell."""
        for f in range(self.F):
            around = []
            for a, k in enumerate(self.faces):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[a], hi[a] = slice(None, -1), slice(1, None)
                top = np.zeros(self.shape)
                top[tuple(lo)] = k[f]
                pad = np.zeros(self.shape); pad[tuple(hi)] = k[f]
                around.append(np.maximum(top, pad))
            r = np.abs(self.C[..., f, f])
            for a in range(3):
                if self.shape[a] < 2:
                    continue
                other = sum(around[b] for b in range(3) if b != a) + r
                cell = around[a] > ANISO * other
                run = LINE_RUN if self.shape[a] >= LINE_RUN else self.shape[a]
                windows = np.lib.stride_tricks.sliding_window_view(cell, run, axis=a)
                line = windows.all(axis=-1).any(axis=a)
                if line.any():
                    grid = np.indices(line.shape).sum(axis=0) % 2
                    seg = cell.copy()                                         # the anisotropic cells and one cell each way
                    lo, hi = [slice(None)] * 3, [slice(None)] * 3
                    lo[a], hi[a] = slice(None, -1), slice(1, None)
                    seg[tuple(lo)] |= cell[tuple(hi)]
                    seg[tuple(hi)] |= cell[tuple(lo)]
                    self.lines.append((a, f, (line & (grid == 0), line & (grid == 1)), seg))

    def neighbours(self, x):
        """sum over faces of k x_nb, per field."""
        out = np.zeros_like(x)
        for a, k in enumerate(self.faces):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[a + 1], hi[a + 1] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            out[lo] += k * x[hi]
            out[hi] += k * x[lo]
        return out

    def apply(self, x):
        """A x in flux form."""
        out = np.einsum("zyxfg,gzyx->fzyx", self.C, x, optimize=True)
        for a, k in enumerate(self.faces):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[a + 1], hi[a + 1] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            flux = k * (x[lo] - x[hi])
            out[lo] += flux
            out[hi] -= flux
        return out

    def sweep(self, x, f, order):
        F = self.F
        for c in order:
            mask = self.color[c]
            rhs = f + self.neighbours(x)
            if F == 1:
                x[0][mask] = self.Dinv[..., 0, 0][mask] * rhs[0][mask]
            else:
                x[:, mask] = np.einsum("nfg,gn->fn", self.Dinv[mask], rhs[:, mask], optimize=True)
        if self.lines:
            for a, field, masks, seg in self.lines if order[0] == 0 else reversed(self.lines):
                for c in order:
                    self.line_sweep(x, f, field, a, masks[c], seg)
        return x

    def line_sweep(self, x, rhs_all, field, axis, mask, seg):
        """Solve exactly along the marked segments of selected lines of one field; elimination uses additions only (GTH-like).
        Scalar loops: this is the reference for the C version, used on small grids."""
        F = self.F
        xf = x[field]
        kz, ky, kx = (k[field] for k in self.faces)
        shape = self.shape
        rhs = rhs_all[field]
        C = self.C
        step = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][axis]

        def along(c):                                                     # face from cell c to c + step along the axis
            return self.faces[axis][field][c]

        def cell(pos, t):
            c = list(pos)
            c.insert(axis, t)
            return tuple(c)
        n = shape[axis]
        for pos in zip(*np.nonzero(mask)):
            t = 0
            while t < n:
                if not seg[cell(pos, t)]:
                    t += 1
                    continue
                t0 = t
                while t + 1 < n and seg[cell(pos, t + 1)]:
                    t += 1
                t1 = t
                t += 1
                m = t1 - t0 + 1
                b, d, u = np.zeros(m), np.zeros(m), np.zeros(m)
                q = 0.0
                for i in range(m):
                    c = cell(pos, t0 + i)
                    z, y, xx = c
                    sv, rr = C[c][field, field], rhs[c]
                    for g in range(F):
                        if g != field:
                            rr -= C[c][field, g] * x[g][c]
                    if axis != 0:
                        if z > 0:
                            rr += kz[z - 1, y, xx] * xf[z - 1, y, xx]; sv += kz[z - 1, y, xx]
                        if z < shape[0] - 1:
                            rr += kz[z, y, xx] * xf[z + 1, y, xx]; sv += kz[z, y, xx]
                    if axis != 1:
                        if y > 0:
                            rr += ky[z, y - 1, xx] * xf[z, y - 1, xx]; sv += ky[z, y - 1, xx]
                        if y < shape[1] - 1:
                            rr += ky[z, y, xx] * xf[z, y + 1, xx]; sv += ky[z, y, xx]
                    if axis != 2:
                        if xx > 0:
                            rr += kx[z, y, xx - 1] * xf[z, y, xx - 1]; sv += kx[z, y, xx - 1]
                        if xx < shape[2] - 1:
                            rr += kx[z, y, xx] * xf[z, y, xx + 1]; sv += kx[z, y, xx]
                    if i == 0 and t0 > 0:                                 # neighbours beyond the segment stay fixed
                        k = along(cell(pos, t0 - 1)); rr += k * xf[cell(pos, t0 - 1)]; sv += k
                    if i == m - 1 and t1 < n - 1:
                        k = along(cell(pos, t1)); rr += k * xf[cell(pos, t1 + 1)]; sv += k
                    u[i] = along(c) if i < m - 1 else 0.0
                    if i == 0:
                        q, b[0] = sv, rr
                    else:
                        b[i] = rr + u[i - 1] * b[i - 1] / d[i - 1]
                        q = sv + u[i - 1] * (q / d[i - 1])
                    d[i] = max(q + u[i], 1e-300)
                nxt = 0.0
                for i in range(m - 1, -1, -1):
                    nxt = (b[i] + u[i] * nxt) / d[i]
                    xf[cell(pos, t0 + i)] = nxt

    def box_index(self):
        return [np.arange(n) // self.ratio[a] for a, n in enumerate(self.shape)]

    def coarse_shape(self):
        return tuple((n + self.ratio[a] - 1) // self.ratio[a] for a, n in enumerate(self.shape))

    def coarsen(self):
        F = self.F
        cshape = self.coarse_shape()
        idx = self.box_index()
        C = np.zeros(cshape + (F, F))
        fine = self.C
        if REACTION == "cluster" and self.ratio == (2, 2, 2) and not any(n % 2 for n in self.shape):
            w = self.series_weights()
            fine = self.C * np.sqrt(np.einsum("zyxf,zyxg->zyxfg", w, w))
        np.add.at(C, np.ix_(idx[0], idx[1], idx[2]), fine)
        faces = []
        for a, k in enumerate(self.faces):
            n = self.shape[a]
            cross = np.arange(1, n - 1, 2) if self.ratio[a] == 2 else np.arange(n - 1)   # faces between neighbouring boxes
            sel = [slice(None)] * 4
            sel[a + 1] = cross
            ks = k[tuple(sel)]
            if COARSE_FACE == "series" and self.ratio[a] == 2 and cross.size:
                kl = np.moveaxis(k, a + 1, -1)
                left = 2.0 * kl[..., cross - 1]                                  # inner face of the lower box, half of it
                right = np.full(left.shape, np.inf)
                two = cross + 1 < n - 1                                          # the upper box has two cells
                right[..., two] = 2.0 * kl[..., cross[two] + 1]
                ks = np.moveaxis(_series(_series(left, kl[..., cross]), right), -1, a + 1)
            cs = list(cshape)
            cs[a] = cshape[a] - 1
            out = np.zeros((F,) + tuple(cs))
            ii = [idx[0], idx[1], idx[2]]
            ii[a] = np.arange(len(cross))
            for f in range(F):
                np.add.at(out[f], np.ix_(*ii), ks[f])
            if COARSE_FACE == "slab" and self.ratio[a] == 2 and cross.size:
                kl = np.moveaxis(k, a + 1, -1)                                   # each box's inner faces summed across, in series
                two = cross + 1 < n - 1
                sides = []
                for faces_at, valid in ((cross - 1, np.ones(cross.size, bool)), (np.where(two, cross + 1, 0), two)):
                    s = np.zeros_like(out)
                    v = np.moveaxis(kl[..., faces_at], -1, a + 1)
                    for f in range(F):
                        np.add.at(s[f], np.ix_(*ii), v[f])
                    sm = np.moveaxis(s, a + 1, -1)
                    sm[..., ~valid] = np.inf                                     # a single-cell box adds no resistance
                    sides.append(np.moveaxis(2.0 * sm, -1, a + 1))
                out = _series(_series(sides[0], out), sides[1])
            scale = {"series": 1.0, "slab": 1.0, "scaled": 1.0 / self.ratio[a]}.get(COARSE_FACE, ALPHA)
            faces.append(scale * out)
        if INTERP == "linear":
            self.setup_interp()
        return Level(faces, C)

    def setup_interp(self):
        """Weights of the operator-dependent linear prolongation, applied axis by axis: x, then y, then z.
        The step along axis a works on a grid still coarse in the axes before a; its faces are the fine faces along a, summed
        over the boxes of those axes."""
        idx = self.box_index()
        self.pw = []
        for a in (2, 1, 0):
            if self.ratio[a] != 2:
                continue
            kb = self.faces[a]
            sb = np.maximum(np.moveaxis(np.einsum("zyxff->zyxf", self.C), -1, 0), 0.0) * SINK if SINK else None
            for b in [b for b in range(a) if self.ratio[b] == 2]:             # transverse axes still coarse at this step
                kb = _box_sum(kb, b + 1, idx[b])
                if sb is not None:
                    sb = _box_sum(sb, b + 1, idx[b])
            own, nb = _interp_weights(np.moveaxis(kb, a + 1, -1), self.shape[a],
                                      None if sb is None else np.moveaxis(sb, a + 1, -1))
            self.pw.append((a, np.ascontiguousarray(np.moveaxis(own, -1, a + 1)), np.ascontiguousarray(np.moveaxis(nb, -1, a + 1))))

    def series_weights(self):
        """Weight (nz, ny, nx, F) of each cell's reaction in its box: 1 inside the cluster holding the box's strongest face,
        g / (g + r) outside it, g being the conductance from the cell to that cluster. All additions and divisions."""
        F = self.F
        nz, ny, nx = self.shape
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        w = np.ones(self.shape + (F,))
        for f in range(F):
            kz, ky, kx = (k[f] for k in self.faces)
            edges = np.zeros((cz, cy, cx, 12))
            for e, (a, b, axis) in enumerate(_EDGES):
                bz, by, bx = a // 4, (a // 2) % 2, a % 2
                k = (kz, ky, kx)[axis]
                edges[..., e] = k[bz::2, by::2, bx::2][:cz, :cy, :cx]
            kmax = edges.max(axis=-1)
            strong = (edges >= THETA * kmax[..., None]) & (kmax[..., None] > 0)
            lab = np.broadcast_to(np.arange(8), (cz, cy, cx, 8)).copy()
            for _ in range(4):
                for e, (a, b, _ax) in enumerate(_EDGES):
                    m = np.minimum(lab[..., a], lab[..., b])
                    lab[..., a] = np.where(strong[..., e], m, lab[..., a])
                    lab[..., b] = np.where(strong[..., e], m, lab[..., b])
            top = np.argmax(edges, axis=-1)
            ends = np.array([_EDGES[e][0] for e in range(12)])[top]
            head = np.take_along_axis(lab, ends[..., None], axis=-1)
            inside = lab == head
            g = np.zeros((cz, cy, cx, 8))
            for e, (a, b, _ax) in enumerate(_EDGES):
                g[..., b] += np.where(inside[..., a] & ~inside[..., b], edges[..., e], 0.0)
                g[..., a] += np.where(inside[..., b] & ~inside[..., a], edges[..., e], 0.0)
            r = np.zeros((cz, cy, cx, 8))
            for c in range(8):
                r[..., c] = self.C[c // 4::2, (c // 2) % 2::2, c % 2::2, f, f][:cz, :cy, :cx]
            box = np.where(inside | (r <= 0), 1.0, g / np.maximum(g + r, 1e-300))
            for c in range(8):
                w[c // 4::2, (c // 2) % 2::2, c % 2::2, f] = box[..., c]
        return w

    def restrict(self, r):
        if INTERP == "linear" and RESTRICT == "transpose":
            idx = self.box_index()
            for a, own, nb in reversed(self.pw):
                n = self.shape[a]
                i = np.arange(n)
                nc = (n + 1) // 2
                J = np.clip(np.where(i % 2 == 0, idx[a] - 1, idx[a] + 1), 0, nc - 1)
                rm = np.moveaxis(r, a + 1, -1)
                wo, wn = np.moveaxis(own, a + 1, -1) * rm, np.moveaxis(nb, a + 1, -1) * rm
                out = np.zeros(rm.shape[:-1] + (nc,))
                for t in range(n):                                             # same order as the C loop
                    out[..., idx[a][t]] += wo[..., t]
                    out[..., J[t]] += wn[..., t]
                r = np.moveaxis(out, -1, a + 1)
            return r
        cshape = self.coarse_shape()
        idx = np.ix_(*self.box_index())
        out = np.zeros((self.F,) + cshape)
        for f in range(self.F):
            np.add.at(out[f], idx, r[f])
        return out

    def prolong(self, e):
        if INTERP == "linear":
            idx = self.box_index()
            for a, own, nb in self.pw:
                n = self.shape[a]
                i = np.arange(n)
                I = idx[a]
                J = np.clip(np.where(i % 2 == 0, I - 1, I + 1), 0, e.shape[a + 1] - 1)
                e = own * np.take(e, I, axis=a + 1) + nb * np.take(e, J, axis=a + 1)
            return e
        idx = np.ix_(*self.box_index())
        return np.stack([e[f][idx] for f in range(self.F)])

    def dense(self):
        """Off-diagonal magnitudes and row excess of the level, for GTH."""
        F = self.F
        n = int(np.prod(self.shape)) * F
        id3 = np.arange(n // F).reshape(self.shape)
        off = np.zeros((n, n))
        excess = np.zeros(n)
        for c in range(n // F):
            i = np.unravel_index(c, self.shape)
            C = self.C[i]
            excess[c * F:(c + 1) * F] = C.sum(axis=1)
            off[c * F:(c + 1) * F, c * F:(c + 1) * F] = np.maximum(-C, 0.0)
        for a, k in enumerate(self.faces):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a] = slice(None, -1), slice(1, None)
            p, q = id3[tuple(lo)].ravel(), id3[tuple(hi)].ravel()
            for f in range(F):
                v = k[f].ravel()
                I, J = p * F + f, q * F + f
                off[I, J] += v; off[J, I] += v
        return off, excess


def parse_matrix(A, dims, fields):
    """Face conductances and cell blocks of a field-blocked 7-point matrix."""
    A = A.tocsr()
    nz, ny, nx = dims
    N = nz * ny * nx
    assert A.shape[0] == fields * N
    faces = [np.zeros((fields, nz - 1, ny, nx)), np.zeros((fields, nz, ny - 1, nx)), np.zeros((fields, nz, ny, nx - 1))]
    C = np.zeros((nz, ny, nx, fields, fields))
    if FAST:
        lib = _kernel()
        lib.flux_parse.restype = ct.c_long
        lib.flux_parse.argtypes = [ct.c_int] * 4 + [ct.POINTER(ct.c_int)] * 2 + [_PD] * 5
        pi = ct.POINTER(ct.c_int)
        indptr, indices = A.indptr.astype(np.intc, copy=False), A.indices.astype(np.intc, copy=False)
        bad = lib.flux_parse(fields, nz, ny, nx, indptr.ctypes.data_as(pi), indices.ctypes.data_as(pi), _p(A.data), _p(faces[0]), _p(faces[1]), _p(faces[2]), _p(C))
        assert bad == 0, f"row {bad - 1} is not a 7-point, point-coupled row"
        return faces, C
    for f in range(fields):
        for g in range(fields):
            B = A[f * N:(f + 1) * N, g * N:(g + 1) * N].tocsr()
            if f == g:
                C[..., f, f] = row_excess(B).reshape(dims)
                faces[0][f] = -B.diagonal(nx * ny).reshape(nz - 1, ny, nx)
                faces[1][f][:] = -B.diagonal(nx)[np.flatnonzero((np.arange(N - nx) // nx) % ny != ny - 1)].reshape(nz, ny - 1, nx)
                faces[2][f][:] = -B.diagonal(1)[np.flatnonzero((np.arange(N - 1) % nx) != nx - 1)].reshape(nz, ny, nx - 1)
            elif B.nnz:
                assert B.nnz == np.count_nonzero(B.diagonal()), "off-diagonal field blocks must be diagonal"
                C[..., f, g] = B.diagonal().reshape(dims)
    return faces, C


class _RegionSet:
    """Region vectors of one coarse level."""

    def __init__(self, L):
        global MIN_REGION
        self.levels = [L]
        keep = MIN_REGION
        MIN_REGION = ML_MIN_REGION
        try:
            Core.build_regions(self)
        finally:
            MIN_REGION = keep


class Core:
    def __init__(self, A, dims, fields=1):
        """A: CSR, blocked by field, each block 7-point on dims = (nz, ny, nx), lexicographic with x fastest."""
        faces, C = parse_matrix(A, dims, fields)
        self.fields, self.dims, self.N = fields, dims, int(np.prod(dims))
        self.levels = [Level(faces, C)]
        while np.prod(self.levels[-1].shape) > COARSEST:
            self.levels.append(self.levels[-1].coarsen())
        self.bottom = GTH(*self.levels[-1].dense())
        self.applications = 0
        self.K = 0
        self.level_regions = {}
        if FAST:
            self.bind()
        self.setup_cycles = 0
        if DEFLATE and ML_DEFLATE:
            for l in range(1, len(self.levels) - 1):
                self.level_regions[l] = _RegionSet(self.levels[l])
        if DEFLATE:
            self.build_regions()
            for round_ in range(ADAPT_ROUNDS):
                if not self.adapt_regions(round_):
                    break
        if FAST:
            self.bind_regions()

    def bind(self):
        """Hand the hierarchy to the C cycle. All arrays stay owned by Python."""
        self.lib = _kernel()
        keep = self._keep = []
        levels = (_CLevel * len(self.levels))()
        for cl, L in zip(levels, self.levels):
            cl.F, (cl.nz, cl.ny, cl.nx) = L.F, L.shape
            cl.n = L.F * int(np.prod(L.shape))
            cl.rz, cl.ry, cl.rx = L.ratio
            kz, ky, kx = [np.ascontiguousarray(k, dtype=float) for k in L.faces]
            arrays = [kz, ky, kx, np.ascontiguousarray(L.C), np.ascontiguousarray(L.Dinv), np.ascontiguousarray(L.scale)]
            keep += arrays
            cl.kz, cl.ky, cl.kx, cl.C, cl.Dinv, cl.scale = [_p(a) for a in arrays]
            work = [np.zeros(cl.n) for _ in range(10)]
            keep += work
            for i, w in enumerate(work):
                setattr(cl, f"w{i + 1}", _p(w))
            pw = getattr(L, "pw", []) if INTERP == "linear" else []
            cl.np, cl.ptrans = len(pw), int(RESTRICT == "transpose")
            for s, (a, own, nb) in enumerate(pw):
                keep += [own, nb]
                cl.pax[s], cl.pown[s], cl.pnb[s] = a, _p(own), _p(nb)
            if pw:
                pb = [np.zeros(cl.n) for _ in range(2)]
                keep += pb
                cl.pb1, cl.pb2 = _p(pb[0]), _p(pb[1])
            cl.nlines = len(L.lines)
            if L.lines:
                ax = np.array([e[0] for e in L.lines], dtype=np.intc)
                fd = np.array([e[1] for e in L.lines], dtype=np.intc)
                masks = [np.ascontiguousarray(m, dtype=np.uint8) for e in L.lines for m in e[2]]
                ptr = (ct.POINTER(ct.c_ubyte) * len(masks))(*[m.ctypes.data_as(ct.POINTER(ct.c_ubyte)) for m in masks])
                segs = [np.ascontiguousarray(e[3], dtype=np.uint8) for e in L.lines]
                sptr = (ct.POINTER(ct.c_ubyte) * len(segs))(*[m.ctypes.data_as(ct.POINTER(ct.c_ubyte)) for m in segs])
                keep += [ax, fd, ptr, sptr] + masks + segs
                cl.line_seg = sptr
                cl.line_axis = ax.ctypes.data_as(ct.POINTER(ct.c_int))
                cl.line_field = fd.ctypes.data_as(ct.POINTER(ct.c_int))
                cl.line_mask = ptr
        h = self.ch = _CHier()
        h.nlev = len(self.levels)
        h.cycle = {"V": 1, "W": 2, "K": 3}[CYCLE]
        h.nu_pre, h.nu_post, h.k_thresh = NU_PRE, NU_POST, K_THRESH
        h.lev = levels
        keep.append(levels)
        self._clevels = levels
        ba, bd = np.ascontiguousarray(self.bottom.a), self.bottom.d
        keep += [ba, bd]
        h.bn, h.ba, h.bd = self.bottom.n, _p(ba), _p(bd)
        n = self.levels[0].F * int(np.prod(self.levels[0].shape))
        vecs = [np.zeros(n) for _ in range(4)]
        keep += vecs
        h.q, h.r1, h.z, h.r2 = [_p(v) for v in vecs]
        h.K = 0
        self.lib.flux_precond.argtypes = [ct.POINTER(_CHier), _PD, _PD]
        self.lib.flux_op.argtypes = [ct.POINTER(_CHier), _PD, _PD]
        self.bind_regions()

    def bind_regions(self):
        if not hasattr(self, "ch"):
            return
        for l, reg in self.level_regions.items():
            cl = self._clevels[l]
            cl.rK = reg.K
            if reg.K:
                keepi = np.ascontiguousarray(reg.flat_keep, dtype=np.int64)
                lab = np.ascontiguousarray(reg.flat_lab, dtype=np.int64)
                wt = np.ascontiguousarray(reg.flat_w, dtype=float)
                ea, ed = np.ascontiguousarray(reg.E.a), reg.E.d
                coef = np.zeros(reg.K)
                self._keep += [keepi, lab, wt, ea, ed, coef]
                cl.rnkeep = keepi.size
                cl.rkeep, cl.rlab = keepi.ctypes.data_as(ct.POINTER(ct.c_long)), lab.ctypes.data_as(ct.POINTER(ct.c_long))
                cl.rwt, cl.rea, cl.red, cl.rcoef = _p(wt), _p(ea), _p(ed), _p(coef)
        h = self.ch
        h.K = self.K
        if not self.K:
            return
        keepi = np.ascontiguousarray(self.flat_keep, dtype=np.int64)
        lab = np.ascontiguousarray(self.flat_lab, dtype=np.int64)
        wt = np.ascontiguousarray(self.flat_w, dtype=float)
        ea, ed = np.ascontiguousarray(self.E.a), self.E.d
        coef = np.zeros(self.K)
        self._reg = [keepi, lab, wt, ea, ed, coef]
        h.wt = _p(wt)
        h.nkeep = keepi.size
        h.keep, h.lab = keepi.ctypes.data_as(ct.POINTER(ct.c_long)), lab.ctypes.data_as(ct.POINTER(ct.c_long))
        h.ea, h.ed, h.coef = _p(ea), _p(ed), _p(coef)

    def op(self, x):
        """A x in flux form on the flat vector."""
        if FAST:
            y = np.empty(x.size)
            self.lib.flux_op(ct.byref(self.ch), _p(np.ascontiguousarray(x)), _p(y))
            return y
        L = self.levels[0]
        return L.apply(x.reshape((L.F,) + L.shape)).ravel()

    def adapt_regions(self, round_):
        """Find what the current preconditioner cannot reduce, and cut regions where that error jumps."""
        L = self.levels[0]
        F, shape = L.F, L.shape
        self.bind_regions()
        rng = np.random.default_rng(round_)
        x = rng.standard_normal((F,) + shape).ravel()
        for _ in range(ADAPT_CYCLES):
            x = x - self(self.op(x))
            self.setup_cycles += 1
        x = x.reshape((F,) + shape)
        changed = False
        n = int(np.prod(shape))
        idx = np.arange(n).reshape(shape)
        for f in range(F):
            lab = self.labels[f]
            if not (lab >= 0).any():
                continue
            e = x[f]
            size = np.abs(e).max()
            if size == 0:
                continue
            I, J = [], []
            for a in range(3):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[a], hi[a] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lo), tuple(hi)
                keep = (lab[lo] == lab[hi]) & (lab[lo] >= 0) & (np.abs(e[lo] - e[hi]) <= ADAPT_JUMP * size)
                I.append(idx[lo][keep]); J.append(idx[hi][keep])
            I, J = np.concatenate(I), np.concatenate(J)
            _, comp = connected_components(sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n)), directed=False)
            comp = comp.reshape(shape)
            old = len(np.unique(lab[lab >= 0]))
            sizes = np.bincount(comp[lab >= 0].ravel())
            big = np.flatnonzero(sizes >= MIN_REGION)
            if big.size > old:
                changed = True
                new = -np.ones_like(lab)
                remap = -np.ones(sizes.size + 1, dtype=np.int64)
                remap[big] = np.arange(big.size)
                inside = lab >= 0
                new[inside] = remap[comp[inside]]
                self.labels[f] = new
        if changed:
            self.assemble_regions(self.labels)
        return changed

    def build_regions(self):
        """Labels (F, nz, ny, nx), -1 outside any region."""
        L = self.levels[0]
        F, shape = L.F, L.shape
        n = int(np.prod(shape))
        idx = np.arange(n).reshape(shape)
        labels = -np.ones((F,) + shape, dtype=np.int64)
        K = 0
        for f in range(F):
            r = np.maximum(L.C[..., f, f], 0.0)
            kmax = np.zeros(shape)
            for a, k in enumerate(L.faces):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[a], hi[a] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lo), tuple(hi)
                pad = np.zeros(shape); pad[lo] = k[f]; kmax = np.maximum(kmax, pad)
                pad = np.zeros(shape); pad[hi] = k[f]; kmax = np.maximum(kmax, pad)
            for rho in (RHO, 0.0):                                           # too many pieces means no island structure: drop the cut
                I, J = [], []
                for a, k in enumerate(L.faces):
                    lo, hi = [slice(None)] * 3, [slice(None)] * 3
                    lo[a], hi[a] = slice(None, -1), slice(1, None)
                    lo, hi = tuple(lo), tuple(hi)
                    keep = (k[f] >= TAU * np.maximum(r[lo], r[hi])) & (k[f] >= rho * np.maximum(kmax[lo], kmax[hi]))
                    I.append(idx[lo][keep]); J.append(idx[hi][keep])
                I, J = np.concatenate(I), np.concatenate(J)
                G = sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n))
                _, comp = connected_components(G, directed=False)
                sizes = np.bincount(comp)
                big = np.flatnonzero(sizes >= MIN_REGION)
                if big.size <= MAX_REGIONS:
                    break
            if big.size > MAX_REGIONS:
                big = big[np.argsort(sizes[big])[-MAX_REGIONS:]]
            remap = -np.ones(sizes.size, dtype=np.int64)
            remap[big] = K + np.arange(big.size)
            labels[f] = remap[comp].reshape(shape)
            K += big.size
        self.assemble_regions(labels)

    def assemble_regions(self, labels):
        """Number regions consecutively and form E = W^T A W, summed without any cancellation."""
        L = self.levels[0]
        F = L.F
        K = 0
        labels = labels.copy()
        for f in range(F):
            ids = np.unique(labels[f][labels[f] >= 0])
            mask = labels[f] >= 0
            labels[f][mask] = np.searchsorted(ids, labels[f][mask]) + K
            K += ids.size
        self.K, self.labels = K, labels
        if not K:
            return
        N = int(np.prod(L.shape))
        idx = np.arange(N).reshape(L.shape)
        cells, labs, wts = [], [], []
        for f in range(F):                                                   # W: 1 inside a region, g / (sink + sum g) on the next cells
            mine = labels[f] >= 0
            sink = np.maximum(L.C[..., f, f], 0.0).ravel().copy()            # reaction + faces to other free cells
            fc, fl, fk = [], [], []
            for a, k in enumerate(L.faces):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[a], hi[a] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lo), tuple(hi)
                p, q, v = labels[f][lo].ravel(), labels[f][hi].ravel(), k[f].ravel()
                ip, iq = idx[lo].ravel(), idx[hi].ravel()
                pf, qf = (p >= 0) & (q < 0), (q >= 0) & (p < 0)
                fc += [iq[pf], ip[qf]]; fl += [p[pf], q[qf]]; fk += [v[pf], v[qf]]
                free = (p < 0) & (q < 0)
                np.add.at(sink, ip[free], v[free]); np.add.at(sink, iq[free], v[free])
            cells.append(f * N + idx[mine]); labs.append(labels[f][mine]); wts.append(np.ones(int(mine.sum())))
            fc, fl, fk = np.concatenate(fc), np.concatenate(fl), np.concatenate(fk)
            if fc.size:
                key, inv = np.unique(fc * (K + 1) + fl, return_inverse=True)     # one entry per (free cell, region)
                g = np.bincount(inv, weights=fk)
                cell, lab = key // (K + 1), key % (K + 1)
                ucell, ci = np.unique(cell, return_inverse=True)
                D = sink[ucell] + np.bincount(ci, weights=g)
                cells.append(f * N + cell); labs.append(lab); wts.append(g / np.where(D > 0, D, 1.0)[ci])
        self.flat_keep, self.flat_lab, self.flat_w = np.concatenate(cells), np.concatenate(labs), np.concatenate(wts)
        W = sp.csr_matrix((self.flat_w, (self.flat_keep, self.flat_lab)), shape=(F * N, K))
        u = np.asarray(W.sum(axis=1)).ravel()
        E, rows = np.zeros((K, K)), np.zeros(K)
        for f in range(F):                                                   # E = (G W)^T K (G W) + W^T C W, in differences
            Wf = W[f * N:(f + 1) * N].tocsr()
            uf = u[f * N:(f + 1) * N]
            has = np.diff(Wf.indptr) > 0
            for a, k in enumerate(L.faces):
                lo, hi = [slice(None)] * 3, [slice(None)] * 3
                lo[a], hi[a] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lo), tuple(hi)
                ip, iq, v = idx[lo].ravel(), idx[hi].ravel(), k[f].ravel()
                lp, lq = labels[f][lo].ravel(), labels[f][hi].ravel()
                act = (has[ip] | has[iq]) & ~((lp == lq) & (lp >= 0))        # faces inside a region contribute exactly zero
                if not act.any():
                    continue
                Dm = (Wf[ip[act]] - Wf[iq[act]]).tocsr()
                Dm.eliminate_zeros()
                E += (Dm.T @ sp.diags(v[act]) @ Dm).toarray()
                rows += Dm.T @ (v[act] * (uf[ip[act]] - uf[iq[act]]))
            for g in range(F):
                Wg = W[g * N:(g + 1) * N].tocsr()
                c = L.C[..., f, g].ravel()
                E += (Wf.T @ sp.diags(c) @ Wg).toarray()
                rows += Wf.T @ (c * u[g * N:(g + 1) * N])
        off = np.maximum(-E, 0.0)
        np.fill_diagonal(off, 0.0)
        self.E_positive = float(np.maximum(E - np.diag(np.diag(E)), 0.0).max()) if K > 1 else 0.0
        self.E = GTH(off, rows + np.maximum(E - np.diag(np.diag(E)), 0.0).sum(axis=1) * 0.0)

    def Q(self, r):
        """W E^-1 W^T r on the flat vector."""
        coef = self.E.solve(np.bincount(self.flat_lab, weights=self.flat_w * r.ravel()[self.flat_keep], minlength=self.K))
        z = np.zeros(r.size)
        np.add.at(z, self.flat_keep, self.flat_w * coef[self.flat_lab])
        return z.reshape(r.shape)

    def solve_coarsest(self, f):
        L = self.levels[-1]
        F = L.F
        v = self.bottom.solve(np.moveaxis(f, 0, -1).ravel()).reshape(L.shape + (F,))
        return np.moveaxis(v, -1, 0)

    def cycle(self, l, f):
        if l == len(self.levels) - 1:
            return self.solve_coarsest(f)
        L = self.levels[l]
        x = np.zeros_like(f)
        for _ in range(NU_PRE):
            L.sweep(x, f, (0, 1))
        r = f - L.apply(x)
        rc = L.restrict(r)
        e = self.coarse_solve(l + 1, rc)
        x += L.prolong(e)
        for _ in range(NU_POST):
            L.sweep(x, f, (1, 0))
        return x

    def coarse_solve(self, l, r):
        x = self.coarse_inner(l, r)
        reg = self.level_regions.get(l)
        if reg is not None and reg.K:
            x = x + reg.Q(r - self.levels[l].apply(x))
        return x

    def coarse_inner(self, l, r):
        if l == len(self.levels) - 1:
            return self.solve_coarsest(r)
        if CYCLE == "V":
            return self.cycle(l, r)
        L = self.levels[l]
        if CYCLE == "W":
            x = self.cycle(l, r)
            return x + self.cycle(l, r - L.apply(x))
        ip = lambda a, b: np.sum(L.scale * a * b)                        # K-cycle: two GCR steps, scaled inner product
        c1 = self.cycle(l, r)
        v1 = L.apply(c1)
        v11 = ip(v1, v1)
        if not v11 > 0:
            return c1
        a1 = ip(v1, r) / v11
        x = a1 * c1
        r1 = r - a1 * v1
        if ip(r1, r1) <= K_THRESH**2 * ip(r, r):
            return x
        c2 = self.cycle(l, r1)
        v2 = L.apply(c2)
        b = ip(v2, v1) / v11
        v2 = v2 - b * v1
        c2 = c2 - b * c1
        v22 = ip(v2, v2)
        if v22 > 0:
            x = x + ip(v2, r1) / v22 * c2
        return x

    def __call__(self, r):
        self.applications += 1
        if FAST:
            out = np.empty(r.size)
            self.lib.flux_precond(ct.byref(self.ch), _p(np.ascontiguousarray(r)), _p(out))
            return out
        f = r.reshape((self.fields,) + self.dims)
        if not self.K:
            return self.cycle(0, f).ravel()
        L = self.levels[0]
        q = self.Q(f)                                                     # balanced deflation around one cycle
        r1 = f - L.apply(q)
        z = self.cycle(0, r1)
        r2 = r1 - L.apply(z)
        return (q + z + self.Q(r2)).ravel()

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))

    def memory_doubles(self):
        return sum(sum(k.size for k in L.faces) + L.C.size + L.Dinv.size + L.sumk.size + L.scale.size for L in self.levels)


_RegionSet.assemble_regions = Core.assemble_regions
_RegionSet.Q = Core.Q


class FluxOp:
    """A x in flux form, needing only the faces and cell blocks."""

    def __init__(self, faces, C):
        self.faces, self.C = faces, C
        self.F = C.shape[3]
        self.shape = C.shape[:3]
        self.n = self.F * int(np.prod(self.shape))
        if FAST:
            self.lib = _kernel()
            self.lib.flux_apply.argtypes = [ct.POINTER(_CLevel), _PD, _PD]
            arrays = [np.ascontiguousarray(k, dtype=float) for k in faces] + [np.ascontiguousarray(C)]
            self._keep = arrays
            cl = self.cl = _CLevel()
            cl.F, (cl.nz, cl.ny, cl.nx), cl.n = self.F, self.shape, self.n
            cl.kz, cl.ky, cl.kx, cl.C = [_p(a) for a in arrays]

    def __call__(self, x):
        y = np.empty(x.size)
        if FAST:
            self.lib.flux_apply(ct.byref(self.cl), _p(np.ascontiguousarray(x)), _p(y))
            return y
        out = np.einsum("zyxfg,gzyx->fzyx", self.C, x.reshape((self.F,) + self.shape), optimize=True)
        for a, k in enumerate(self.faces):
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[a + 1], hi[a + 1] = slice(None, -1), slice(1, None)
            flux = k * (x.reshape((self.F,) + self.shape)[tuple(lo)] - x.reshape((self.F,) + self.shape)[tuple(hi)])
            out[tuple(lo)] += flux
            out[tuple(hi)] -= flux
        return out.ravel()


def probe_contraction(B, order, dims):
    """Contraction of the block-triangular solve on the system aggregated to PROBE_BOXES boxes per axis: power iteration on
    I - P^-1 A, with P the groups in solve order, solved exactly. Sees what pointwise gains miss: fields coupled through a
    collective mode, also when the coupling holds on only part of the domain, and conduction that relieves it."""
    shape = [min(PROBE_BOXES, d) for d in dims]
    boxes = np.meshgrid(*[np.arange(d) * c // d for d, c in zip(dims, shape)], indexing="ij")
    idx = np.ravel_multi_index(boxes, shape).ravel()
    n = int(np.prod(shape))
    R = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(n, idx.size))
    fields = [f for G in order for f in G]
    C = sp.bmat([[R @ B[i, j] @ R.T if (i, j) in B else None for j in fields] for i in fields]).tocsr()
    sweep, start = [], 0
    for G in order:
        stop = start + len(G) * n
        sweep.append((start, stop, splu(C[start:stop, start:stop].tocsc()), C[start:stop, :start]))
        start = stop
    d = np.sqrt(np.abs(C.diagonal()))
    e = np.random.default_rng(0).standard_normal(C.shape[0]) / d
    e /= np.linalg.norm(d * e)
    rates = []
    for _ in range(20):
        r, z = C @ e, np.zeros(e.size)
        for start, stop, lu, low in sweep:
            z[start:stop] = lu.solve(r[start:stop] - low @ z[:start])
        e -= z
        rates.append(np.linalg.norm(d * e))
        if rates[-1] == 0.0:
            return 0.0
        e /= rates[-1]
    return float(np.exp(np.mean(np.log(rates[-8:]))))


class Reduction:
    """Automatic reduction of a multi-field matrix (the rules of multifield/amfr.py, H50): eliminate fields solved by their
    diagonal, group the rest by constant-mode loop gain, solve the groups in block lower-triangular order. New here: if the
    probe finds that order contracting slower than PROBE_TAU, all coupled fields form one group.
    With PROBE_TAU = inf this is amfr.AMFR, bit for bit (test_grouping.py). solver(M) returns the solve r -> z of a group."""

    def __init__(self, A, dims, k, solver):
        A = A.tocsr()
        N = A.shape[0] // k
        self.N, self.k = N, k
        sl = [slice(i * N, (i + 1) * N) for i in range(k)]
        B = {}
        for i in range(k):
            for j in range(k):
                blk = A[sl[i], sl[j]].tocsr()
                blk.eliminate_zeros()
                if blk.nnz or i == j:
                    B[i, j] = blk
        alive = list(range(k))
        self.elim = []                                # (e, d_e, {i: a_ie}, {j: a_ej}) in elimination order

        def is_point(M):
            return M.nnz == np.count_nonzero(M.diagonal())

        def share(M):
            d = np.abs(M.diagonal())
            return float(np.quantile((np.asarray(abs(M).sum(axis=1)).ravel() - d) / d, Q))
        while len(alive) > 1:
            cand = [(share(B[e, e]), e) for e in alive
                    if all(is_point(B[key]) for j in alive if j != e for key in ((e, j), (j, e)) if key in B)]
            cand = [c for c in cand if c[0] <= TAU_D]
            if not cand:
                break
            e = min(cand)[1]
            alive.remove(e)
            d_e = B[e, e].diagonal().copy()
            col = {i: B[i, e].diagonal().copy() for i in alive if (i, e) in B}     # a_ie
            row = {j: B[e, j].diagonal().copy() for j in alive if (e, j) in B}     # a_ej
            for i, a_ie in col.items():
                for j, a_ej in row.items():
                    if i == j:                                                     # cancellation-free form of the diagonal
                        dd = B[i, i].diagonal()
                        B[i, i] = B[i, i].tocsr().copy()
                        B[i, i].setdiag((dd + a_ie) - a_ie * (d_e + a_ej) / d_e)
                    else:
                        upd = sp.diags(-a_ie * a_ej / d_e).tocsr()
                        B[i, j] = (B[i, j] + upd).tocsr() if (i, j) in B else upd
            self.elim.append((e, d_e, col, row))
            for key in [kk for kk in B if e in kk]:
                del B[key]
        rho = {i: np.abs(np.asarray(B[i, i].sum(axis=1)).ravel()) for i in alive}
        c = {key: np.minimum(1.0, np.asarray(abs(B[key]).sum(axis=1)).ravel() / np.maximum(rho[key[0]], np.finfo(float).tiny))
             for key in B if key[0] != key[1]}
        order = self._order(self._groups(alive, c, TAU_G), c)
        if len(order) > 1 and np.isfinite(PROBE_TAU):
            self.probe = probe_contraction(B, order, dims)
            if self.probe > PROBE_TAU:
                order = self._order(self._groups(alive, c, -1.0), c)    # gain >= 0 > -1: every coupled pair joins
        self.groups = order
        self.solvers, self.lower = [], []
        for gi, G in enumerate(order):
            M = sp.bmat([[B.get((i, j)) if (i, j) in B else None for j in G] for i in G]).tocsr()
            M.sort_indices()
            self.solvers.append(solver(M))
            low = []
            for G2 in order[:gi]:
                if any((i, j) in B for i in G for j in G2):
                    low.append((G2, sp.bmat([[B.get((i, j)) if (i, j) in B else sp.csr_matrix((N, N))
                                              for j in G2] for i in G]).tocsr()))
            self.lower.append(low)

    @staticmethod
    def _groups(alive, c, tau):
        """Join i, j when the pairwise loop gain exceeds tau on at least a fraction PHI of the cells."""
        parent = {i: i for i in alive}

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for a in alive:
            for b in alive:
                if a < b and (a, b) in c and (b, a) in c and float(np.mean(c[a, b] * c[b, a] > tau)) >= PHI:
                    parent[find(a)] = find(b)
        groups = {}
        for i in alive:
            groups.setdefault(find(i), []).append(i)
        return [sorted(g) for g in groups.values()]

    @staticmethod
    def _order(groups, c):
        """Next comes the group receiving the least influence from the groups not yet solved."""
        def infl(P, S):
            vals = [c[p, s] for p in P for s in S if (p, s) in c]
            return max((float(np.quantile(v, Q)) for v in vals), default=0.0)
        order, rest = [], sorted(groups)
        while rest:
            nxt = min(rest, key=lambda P: sum(infl(P, S) for S in rest if S is not P))
            order.append(nxt)
            rest.remove(nxt)
        return order

    def structure(self):
        e = "".join(f"e{f}" for f, *_ in self.elim)
        return e + "|" + "|".join("".join(map(str, G)) for G in self.groups)

    def __call__(self, r):
        N = self.N
        f = [r[i * N:(i + 1) * N].copy() for i in range(self.k)]
        for e, d_e, col, row in self.elim:             # forward: r_i -= a_ie r_e / d_e
            t = f[e] / d_e
            for i, a_ie in col.items():
                f[i] -= a_ie * t
        x = [None] * self.k
        for G, solve, low in zip(self.groups, self.solvers, self.lower):
            rhs = np.concatenate([f[i] for i in G])
            for G2, L in low:
                rhs -= L @ np.concatenate([x[j] for j in G2])
            z = solve(rhs)
            for n, i in enumerate(G):
                x[i] = z[n * N:(n + 1) * N]
        for e, d_e, col, row in reversed(self.elim):   # back: x_e = (r_e - sum a_ej x_j) / d_e
            t = f[e].copy()
            for j, a_ej in row.items():
                t -= a_ej * x[j]
            x[e] = t / d_e
        return np.concatenate(x)


class FluxMG:
    """The solver: multi-field matrices first go through Reduction, and each group is solved by one Core cycle."""

    def __init__(self, A, dims, fields=1):
        self.op = FluxOp(*parse_matrix(A, dims, fields))
        if fields == 1 or not REDUCE:
            self.P = Core(A, dims, fields)
            self.cores = [self.P]
        else:
            N = int(np.prod(dims))
            self.cores = []

            def group_solver(M):
                if M.nnz == M.shape[0] and np.count_nonzero(M.diagonal()) == M.shape[0]:
                    d = M.diagonal().copy()
                    return lambda r: r / d
                core = Core(M, dims, M.shape[0] // N)
                self.cores.append(core)
                return core
            self.P = Reduction(A, dims, fields, group_solver)
        self.K = sum(c.K for c in self.cores)
        self.levels = self.cores[0].levels if self.cores else []

    def __call__(self, r):
        return self.P(r)

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))
