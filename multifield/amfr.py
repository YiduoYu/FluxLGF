"""AMFR — automatic algebraic multi-field reduction preconditioner (H50).

Input: a sparse matrix A and a partition of its unknowns into k fields of equal size with cell correspondence
(row c of every field belongs to the same cell; the usual blocked multiphysics layout). Nothing else is used.
bt-v (H41), bt-j (H44) and monolithic BoomerAMG are special cases of what the rule below builds.

1. Elimination. A field e is eliminated by its diagonal when
     - it is not the last remaining field,
     - all its couplings to other fields are point couplings (diagonal blocks), so the Schur complement keeps the
       sparsity of the remaining blocks, and
     - its off-diagonal share  s_e = q_Q( sum_j |a_ee[c,j]| (j != c) / |a_ee[c,c]| )  is <= TAU_D
       (s_e bounds the contraction factor of replacing A_ee by its diagonal, ||D^-1 (A_ee - D)||_inf over the cells).
   Fields are eliminated one at a time (smallest share first); after each step the Schur complement
     S_ij = A_ij - diag(a_ie a_ej / d_e)
   is formed and the shares are recomputed. Diagonal entries use the cancellation-free form
     S_ii[c,c] = (a_ii + a_ie) - a_ie (d_e + a_ei) / d_e        (an identity; exact subtractions for M-matrix rows).
2. Grouping. For the remaining fields, the per-cell loop gain of the constant (slowest) mode is
     g_ij[c] = min(1, b_ij[c] / rho_i[c]) * min(1, b_ji[c] / rho_j[c]),
   b_ij = row sums of |S_ij|, rho_i = |row sum of S_ii| (the block's action on a constant). Fields i, j are joined
   (union-find) when g_ij > TAU_G on at least a fraction PHI of the cells. Joined fields are solved together.
3. Ordering. Groups are solved in block lower-triangular order; the next group is the one whose equations receive the
   least influence from the groups not yet solved (influence P <- Q = q_Q of min(1, b_PQ / rho_P)); couplings to
   later groups are dropped.
4. Each group block is solved by one BoomerAMG V-cycle on the group as one scalar matrix (field-blocked order,
   H40 amg-3d settings; per-field median-diagonal scaling of joint groups was tried and tripled the iterations on
   gen3t G = 1, so the group is used in the given units); a group whose matrix is diagonal is solved by its diagonal.
Apply: forward elimination of the residual, block triangular sweep over the groups, back substitution.

The elimination / grouping / ordering tests are invariant to a scaling of the rows or columns of a field (units).
"""
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

AMG3D = (("coarsen_type", "HMIS"), ("interp_type", "ext+i"), ("P_max", 4), ("strong_threshold", 0.5), ("agg_nl", 1))
TAU_D, TAU_G, PHI, Q = 2e-2, 0.7, 1e-3, 0.95


def amg_defaults(prefix):
    """Write AMG3D as defaults under this options prefix: a value already in the database (caller, PETSC_OPTIONS) wins."""
    opts = PETSc.Options()
    for k, v in AMG3D:
        key = prefix + "pc_hypre_boomeramg_" + k
        if not opts.hasName(key):
            opts[key] = v


_count = [0]


def vcycle(M):
    """One BoomerAMG V-cycle (amg-3d settings) as a function r -> z."""
    M = M.tocsr()
    if M.nnz == M.shape[0] and np.count_nonzero(M.diagonal()) == M.shape[0]:
        d = M.diagonal().copy()
        return lambda r: r / d
    _count[0] += 1
    prefix = f"amfr{_count[0]}_"
    P = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(P)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    amg_defaults(prefix)
    ksp.setType("preonly")
    ksp.setFromOptions()
    ksp.setUp()
    b, x = P.createVecs()

    def solve(r):
        b.setArray(r)
        ksp.solve(b, x)
        return x.getArray().copy()
    solve.keep = (ksp, P, b, x)
    return solve


def is_point(B):
    return B.nnz == np.count_nonzero(B.diagonal())


def share(B):
    d = np.abs(B.diagonal())
    off = np.asarray(abs(B).sum(axis=1)).ravel() - d
    return float(np.quantile(off / d, Q))


class AMFR:
    def __init__(self, A, k, tau_d=TAU_D, tau_g=TAU_G, phi=PHI, join="pair", verbose=True):
        A = A.tocsr()
        N = A.shape[0] // k
        assert N * k == A.shape[0]
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
        log = []
        # ---- 1. elimination
        while len(alive) > 1:
            cand = []
            for e in alive:
                if all(is_point(B[key]) for j in alive if j != e for key in ((e, j), (j, e)) if key in B):
                    s = share(B[e, e])
                    if s <= tau_d:
                        cand.append((s, e))
            if not cand:
                break
            s, e = min(cand)
            alive.remove(e)
            d_e = B[e, e].diagonal().copy()
            col = {i: B[i, e].diagonal().copy() for i in alive if (i, e) in B}     # a_ie
            row = {j: B[e, j].diagonal().copy() for j in alive if (e, j) in B}     # a_ej
            for i, a_ie in col.items():
                for j, a_ej in row.items():
                    if i == j:
                        dd = B[i, i].diagonal()
                        B[i, i] = B[i, i].tocsr().copy()
                        B[i, i].setdiag((dd + a_ie) - a_ie * (d_e + a_ej) / d_e)
                    else:
                        upd = sp.diags(-a_ie * a_ej / d_e).tocsr()
                        B[i, j] = (B[i, j] + upd).tocsr() if (i, j) in B else upd
            self.elim.append((e, d_e, col, row))
            log.append(f"eliminate field {e} (share {s:.1e})")
            for key in [kk for kk in B if e in kk]:
                del B[key]
        # ---- 2. grouping by loop gain of the constant mode
        rho = {i: np.abs(np.asarray(B[i, i].sum(axis=1)).ravel()) for i in alive}
        rs = {key: np.asarray(abs(B[key]).sum(axis=1)).ravel() for key in B if key[0] != key[1]}
        tiny = np.finfo(float).tiny
        c = {key: np.minimum(1.0, v / np.maximum(rho[key[0]], tiny)) for key, v in rs.items()}
        self.gains = {}
        if join == "spectral":     # H51: unclipped couplings (only products / radii are unit-invariant)
            craw = {key: v / np.maximum(rho[key[0]], tiny) for key, v in rs.items()}
            groups = self._spectral_groups(alive, craw, tau_g, phi, log)
        else:
            groups = self._pair_groups(alive, c, tau_g, phi)
        # ---- 3. ordering
        def infl(P, Qg):
            vals = [c[p, q] for p in P for q in Qg if (p, q) in c]
            return max((float(np.quantile(v, Q)) for v in vals), default=0.0)
        order, rest = [], sorted(groups)
        while rest:
            nxt = min(rest, key=lambda P: sum(infl(P, Qg) for Qg in rest if Qg is not P))
            order.append(nxt)
            rest.remove(nxt)
        self.groups = order
        # ---- 4. group solvers and lower couplings
        self.solvers, self.lower = [], []
        for gi, G in enumerate(order):
            M = sp.bmat([[B.get((i, j)) if (i, j) in B else None for j in G] for i in G]).tocsr()
            M.sort_indices()
            self.solvers.append(vcycle(M))
            low = []
            for G2 in order[:gi]:
                if any((i, j) in B for i in G for j in G2):
                    low.append((G2, sp.bmat([[B.get((i, j)) if (i, j) in B else sp.csr_matrix((N, N))
                                              for j in G2] for i in G]).tocsr()))
            self.lower.append(low)
        log.append("groups (solve order): " + " -> ".join("{" + ",".join(map(str, G)) + "}" for G in order))
        log.append("loop gains q95/frac>tau: " + ", ".join(f"{a}-{b}: {g:.2f}/{f:.1e}" for (a, b), (g, f) in self.gains.items()))
        self.log = log
        if verbose:
            print("AMFR: " + "; ".join(log), flush=True)

    def _pair_groups(self, alive, c, tau_g, phi):
        """H50: join i, j when the pairwise loop gain c_ij c_ji > tau_g on >= phi of the cells."""
        parent = {i: i for i in alive}

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for a in alive:
            for b in alive:
                if a < b and (a, b) in c and (b, a) in c:
                    g = c[a, b] * c[b, a]
                    frac = float(np.mean(g > tau_g))
                    self.gains[a, b] = (float(np.quantile(g, Q)), frac)
                    if frac >= phi:
                        parent[find(a)] = find(b)
        groups = {}
        for i in alive:
            groups.setdefault(find(i), []).append(i)
        return [sorted(g) for g in groups.values()]

    def _spectral_groups(self, alive, c, tau_g, phi, log):
        """H51: per cell, C_ij = b_ij / rho_i (unclipped normalised constant-mode coupling); with the current groups, the block-Jacobi
        constant-mode iteration matrix is C with the intra-group entries removed. While its squared spectral radius
        (= the pairwise loop gain for two fields; block Gauss-Seidel contraction) exceeds tau_g on >= phi of the
        cells, merge the two groups with the largest summed c_pq c_qp over those cells. Catches star couplings."""
        K, N = len(alive), self.N
        pos = {f: n for n, f in enumerate(alive)}
        Cm = np.zeros((N, K, K))
        for (i, j), v in c.items():
            Cm[:, pos[i], pos[j]] = v
        label = np.arange(K)
        while len(set(label)) > 1:
            Ci = Cm * (label[:, None] != label[None, :])
            cand = np.nonzero(Ci.sum(axis=2).max(axis=1) ** 2 > tau_g)[0]       # inf-norm bound on the radius
            flagged = cand[np.abs(np.linalg.eigvals(Ci[cand])).max(axis=1) ** 2 > tau_g] if cand.size else cand
            frac = flagged.size / N
            log.append(f"spectral: {len(set(label))} groups, frac(rho^2 > tau) {frac:.1e}")
            if frac < phi:
                break
            Pp = (Ci[flagged] * np.transpose(Ci[flagged], (0, 2, 1))).sum(axis=0)
            labs = sorted(set(label))
            best = max(((a, b) for a in labs for b in labs if a < b),
                       key=lambda ab: Pp[np.ix_(label == ab[0], label == ab[1])].sum())
            label[label == best[1]] = best[0]
        groups = {}
        for f in alive:
            groups.setdefault(label[pos[f]], []).append(f)
        return [sorted(g) for g in groups.values()]

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

    def apply(self, pc, x, y):
        y.setArray(self(x.getArray(readonly=True)))
