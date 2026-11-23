"""Threshold sweep for the global constant-mode rule. Prototype: reads AMFR's blocks by frame access; diagnostic only."""
import sys, json
from time import process_time
sys.path[:0] = ["/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/fluxmg", "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/multifield"]
import numpy as np, scipy.sparse as sp
import flux_mg
from try_one import load, solve_ir
import amfr, mgrd
ORIG = amfr.AMFR
def jac_radius(M, label):
    """radius of the block-Jacobi iteration on M (k x k), blocks = equal labels."""
    d = np.diag(M); C = -M / d[:, None]; np.fill_diagonal(C, 0.0)
    return float(np.abs(np.linalg.eigvals(C * (label[:, None] != label[None, :]))).max())
class G(ORIG):
    tau = None; info = {}
    def _pair_groups(self, alive, c, tau_g, phi):
        groups = super()._pair_groups(alive, c, tau_g, phi)
        B = sys._getframe(1).f_locals["B"]
        k = len(alive); pos = {f: n for n, f in enumerate(alive)}
        M = np.zeros((k, k))
        for (i, j), S in B.items():
            M[pos[i], pos[j]] = S.sum() if i == j else -abs(S).sum()
        label = np.zeros(k, int)
        for n, g in enumerate(groups):
            for f in g: label[pos[f]] = n
        G.info["r0"] = jac_radius(M, np.arange(k))
        G.info["r"] = jac_radius(M, label)
        if self.tau is not None:
            d = np.diag(M)
            while len(set(label)) > 1 and jac_radius(M, label) > self.tau:
                labs = sorted(set(label))
                def gain(ab):
                    I, J = label == ab[0], label == ab[1]
                    return (M[np.ix_(I, J)] / d[I][:, None]).__abs__().sum() * (M[np.ix_(J, I)] / d[J][:, None]).__abs__().sum()
                a, b_ = max(((a, b_) for a in labs for b_ in labs if a < b_), key=gain)
                label[label == b_] = a
            groups = [sorted(alive[n] for n in np.nonzero(label == l)[0]) for l in sorted(set(label))]
        G.info["s"] = self.structure_of(groups)
        return groups
    @staticmethod
    def structure_of(gr): return "|".join(",".join(map(str, g)) for g in gr)
def flux(A, b, dims, F, tau):
    G.tau = tau
    amfr.AMFR = G
    t0 = process_time(); P = flux_mg.FluxMG(A, dims, F); build = process_time() - t0
    q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=200, max_steps=6)
    return dict(ok=bool(q["ok"]), its=q["its"], t=build + q["cpu"], K=P.K, s=G.info["s"], r=G.info["r"], r0=G.info["r0"])
def blob(m, ng, sbar, zeta, lam, seed, frac=0.05):
    """mgrd, but the small heat capacity holds only in a blob of about frac of the cells; elsewhere c_v is 1e2 * sbar."""
    from scipy.ndimage import gaussian_filter
    keep = mgrd.generate
    A, b = keep(m, ng, sbar, 1e2, lam, seed)            # easy system
    A = A.tolil() if False else A.tocsr()
    n = m**3
    rng = np.random.default_rng(seed + 1)
    f = gaussian_filter(rng.standard_normal((m, m, m)), sigma=m / 12, mode="wrap").ravel()
    in_blob = f > np.quantile(f, 1 - frac)
    z, y, x = np.indices((m, m, m)) / m
    ball = (np.sqrt(x**2 + y**2 + z**2) < 0.35).ravel()
    drop = (1e2 - zeta) * sbar * np.where(ball, 10.0, 1.0) * in_blob
    D = sp.diags(np.concatenate([np.zeros(ng * n), drop]))
    A = (A - D).tocsr(); A.sort_indices()
    return A, b
out = []
cases = [f"mg:32:{ng}:{s}:{z}:{l}:11" for ng in (4, 8, 16) for s in ("1", "1e2") for z in ("0.02", "0.05", "0.1", "0.2", "0.4") for l in ("0", "1")]
cases += [f"mg:64:8:{s}:{z}:0:12" for s in ("1", "1e2") for z in ("0.02", "0.1", "0.4")]
cases += [f"blob:32:8:{s}:{z}:{l}:13" for s in ("1", "1e2") for z in ("1e-2",) for l in ("0", "1")]
for c in cases:
    if c.startswith("blob"):
        p = c.split(":"); A, b = blob(int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), int(p[6]))
        dims = (int(p[1]),) * 3; F = int(p[2]) + 1
    else:
        A, b, dims, F = load(c)
    a = solve_ir(A, b, "amg", theta=0.25)
    row = dict(case=c, amg=a["cpu"], amg_its=a["its"], amg_ok=bool(a["ok"]))
    for name, tau in (("pair", None), ("t8", .8), ("t6", .6), ("t4", .4), ("one", -1.0)):
        row[name] = flux(A, b, dims, F, tau)
    out.append(row)
    print(f"{c:24s} amg {a['its']:4d} {a['cpu']:6.2f} r0 {row['pair']['r0']:.3f} r {row['pair']['r']:.3f} | " + " ".join(f"{v} {row[v]['its']:4d}{'' if row[v]['ok'] else 'F'} {row[v]['t']:6.2f}" for v in ("pair","t8","t6","t4","one")) + " | " + row["t6"]["s"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=0)
