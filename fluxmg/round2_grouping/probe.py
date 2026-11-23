"""Coarse probe: contraction of the block-triangular preconditioner on the aggregated system. Prototype; diagnostic only."""
import sys, json
from time import process_time, perf_counter
sys.path[:0] = ["/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/fluxmg", "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/multifield"]
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl
import flux_mg
from try_one import load, solve_ir
import amfr, mgrd
cap = {}
class Cap(amfr.AMFR):
    def _pair_groups(self, alive, c, tau_g, phi):
        f = sys._getframe(1).f_locals
        cap.update(B=f["B"], alive=list(alive))
        return super()._pair_groups(alive, c, tau_g, phi)
def restrict(dims, cn):
    e = [np.arange(d) * min(cn, d) // d for d in dims]
    z, y, x = np.meshgrid(*e, indexing="ij")
    idx = (z * cn * cn + y * cn + x).ravel()
    return sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(cn**3, idx.size))
def probe(B, order, dims, cn, its=20, seed=0):
    """Estimate the spectral radius of I - P^-1 A on the aggregated system: power iteration in the diagonally scaled norm."""
    R = restrict(dims, cn)
    C = {key: (R @ S @ R.T).tocsc() for key, S in B.items()}
    lu, low = [], []
    for gi, G in enumerate(order):
        M = sp.bmat([[C.get((i, j)) for j in G] for i in G], format="csc")
        lu.append(sl.splu(M))
        low.append([(G2, sp.bmat([[C.get((i, j), sp.csr_matrix((cn**3, cn**3))) for j in G2] for i in G], format="csr")) for G2 in order[:gi] if any((i, j) in C for i in G for j in G2)])
    fields = [f for G in order for f in G]
    n = cn**3
    d = {f: np.sqrt(np.abs(C[f, f].diagonal())) for f in fields}
    rng = np.random.default_rng(seed)
    e = {f: rng.standard_normal(n) / d[f] for f in fields}
    def norm(v): return np.sqrt(sum(np.sum((d[f] * v[f])**2) for f in fields))
    rates = []
    a = norm(e)
    for _ in range(its):
        # new error = e - P^-1 A e
        r = {f: sum(C[f, g] @ e[g] for g in fields if (f, g) in C) for f in fields}
        z = {}
        for G, F, lo in zip(order, lu, low):
            rhs = np.concatenate([r[f] for f in G])
            for G2, L in lo:
                rhs = rhs - L @ np.concatenate([z[f] for f in G2])
            s = F.solve(rhs)
            for m, f in enumerate(G):
                z[f] = s[m * n:(m + 1) * n]
        e = {f: e[f] - z[f] for f in fields}
        b = norm(e)
        if b == 0: return 0.0
        rates.append(b / a)
        e = {f: e[f] / b for f in fields}; a = 1.0
    return float(np.exp(np.mean(np.log(rates[-8:]))))
def indicators(A, dims, F):
    keep = amfr.vcycle; amfr.vcycle = lambda M: (lambda r: r)
    P = Cap(A, F, verbose=False); amfr.vcycle = keep
    out = {}
    for cn in (4, 8):
        t0 = perf_counter()
        out[f"p{cn}"] = probe(cap["B"], P.groups, dims, cn) if len(P.groups) > 1 else 0.0
        out[f"t{cn}"] = perf_counter() - t0
    return out, "|".join(",".join(map(str, g)) for g in P.groups)
def blob(m, ng, sbar, zeta, lam, seed, frac):
    from scipy.ndimage import gaussian_filter
    A, b = mgrd.generate(m, ng, sbar, 1e2, lam, seed)
    n = m**3
    rng = np.random.default_rng(seed + 1)
    f = gaussian_filter(rng.standard_normal((m, m, m)), sigma=m / 12, mode="wrap").ravel()
    in_blob = f > np.quantile(f, 1 - frac)
    z, y, x = np.indices((m, m, m)) / m
    ball = (np.sqrt(x**2 + y**2 + z**2) < 0.35).ravel()
    drop = (1e2 - zeta) * sbar * np.where(ball, 10.0, 1.0) * in_blob
    A = (A - sp.diags(np.concatenate([np.zeros(ng * n), drop]))).tocsr(); A.sort_indices()
    return A, b
def timed(A, b, dims, F, reduce_all):
    orig = amfr.AMFR
    amfr.AMFR = (lambda A, k, verbose=False: orig(A, k, tau_g=-1.0, verbose=False)) if reduce_all else orig
    t0 = process_time(); P = flux_mg.FluxMG(A, dims, F); bt = process_time() - t0
    q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=200, max_steps=6)
    amfr.AMFR = orig
    return dict(ok=bool(q["ok"]), its=q["its"], t=bt + q["cpu"])
rows = []
known = {r["case"]: r for r in json.load(open("sweep.json")) + json.load(open("sweep2.json"))}
for c, r in known.items():
    if c.startswith("blob"): continue
    A, b, dims, F = load(c)
    ind, s = indicators(A, dims, F)
    rows.append(dict(case=c, pair=r["pair"]["its"], one=r["one"]["its"], tp=r["pair"]["t"], to=r["one"]["t"], amg=r["amg"], **ind, s=s))
    print(f"{c:26s} pair {r['pair']['its']:4d} one {r['one']['its']:4d} tp/to {r['pair']['t']/r['one']['t']:5.2f} p4 {ind['p4']:.3f} p8 {ind['p8']:.3f} t4 {ind['t4']*1e3:5.0f}ms t8 {ind['t8']*1e3:5.0f}ms", flush=True)
for ng in (4, 8):
  for sbar in ("1", "1e2"):
    for lam in ("0", "1"):
      for frac in (0.05, 0.01, 0.002):
        for z in ("1e-2", "1"):
          c = f"blob:32:{ng}:{sbar}:{z}:{lam}:13:{frac}"
          A, b = blob(32, ng, float(sbar), float(z), float(lam), 13, frac); dims = (32,)*3; F = ng + 1
          ind, s = indicators(A, dims, F)
          a = solve_ir(A, b, "amg", theta=0.25)
          tp, to = timed(A, b, dims, F, False), timed(A, b, dims, F, True)
          rows.append(dict(case=c, pair=tp["its"], one=to["its"], tp=tp["t"], to=to["t"], amg=a["cpu"], **ind, s=s))
          print(f"{c:30s} amg {a['its']:4d} {a['cpu']:5.2f} pair {tp['its']:4d} {tp['t']:5.2f} one {to['its']:4d} {to['t']:5.2f} tp/to {tp['t']/to['t']:5.2f} p4 {ind['p4']:.3f} p8 {ind['p8']:.3f} {s}", flush=True)
for c in ["g3t:64:2e-11:2e-4","g3t:64:2e-11:1","g3t:64:1:1e-2","gen:64:1:8:0.5:0:2e-2:1","gen:64:1:2:1:4:2e-2:1","gen:64:3:2:1:0:2e-2:1","gen:64:10:2:1:0:2e-2:1","gen:64:10:2:1:0:2e-2:2","gen:64:10:8:0.5:2:2e-2:3","gen:128:10:2:1:0:2e-2:2","real"]:
    A, b, dims, F = load(c)
    ind, s = indicators(A, dims, F)
    print(f"{c:30s} p4 {ind['p4']:.3f} p8 {ind['p8']:.3f} t8 {ind['t8']*1e3:5.0f}ms  {s}", flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=0)
