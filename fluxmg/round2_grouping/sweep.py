"""Grouping variants of FluxMG on a grid of multigroup systems. Diagnostic only; touches no project file."""
import sys, json
from time import process_time
sys.path.insert(0, "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/fluxmg")
import numpy as np
import flux_mg
from try_one import load, solve_ir
import amfr
ORIG = amfr.AMFR
last = {}
def variant(name):
    flux_mg.REDUCE = name != "mono"
    def make(A, k, verbose=False):
        kw = dict(pair={}, spec=dict(join="spectral"), one=dict(tau_g=-1.0))[name]
        P = ORIG(A, k, verbose=False, **kw)
        last["s"] = P.structure()
        return P
    amfr.AMFR = make
def flux(case, name):
    A, b, dims, F = load(case)
    variant(name); last["s"] = "mono"
    t0 = process_time(); P = flux_mg.FluxMG(A, dims, F); build = process_time() - t0
    q = solve_ir(A, b, "flux", P=P, flux_op=True, max_it=200, max_steps=6)
    return dict(ok=bool(q["ok"]), its=q["its"], t=build + q["cpu"], build=build, K=P.K, s=last["s"])
out = []
for ng in (4, 8):
  for sbar in ("1e-2", "1", "1e2"):
    for zeta in ("1e-2", "1", "1e2"):
      for lam in ("0", "1"):
        c = f"mg:32:{ng}:{sbar}:{zeta}:{lam}:7"
        A, b, dims, F = load(c)
        a = solve_ir(A, b, "amg", theta=0.25)
        row = dict(case=c, amg=a["cpu"], amg_its=a["its"], amg_ok=bool(a["ok"]))
        for v in ("pair", "spec", "one", "mono"):
            row[v] = flux(c, v)
        out.append(row)
        print(f"{c:24s} amg {a['its']:4d} {a['cpu']:6.2f} |" + " ".join(f"{v} {row[v]['its']:4d}{'' if row[v]['ok'] else 'F'} {row[v]['t']:6.2f} K{row[v]['K']:<4d}" for v in ("pair","spec","one","mono")) + " | " + row["pair"]["s"] + " / " + row["spec"]["s"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=0)
