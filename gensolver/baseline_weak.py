"""BoomerAMG on the weak-spot suite, with the configurations a user would try for each class.

Besides the 8 scalar configurations of krylov.AMG:
  air.1, air.2    AIR restriction (distance 1 and 2), no pre-relaxation, F-F-C Jacobi post-relaxation (nonsymmetric cases)
  sys.u, sys.n    multigroup: unknowns reordered cell by cell (not timed) and numfunctions set, unknown-based or nodal coarsening
One subprocess per (case, configuration). usage: baseline_weak.py all [--hold] | one CASE CFG     writes amg_weak.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "amg_weak.json"
_AIR = dict(coarsen_type="PMIS", strong_threshold=0.25, relax_type_all="Jacobi", grid_sweeps_down=0, grid_sweeps_up=3, postrelax="F,F,C")
EXTRA = {
    "air.1": dict(_AIR, restriction_type=1),
    "air.2": dict(_AIR, restriction_type=2),
    "sys.u": dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=1, strong_threshold=0.25),
    "sys.n": dict(coarsen_type="HMIS", interp_type="ext+i", P_max=4, agg_nl=1, strong_threshold=0.25, nodal_coarsen=1),
}


def configs(case, sym):
    from krylov import AMG
    c = list(AMG)
    if not sym:
        c += ["air.1", "air.2"]
    if case.startswith("mg"):
        c += ["sys.u", "sys.n"]
    return c


def one(case, cfg):
    sys.path.insert(0, str(HERE))
    import numpy as np
    from krylov import AMG, run
    from problems_weak import load
    A, b, _ = load(case)
    table = {**AMG, **EXTRA}
    c = dict(table[cfg])
    if cfg.startswith("sys"):
        nf = int(case.split(":")[2]) + 1
        N = A.shape[0] // nf
        perm = (np.arange(N)[:, None] + N * np.arange(nf)[None, :]).ravel()     # new index cell*nf + field
        A = A[perm][:, perm].tocsr(); b = b[perm]
        c["numfunctions"] = nf
    print("RESULT " + json.dumps(run(A, b, "amg", amg_cfg=c)))


def main():
    sys.path.insert(0, str(HERE))
    from problems_weak import WEAK, WEAK_HOLD, WEAK_HOLD2, load
    cases = WEAK_HOLD2 if "--hold2" in sys.argv else WEAK_HOLD if "--hold" in sys.argv else WEAK
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c, label in cases:
        A, _, _ = load(c)
        sym = abs(A - A.T).max() <= 1e-10 * abs(A).max()
        d = data.setdefault(c, {})
        for cfg in configs(c, sym):
            if cfg in d:
                continue
            try:
                p = subprocess.run([sys.executable, __file__, "one", c, cfg], capture_output=True, text=True, timeout=240)
                line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
                d[cfg] = json.loads(line[0][7:]) if line else dict(ok=False, its=-1, cpu=None, err=p.stderr[-200:])
            except subprocess.TimeoutExpired:
                d[cfg] = dict(ok=False, its=-1, cpu=None, err="timeout 240 s")
            q = d[cfg]
            print(f"{c:28s} {cfg:11s} ok={q['ok']!s:5s} its={q['its']:5d} cpu={q['cpu'] if q['cpu'] is None else round(q['cpu'], 2)}", flush=True)
            OUT.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3])
    else:
        main()
