"""BoomerAMG baselines on the generalization suite. One subprocess per (case, config), so memory and options stay isolated.
usage: baseline.py one CASE CFG | baseline.py all [--holdout] [--repeats R]    writes amg_baseline.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "amg_baseline.json"


def one(case, cfg):
    sys.path.insert(0, str(HERE))
    from krylov import AMG, NODAL, run
    from problems import load
    A, b, info = load(case)
    table = {**AMG, **NODAL}
    bs = info["bs"] if cfg.startswith("nodal") else 1
    print("RESULT " + json.dumps(run(A, b, "amg", amg_cfg=table[cfg], bs=bs)))


def main():
    sys.path.insert(0, str(HERE))
    from krylov import AMG, NODAL
    from problems import DEV, HOLDOUT, STRETCH, load
    cases = (HOLDOUT if "--holdout" in sys.argv else DEV) + STRETCH
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    for c, label in cases:
        load(c)
        cfgs = list(AMG) + (list(NODAL) if c.startswith("elast") else [])
        d = data.setdefault(c, {})
        for cfg in cfgs:
            if cfg in d:
                continue
            try:
                p = subprocess.run([sys.executable, __file__, "one", c, cfg], capture_output=True, text=True, timeout=240)
                line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
                d[cfg] = json.loads(line[0][7:]) if line else dict(ok=False, its=-1, cpu=None, err=p.stderr[-200:])
            except subprocess.TimeoutExpired:
                d[cfg] = dict(ok=False, its=-1, cpu=None, err="timeout 240 s")
            q = d[cfg]
            print(f"{c:26s} {cfg:11s} ok={q['ok']!s:5s} its={q['its']:4d} cpu={q['cpu'] if q['cpu'] is None else round(q['cpu'], 2)}", flush=True)
            OUT.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3])
    else:
        main()
