"""try_amg.py CASE 'k=v,k=v' ...   one BoomerAMG configuration per argument, same driver as everything else."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from krylov import run
from problems import load
case = sys.argv[1]; A, b, info = load(case)
for spec in sys.argv[2:]:
    cfg = {}
    bs = 1
    for kv in spec.split(";"):
        k, v = kv.split("=")
        if k == "bs": bs = int(v); continue
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: pass
        cfg[k] = v
    q = run(A, b, "amg", amg_cfg=cfg, bs=bs)
    print(f"{case:24s} ok={q['ok']!s:5s} its={q['its']:5d} cpu={q['cpu']:7.2f} setup={q['setup']:6.2f}  {spec}", flush=True)
