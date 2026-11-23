"""Side checks of the final SSR: seeds, grid growth, real matrix. Prints one line per run."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(case, *kw, task="ssr", limit=400):
    cmd = [sys.executable, str(HERE / "eval_ssr.py"), "one", case, *kw] if task == "ssr" else [sys.executable, str(HERE / "baseline.py"), "one", case, task]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
        return json.loads(line[0][7:]) if line else dict(ok=False, its=-1, err=p.stderr[-200:])
    except subprocess.TimeoutExpired:
        return dict(ok=False, its=-1, err="timeout")


def show(tag, q):
    t = q.get("time", q.get("cpu"))
    print(f"{tag:44s} ok={q['ok']!s:5s} its={q['its']:5d} t={t if t is None else round(t, 2)} lev={q.get('lev')} cx={None if q.get('cx') is None else round(q['cx'], 2)} rss={None if q.get('rss') is None else round(q['rss'], 2)}", flush=True)


print("== seeds")
for c in ("poisson3d:64", "tubes", "cd2d:512:1e-3:rec", "g3t:64:2e-11:2e-4", "mg:32:8:1e2:1e-2:1:7"):
    for s in (1, 2, 3, 4, 5):
        show(f"{c} seed {s}", run(c, f"SEED={s}"))
print("== growth")
for m in (32, 64, 100, 128):
    c = f"poisson3d:{m}"
    show(c + " ssr", run(c))
    show(c + " fast.25", run(c, task="fast.25"))
for m in (256, 512, 1024):
    c = f"cd2d:{m}:1e-3:rec"
    show(c + " ssr", run(c))
    show(c + " fast.25", run(c, task="fast.25"))
print("== real matrix")
show("real ssr", run("real"))
show("real fast.25", run("real", task="fast.25"))
print("== stretch")
show("helm ssr", run("helm:512:0.2"))
print("DONE")
