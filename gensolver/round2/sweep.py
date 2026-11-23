"""sweep.py LIMIT "CASE,CASE" "K=V K=V" "K=V" ...   one subprocess per case and configuration; iteration counts and complexity, no timing claims."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
lim = int(sys.argv[1]); cases = sys.argv[2].split(",")
for cfg in sys.argv[3:]:
    print("==", cfg, flush=True)
    for c in cases:
        try:
            p = subprocess.run([sys.executable, str(HERE / "round3" / "ssr_one.py"), c, *cfg.split()], capture_output=True, text=True, timeout=lim)
            l = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
            if l:
                q = json.loads(l[0][7:])
                print(f"  {c:26s} ok={q['ok']!s:5s} its={q['its']:5d} t={q['time']:7.2f} build={q['build']:6.2f} lev={q['lev']:3d} cx={q['cx']:6.2f} rss={q['rss']:.2f}", flush=True)
            else:
                print(f"  {c:26s} error: {p.stderr.strip().splitlines()[-1][:90] if p.stderr.strip() else p.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  {c:26s} timeout {lim}", flush=True)
