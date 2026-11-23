"""H56 figure: 5-step m=128 sequences (fixed geometry, k(T) drift) -- total time per strategy (results/explore_seq128b.log)."""
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
t = {}
for line in open(R / "explore_seq128b.log"):
    m = re.match(r"128:(\d+):\S+ (fdd?) (\S+)\s+its/step \[([^\]]*)\] total ([\d.]+) s", line)
    if m:
        t[(m[1], m[2], m[3])] = (float(m[5]), [int(v) for v in m[4].split(",")])
bars = [("BoomerAMG\nrebuilt", "fd", "amg", "0.75"), ("BoomerAMG\nreused", "fd", "amgf", "0.5"),
        ("bt-fd\nreused", "fd", "frozen", "C0"), ("bt-fd reused\n+ recycling k=5", "fd", "rec5", "#9ecae1"),
        ("bt-fdd\nreused", "fdd", "frozen", "C3"), ("bt-fdd reused\n+ recycling k=5", "fdd", "rec5", "#fc9272")]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
for a, (n, title) in zip(ax, (("1", "one capsule (base), F 4.7 % / 1.8 %"), ("10", "10 random capsules, F 13.5 % / 10.1 %"))):
    x = np.arange(len(bars))
    for i, (lab, base, meth, col) in enumerate(bars):
        tt, its = t[(n, base, meth)]
        a.bar(i, tt, color=col)
        a.text(i, tt, f"{tt:.0f} s\n{np.mean(its):.0f} its", ha="center", va="bottom", fontsize=7)
    a.set_xticks(x, [b[0] for b in bars], fontsize=7); a.set_ylabel("total time, 5 steps (s)")
    a.set_title(f"m = 128, {title}", fontsize=9); a.set_ylim(0, 1.25 * max(t[(n, b[1], b[2])][0] for b in bars))
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "results56.png", dpi=160)
