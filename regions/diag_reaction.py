"""Audit item 5: does a stored gen_gen matrix keep the reaction terms it was built with? For each scalar diagonal block,
row sum of the STORED entries (math.fsum, exact) minus the nominal reaction, per layer (core / shell / outside; interface
cells of graded geometries go to the nearest layer). Far-face Robin cells are excluded for the radiation block.
usage: python diag_reaction.py GEOMETRY [PHASE ...]     GEOMETRY = m:n:p:aspect:width:delta:seed"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_gen import C1, C2, C3, E12, W, generate, level, per_cell  # noqa: E402

geom = sys.argv[1]
m_, n_, p_, a_, w_, d_, sd_ = geom.split(":")
m, n, p, asp, width, delta, seed = int(m_), int(n_), float(p_), float(a_), float(w_), float(d_), int(sd_)
F = 2e-11                                                       # generate()'s default, as in every H56 sequence
s = level(m, n, p, asp, width, seed)
N = m**3
c1, e, c2, w, c3 = (per_cell(t, s) for t in (C1, E12, C2, W, C3))
nominal = {"radiation": c1 + e, "electron": c2 + w + F * (c2 + w), "ion": c3 + w}
z, y, x = np.indices((m, m, m)).reshape(3, -1)
interior_far = (x < m - 1) & (y < m - 1) & (z < m - 1)
rows = []
for phase in [float(v) for v in sys.argv[2:]] or [0.0]:
    A, _ = generate(m, n, p, asp, width, delta, seed, phase=phase)
    for k, name in enumerate(nominal):
        B = A[k * N:(k + 1) * N, k * N:(k + 1) * N].tocsr()
        got = np.fromiter((math.fsum(B.data[B.indptr[i]:B.indptr[i + 1]]) for i in range(N)), float, N)
        for lay, label in enumerate(("core", "shell", "outside")):
            mask = (np.rint(s) == lay) & (interior_far if name == "radiation" else True)
            if not mask.any():
                continue
            dev = (got[mask] - nominal[name][mask]) / nominal[name][mask]
            rows.append(dict(geometry=geom, phase=phase, block=name, layer=label, cells=int(mask.sum()),
                             nominal_median=float(np.median(nominal[name][mask])), rel_dev_min=float(dev.min()),
                             rel_dev_max=float(dev.max()), rel_dev_rms=float(np.sqrt(np.mean(dev**2))),
                             max_diag=float(B.diagonal()[mask].max())))
            print(json.dumps(rows[-1]), flush=True)
