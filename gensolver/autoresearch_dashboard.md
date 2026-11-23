# Autoresearch dashboard, round 2

Constrained files: `ssr.c`, `ssr.py`. Default after the round: `LUMP=1`, unchanged. New path: `LUMP=4 DMODE=2 NTV=1 BLOCK=2 KEEP=2 FILT=0.01`, off by default.

Eval SCORE without changes: -0.186, -0.160, -0.127; min-delta 0.15. This SCORE uses the stale `amg_baseline.json` and reads about 0.1 above `final.py` (locked dev -0.335).

| # | Status | Change | Result |
|---|--------|--------|--------|
| 1 | diagnostic | Cluster pivots, second-order dropping | rot2d:200 66 its at cx 44; elast and stretch exceed cx 60 |
| 2 | discard | Coarse filter, scaled diagonal | cx 1-4, but its 73-388 |
| 3 | promising | Truncated rows keep the constant | rot2d:200 66 -> 24 at cx 44 |
| 4 | promising | Reroute weak edges | cx 3-6, its at or below default, time above default |
| 5 | discard | Several test vectors | no fewer its, slower build |
| 6 | discard | Weak F-F edges for faster coarsening | fe3d jump 379 to fail |
| 7 | invalid | THETA=0 reached the scalar path by mistake | -0.741, -0.600 |
| 8 | discard | Tighter filter on elast jump | still not converged |
| 9 | discard | Corrected full eval | -0.176 against default -0.144 |

Kept: 0. Holdout: not run, nothing to lock.
