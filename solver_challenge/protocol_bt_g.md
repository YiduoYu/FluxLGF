# H42 — FD-LGF (FDM background) correction for the radiation block inside the H41 preconditioner

Why: H41 diagnosis — the A11 (radiation) V-cycle is the main limiter (A11 exact: 23 -> 17 outer its). Exploratory checks
before this lock (disclosed): the error left by the amg-3d V-cycle on A11 is 99 % in the constant-coefficient outside
region, smooth (e'Ae/e'De ~ 1e-4, first Neumann modes), asymptotic factor 0.711 per V-cycle (0.162 s). The exact FDM
inverse G of the fitted A11 background (kappa 6.0814e9, r 972.93, far-face Robin beta 1.9184e6; fdlgf_real.FDM) removes
these modes: stationary factors (single thread) V 0.711 / 0.16 s, G+V 0.481 / 0.27 s, V+G+V 0.256 / 0.44 s, G+V+G 0.480 /
0.42 s. Other BoomerAMG settings on A11 alone: default Falgout 0.198 / 0.61 s (setup 9.9 s), HMIS th 0.25 no-agg
0.573 / 0.22 s (setup 6.3 s). No outer (full-system) solve with any of these has been run.

Method (fixed here):
- bt-g: H41 bt-v with A11^-1 replaced by the symmetric multiplicative V+G+V:
      z = V r;  z += G (r - A11 z);  z += V (r - A11 z)
  V = one amg-3d BoomerAMG V-cycle on A11, G = FDM of the fitted background. S3: one V-cycle as in bt-v. GMRES(50).
Baselines in the same batch: amg-3d, bt-v. Criterion as H40/H41 (solver-reported residual, rtol 1e-4 and 1e-8, x0 = 0),
single thread, sequential. Secondary: ir_fp128.py with inner bt-g vs inner bt-v (rtol 1e-6), both in this batch.

Predictions:
1. bt-g reaches 1e-8 in <= 19 iterations.
2. bt-g total time at 1e-8 < bt-v total time at 1e-8 (same batch).
3. bt-g total time at 1e-8 <= 0.8 x amg-3d total time at 1e-8 (same batch).
4. IR + bt-g reaches true ||b-Ax||/||b|| < 1e-8 in <= 4 steps and in less total time than IR + bt-v (same batch).
