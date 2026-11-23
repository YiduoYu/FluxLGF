# H55 — bt-fdd: multi-region background + island coarse space (locked test)

Method (rsolve.DeflatedFDLGF, used for both scalar blocks of bt-fd; run_real/run_ir name "bt-fdd"):
  regions = 6-connected components (>= m^3/1000 cells) of cells whose 6 face coefficients agree to 1e-6; feature F =
  cells not matching their region's constant row (1e-6 of row max) + 2-cell ring, solved exactly; largest region:
  whole-cube FDM; every other region: FDM on its bounding box with Dirichlet at interior box faces (FDMBox dirichlet);
  coarse space W = [I_F, v_k] with v_k = indicator of each non-largest region (no island test, no linear modes);
  W^T M W factored by one MUMPS LU (bordered, sparse); M v, M^T v, v^T M v by exact double-double sums;
  balanced form P = Q + (I - QM) G (I - MQ). Defaults frozen: LIN False, FORM "bal", ADMIT "dbox".
The bt-fd (H53) code path (ScalarFDLGF) is unchanged (MultiFDLGF factorisation was moved into a _factor method only).

Seen during exploration (m = 128): real SC26_01, geo 30:70:1, 60:66:1, 43:54:1 (2e-11:2e-4). The other 9 gen matrices
have never been run with bt-fdd. Exploration numbers: gen 30/70 7.7/9.7 s, 60/66 8.5/12.1, 43/54 7.1/9.1 (AMG
13.1-13.4 / 17.3-18.2); real reported 1e-4 7.8 s, reported 1e-8 60 its (residual-estimate plateau ~1-2e-8, bt-fd's
plateau is just below 1e-8); real official 17.2 / 23.0 s (AMG 23.5 / 29.6, bt-fd 16.3 / 20.4).

Tests (single thread, sequential; GMRES(50), right preconditioning, x0 = 0):
  R   real, solver-reported residual, rtol 1e-4 and 1e-8, amg-3d / bt-fd / bt-fdd, 3 passes, median total time
  IR  real, official ||b - Ax||/||b|| by double-double refinement, inner rtol 1e-6, amg-3d / bt-fd / bt-fdd
  G   the 12 H53 gen matrices geo:128:..., amg-3d / bt-fd / bt-fdd, both tolerances (one pass)
  L   linearity defect of bt-fdd on real + 12 gen (lin55.py)

Predictions:
1. G: all 24 bt-fdd runs converge; bt-fdd faster than amg-3d in >= 22 of 24 runs.
2. G: on the two large-feature matrices (60:66, 30:70) bt-fdd is faster than both amg-3d and bt-fd at both tolerances
   (8 comparisons), and <= 0.75x bt-fd in each of the 4 runs.
3. G: bt-fdd faster than bt-fd in >= 14 of 24 runs (it may lose on small-feature cases such as 25:32 and 35:45).
4. R: median bt-fdd time at 1e-4 <= 0.65x amg-3d.
5. IR: bt-fdd reaches the official 1e-8 in <= 0.85x amg-3d's time.
6. L: linearity defect <= 1e-14 on all 13 matrices.
7. R (expected negative): bt-fdd needs > 40 its to the reported 1e-8 on the real matrix (double-precision plateau).
Locked: 2026-11-05T09:22:38Z
