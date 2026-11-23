# H53 — bt-fd with MUMPS LU feature solves (re-lock of H52 after the CHOLMOD finding)

Change vs H52 (0fece60e): the feature solve in rsolve.ScalarFDLGF is MUMPS LU for every block (H52 used CHOLMOD Cholesky
for symmetric blocks). Nothing else changed. Reason (H52 post-lock, exploratory): on the real matrix CHOLMOD gave 53 outer
its to the reported 1e-8 vs 16 with LU (both linear); LU setup is ~1-2 s larger. Smoke test of this code: gen3t m=32 only.
The real matrix and the 7 H52 gen3t cases have been seen (with CHOLMOD); 5 new gen3t matrices have never been run.

Tests (all single thread, sequential; GMRES(50), right preconditioning, x0 = 0):
  R   real SC26_01, solver-reported residual, rtol 1e-4 and 1e-8, amg-3d / bt-v / bt-fd, 3 passes, median total time
  IR  real SC26_01, official ||b - Ax||/||b|| by double-double refinement, inner rtol 1e-6, amg-3d / bt-v / bt-fd
  G   own 3T m = 128, geo:128:r_core:r_shell:cap:F:G, amg-3d / bt-v / bt-fd, both tolerances
      seen (H52): 43:54:1:2e-11:2e-4, 25:32:1, 60:66:1, 30:70:1, 43:54:0.1, 43:54:10, 43:54:1:1e-3:1e-2
      new:        35:45:1:2e-11:2e-4, 50:58:1:2e-11:2e-4, 43:54:3:2e-11:2e-4, 43:54:0.3:2e-11:2e-4, 43:54:1:1e-6:1e-3
  L   linearity defect of bt-fd on the real matrix and the 12 gen3t matrices (lin.py)

Predictions:
1. R: median bt-fd time <= 0.75x amg-3d at 1e-4 and <= 0.85x at 1e-8; bt-fd faster than bt-v at both.
2. IR: bt-fd reaches the official 1e-8 in less time than amg-3d and bt-v.
3. G: all 24 runs converge; bt-fd iterations to 1e-8 <= 0.7x amg-3d's in >= 11 of 12 cases.
4. G: bt-fd faster than amg-3d in >= 18 of 24 runs (losses expected where the feature is >= ~8 %).
5. L: linearity defect <= 1e-14 on all 13 matrices.
