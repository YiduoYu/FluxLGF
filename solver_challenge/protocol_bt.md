# H41 — electron-elimination block-triangular preconditioner on the real SolverChallenge26_01 matrix

Why (exploratory evidence, report 06): T1 = radiation, T2 = electron (hub), T3 = ion. On the BoomerAMG solution,
||A21 x1|| / ||A22 x2|| = 9.3e-13 and the T2 off-diagonal part is 2e-8 of its diagonal part; A23 = A32 are diagonal.
So, dropping A21 and the T2 off-diagonals, the system is block lower triangular with a diagonal electron block:
    S3 x3 = r3 - A32 D22^-1 r2,   S3 = A33 - A32 D22^-1 A23   (symmetric 7-point, ion)
    x2 = D22^-1 (r2 - A23 x3)
    A11 x1 = r1 - A12 x2                                       (radiation)
Diagonal of S3 is formed as (a33 - |a23|) + |a23| (d22 - |a23|) / d22 to avoid the cancellation a33 ~ |a23| in the shell.

Methods (fixed here, no tuning on this matrix):
- bt-v : the above with ONE BoomerAMG V-cycle (amg-3d settings: HMIS, ext+i, P_max 4, theta 0.5, 1 aggressive level)
         for S3^-1 and for A11^-1. Linear preconditioner, outer GMRES(50) right-preconditioned.
- bt-ex: same, but S3^-1 and A11^-1 by CG + the same BoomerAMG to rtol 1e-10 (max 200 its); outer FGMRES(50).
         Measures the quality of the block approximation itself.
Baseline: amg-3d from H40 (h40_main2: 13.8 s / 15 its at 1e-4, 18.1 s / 23 its at 1e-8).
Criterion as H40 (user choice): solver-reported unpreconditioned residual, rtol 1e-4 and 1e-8, x0 = 0, MAXIT 2000;
true and diag-scaled residuals logged. Single thread (OMP/VECLIB/OPENBLAS_NUM_THREADS=1), sequential runs; amg-3d rerun
in the same batch for timing reference.
Secondary (IR): ir_fp128.py with inner solver GMRES + bt-v at rtol 1e-6, double-double x and residual; compared with the
exploratory IR + amg-3d run (19+9+9 its, 41.5 s, true 4.5e-10).

Predictions:
1. bt-ex reaches 1e-8 in <= 3 outer iterations.
2. bt-v reaches both targets, <= 15 iterations to 1e-8.
3. bt-v total time at 1e-8 < amg-3d total time at 1e-8 in the same batch.
4. bt-v 1e-8 solution has diag-scaled residual < 1e-6.
5. IR with bt-v inner reaches true ||b-Ax||/||b|| < 1e-8 in <= 4 refinement steps, in less total time than IR + amg-3d
   (rerun in the same batch).
