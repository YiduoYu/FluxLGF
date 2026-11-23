# H44 — joint electron-ion solve for the hard regime of H43

Why: H43 showed every bt variant is 1.8-2x slower than monolithic BoomerAMG when electron conduction is moderate
(G = 1) and e-i exchange is strong (shell): diag(A22) in the ion Schur complement is then poor. Remedy: solve the
electron-ion block jointly; keep dropping only A21 (radiation feedback; harmless, loop gain < 1).
Unit check before lock (test_bt_elim.py): both variants converge as stationary iterations on a small G = 1 type system.
No solve on generated or real matrices yet.

Methods (fixed, bt_elim.BTJoint):
- bt-j : [A22 A23; A32 A33] by one BoomerAMG V-cycle (amg-3d settings) as a scalar matrix; then A11 by one V-cycle.
- bt-jn: same with interleaved ordering, block size 2 (num_functions 2) and nodal coarsening (nodal_coarsen 1).
Re-run in the same batch: amg-3d, bt-v, bt-v2. Generated matrices: gen3t.py, m = 64, the H43 grid of 20 (F, G) cases.
Real SC26_01: run_sc.py, criterion as H40 (solver-reported residual), rtol 1e-4 and 1e-8, methods amg-3d, bt-v, bt-j,
bt-jn. All GMRES(50), right preconditioning, x0 = 0, single thread, sequential.

Predictions:
1. Hard regime (G = 1, F <= 1e-3; 3 cases): min(its of bt-j, bt-jn) <= 25 in all 3 (bt-v2 had 50-59).
2. bt-j converges in all 20 generated cases with iterations <= 1.3x those of amg-3d.
3. The fastest of {bt-v, bt-v2, bt-j, bt-jn} beats amg-3d in total time in >= 18 of the 20 generated cases.
4. Real SC26_01: bt-j reaches 1e-8 in <= 23 iterations and its total time is <= amg-3d's (same batch).
