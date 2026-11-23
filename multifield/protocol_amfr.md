# H50 — automatic algebraic multi-field reduction (AMFR): one rule that contains bt-v, bt-j and monolithic AMG

User request (2026-11-05): "尝试一下这个" = the generic route from report 12: automatic algebraic multi-field reduction.
Method: amfr.py (fixed). Input A + number of equal-size cell-aligned fields k; nothing problem-specific.
  1. eliminate by diagonal every field (one at a time, smallest share first) whose cross-field couplings are point
     couplings and whose off-diagonal share q95(sum|offdiag|/|diag|) <= tau_d = 2e-2; Schur complement with the
     cancellation-free diagonal; recompute.
  2. join remaining fields i, j when the constant-mode loop gain g_ij = min(1, b_ij/rho_i) min(1, b_ji/rho_j)
     exceeds tau_g = 0.7 on >= phi = 1e-3 of the cells.
  3. block lower-triangular order (least incoming influence first); 4. one BoomerAMG V-cycle (amg-3d settings) per group.
Unit checks (test_amfr.py): reproduces BTElim on gen3t base (diff 5e-16) and BTJoint on G = 1 (diff 0); exact inverse when
nothing is dropped (3e-16, well-scaled test).

Dev data (seen; used to set the thresholds): H43/H44 gen3t grid m = 64 (20 cases) and mgrd.py dev cases m = 32, ng = 3,
sbar {0.1, 10}, zeta {0.1, 10}, lam {0, 1}, seed 1 (8 cases). tau_g was 0.9 first; dev showed gen3t F = 0.1, G = 1
(e-i loop gain 0.89, not joined: 31 its vs 17) and mg 0.1/0.1/1 (46 vs 32) -> changed to 0.7 (both fixed: 17 and 25 its).
Per-field median-diagonal scaling of joint groups tried on dev (36 vs 15 its) and removed. tau_d, phi, q not changed.

Test cases (never solved before this lock):
  T-A gen3t m = 64, F in {1e-8, 1e-4, 3e-2, 0.3, 3} x G in {1e-3, 3e-2, 0.3, 10}  (20) + m = 128 (1e-8, 1e-3), (1e-8, 0.3)
  T-B mgrd m = 40, ng in {2, 4, 8} x sbar in {1e-2, 1, 1e2} x zeta in {1e-2, 1e2} x lam in {0, 1}, seed 7  (36)
  T-C real SolverChallenge26_01 (amg-3d, amfr only)
Methods: amg-3d (monolithic BoomerAMG, H40 settings), amg-unk (hypre systems AMG, unknown approach: interleaved, block size
k, same settings), amfr, and ablations amfr-sep (no joining), amfr-noelim (no elimination), amfr-all (one joint group
after elimination). All GMRES(50), right preconditioning, unpreconditioned residual estimate rtol 1e-8 (solver-reported
criterion, as H40/H43), x0 = 0, max 500 its, single thread, sequential (run_mf.py, test.sh). Time = setup + solve.

Predictions (generated = T-A + T-B, 58 cases):
1. Robustness: amfr converges in all 58 and needs <= 1.5x the iterations of amg-3d in >= 55 of 58.
2. Speed: time ratio amfr / amg-3d has median <= 0.90 on T-A, median <= 0.75 on T-B, and is <= 1.10 in >= 52 of 58.
3. Rule quality: amfr time <= 1.15 x min(amfr, amfr-sep, amfr-noelim, amfr-all) in >= 49 of 58 (85 %).
4. amfr is faster than amg-unk in >= 46 of 58 (80 %).
5. Real SC26_01: amfr builds e1|2|0 (= bt-v) and its total time is below amg-3d's (same batch).
6. Structure on T-A (m = 64): the electron field is eliminated in all 5 cases with G = 1e-3 and in none of the 15 with
   G >= 3e-2.
