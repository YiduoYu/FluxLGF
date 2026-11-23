# H43 — does bt-v generalise? Regime sweep on our own 3T discretization

Matrices: gen3t.py (own linearised 3T discretization: 1/8 capsule, 3 layers with the real matrix's per-layer values,
backward Euler, cell-centred FV 7-point, Neumann symmetry planes, radiation Robin far faces). Knobs:
F = radiation feedback into the electron row (f = F (c2 + w)), G = electron diffusion (k2 = G (c2 + w + f) / 6).
Calibration before this lock (coefficient level only, no solve run): base F = 2e-11, G = 2e-4 gives median |a21|/d22 =
2.0e-11 and electron off-diagonal share 2.0e-4 (real SC26_01: 1.6e-11, 2.1e-4), same nnz at m = 128.
Grid: m = 64, F in {2e-11, 1e-6, 1e-3, 1e-1, 1}, G in {2e-4, 1e-2, 1, 1e2} (20 cases); plus m = 128 base as fidelity
anchor. Unit tests (test_bt_elim.py) of the variants were run; no solve on generated matrices yet.

Methods (fixed): amg-3d (monolithic BoomerAMG, H40 settings); bt-v (H41); bt-v2 = bt-v with the electron block solved
by one BoomerAMG V-cycle instead of its diagonal; bt-v2x2 = bt-v2 applied twice as a stationary iteration
(z = P r; z += P (r - A z)), which brings in the dropped feedback A21. All: GMRES(50), right preconditioning,
unpreconditioned residual, rtol 1e-8, x0 = 0, max 500 its, single thread, sequential (run_gen.py).
"Base case" = F 2e-11, G 2e-4 at m = 64.

Predictions:
1. Fidelity: at m = 128 base, amg-3d and bt-v converge with iteration counts within +-30 % of the real matrix (23 each).
2. In the 4 cases F <= 1e-6 and G <= 1e-2, bt-v converges and has lower total time than amg-3d in >= 3 of 4.
3. bt-v degrades when its assumptions break: in the 14 cases with F >= 1e-1 or G >= 1, bt-v needs >= 1.5x its base-case
   iterations or fails to converge, in >= 11 of 14.
4. bt-v2 restores robustness to electron diffusion: in the 12 cases with F <= 1e-3, bt-v2 converges with <= 1.5x its
   base-case iterations in >= 10 of 12.
5. bt-v2x2 handles strong feedback: in the 8 cases with F >= 1e-1, bt-v2x2 converges in all 8 with <= 2x its base-case
   iterations.
6. The fastest bt variant beats amg-3d in total time in >= 15 of the 20 m = 64 cases.
