# H48 protocol: adaptive coarse-grid deflation (CGD2) — LOCKED before any test-seed solve

## Changes from H47 (report 14)
1. Fine-grid probe instead of the coarse detector: 10 plain PCG iterations (kept), smallest Ritz value of B A from the
   PCG tridiagonal; deflate only if it is < 0.035, otherwise continue the same PCG (no wasted work).
2. Coarse grid n_c = (n+1)/4 - 1 (127 / 255 / 511 for n = 511 / 1023 / 2047) instead of n/8.
3. Lanczos K = 40 on the coarse grid, Ritz vectors with value < 0.5, at most 24; deflated PCG restarted from the
   current iterate (x <- x + W W^T r).
Everything else as H47: same families (circ, disk, band, rot), BoomerAMG theta 0.25 agg_nl 0, PCG rtol 1e-8, single
thread. Code: snn/cgd2.py, snn/run_cgd2.py.

## Test set
TEST_SEED = 3030 (unused so far). n = 511, 1023: 6 instances per family; n = 2047: 3 per family. Runs strictly
sequential (one process) to reduce timing noise (H47 ran two streams in parallel).

## Exploration disclosed
Dev seed 555 only: results/dev2.csv (n = 511, 6/family), results/dev2_1023.csv (n = 1023, 3/family),
results/dev2047.log (n = 2047, 1 circ + 1 disk, forced deflation: 0.94 / 0.94 — only 162 -> 108 and 115 -> 80 its).
Probe threshold 0.035 chosen from the dev values: gains at <= 0.033, break-even 0.036-0.05, losses above.
The probe is resolution dependent (dev circ: 0.033 with 112 its at n = 1023 vs 0.041 with 49 its at n = 511).

## Predictions
1. circ and disk combined: median time ratio <= 0.85 at n = 511 and at n = 1023; <= 1.00 at n = 2047.
2. Deflated instances with n <= 1023: ratio <= 0.90 in at least 80 % of them.
3. Instances that are not deflated (probe >= 0.035 or converged within 10 its): median ratio in [0.95, 1.05].
4. band and rot: median ratio <= 1.05 at every n.
5. No instance has ratio > 1.20.
6. True residual <= 1.1e-8 for both solvers on every instance.
