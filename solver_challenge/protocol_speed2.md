# H37b — block-scaled aggregation W-cycle (agg3t-w), locked before the m = 128 run

H37 confirm (m = 128, locked e212a666) showed the balanced-form variants are still 1.3–1.4× slower than BoomerAMG at
1e-8. EXPLORATORY follow-up (m = 64, see research-log): analytic 3×3 Cholesky + sparse restriction cut setup to
0.15 s; the same pieces as a plain symmetric V(1,1) cycle need 2 fine matvecs per application instead of 4.
Best: box 2, ω = 0.8, ν = 1, W-cycle coarse levels: 22 / 36 its, total 0.75 / 1.14 s vs BoomerAMG (3D) 1.52 / 2.04 s.
This variant contains no LGF; it descends from FD-LGF-3T through the ablations (LGF → block Jacobi, balanced form →
V-cycle).

Run: `run_sc.py h37b_dt<dt> 128@<dt> amg-3d agg3t-w bnn3t-fast`, dt ∈ {1e-3, 1e1}, single thread, nothing else running.

Predictions (m = 128)
1. agg3t-w reaches both targets for both dt with ≤ 30 / ≤ 50 iterations to 1e-8.
2. agg3t-w total time ≤ 0.8× BoomerAMG (3D settings) for both targets and both dt.
3. bnn3t-fast with the new setup / restriction code is ≥ 15 % faster than in H37 (16.3 / 23.1 s at 1e-8).
4. Peak memory ≤ 9 GB.
