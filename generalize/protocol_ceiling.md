# H45 — ceiling diagnostics for generic add-ons to BoomerAMG (generalisation study)

Goal (user/professor 2026-11-04): a method that is NOT special to SC26_01, generalises across PDE classes, may use neural
networks (SNN = subspace NN, Xu & Sheng arXiv 2404.08223; DeepONet trunk-basis coarse spaces, Kopanicakova-Karniadakis
2401.02016 / 2508.00101; learned AMG, Luz et al. ICML 2020; AMG-parameter learning, Caldana-Antonietti-Dede), and beats
BoomerAMG. Before building any learned component, measure the best gain each family could give (oracle ceilings).

Suite (suite.py, own discretizations): lognorm3, tubes, insul, perc3 (3D scalar, contrast 1e6, n=64); circ (2D aligned
anisotropy 1e4, n=511); 3t-base, 3t-hard (G=1), 3t-feed (F=1, G=1e-2) (gen3t m=64).
Solver: numpy right-preconditioned GMRES(50) with one re-orthogonalisation pass, BoomerAMG (H40 amg-3d settings) V-cycle,
rtol 1e-8 on the GMRES residual estimate (solver-reported criterion), x0 = 0, max 500, single thread.
Diagnostics: D0 baseline; D1 oracle deflation k in {4,16,48} (harmonic Ritz vectors of A M^-1 from 100 Arnoldi steps,
z = M r + Q (r - A M r)); D2 x0 = x*(1 + eps eta), eta smooth random field, eps in {1e-1, 1e-2}; D3 36-point AMG grid
theta {0.25,0.5,0.7} x agg_nl {0,1,2} x P_max {2,4} x relax {sym-SOR/Jacobi, l1-scaled SOR/Jacobi}.
Before this lock: smoke test only at small sizes (lognorm3 n=24, circ n=127, 3t-hard/base m=24): 3T smallest |theta| 0.41,
deflation 16 -> 15 its; circ |theta_1| 1.5e-3, deflation 178 -> 48 (k=16); lognorm3 12 -> 10.

Predictions:
1. D1: on the 7 non-circ cases, k=16 deflation cuts iterations by <= 20 % (median); on circ by >= 50 %.
2. D2: eps = 1e-2 saves >= 25 % of iterations (median over 8); eps = 1e-1 saves <= 20 %.
3. D3: the best setting per case has total time <= 0.8x the default (median over 8), and no single setting is within 10 %
   of the per-case best on all 8 cases.
4. Smallest harmonic Ritz |theta_1| > 0.2 on all 7 non-circ cases and < 0.01 on circ.
