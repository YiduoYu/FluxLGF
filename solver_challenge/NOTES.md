# SolverChallenge26_01 — can FD-LGF solve it? (H35)

User request (2026-11-03): "用这个数据进行测试，看看FD-LGF可不可以，如果不可以，请你研究相关文献和工作，尝试进行改进".
Challenge: 3D three-temperature (3T) radiation-hydrodynamics energy equations (laser fusion), 128³ structured grid,
FV 7-point, order 6,291,456, nnz 52,133,888, MD5 c0d0a33e199a7746b0d9ba62396a0e79; score = end-to-end time to
‖r‖/‖b‖ < 1e-4 and < 1e-8 with ≤ 9 GB memory.

## Status
- The real matrix is NOT on this machine and not public (web search found no download). Waiting for the user's file.
- The order and nnz pin down the structure: 3 × 128³ unknowns (Te, Ti, Tr); each temperature a 7-point block
  (7N − 6·128² = 14,581,760 nnz) plus Te–Ti and Te–Tr exchange couplings in the same cell (4N). 3 × 14,581,760 +
  4 × 2,097,152 = 52,133,888 exactly. No Ti–Tr coupling.
- `surrogate3t.py` builds a matrix with exactly this sparsity from a physical capsule + radiation-wave state
  (K_e ∝ Te^2.5, K_i ∝ Ti^2.5, K_r ∝ Tr^3/ρ², exchange ∝ ρ² Te^-1.5 / Te^-0.5). SPD M-matrix.
- `load_real.py` (MD5 check, ordering detection, symmetry / M-matrix check) and `run_sc.py OUT MATRIX[,RHS]` are ready.

## Exploratory results on the surrogate
- m = 32 and 64: whole-matrix BoomerAMG + CG: 2 its to 1e-4, 5 its to 1e-8 (m = 64: 1.4 / 1.8 s, 0.94 GB).
  Usual 3D settings (HMIS, ext+i, θ = 0.5): more its (5 / 11) but cheaper setup (1.0 / 1.4 s).
- FD-LGF does not work: K varies by 1e7 and is not separable, so 98–99 % of cells fall in the feature mask (the
  local LU would be the whole 6.3M-unknown 3D system: does not fit 9 GB). Background-only P = G: no convergence in
  2000 its (residual 0.2). Green–Jacobi scaled background (D^-1/2 G D^-1/2): converges but 1500–1700 its.
- Literature methods (physics blocks): block SGS over temperatures + BoomerAMG per block 800+ its; PCTL (equal-
  temperature coarse space + block AMG smoother) 634 its at m = 32; with exact block solves 163 its at m = 12.
  The exchange terms dominate the diagonals (w/diag ≈ 0.99–0.9999), and whole-matrix classical AMG treats the
  exchange links as strong connections, which the blockwise methods cannot.
- Systems ("nodal") BoomerAMG is much worse (177 / 580 its).
- Full size m = 128 (order 6,291,456, same nnz as the challenge), single thread, M1 Pro:
  BoomerAMG defaults 3 / 5 its, 17.6 / 21.6 s (setup 13 s); BoomerAMG 3D settings 5 / 12 its, 9.4 / 14.9 s
  (setup 6–7 s); peak RSS 4.2–4.5 GB (< 9 GB). m = 64: FD-LGF background only 2000 its, residual 0.61;
  Green–Jacobi 666 / 1801 its (15.9 / 42.1 s vs BoomerAMG 1.0 / 1.4 s).

## Caveat
The surrogate is an SPD M-matrix, the best case for classical AMG. Published 3T matrices (Yue et al. 2022; Shu et al.
2021) are often nonsymmetric / not M-matrices after linearisation and row scaling; that is where PCTL-type and
block-triangular AMG preconditioners beat plain BoomerAMG. Conclusions about the real matrix need the real file.

## H36: improved FD-LGF (FD-LGF-3T, `fdlgf3t.py`) — report `report_sc/report_sc.pdf`
- Balanced form kept; M = 3×3 point-block Green–Jacobi LGF (DST-II) + block-Jacobi smoothing ω = 0.6;
  Q = box-constant (4³) Galerkin coarse space, recursive W-cycle (128³ → 32³ → 8³, LU).
- Ablation (`ablation_fdlgf3t.py`, m = 16/32/64): 3000+ → 1182–1801 (scalar GJ) → 53/112/226 (3×3 scaling)
  → 40/50/47 (+ coarse) → 21/27/25 (+ smoothing) → 21/27/32 (multilevel).
- m = 128 confirm (protocol eb34886e, all 3 predictions supported): to 1e-8 37 its / 26.2 s (dt 1e-3), 53 its / 33.8 s
  (dt 10); BoomerAMG 3D settings 14.0 / 17.3 s; default BoomerAMG 19.6 / 40.8 s. Peak RSS ≤ 5.7 GB.
- Next: real matrix (nonsymmetric → 3×3 block LU scaling + GMRES); speed-ups (float32 / threaded DST, compiled kernels).

## H37 / H37b: speed-up (protocols `protocol_speed.md` e212a666, `protocol_speed2.md` f3f5516d)
- The LGF part contributes nothing here (m = 64: block Jacobi instead of the Green function 32 vs 32, 56 vs 54 its;
  percentile-fitted or masked Green functions worse). What works: 3×3 block scaling, block smoothing, box coarse space.
- Rewrite (analytic 3×3 Cholesky, row sums by formula, sparse restriction): setup 0.15 s at m = 64; float32 breaks at dt = 10.
- `agg3t-w` = form="vcycle", box 2, ω 0.8, ν 1, W-cycle, no LGF. m = 128: 4.6 / 7.4 s (dt 1e-3) and 4.4 / 10.0 s (dt 10)
  to 1e-4 / 1e-8 vs BoomerAMG-3D 9.4 / 13.0 and 10.7 / 17.2 s (0.41–0.58×), 25 / 36 its, ≤ 4.3 GB. H37b 4/4 predictions.
- m = 64 dt sweep 1e-5…1e3: 14–37 its, 0.55–0.74× BoomerAMG time.

## H38: nonsymmetric surrogates (protocol_nonsym.md, re-locked 2ed4bcd5) — report section 8
- Variants: `64@dt@emis` (emission-linearised Te–Tr exchange) and `64@dt@scaled` (rows divided by c/dt). agg3t-w uses
  inv3 (plain 3×3 inverse) and GMRES(50); run_sc.solve restarts until the TRUE residual meets the target.
- m = 128: emis — agg3t-w 0.39–0.57× BoomerAMG-3D; at dt 10 / 1e-8 BoomerAMG stalls (500 its, 3.2e-8, 376 s) while
  agg3t-w needs 35 its / 11.9 s. scaled — agg3t-w 15.5 / 28.0 s (dt 1e-3), 8.9 / 41.9 s (dt 10); BoomerAMG-3D + PETSc
  diagonal scaling 52.5 / 49.8 s (dt 1e-3) and no convergence in 500 its at dt 10. Both predictions supported.

## H39: the REAL matrix (2026-11-04) — protocol_real2.md 8b86920e
- File: ~/Downloads/SolverChallenge26-tenproblems/solverchallenge26_01.7z (MD5 of the .7z = challenge MD5 c0d0a33e…;
  the extracted A.dat is c26a1d64…). Contents: PETSc binary A (6,291,456, nnz 52,133,888) and b. Cached as
  extracted/A.npz, b.npy (load 3 s, 2.9 GB RSS).
- Structure: blocked [T1; T2; T3], each a symmetric 7-point block; couplings T1–T2 and T2–T3 only (T2 is the hub —
  in the surrogate the hub was the first block). No positive off-diagonals (M-matrix sign pattern).
- NONSYMMETRIC: diagonal blocks and the T2–T3 coupling are exactly symmetric, but a12/a21 ≈ 1.2e12 (2.3e10–1.2e12,
  varies by cell), so it is not a row-scaled symmetric matrix. Global max|A−Aᵀ|/max|A| = 9.8e-12 only because
  max|A| = 1.8e17 — that number hides the asymmetry.
- Far more extreme than the surrogate: face-coefficient contrast 1e16 (T1) and 4e19 (T3); T1, T3 are almost pure
  diffusion (reaction/diag median 3e-10, 1e-10); T2 is reaction-dominated (diffusion/diag 2e-4, contrast 95).
  Exchange/diag median ≈ 0 (max 0.99–0.999999): coupling is strong only in a few cells.
- b: T1 entries 6e-16…7.6e-8 (negligible in ||b||), T2 0.0017…0.15, T3 0.057…0.15.
- Coefficient map: near-constant background (each block constant face coefficient + tiny reaction, Neumann/symmetry
  faces; T1 has a weak Robin term on the x/y/z = 127 faces); all variation and all exchange coupling sit in a 1/8 ball of
  radius ≈ 57 cells at the origin corner (1.8 / 3.8 / 4.4 % of cells differ > 2x from the median). This is the setting
  original FD-LGF (DCT background + local feature solve) was built for — the surrogate (98 % features) was wrong here.
- PRECISION FLOOR: ||b − Ax||/||b|| < 1e-2 is NOT attainable in float64. ~26k T3 rows with diag 1.5–1.8e17 nearly cancel;
  a 1-ulp change of x changes the exact residual by 2.2e-2 ||b|| (diag_floor.py, exact_residual.py with Fraction sums).
  BoomerAMG-3D + GMRES: estimate 1.2e-11 after 56 its, exact residual 7.3e-3 (all in T3) — it is at the floor; its
  solution has diag-scaled residual 2e-16, normwise backward error 4e-17, componentwise 2e-14 (i.e. it is solved).
- real_main (stopped): amg-3d 8.3e-3 / 8.2e-3 after 10 restarts (floor); agg3t-w DIVERGES (350 its, GMRES breakdown,
  true residual 1.15) — box aggregation does not handle the 1e16–1e19 jumps. FD-LGF-3T skipped (user, 2026-11-04).

## H40 — real matrix, solver-reported residual criterion (2026-11-04)
- amg-3d 13.8 s / 18.1 s (15 / 23 its), amg default 38.2 / 36.6 s (6 / 10 its); true res 1.3e-2 (float64 floor), diag-scaled 4e-13 at 1e-8.
- fdlgf-real 58.2 / 81.9 s (6 / 14 its), diag-scaled 1.2e-9, RSS 5.76 GB. Setup 45 s of which MUMPS LU of A_FF (403,275 unknowns, 3.25M nnz) 42 s; one apply 1.15 s (Q ~0.2-0.3 s x2, G 0.34 s, A@ 0.12 s x2).
- agg3t-w, pctl, split-sym: GMRES breakdown; fdlgf-G, fdlgf-GJ: no progress in 2000 its.
- H40 preds 1,2,3 hold; 4 refuted (4.5x slower). H35: 1 refuted (mask part), 2,3,4 hold.
- First fdlgf-real run crashed (read-only Vec, getArray); fixed to readonly=True, no design change.
- Feature mask outside the ball = ~2,000 cells near cube edges (background fit misses edge rows).
