# H35 — predictions for the real SolverChallenge26_01 matrix (locked before the file is available)

Run: `load_real.py MATRIX` then `run_sc.py real MATRIX[,RHS] amg amg-3d pctl split-sym fdlgf-G fdlgf-GJ`
(hypre_env, single process; CG if symmetric, otherwise right-preconditioned GMRES(50); maxiter 2000).

1. FD-LGF background only (fdlgf-G) does not reach ‖r‖/‖b‖ < 1e-4 within 2000 iterations; ≥ 90 % of cells of every
   temperature block fall outside a factor 2 of the block's geometric-mean face coefficient (mask fraction ≥ 0.9).
2. Green–Jacobi scaled background (fdlgf-GJ) needs ≥ 10× the iterations of BoomerAMG for 1e-8.
3. Whole-matrix BoomerAMG (amg or amg-3d) is the fastest of the tested solvers for both targets and fits 9 GB.
4. If the matrix is an M-matrix (no positive off-diagonals): BoomerAMG ≤ 30 its to 1e-8. If it is not, PCTL needs
   fewer iterations than whole-matrix BoomerAMG on at least one target.
