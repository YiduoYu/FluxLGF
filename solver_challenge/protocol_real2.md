# H39 — agg3t-w / FD-LGF-3T on the REAL SolverChallenge26_01 matrix (locked before reading the matrix contents)

File: ~/Downloads/SolverChallenge26-tenproblems/solverchallenge26_01.7z (MD5 c0d0a33e199a7746b0d9ba62396a0e79 = challenge)
-> solverchallenge26_01_A.dat, _b.dat (PETSc binary). Only the headers (order 6,291,456, nnz 52,133,888) have been seen.

Run: load_real.py (describe), then run_sc.py with the given b (not A·1): amg, amg-3d, agg3t-w, fdlgf3t, plus the
H35 list (pctl, split-sym, fdlgf-G, fdlgf-GJ). CG if symmetric, else right-preconditioned GMRES(50); true-residual
restart rule; MAXIT 2000; single process, hypre_env. Targets ||r||/||b|| < 1e-4 and < 1e-8. If the matrix is not in a
form agg3t-w can use (not 3 x 128^3 temperatures, or ordering not detected), report that and adapt before predicting
anything about speed (adaptation is EXPLORATORY).

Predictions:
1. agg3t-w reaches both targets with peak RSS < 9 GB.
2. agg3t-w total time <= 0.8x the faster BoomerAMG setting (amg / amg-3d) for both targets.
3. FD-LGF-3T reaches both targets and is slower than agg3t-w.
4. H35 predictions (protocol_real.md) are evaluated as locked.
