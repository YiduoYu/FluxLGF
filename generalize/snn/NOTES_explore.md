# H46 exploration notes (before the protocol lock; all EXPLORATORY)

1. torch_env `import torch` segfault (rc 139): libomp `pthread_mutex_init failed` on this macOS. Fix: `OMP_NUM_THREADS=1`.
   torch 2.7.1 also installed into hypre_env (pip) so torch and petsc4py/BoomerAMG share one process.
2. Baseline correction for circ: H45 used GMRES(50) + H40 settings (agg_nl 1) -> 429 its at n=511. A is SPD, so PCG is
   the right Krylov method, and agg_nl 0 is 3-4x better here: PCG theta 0.25 agg 0: 33 / 55 / 86 its (n 127/255/511).
3. H45 D1 was inconclusive because of a Lanczos bug: A w was updated by recurrence and lost A-orthogonality
   (cond(A) ~ 1e8). Recomputing A w after each re-orthogonalisation fixes it (checked against dense B A, n = 31).
4. Accurate ceiling (exact slow modes, n = 511): deflating k = 8/16/32 -> 45/30/23 its (base 86), solve 2.13 -> 0.84 s.
   The first 16 slow modes are purely radial (ring fraction 1.00); # eigenvalues < 0.1 grows with n (4, 8, 17).
5. Low eigenvectors of A capture the circ slow modes (m = 32: 29 its) but NOT disk/band (disk m = 16: 35 vs 13 with the
   true slow modes) -> an A-only loss is not generic; the target must be the AMG-preconditioned operator.
6. Slow subspace transfers across grids (127 -> 255 almost lossless; 63 -> 255 loses some) and is insensitive to chi
   (chi/10, chi*10 identical), but depends strongly on geometry (nominal-geometry basis on shifted circ: 44 vs base 46).
7. SNN training with the preconditioned Ritz loss: (a) ridge on G only -> network fakes small Ritz values with
   near-collinear columns (loss 0.2, true Ritz 0.85); (b) A-normalised columns + eps on both -> honest but slow (slow
   modes are low-energy combinations, i.e. exactly the ill-conditioned directions); (c) Euclidean QR + exact Ritz ->
   honest, but loss 0.83 -> 0.82 in 200 steps: unsupervised Ritz training is too slow.
8. Switched to supervised subspace least squares (SNN-style): targets = slowest modes of training instances.
