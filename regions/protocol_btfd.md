# H52 — bt-fd: an AMG-free solver for SC26_01-class 3T matrices (electron elimination + per-block scalar FD-LGF)

User request (2026-11-05): beat BoomerAMG with a method that does not use AMG; follow the region plan; borrow from papers.
Method (fixed; btfd.py, rsolve.py): bt-v block structure (drop A21; eliminate the electron field by its diagonal;
ion Schur S3 with the cancellation-free diagonal; S3 first, then radiation A11). Each scalar block M is solved by ONE
application of the scalar FD-LGF: fitted constant-coefficient background (median face coefficient, median reaction,
far-face Robin) inverted by the fast diagonalization method; feature F = cells whose row differs from the background by
> 1e-6 of the row's largest entry, dilated by 2 cells; exact CHOLMOD Cholesky (MUMPS LU if unsymmetric) on F; balanced
form written without products with the feature rows: P r = [q_F - M_FF^-1 M_FC z_C ; z_C], z = G(r_C - M_CF q_F),
q_F = M_FF^-1 r_F. No AMG anywhere.

Exploration before this lock (real matrix + gen3t m = 64 base and (1e-3, 1e-2); all exploratory):
- feasibility: S3 splits into outside (2.02M cells, leak 6e-8, 99.4 % faces within 2x) / core+layer / diagonal shell;
  A11 one component, 98.2 % faces within 2x of 6.1e9. DCT 128^3 0.046 s.
- oracle: S3 11 PCG its, A11 10 its to 1e-10 (background alone: no convergence in 300).
- first bt-fd: 6 its to 1e-4 but stalled at 1e-8 (44-55 its): P was nonlinear at 6e-8 (products with the 1e17 core rows);
  rewritten -> linearity 2e-16, 16 its. ring 2 > ring 1 (37 its) > ring 0 (53) at 1e-8. CHOLMOD 1.35 s vs MUMPS 1.9 s.
- real (H40 criterion): bt-fd 8.9 / 13.4 s vs amg-3d 14.4 / 18.4 s, bt-v 11.9 / 15.6 s. gen3t m=64: 0.6/0.7 vs 1.2/1.7 s.

Tests (never run with bt-fd before this lock):
  R  real SC26_01, H40 criterion (GMRES(50) residual estimate, rtol 1e-4 and 1e-8, x0 = 0), methods amg-3d, bt-v, bt-fd;
     3 sequential passes, median total time. (run_real.py)
  IR real SC26_01, official criterion ||b - Ax||/||b|| by double-double iterative refinement, inner GMRES rtol 1e-6,
     methods amg-3d, bt-v, bt-fd, time to 1e-4 and 1e-8 (run_ir.py)
  G  own 3T discretization m = 128 (6.3M unknowns), geo:128:r_core:r_shell:cap:F:G =
     43:54:1:2e-11:2e-4 (base), 25:32:1, 60:66:1, 30:70:1 (geometry), 43:54:0.1 and 43:54:10 (heat capacity / dt),
     43:54:1:1e-3:1e-2 (stronger feedback and electron conduction); methods amg-3d, bt-v, bt-fd, both tolerances.
All single thread, sequential, Mac M1 Pro.

Predictions:
1. R: median bt-fd total time <= 0.75x amg-3d at 1e-4 and <= 0.85x at 1e-8; bt-fd faster than bt-v at both.
2. IR: bt-fd reaches ||b - Ax||/||b|| < 1e-8 in less time than both amg-3d and bt-v.
3. G: bt-fd converges in all 14 (case x tol) runs and is faster than amg-3d in >= 12 of 14.
4. G: bt-fd iterations <= 0.7x those of amg-3d at 1e-4 in >= 6 of 7 cases.
5. Feature fraction before the ring <= 12 % for both scalar blocks in all 8 matrices.
