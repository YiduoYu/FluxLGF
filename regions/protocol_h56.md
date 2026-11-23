# H56 — time-step sequences with setup reuse: bt-fd vs BoomerAMG on generalized geometries (locked test)

Question: in a time-stepping setting (fixed geometry, coefficients drifting, new right-hand side every step) does bt-fd
with its preconditioner built once (factorisation reuse) beat BoomerAMG with its setup reused, beyond concentric spheres?
No AMG inside our method; BoomerAMG is the yardstick only.

Methods (seq_lock.py; both PETSc GMRES(50), right preconditioning, x0 = 0, solver-reported rtol 1e-8, single thread):
  bt-fd : BTFD(A_0, multi=False, tol=0.1) built at step 0, kept for all steps (python PC, reuse)
  amg   : BoomerAMG with the AMG3D settings set up at step 0, kept (PCSetReusePreconditioner)
Metric: total time over the sequence (setup at step 0 + all solves); matrix generation not timed.
Sequences: gen_gen geometries at m = 128, 10 steps, phase 0.15 per step, rhs seed 100 + step.
Changes since H55 lock: rsolve gains MIN_REGION / BLR / ORDERING (defaults unchanged), K = 0 fix in DeflatedFDLGF,
ScalarFDLGF factorisation moved to a _factor method (same calls); tol 0.1 chosen during H56 exploration.

  G1  128:1:2:1:0:2e-2:1    sphere                    (SEEN: identical to the 5-step dev sequence)
  G2  128:1:1:1:0:2e-2:1    octahedron                (dev at m = 64 only)
  G3  128:1:8:0.5:0:2e-2:1  oblate box                (dev at m = 64 only)
  G4  128:1:2:2:0:2e-2:1    prolate sphere, aspect 2  (new)
  G5  128:1:2:1:4:2e-2:1    graded transition, 4 cells (dev single solve at m = 128)
  G6  128:1:2:1:0:1e-1:1    sphere, 10 % k(T) field    (new)
  G7  128:3:2:1:0:2e-2:1    3 random capsules         (new)
  G8  128:10:2:1:0:2e-2:1   10 random capsules        (new seed)
  G9  128:10:2:1:0:2e-2:2   10 random capsules        (new seed)
  G10 128:10:8:0.5:2:2e-2:3 10 oblate boxes, graded 2 (new)
  G11 128:20:2:1:0:2e-2:1   20 random capsules        (new; expected loss)

Predictions:
1. All 220 solves converge (reason > 0) with diagonally scaled true residual <= 1e-8.
2. G1-G6 (one capsule): bt-fd total <= 0.75x amg total in each of the 6 sequences.
3. G7-G10 (3-10 capsules): bt-fd faster than amg in >= 3 of 4.
4. G11 (20 capsules): bt-fd slower than amg (expected negative).
5. Frozen preconditioner lasts: in every sequence bt-fd its at step 9 <= its at step 0 + 2.
6. Median bt-fd / amg total-time ratio over G1-G10 <= 0.80.
Locked: 2026-11-05T12:33:47Z
