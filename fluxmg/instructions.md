# Autoresearch round 1: FluxMG against BoomerAMG

## Goal

A solver that uses no AMG and beats BoomerAMG (theta 0.25) on most equations, at the same accuracy.

## System

FluxMG solves 7-point cell-coupled diffusion-reaction systems on tensor grids. It stores face conductances and the row excess, never a diagonal, so 1e17 entries do not cancel. Parts:

- piecewise-constant 2x2x2 hierarchy with added-up coarse operators, semi-coarsening, K-cycle
- block red-black smoothing, adaptive zebra line smoothing
- one constant vector per diffusion-dominated region, decaying through neighbouring free cells, solved by GTH elimination

## Files

**Constrained, may be edited:** `flux_mg.py`, `kernel.c`. They must implement the same algorithm; `EQUIVALENCE` checks this.

**Frozen:** `eval_fluxmg.py`, `try_one.py`, `amg_baseline.json`, and everything outside `FD_LGF/fluxmg/`. That covers the generators and the suite.

**Levers in `flux_mg.py`:** `CYCLE`, `NU_PRE`, `NU_POST`, `ALPHA`, `K_THRESH`, `COARSEST`, `REACTION`, `THETA`, `DEFLATE`, `TAU`, `RHO`, `MIN_REGION`, `MAX_REGIONS`, `ADAPT_ROUNDS`, `ADAPT_CYCLES`, `ADAPT_JUMP`, `LINE`, `ANISO`, `SEMI`, `SEMI_RATIO`, `PROBE_BOXES`, `PROBE_TAU`. Changes to the algorithm are allowed too: interpolation, coarse faces, smoother, region vectors, porting setup to C.

## Eval

```sh
cd FD_LGF/fluxmg
OMP_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python -u eval_fluxmg.py
```

It takes about 90 s while the graded-box case fails, and less once it converges.

- **SCORE**, higher is better: minus the mean of log2(FluxMG time / BoomerAMG time) over 19 cases. A failed case counts as ratio 8.
- **Baseline:** -0.0082, the median of 3 runs (-0.0160, -0.0082, +0.0323). 13 or 14 wins of 19, 1 failure.
- **Noise:** sd about 0.015 from 4 runs of one code. **Min-delta is 0.04**, changed from 0.10 once noise was measured.

## Each iteration

1. Make one change. Say what you expect and why.
2. Run the eval. If SCORE improved by more than the min-delta, run it again as confirmation.
3. **Keep** only if all of these hold:
   - both runs beat the current best by more than the min-delta
   - `FAILS` did not increase
   - `EQUIVALENCE` is below 1e-12
   - `PEAK_RSS_GB` did not grow by more than 10 %
4. To keep: `cp flux_mg.py kernel.c best/`. Otherwise revert: `cp best/flux_mg.py best/kernel.c .`
5. Append a line to `autoresearch.jsonl` and regenerate `autoresearch_dashboard.md`.
6. **Tie-break (user decision, 2026-11-06):** if the suite SCORE is within 0.04 of the best and the real SC26_01 time improves by at least 10 %, that also counts as a keep. Run `try_one.py real` for this. A setting counts only if neighbouring parameter values also do well on the real matrix, because its iteration count is sensitive.
7. Every 5 keeps, run `eval_fluxmg.py --real` and log the real-matrix ratio. A large regression stops the loop for review.

## Strategy, in order

1. **Convergence on hard cases.** `gen:64:10:8:0.5:2:2e-2:3` fails, and the real SC26_01 matrix takes about 110 iterations against 34. What is known so far:
   - the slow mode is made of near-constant pieces around graded transitions, and diagnose scripts show that
   - exact block solves on free cells did not help
   - `RHO` changes the result strongly
   - Try coefficient-dependent interpolation (Kwak; Khalil and Wesseling) and upscaled coarse faces (series-parallel) before more region logic.
2. **Weak classes.** `insul`, `perc3`, `mg easy` (setup cost, K=large), `aniso:64:100` (line smoothing is slow to set up in Python).
3. **Cost.** Port setup to C. Try V or W cycles with a stronger smoother, and fewer matvecs around deflation.

Stop early if SCORE has not improved for 8 iterations in a row; report that the ceiling is likely architectural.

## Do not

- Use BoomerAMG or any AMG package inside the method.
- Detect a case by name, size or known coefficient values; every rule must work from the matrix alone.
- Edit the suite, the baseline or the generators, or read held-out results during the loop.
- Accept a change on iteration count alone; SCORE is time.
- Use `-ffast-math`.
- Commit to git; the user decides that.

Any report must state what is single-thread, what is on our own generated matrices, and that the multigroup cases are analogues of Yue et al. 2026, not their matrices.

## Resuming

If you lose context, read `autoresearch.jsonl` and `autoresearch_dashboard.md`; the header has the parameters. Continue from the last iteration number.

## At the end

Run `eval_fluxmg.py --holdout` and `--real` on `best/` and on the round-start code. Write the comparison with what worked and what did not, and say plainly if the held-out cases gain less than the suite: that is overfitting.
