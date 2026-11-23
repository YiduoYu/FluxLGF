# Autoresearch round 2: cluster pivots for SSR

## Goal
Keep the convergence of signed second-order elimination (LUMP=3, THETA=0 gave elasticity 169 -> 58 its) while bringing operator complexity from above 100 down to about 5. Without that it is not usable.

## Files
Constrained: `ssr.c`, `ssr.py`. Frozen: `problems.py`, `krylov.py`, `baseline.py`, `eval_ssr.py`, `final.py`, `amg_baseline.json`, `v1_locked/`.
Revert: restore both constrained files from the last kept copy. Never look at `HOLDOUT` during the loop.

## Eval
`OMP_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python -u eval_ssr.py`
SCORE = -mean log2(t_ssr / t_fast.25), failure counts as ratio 8. Start: dev -0.335 from `v1_locked/`. Min-delta is not set yet: run the eval 3 times first and use 2-3 times the spread. Run nothing else during a timed run, and watch vm_stat Swapouts.

## Keep a change only if
SCORE rises by more than min-delta on a confirmation run; no new failed case; multigroup, graph and convection wins stay within 10 %; peak memory stays below 2 times BoomerAMG.

## Order
1. Group-independent sets (ARMS): small mutually decoupled clusters, exact block pivot, so a level eliminates about half of the unknowns.
2. Second-order dropping per row of L^-1 A_FC (Tismenetsky-Kaporin).
3. Preserve the constant and relaxed vectors in what is dropped (Klockiewicz-Darve), because plain positive semidefinite dropping loses the near kernel.
4. Only then tune cluster size, KEEP, smoothing.

## Do not retry (see autoresearch.jsonl)
Signed fill with chain sampling, keeping a fraction of same-sign entries, midpoint collapse, moment-fitted M-matrices, exact elimination on the first levels only, half-circle plan on symmetric matrices, coarse DROP.

## Log
Append each iteration to `autoresearch.jsonl`, regenerate `autoresearch_dashboard.md`. After context loss, read those two first. Stop after 5 flat iterations and say the ceiling is architectural.
