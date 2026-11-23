# H33 — FD-LGF vs BoomerAMG vs GMG on high-contrast coefficients that are not functions of x alone

Locked 2026-11-03. User request: "以后的对比试验去掉传统LGF。再找找其他a … 高对比度 … 再找找有没有boomerAMG算不好的例子，
用方法二试试". Traditional LGF is no longer a baseline; 方法二 is called FD-LGF.

## Already seen (EXPLORATORY, before locking)
- 2D scan (`results/scan.csv`), n = 255, 511, all 10 families of `cases_hc.py`, χ ∈ {1e2, 1e4, 1e6, 1e8}:
  BoomerAMG 7–13 its everywhere; GMG up to 398 its (diagfine) and 101 (perc); FD-LGF fastest only on sep2;
  FD-LGF local fraction 0.59–1.0 on the non-separable families (degenerates to a sparse direct solve).
- 3D scan (`results/scan3d.csv`), n = 31, 63, 127, families of `cases_hc3d.py`: BoomerAMG 7–12 its but 8–17 s at
  n = 127 (setup 5–8 s), GMG 2–5 s; FD-LGF 0.3–0.5 s on sheets, 4.6–8.5 s on tubes, 115 s on insul.

## Confirmatory run
2D, n = 1023, all 10 families, χ ∈ {1e2, 1e4, 1e6, 1e8}, methods GMG (ω = 4/5), BoomerAMG (PETSc defaults),
FD-LGF w = 1/16 and 1/64 with the construction split of `cases_hc.py`. `bench_hc.py hc1023 1023`; same PCG, stopping
rule, timing and error trace as H31.

## Predictions (n = 1023)
1. BoomerAMG ≤ 15 iterations for every family and χ.
2. GMG needs > 100 iterations on diagfine and on perc for at least one χ, and ≥ 2× BoomerAMG's iterations on at
   least 6 of the 10 families (worst χ).
3. sep2: FD-LGF ≤ 4 iterations and total time ≤ 0.8× BoomerAMG for every χ.
4. diag, diagfine, layered, lognorm, perc, rough: FD-LGF (w = 1/16) local fraction ≥ 0.8 and best FD-LGF total time
   ≥ 3× BoomerAMG for every χ.
5. chan, walls: best FD-LGF total time ≤ 2× BoomerAMG; FD-LGF (w = 1/16) iterations ≤ BoomerAMG + 1.
