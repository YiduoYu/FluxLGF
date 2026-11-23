# H47 protocol: generic coarse-grid deflation (CGD) for BoomerAMG-PCG — LOCKED before any test-seed solve

## Question
Does deflating BoomerAMG-PCG with slow modes computed on a coarse rediscretization of the same PDE (no training, no
knowledge of the field lines) beat plain BoomerAMG-PCG in total time on AMG-weak anisotropic families, while costing
little where BoomerAMG is already good?

## Method (code: snn/cgd.py, snn/run_cgd.py)
1. BoomerAMG (HMIS, ext+i, P_max 4, theta 0.25, agg_nl 0, symmetric-SOR/Jacobi) on the target A (grid n).
2. Coarse rediscretization A_c of the same family/parameters on n_c = max(127, (n+1)/8 - 1); BoomerAMG on A_c.
3. Detector: 20-step Lanczos for B_c A_c (A_c inner product, full reorthogonalisation). If the smallest Ritz value
   >= tau_det = 0.06: plain PCG.
4. Otherwise 40-step Lanczos (same start vector); keep Ritz vectors with value < 0.5, at most 32; bilinear (sparse
   tensor-product) interpolation to the fine grid; A-orthonormalise; deflated PCG (x0 = W W^T b, z = (I - W (AW)^T) B r).
5. Stop ||r|| <= 1e-8 ||b||; single thread (OMP/VECLIB/OPENBLAS_NUM_THREADS=1).
Time ratio = CGD total (fine AMG setup + steps 2-4 + solve) / base total (AMG setup + PCG solve), same process.

## Test set
Families (snn/cases_param.py): circ, disk, band, rot (rot = AMG-easy control). Geometry p ~ U[0,1]^3,
chi log-uniform [1e3, 1e5], TEST_SEED = 2026 (never used before; exploration used 777 and the dev sweep 555).
n = 511 and 1023: 6 instances per family; n = 2047: 3 instances per family.

## Exploration disclosed (not confirmatory)
NOTES_explore.md, report 13 (seed 777); dev sweep results/dev_sweep.csv (seed 555, n = 511: n_c in {127, 255},
K in {40, 80}, tau in {0.3, 0.5}) and results/dev1023.log (n = 1023). Chosen: n_c 127 (255 not better at 1023),
K 40, tau 0.5. tau_det = 0.06 was chosen from the dev values (circ/disk 0.005-0.055 gain; band 0.071-0.073 lose
15-18 %, band 0.018 roughly even; rot 0.31-0.52): a narrow margin, fitted on 3 instances per family.

## Predictions
1. circ and disk: median time ratio <= 0.75 at n = 511, <= 0.65 at n = 1023, <= 0.60 at n = 2047.
2. circ and disk, n >= 1023: median iteration reduction >= 2x.
3. rot: detector skips in >= 5 of 6 instances at n = 511 and 1023; median ratio <= 1.10 at every n.
4. band: median ratio in [0.85, 1.10] at n = 511 and 1023 (no reliable gain either way).
5. No instance has ratio > 1.30.
6. Both solvers reach ||b - A x|| / ||b|| <= 1.1e-8 (true residual) on every instance.
