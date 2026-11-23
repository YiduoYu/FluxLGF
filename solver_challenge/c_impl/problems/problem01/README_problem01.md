# Problem 01 solver (MPI + hypre)

Electron-elimination block-triangular preconditioner (two scalar BoomerAMG V-cycles: ion Schur complement S3 and
radiation block A11) inside GMRES(50), wrapped in iterative refinement with a double-double residual and solution.
With `SC26_SOLUTION_FP128=ON` the solution is handed to the harness as `__float128`, so the official float128 check sees
the refined residual (about 1e-10) instead of the float64 representation floor (about 1e-2).

## Build on the competition machine (Linux, GCC, MPICH)

```bash
module load mpich
# hypre built with the same MPICH, e.g. ./configure --prefix=$HOME/hypre && make install
cmake -S . -B build -DSC26_PROBLEM=01 -DSC26_BUILD_MPI=ON -DSC26_SOLUTION_FP128=ON -DHYPRE_DIR=$HOME/hypre
cmake --build build --parallel
```

## Run

```bash
SC26_P01_TOL=1e-8 mpirun -np 96 ./build/bin/MPI_iterative_solver A.dat b.dat 0 0   # criterion 2
SC26_P01_TOL=1e-4 mpirun -np 96 ./build/bin/MPI_iterative_solver A.dat b.dat 0 0   # criterion 1
```

Optional environment: `SC26_P01_INNER` (inner GMRES rtol, default 1e-6), `SC26_P01_VERBOSE=1` (per-step report),
`SC26_P01_DUMP=prefix` (write the double-double solution from rank 0; diagnostics only, slows the timed runs).

## Local build (macOS, no CMake, no __float128)

`../../build_local.sh` compiles with the conda `hypre_env` OpenMPI + hypre and `SC26_DISABLE_FLOAT128`.
`../../../check_dd_solution.py` checks a dumped double-double solution independently.

## Not yet verified

- The `SC26_SOLUTION_FP128` build and the float128 check (no `__float128` on macOS arm64).
- Whether the competition allows `SC26_SOLUTION_FP128` and per-criterion runs with `SC26_P01_TOL` (ask the organisers).
- Scaling and memory at 96 ranks.
