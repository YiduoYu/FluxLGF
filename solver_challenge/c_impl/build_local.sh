#!/bin/sh
# Local build of MPI_iterative_solver for problem 01 without CMake (macOS, conda hypre_env with OpenMPI + hypre).
# No __float128 on macOS arm64, so the float128 checks are disabled here (SC26_DISABLE_FLOAT128).
set -e
cd "$(dirname "$0")"
E=/opt/anaconda3/envs/hypre_env
mkdir -p build-local/bin
OMPI_CC=clang "$E/bin/mpicc" -std=gnu99 -O3 -ffp-contract=off -Wall -Wextra \
    -DITERATIVE_SOLVER -DMPI_USE -DSC26_DISABLE_FLOAT128 \
    -Iinclude -Iproblems/problem01 -I"$E/include" \
    apps/MPI_iterative_solver.c problems/problem01/solver.c \
    src/mpi_system_loader.c src/diagnostics.c src/petsc_binary_reader.c src/system_loader.c \
    src/matrix_market/highlevel.c src/matrix_market/mmio.c \
    -L"$E/lib" -Wl,-rpath,"$E/lib" -lHYPRE -lm \
    -o build-local/bin/MPI_iterative_solver
echo "built build-local/bin/MPI_iterative_solver"
