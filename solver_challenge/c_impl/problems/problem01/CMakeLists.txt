# Problem 01-specific dependencies: hypre (BoomerAMG), built with the same MPI.
# Configure with -DHYPRE_DIR=/path/to/hypre/install (containing include/ and lib/).
set(HYPRE_DIR "" CACHE PATH "hypre installation prefix")
find_path(HYPRE_INCLUDE_DIR HYPRE.h HINTS "${HYPRE_DIR}/include" REQUIRED)
find_library(HYPRE_LIBRARY HYPRE HINTS "${HYPRE_DIR}/lib" "${HYPRE_DIR}/lib64" REQUIRED)
add_library(sc26_problem_dependencies INTERFACE)
target_include_directories(sc26_problem_dependencies INTERFACE "${HYPRE_INCLUDE_DIR}")
target_link_libraries(sc26_problem_dependencies INTERFACE "${HYPRE_LIBRARY}" m)
# the double-double residual relies on exactly rounded products: no FMA contraction
target_compile_options(sc26_problem_dependencies INTERFACE -ffp-contract=off -O3)
