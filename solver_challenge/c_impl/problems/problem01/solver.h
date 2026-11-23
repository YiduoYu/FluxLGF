/* SolverChallenge26 problem 01 solver interface.
 *
 * MPI iterative solver: electron-elimination block-triangular preconditioner (H41 "bt-v") inside GMRES(50), wrapped in
 * mixed-precision iterative refinement with a double-double residual and solution (written out as __float128 when the
 * harness is built with SC26_SOLUTION_FP128). Requires hypre (BoomerAMG). The serial entry points are stubs. */
#ifndef SC26_PROBLEM01_SOLVER_H
#define SC26_PROBLEM01_SOLVER_H

#include <string.h>

#include "solverchallenge26/solver_input.h"

#ifdef MPI_USE
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#endif

struct MySolver {
    int ready;
#ifdef MPI_USE
    MPI_Comm comm;
    int rank, size;
    int N, c0, nloc, n;               /* cells per field, first owned cell, owned cells, local unknowns (3 nloc) */
    int nin;                          /* rows of the harness partition on this rank */
    /* redistribution harness rows -> cell owner (all three fields of a cell on one rank) */
    int *rd_sc, *rd_sd, *rd_rc, *rd_rd;
    int *rd_sorder;                   /* harness row index, in send order */
    int *rd_rpos;                     /* local unknown of each received row, in receive order */
    /* local CSR of the full system: columns < n are owned, >= n are halo entries */
    int *rp, *ci;
    double *va, *b;
    int *gci;                         /* global column of every entry (kept for the hypre matrices) */
    /* halo exchange */
    int nhalo;
    int *h_sc, *h_sd, *h_rc, *h_rd, *h_sidx;
    double *h_sbuf, *h_rbuf;
    /* diagonal inter-field couplings per owned cell */
    double *a12, *d22, *a23, *a32;
    /* hypre: radiation block A11 and ion Schur complement S3 */
    HYPRE_IJMatrix ijA11, ijS3;
    HYPRE_ParCSRMatrix A11, S3;
    HYPRE_IJVector ijb1, ijx1, ijb3, ijx3;
    HYPRE_ParVector b1, x1, b3, x3;
    HYPRE_Solver amg11, amg3;
    /* GMRES and refinement work space */
    int restart;
    double *V, *Z, *H, *cs, *sn, *g, *w, *xe;
    double *xh, *xl, *r, *d;
#endif
};

struct MySolver_complex {
    int unused;
};

/* the harness keeps the solver struct on the stack without initialising it */
#define SC26_SOLVER_CREATE(solver, input) memset((solver), 0, sizeof(*(solver)))

#ifdef DIRECT_SOLVER
void preprocess(struct MySolver* solver, const struct SC26_SolverInput* input);
void direct_solver(struct MySolver* solver, const struct SC26_SolverInput* input);
void solve(struct MySolver* solver, const struct SC26_SolverInput* input);
void preprocess_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
void direct_solver_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
void solve_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
#endif

#ifdef ITERATIVE_SOLVER
void analyse(struct MySolver* solver, const struct SC26_SolverInput* input);
void preprocess(struct MySolver* solver, const struct SC26_SolverInput* input);
void iterative_solver(struct MySolver* solver, const struct SC26_SolverInput* input);
void analyse_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
void preprocess_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
void iterative_solver_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input);
#endif

#endif
