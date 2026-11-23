/* Thin interface to hypre's structured multigrid, one V-cycle per apply.
   build: clang -O3 -shared -fPIC hypre_struct.c -I$ENV/include -L$ENV/lib -lHYPRE -lmpi -Wl,-rpath,$ENV/lib -o libhypre_struct.dylib

   Grid: hypre index (x, y, z) with x fastest, the cell order of the matrices here (z slowest). Serial, MPI_COMM_SELF.
   st: 7 values per cell: centre, -x, +x, -y, +y, -z, +z; entries leaving the domain must be zero. */
#include <mpi.h>
#include <stdlib.h>
#include "HYPRE_utilities.h"
#include "HYPRE_struct_ls.h"

static HYPRE_Int OFF[7][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

typedef struct {
    int smg;
    HYPRE_Int lo[3], hi[3];
    HYPRE_StructGrid grid;
    HYPRE_StructStencil stencil;
    HYPRE_StructMatrix A;
    HYPRE_StructVector b, x;
    HYPRE_StructSolver s;
} Struct;

void *struct_new(int smg, int nz, int ny, int nx, double *st, int relax, int rap, int npre, int npost, int skip, int maxit, double tol)
{
    Struct *h = malloc(sizeof(Struct));
    HYPRE_Int entries[7] = {0, 1, 2, 3, 4, 5, 6};
    if (!HYPRE_Initialized()) HYPRE_Initialize();
    h->smg = smg;
    h->lo[0] = h->lo[1] = h->lo[2] = 0;
    h->hi[0] = nx - 1; h->hi[1] = ny - 1; h->hi[2] = nz - 1;
    HYPRE_StructGridCreate(MPI_COMM_SELF, 3, &h->grid);
    HYPRE_StructGridSetExtents(h->grid, h->lo, h->hi);
    HYPRE_StructGridAssemble(h->grid);
    HYPRE_StructStencilCreate(3, 7, &h->stencil);
    for (int e = 0; e < 7; e++) HYPRE_StructStencilSetElement(h->stencil, e, OFF[e]);
    HYPRE_StructMatrixCreate(MPI_COMM_SELF, h->grid, h->stencil, &h->A);
    HYPRE_StructMatrixInitialize(h->A);
    HYPRE_StructMatrixSetBoxValues(h->A, h->lo, h->hi, 7, entries, st);
    HYPRE_StructMatrixAssemble(h->A);
    HYPRE_StructVectorCreate(MPI_COMM_SELF, h->grid, &h->b);
    HYPRE_StructVectorCreate(MPI_COMM_SELF, h->grid, &h->x);
    HYPRE_StructVectorInitialize(h->b);
    HYPRE_StructVectorInitialize(h->x);
    HYPRE_StructVectorSetConstantValues(h->b, 0.0);
    HYPRE_StructVectorSetConstantValues(h->x, 0.0);
    HYPRE_StructVectorAssemble(h->b);
    HYPRE_StructVectorAssemble(h->x);
    if (smg) {
        HYPRE_StructSMGCreate(MPI_COMM_SELF, &h->s);
        HYPRE_StructSMGSetMaxIter(h->s, maxit);
        HYPRE_StructSMGSetTol(h->s, tol);
        HYPRE_StructSMGSetNumPreRelax(h->s, npre);
        HYPRE_StructSMGSetNumPostRelax(h->s, npost);
        HYPRE_StructSMGSetZeroGuess(h->s);
        HYPRE_StructSMGSetup(h->s, h->A, h->b, h->x);
    } else {
        HYPRE_StructPFMGCreate(MPI_COMM_SELF, &h->s);
        HYPRE_StructPFMGSetMaxIter(h->s, maxit);
        HYPRE_StructPFMGSetTol(h->s, tol);
        HYPRE_StructPFMGSetRelaxType(h->s, relax);
        HYPRE_StructPFMGSetRAPType(h->s, rap);
        HYPRE_StructPFMGSetNumPreRelax(h->s, npre);
        HYPRE_StructPFMGSetNumPostRelax(h->s, npost);
        HYPRE_StructPFMGSetSkipRelax(h->s, skip);
        HYPRE_StructPFMGSetZeroGuess(h->s);
        HYPRE_StructPFMGSetup(h->s, h->A, h->b, h->x);
    }
    return h;
}

void struct_apply(void *p, double *r, double *z)
{
    Struct *h = p;
    HYPRE_StructVectorSetBoxValues(h->b, h->lo, h->hi, r);
    HYPRE_StructVectorAssemble(h->b);
    HYPRE_StructVectorSetConstantValues(h->x, 0.0);
    if (h->smg) HYPRE_StructSMGSolve(h->s, h->A, h->b, h->x);
    else HYPRE_StructPFMGSolve(h->s, h->A, h->b, h->x);
    HYPRE_StructVectorGetBoxValues(h->x, h->lo, h->hi, z);
}
