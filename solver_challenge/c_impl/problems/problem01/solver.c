/* SolverChallenge26 problem 01 solver implementation (see solver.h).
 *
 * Blocks [T1 radiation; T2 electron; T3 ion], each a symmetric 7-point operator on the same 128^3 cells; the couplings
 * A12, A21, A23, A32 are diagonal. Preconditioner (drops A21 ~ 1e-12 and the T2 off-diagonals ~ 2e-8):
 *     S3 z3 = v3 - a32 v2 / d22,   S3 = A33 - diag(a32 a23 / d22)        (one BoomerAMG V-cycle)
 *     z2 = (v2 - a23 z3) / d22
 *     A11 z1 = v1 - a12 z2                                                (one BoomerAMG V-cycle)
 * inside right-preconditioned GMRES(50). Outer loop: iterative refinement with r = b - A x and x kept in double-double,
 * so ||b - Ax|| / ||b|| is not limited by the float64 representation of x (~1e-2 on this matrix).
 *
 * Environment (optional): SC26_P01_TOL target of ||b - Ax||/||b|| (default 1e-8), SC26_P01_INNER inner GMRES rtol
 * (default 1e-6), SC26_P01_VERBOSE=1 per-step report on rank 0, SC26_P01_DUMP=prefix writes the double-double
 * solution (prefix_hi.bin, prefix_lo.bin, harness row order) from rank 0. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "solver.h"

#if defined(ITERATIVE_SOLVER) && defined(MPI_USE)
#pragma STDC FP_CONTRACT OFF
#include "_hypre_parcsr_mv.h"
/* from the harness MPI loader (its header would clash with hypre's MM_typecode typedef) */
void mpi_row_partition(int rows, int rank, int size, int* begin, int* end);
void mpi_row_counts(int rows, int size, int* counts, int* displs);

#define RESTART 50

static double env_double(const char* name, double fallback)
{
    const char* v = getenv(name);
    return v ? atof(v) : fallback;
}

static double* vdata(HYPRE_ParVector v)
{
    return hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)v));
}

static int cell_owner(int c, int N, int size)
{
    int base = N / size, extra = N % size;
    if (c < extra * (base + 1)) return c / (base + 1);
    return extra + (c - extra * (base + 1)) / base;
}

static void two_sum(double a, double b, double* s, double* e)
{
    double t = a + b, bb = t - a;
    *s = t;
    *e = (a - (t - bb)) + (b - bb);
}

static double dot(const struct MySolver* s, const double* x, const double* y)
{
    double loc = 0.0, glob;
    for (int i = 0; i < s->n; i++) loc += x[i] * y[i];
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, s->comm);
    return glob;
}

/* fill x[n .. n+nhalo) from the owners */
static void halo(struct MySolver* s, double* x)
{
    int nsend = s->h_sd[s->size - 1] + s->h_sc[s->size - 1];
    for (int i = 0; i < nsend; i++) s->h_sbuf[i] = x[s->h_sidx[i]];
    MPI_Alltoallv(s->h_sbuf, s->h_sc, s->h_sd, MPI_DOUBLE, x + s->n, s->h_rc, s->h_rd, MPI_DOUBLE, s->comm);
}

/* y = A x (x has valid halo) */
static void spmv(const struct MySolver* s, const double* x, double* y)
{
    for (int i = 0; i < s->n; i++) {
        double sum = 0.0;
        for (int k = s->rp[i]; k < s->rp[i + 1]; k++) sum += s->va[k] * x[s->ci[k]];
        y[i] = sum;
    }
}

/* r = b - A (xh + xl) in double-double (xh, xl have valid halos) */
static void dd_residual(const struct MySolver* s, const double* xh, const double* xl, double* r)
{
    for (int i = 0; i < s->n; i++) {
        double sh = 0.0, sl = 0.0, t, rh;
        for (int k = s->rp[i]; k < s->rp[i + 1]; k++) {
            double a = s->va[k], xj = xh[s->ci[k]];
            double p = a * xj, e = fma(a, xj, -p);
            two_sum(sh, p, &sh, &t);
            sl += t + e + a * xl[s->ci[k]];
        }
        two_sum(s->b[i], -sh, &rh, &t);
        r[i] = rh + (t - sl);
    }
}

/* z = M^-1 v (electron-elimination block-triangular preconditioner) */
static void precond(struct MySolver* s, const double* v, double* z)
{
    int nl = s->nloc;
    double *b1 = vdata(s->b1), *x1 = vdata(s->x1), *b3 = vdata(s->b3), *x3 = vdata(s->x3);
    for (int k = 0; k < nl; k++) {
        b3[k] = v[2 * nl + k] - s->a32[k] * (v[nl + k] / s->d22[k]);
        x3[k] = 0.0;
    }
    HYPRE_BoomerAMGSolve(s->amg3, s->S3, s->b3, s->x3);
    for (int k = 0; k < nl; k++) {
        double z3 = x3[k], z2 = (v[nl + k] - s->a23[k] * z3) / s->d22[k];
        z[2 * nl + k] = z3;
        z[nl + k] = z2;
        b1[k] = v[k] - s->a12[k] * z2;
        x1[k] = 0.0;
    }
    HYPRE_BoomerAMGSolve(s->amg11, s->A11, s->b1, s->x1);
    for (int k = 0; k < nl; k++) z[k] = x1[k];
}

/* right-preconditioned GMRES(RESTART) for A d = rhs, d0 = 0; returns iterations */
static int gmres(struct MySolver* s, const double* rhs, double* d, double rtol, int maxit, double* relres)
{
    int n = s->n, m = RESTART, its = 0, conv = 0;
    double *w = s->w, *xe = s->xe, *H = s->H, *cs = s->cs, *sn = s->sn, *g = s->g;
    double y[RESTART + 1];
    for (int i = 0; i < n; i++) d[i] = 0.0;
    for (int i = 0; i < n; i++) w[i] = rhs[i];
    double beta = sqrt(dot(s, w, w)), bnorm = beta, res = beta;
    if (bnorm == 0.0) { *relres = 0.0; return 0; }
    while (!conv && its < maxit) {
        int j;
        for (int i = 0; i < n; i++) s->V[i] = w[i] / beta;
        for (int i = 0; i <= m; i++) g[i] = 0.0;
        g[0] = beta;
        for (j = 0; j < m && its < maxit; j++) {
            double* vj = s->V + (size_t)j * n;
            double* vn = s->V + (size_t)(j + 1) * n;
            precond(s, vj, xe);
            halo(s, xe);
            spmv(s, xe, w);
            for (int i = 0; i <= j; i++) {                                      /* modified Gram-Schmidt */
                double h = dot(s, w, s->V + (size_t)i * n);
                H[i * m + j] = h;
                for (int k = 0; k < n; k++) w[k] -= h * s->V[(size_t)i * n + k];
            }
            double hn = sqrt(dot(s, w, w));
            for (int k = 0; k < n; k++) vn[k] = w[k] / hn;
            for (int i = 0; i < j; i++) {                                       /* previous Givens rotations */
                double a = H[i * m + j], b = H[(i + 1) * m + j];
                H[i * m + j] = cs[i] * a + sn[i] * b;
                H[(i + 1) * m + j] = -sn[i] * a + cs[i] * b;
            }
            double a = H[j * m + j], rr = sqrt(a * a + hn * hn);
            cs[j] = a / rr;
            sn[j] = hn / rr;
            H[j * m + j] = rr;
            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            its++;
            res = fabs(g[j + 1]);
            if (res <= rtol * bnorm) { conv = 1; j++; break; }
        }
        for (int i = j - 1; i >= 0; i--) {                                      /* H y = g */
            double t = g[i];
            for (int k = i + 1; k < j; k++) t -= H[i * m + k] * y[k];
            y[i] = t / H[i * m + i];
        }
        for (int k = 0; k < n; k++) w[k] = 0.0;                                 /* d += M^-1 (V y) */
        for (int i = 0; i < j; i++)
            for (int k = 0; k < n; k++) w[k] += y[i] * s->V[(size_t)i * n + k];
        precond(s, w, xe);
        for (int k = 0; k < n; k++) d[k] += xe[k];
        if (!conv) {                                                            /* restart from the true residual */
            for (int k = 0; k < n; k++) xe[k] = d[k];
            halo(s, xe);
            spmv(s, xe, w);
            for (int k = 0; k < n; k++) w[k] = rhs[k] - w[k];
            beta = sqrt(dot(s, w, w));
            res = beta;
            if (beta <= rtol * bnorm) conv = 1;
        }
    }
    *relres = res / bnorm;
    return its;
}

static void p01_free(struct MySolver* s)
{
    if (!s->ready) return;
    if (s->amg11) HYPRE_BoomerAMGDestroy(s->amg11);
    if (s->amg3) HYPRE_BoomerAMGDestroy(s->amg3);
    if (s->ijA11) HYPRE_IJMatrixDestroy(s->ijA11);
    if (s->ijS3) HYPRE_IJMatrixDestroy(s->ijS3);
    if (s->ijb1) HYPRE_IJVectorDestroy(s->ijb1);
    if (s->ijx1) HYPRE_IJVectorDestroy(s->ijx1);
    if (s->ijb3) HYPRE_IJVectorDestroy(s->ijb3);
    if (s->ijx3) HYPRE_IJVectorDestroy(s->ijx3);
    void* arrays[] = {s->rd_sc, s->rd_sd, s->rd_rc, s->rd_rd, s->rd_sorder, s->rd_rpos, s->rp, s->ci, s->va, s->b,
                      s->gci, s->h_sc, s->h_sd, s->h_rc, s->h_rd, s->h_sidx, s->h_sbuf, s->a12, s->d22, s->a23,
                      s->a32, s->V, s->H, s->cs, s->sn, s->g, s->w, s->xe, s->xh, s->xl, s->r, s->d};
    for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++) free(arrays[i]);
    memset(s, 0, sizeof(*s));
}

typedef struct { int key, pos; } KeyPos;

static int cmp_keypos(const void* a, const void* b)
{
    int x = ((const KeyPos*)a)->key, y = ((const KeyPos*)b)->key;
    return (x > y) - (x < y);
}

static int cmp_long(const void* a, const void* b)
{
    long long x = *(const long long*)a, y = *(const long long*)b;
    return (x > y) - (x < y);
}

/* Analysis: move every harness row to the owner of its cell, build the local CSR and the halo exchange. */
void analyse(struct MySolver* s, const struct SC26_SolverInput* in)
{
    static int hypre_started = 0;
    if (!hypre_started) {
        HYPRE_Initialize();
        hypre_started = 1;
    }
    p01_free(s);
    s->ready = 1;
    s->comm = in->comm;
    MPI_Comm_rank(s->comm, &s->rank);
    MPI_Comm_size(s->comm, &s->size);
    int P = s->size, N = in->global_n / 3, c1;
    s->N = N;
    mpi_row_partition(N, s->rank, P, &s->c0, &c1);
    s->nloc = c1 - s->c0;
    s->n = 3 * s->nloc;
    s->nin = in->local_n;
    int n = s->n, nl = s->nloc, c0 = s->c0, base = in->row_ptr[0];

    /* 1. redistribution: counts of rows and nonzeros per destination */
    s->rd_sc = calloc(P, sizeof(int)); s->rd_sd = calloc(P, sizeof(int));
    s->rd_rc = calloc(P, sizeof(int)); s->rd_rd = calloc(P, sizeof(int));
    int *zsc = calloc(P, sizeof(int)), *zsd = calloc(P, sizeof(int)), *zrc = calloc(P, sizeof(int)),
        *zrd = calloc(P, sizeof(int));
    int* dest = malloc((size_t)s->nin * sizeof(int));
    for (int i = 0; i < s->nin; i++) {
        int g = in->row_begin + i;
        dest[i] = cell_owner(g % N, N, P);
        s->rd_sc[dest[i]]++;
        zsc[dest[i]] += in->row_ptr[i + 1] - in->row_ptr[i];
    }
    MPI_Alltoall(s->rd_sc, 1, MPI_INT, s->rd_rc, 1, MPI_INT, s->comm);
    MPI_Alltoall(zsc, 1, MPI_INT, zrc, 1, MPI_INT, s->comm);
    for (int p = 1; p < P; p++) {
        s->rd_sd[p] = s->rd_sd[p - 1] + s->rd_sc[p - 1];
        s->rd_rd[p] = s->rd_rd[p - 1] + s->rd_rc[p - 1];
        zsd[p] = zsd[p - 1] + zsc[p - 1];
        zrd[p] = zrd[p - 1] + zrc[p - 1];
    }
    int nrecv = s->rd_rd[P - 1] + s->rd_rc[P - 1], zrecv = zrd[P - 1] + zrc[P - 1];
    if (nrecv != n) {
        fprintf(stderr, "problem01: rank %d received %d rows, expected %d (matrix is not 3 x N cells?)\n", s->rank,
                nrecv, n);
        MPI_Abort(s->comm, 1);
    }

    /* 2. pack rows in destination order: (global row, length), columns, values, rhs */
    int *fill = calloc(P, sizeof(int)), *zfill = calloc(P, sizeof(int));
    int *smeta = malloc((size_t)2 * s->nin * sizeof(int)), *scol = malloc((size_t)(zsd[P - 1] + zsc[P - 1] + 1) * sizeof(int));
    double *sval = malloc((size_t)(zsd[P - 1] + zsc[P - 1] + 1) * sizeof(double)), *sb = malloc((size_t)s->nin * sizeof(double));
    s->rd_sorder = malloc((size_t)s->nin * sizeof(int));
    for (int i = 0; i < s->nin; i++) {
        int p = dest[i], q = s->rd_sd[p] + fill[p]++, len = in->row_ptr[i + 1] - in->row_ptr[i];
        int z = zsd[p] + zfill[p];
        s->rd_sorder[q] = i;
        smeta[2 * q] = in->row_begin + i;
        smeta[2 * q + 1] = len;
        sb[q] = in->b[i];
        for (int k = 0; k < len; k++) {
            scol[z + k] = in->col_idx[in->row_ptr[i] - base + k];
            sval[z + k] = in->val[in->row_ptr[i] - base + k];
        }
        zfill[p] += len;
    }
    int *msc = malloc(P * sizeof(int)), *msd = malloc(P * sizeof(int)), *mrc = malloc(P * sizeof(int)),
        *mrd = malloc(P * sizeof(int));
    for (int p = 0; p < P; p++) {
        msc[p] = 2 * s->rd_sc[p]; msd[p] = 2 * s->rd_sd[p];
        mrc[p] = 2 * s->rd_rc[p]; mrd[p] = 2 * s->rd_rd[p];
    }
    int *rmeta = malloc((size_t)2 * n * sizeof(int)), *rcol = malloc((size_t)(zrecv + 1) * sizeof(int));
    double *rval = malloc((size_t)(zrecv + 1) * sizeof(double)), *rb = malloc((size_t)n * sizeof(double));
    MPI_Alltoallv(smeta, msc, msd, MPI_INT, rmeta, mrc, mrd, MPI_INT, s->comm);
    MPI_Alltoallv(scol, zsc, zsd, MPI_INT, rcol, zrc, zrd, MPI_INT, s->comm);
    MPI_Alltoallv(sval, zsc, zsd, MPI_DOUBLE, rval, zrc, zrd, MPI_DOUBLE, s->comm);
    MPI_Alltoallv(sb, s->rd_sc, s->rd_sd, MPI_DOUBLE, rb, s->rd_rc, s->rd_rd, MPI_DOUBLE, s->comm);

    /* 3. local CSR in local row order f * nloc + (c - c0) */
    s->rd_rpos = malloc((size_t)n * sizeof(int));
    s->rp = calloc((size_t)n + 1, sizeof(int));
    int* roff = malloc((size_t)n * sizeof(int));
    for (int k = 0, off = 0; k < n; k++) {
        int g = rmeta[2 * k], lr = (g / N) * nl + (g % N - c0);
        s->rd_rpos[k] = lr;
        s->rp[lr + 1] = rmeta[2 * k + 1];
        roff[k] = off;
        off += rmeta[2 * k + 1];
    }
    for (int i = 0; i < n; i++) s->rp[i + 1] += s->rp[i];
    int nnz = s->rp[n];
    s->gci = malloc((size_t)nnz * sizeof(int));
    s->ci = malloc((size_t)nnz * sizeof(int));
    s->va = malloc((size_t)nnz * sizeof(double));
    s->b = malloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) {
        int lr = s->rd_rpos[k], len = rmeta[2 * k + 1];
        for (int t = 0; t < len; t++) {
            s->gci[s->rp[lr] + t] = rcol[roff[k] + t];
            s->va[s->rp[lr] + t] = rval[roff[k] + t];
        }
        s->b[lr] = rb[k];
    }

    /* 4. halo: off-rank columns, grouped by owner (key owner * 3N + global column) */
    long long* keys = malloc((size_t)(nnz + 1) * sizeof(long long));
    int nk = 0;
    for (int k = 0; k < nnz; k++) {
        int c = s->gci[k] % N;
        if (c < c0 || c >= c1) keys[nk++] = (long long)cell_owner(c, N, P) * 3 * N + s->gci[k];
    }
    qsort(keys, nk, sizeof(long long), cmp_long);
    int nh = 0;
    for (int k = 0; k < nk; k++)
        if (nh == 0 || keys[k] != keys[nh - 1]) keys[nh++] = keys[k];
    s->nhalo = nh;
    KeyPos* lookup = malloc((size_t)(nh + 1) * sizeof(KeyPos));
    s->h_sc = calloc(P, sizeof(int)); s->h_sd = calloc(P, sizeof(int));
    s->h_rc = calloc(P, sizeof(int)); s->h_rd = calloc(P, sizeof(int));
    int* hreq = malloc((size_t)(nh + 1) * sizeof(int));
    for (int j = 0; j < nh; j++) {
        int owner = (int)(keys[j] / (3LL * N)), g = (int)(keys[j] % (3LL * N));
        s->h_rc[owner]++;
        hreq[j] = g;
        lookup[j].key = g;
        lookup[j].pos = j;
    }
    qsort(lookup, nh, sizeof(KeyPos), cmp_keypos);
    for (int k = 0; k < nnz; k++) {
        int g = s->gci[k], c = g % N;
        if (c >= c0 && c < c1) {
            s->ci[k] = (g / N) * nl + (c - c0);
        } else {
            KeyPos key = {g, 0}, *hit = bsearch(&key, lookup, nh, sizeof(KeyPos), cmp_keypos);
            s->ci[k] = n + hit->pos;
        }
    }
    MPI_Alltoall(s->h_rc, 1, MPI_INT, s->h_sc, 1, MPI_INT, s->comm);
    for (int p = 1; p < P; p++) {
        s->h_sd[p] = s->h_sd[p - 1] + s->h_sc[p - 1];
        s->h_rd[p] = s->h_rd[p - 1] + s->h_rc[p - 1];
    }
    int nsend = s->h_sd[P - 1] + s->h_sc[P - 1];
    s->h_sidx = malloc((size_t)(nsend + 1) * sizeof(int));
    MPI_Alltoallv(hreq, s->h_rc, s->h_rd, MPI_INT, s->h_sidx, s->h_sc, s->h_sd, MPI_INT, s->comm);
    for (int i = 0; i < nsend; i++) {
        int g = s->h_sidx[i];
        s->h_sidx[i] = (g / N) * nl + (g % N - c0);
    }
    s->h_sbuf = malloc((size_t)(nsend + 1) * sizeof(double));

    free(zsc); free(zsd); free(zrc); free(zrd); free(dest); free(fill); free(zfill);
    free(smeta); free(scol); free(sval); free(sb); free(msc); free(msd); free(mrc); free(mrd);
    free(rmeta); free(rcol); free(rval); free(rb); free(roff); free(keys); free(lookup); free(hreq);
}

static void make_vector(struct MySolver* s, HYPRE_IJVector* ij, HYPRE_ParVector* par)
{
    HYPRE_IJVectorCreate(s->comm, s->c0, s->c0 + s->nloc - 1, ij);
    HYPRE_IJVectorSetObjectType(*ij, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(*ij);
    HYPRE_IJVectorAssemble(*ij);
    HYPRE_IJVectorGetObject(*ij, (void**)par);
    HYPRE_ParVectorSetConstantValues(*par, 0.0);
}

static void make_amg(struct MySolver* s, HYPRE_Solver* amg, HYPRE_ParCSRMatrix A, HYPRE_ParVector b, HYPRE_ParVector x)
{
    HYPRE_BoomerAMGCreate(amg);
    HYPRE_BoomerAMGSetCoarsenType(*amg, 10);         /* HMIS */
    HYPRE_BoomerAMGSetInterpType(*amg, 6);           /* extended+i */
    HYPRE_BoomerAMGSetPMaxElmts(*amg, 4);
    HYPRE_BoomerAMGSetStrongThreshold(*amg, 0.5);
    HYPRE_BoomerAMGSetAggNumLevels(*amg, 1);
    HYPRE_BoomerAMGSetRelaxType(*amg, 6);            /* hybrid symmetric Gauss-Seidel (PETSc default) */
    HYPRE_BoomerAMGSetCycleRelaxType(*amg, 9, 3);    /* Gaussian elimination on the coarsest level */
    HYPRE_BoomerAMGSetMaxIter(*amg, 1);
    HYPRE_BoomerAMGSetTol(*amg, 0.0);
    HYPRE_BoomerAMGSetPrintLevel(*amg, 0);
    HYPRE_BoomerAMGSetup(*amg, A, b, x);
    (void)s;
}

/* Preparation: diagonal couplings, A11 and S3 as hypre matrices, BoomerAMG setups, work space. */
void preprocess(struct MySolver* s, const struct SC26_SolverInput* in)
{
    (void)in;
    int N = s->N, nl = s->nloc, c0 = s->c0, n = s->n;
    s->a12 = malloc((size_t)nl * sizeof(double)); s->d22 = malloc((size_t)nl * sizeof(double));
    s->a23 = malloc((size_t)nl * sizeof(double)); s->a32 = malloc((size_t)nl * sizeof(double));
    HYPRE_BigInt* rows = malloc((size_t)nl * sizeof(HYPRE_BigInt));
    HYPRE_Int* ncols = malloc((size_t)nl * sizeof(HYPRE_Int));
    HYPRE_BigInt* cols = malloc((size_t)7 * nl * sizeof(HYPRE_BigInt));
    double* vals = malloc((size_t)7 * nl * sizeof(double));
    for (int k = 0; k < nl; k++) {
        int c = c0 + k;
        rows[k] = c;
        s->a12[k] = s->d22[k] = s->a23[k] = s->a32[k] = 0.0;
        for (int t = s->rp[k]; t < s->rp[k + 1]; t++)
            if (s->gci[t] == N + c) s->a12[k] = s->va[t];
        for (int t = s->rp[nl + k]; t < s->rp[nl + k + 1]; t++) {
            if (s->gci[t] == N + c) s->d22[k] = s->va[t];
            if (s->gci[t] == 2 * N + c) s->a23[k] = s->va[t];
        }
        for (int t = s->rp[2 * nl + k]; t < s->rp[2 * nl + k + 1]; t++)
            if (s->gci[t] == N + c) s->a32[k] = s->va[t];
    }

    for (int field = 0; field < 3; field += 2) {     /* field 0: A11, field 2: S3 */
        int z = 0;
        for (int k = 0; k < nl; k++) {
            int c = c0 + k, cnt = 0;
            for (int t = s->rp[field * nl + k]; t < s->rp[field * nl + k + 1]; t++) {
                int g = s->gci[t];
                if (g / N != field) continue;
                double v = s->va[t];
                if (field == 2 && g == 2 * N + c) {    /* S3 diagonal without the cancellation a33 ~ |a23| */
                    double cc = -s->a23[k];
                    v = (v - cc) + cc * (s->d22[k] - cc) / s->d22[k];
                }
                cols[z] = g - field * N;
                vals[z] = v;
                z++;
                cnt++;
            }
            ncols[k] = cnt;
        }
        HYPRE_IJMatrix* ij = field == 0 ? &s->ijA11 : &s->ijS3;
        HYPRE_IJMatrixCreate(s->comm, c0, c0 + nl - 1, c0, c0 + nl - 1, ij);
        HYPRE_IJMatrixSetObjectType(*ij, HYPRE_PARCSR);
        HYPRE_IJMatrixSetRowSizes(*ij, ncols);
        HYPRE_IJMatrixInitialize(*ij);
        HYPRE_IJMatrixSetValues(*ij, nl, ncols, rows, cols, vals);
        HYPRE_IJMatrixAssemble(*ij);
        HYPRE_IJMatrixGetObject(*ij, (void**)(field == 0 ? &s->A11 : &s->S3));
    }
    free(rows); free(ncols); free(cols); free(vals);
    make_vector(s, &s->ijb1, &s->b1); make_vector(s, &s->ijx1, &s->x1);
    make_vector(s, &s->ijb3, &s->b3); make_vector(s, &s->ijx3, &s->x3);
    make_amg(s, &s->amg3, s->S3, s->b3, s->x3);
    make_amg(s, &s->amg11, s->A11, s->b1, s->x1);

    size_t nh = (size_t)n + s->nhalo;
    s->V = malloc((size_t)(RESTART + 1) * n * sizeof(double));
    s->H = malloc((size_t)(RESTART + 1) * RESTART * sizeof(double));
    s->cs = malloc(RESTART * sizeof(double)); s->sn = malloc(RESTART * sizeof(double));
    s->g = malloc((RESTART + 1) * sizeof(double));
    s->w = malloc((size_t)n * sizeof(double)); s->xe = malloc(nh * sizeof(double));
    s->xh = calloc(nh, sizeof(double)); s->xl = calloc(nh, sizeof(double));
    s->r = malloc((size_t)n * sizeof(double)); s->d = malloc((size_t)n * sizeof(double));
}

/* Solve: iterative refinement with a double-double residual and solution. */
void iterative_solver(struct MySolver* s, const struct SC26_SolverInput* in)
{
    int n = s->n, verbose = getenv("SC26_P01_VERBOSE") != NULL;
    double target = env_double("SC26_P01_TOL", 1e-8), inner = env_double("SC26_P01_INNER", 1e-6);
    double nb = sqrt(dot(s, s->b, s->b)), t0 = MPI_Wtime();
    for (int i = 0; i < n; i++) {
        s->xh[i] = s->xl[i] = 0.0;
        s->r[i] = s->b[i];
    }
    for (int step = 0; step < 30; step++) {
        double rel = sqrt(dot(s, s->r, s->r)) / nb;
        if (verbose && s->rank == 0)
            fprintf(stdout, "  [p01] step %d  ||b-Ax||/||b|| (dd) %.3e  t %.2f s\n", step, rel, MPI_Wtime() - t0);
        if (rel < target) break;
        double relres;
        int its = gmres(s, s->r, s->d, inner, 1000, &relres);
        if (verbose && s->rank == 0) fprintf(stdout, "  [p01]   inner GMRES %d its, rel %.2e\n", its, relres);
        for (int i = 0; i < n; i++) {
            double hi, lo;
            two_sum(s->xh[i], s->d[i], &hi, &lo);
            lo += s->xl[i];
            two_sum(hi, lo, &s->xh[i], &s->xl[i]);
        }
        halo(s, s->xh);
        halo(s, s->xl);
        dd_residual(s, s->xh, s->xl, s->r);
    }

    /* send (hi, lo) back to the harness row owners */
    int P = s->size;
    double *sbuf = malloc((size_t)2 * n * sizeof(double)), *rbuf = malloc((size_t)(2 * s->nin + 1) * sizeof(double));
    int *sc = malloc(P * sizeof(int)), *sd = malloc(P * sizeof(int)), *rc = malloc(P * sizeof(int)),
        *rd = malloc(P * sizeof(int));
    for (int p = 0; p < P; p++) {
        sc[p] = 2 * s->rd_rc[p]; sd[p] = 2 * s->rd_rd[p];
        rc[p] = 2 * s->rd_sc[p]; rd[p] = 2 * s->rd_sd[p];
    }
    for (int k = 0; k < n; k++) {
        sbuf[2 * k] = s->xh[s->rd_rpos[k]];
        sbuf[2 * k + 1] = s->xl[s->rd_rpos[k]];
    }
    MPI_Alltoallv(sbuf, sc, sd, MPI_DOUBLE, rbuf, rc, rd, MPI_DOUBLE, s->comm);
    for (int q = 0; q < s->nin; q++) {
        int i = s->rd_sorder[q];
#ifdef SC26_SOLUTION_FP128
        in->x[i] = (__float128)rbuf[2 * q] + (__float128)rbuf[2 * q + 1];
#else
        in->x[i] = rbuf[2 * q] + rbuf[2 * q + 1];
#endif
    }
    const char* dump = getenv("SC26_P01_DUMP");
    if (dump) {                                       /* diagnostics only: double-double solution on rank 0 */
        int *cnt = malloc(P * sizeof(int)), *dsp = malloc(P * sizeof(int));
        mpi_row_counts(in->global_n, P, cnt, dsp);
        double *hi = malloc((size_t)(s->nin + 1) * sizeof(double)), *lo = malloc((size_t)(s->nin + 1) * sizeof(double));
        for (int q = 0; q < s->nin; q++) {
            hi[s->rd_sorder[q]] = rbuf[2 * q];
            lo[s->rd_sorder[q]] = rbuf[2 * q + 1];
        }
        double* all = s->rank == 0 ? malloc((size_t)in->global_n * sizeof(double)) : NULL;
        for (int part = 0; part < 2; part++) {
            MPI_Gatherv(part ? lo : hi, s->nin, MPI_DOUBLE, all, cnt, dsp, MPI_DOUBLE, 0, s->comm);
            if (s->rank == 0) {
                char name[1024];
                snprintf(name, sizeof(name), "%s_%s.bin", dump, part ? "lo" : "hi");
                FILE* f = fopen(name, "wb");
                if (f) {
                    fwrite(all, sizeof(double), (size_t)in->global_n, f);
                    fclose(f);
                }
            }
        }
        free(cnt); free(dsp); free(hi); free(lo); free(all);
    }
    free(sbuf); free(rbuf); free(sc); free(sd); free(rc); free(rd);
}

void analyse_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input)
{
    (void)solver; (void)input;
}

void preprocess_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input)
{
    (void)solver; (void)input;
}

void iterative_solver_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input)
{
    (void)solver; (void)input;
}

#else /* serial and direct entry points: this solver needs MPI_iterative_solver */

static void not_supported(void)
{
    fprintf(stderr, "problem01: use MPI_iterative_solver (MPI + hypre)\n");
}

#ifdef DIRECT_SOLVER
void preprocess(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; not_supported(); }
void direct_solver(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void solve(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void preprocess_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; not_supported(); }
void direct_solver_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void solve_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
#endif

#ifdef ITERATIVE_SOLVER
void analyse(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; not_supported(); }
void preprocess(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void iterative_solver(struct MySolver* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void analyse_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; not_supported(); }
void preprocess_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
void iterative_solver_complex(struct MySolver_complex* solver, const struct SC26_SolverInput* input) { (void)solver; (void)input; }
#endif

#endif
