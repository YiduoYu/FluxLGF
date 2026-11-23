/* SSR: sampled Schur reduction. A multilevel block elimination for general sparse matrices.
 *
 * On each level: scale A^ = Y A T by measured near-kernel vectors, pick an independent set F of acceptable pivots, eliminate F
 * exactly in the transfer operators, and replace the fill of the Schur complement by an unbiased random sample of it. The
 * diagonal then restores the exact action of the Schur complement on the near-kernel vector, here the constant after scaling.
 * No strength threshold, no interpolation formula, no triple product.
 *
 * cc -O3 -march=native -shared -fPIC -o libssr.dylib ssr.c
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int n, m; int *ia, *ja; double *a; } CSR;   /* n rows, m columns */

typedef struct {
    int n, nF, nC;
    int *F, *C;
    double *dinv;
    CSR fc, cf;              /* A_FC, rows in F order, columns in C order; A_CF, rows in C order, columns in F order */
    CSR A;                   /* whole level matrix, kept for smoothing and residuals */
    double *diag;
    double *rF, *rC, *yC, *wk;
} Level;

typedef struct {
    int nlev, N;
    Level *lev;
    int cn;                  /* coarsest direct solve */
    double *lu; int *piv;
    double *T, *Y;           /* scaling on the finest level: A^ = diag(Y) A diag(T) */
    double *rs, *zs;
    int nu, every;
    double nnz_sum, nnz0;
    double setup_t[4];
} SSR;

static uint64_t rng_s;
static inline uint64_t rng(void) { uint64_t x = rng_s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; rng_s = x; return x; }
static inline double urand(void) { return (rng() >> 11) * (1.0 / 9007199254740992.0); }

static void csr_alloc(CSR *M, int n, int m, int nnz) {
    M->n = n; M->m = m;
    M->ia = (int *)malloc(sizeof(int) * (n + 1));
    M->ja = (int *)malloc(sizeof(int) * (nnz > 0 ? nnz : 1));
    M->a = (double *)malloc(sizeof(double) * (nnz > 0 ? nnz : 1));
}
static void csr_free(CSR *M) { free(M->ia); free(M->ja); free(M->a); }

static void transpose(const CSR *A, CSR *T) {
    int nnz = A->ia[A->n];
    csr_alloc(T, A->m, A->n, nnz);
    memset(T->ia, 0, sizeof(int) * (A->m + 1));
    for (int p = 0; p < nnz; p++) T->ia[A->ja[p] + 1]++;
    for (int i = 0; i < A->m; i++) T->ia[i + 1] += T->ia[i];
    int *pos = (int *)malloc(sizeof(int) * (A->m + 1));
    memcpy(pos, T->ia, sizeof(int) * (A->m + 1));
    for (int i = 0; i < A->n; i++)
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int q = pos[A->ja[p]]++; T->ja[q] = i; T->a[q] = A->a[p]; }
    free(pos);
}

static int is_symmetric(const CSR *A, const CSR *T, double tol) {
    if (A->n != A->m) return 0;
    double amax = 0;
    for (int p = 0; p < A->ia[A->n]; p++) if (fabs(A->a[p]) > amax) amax = fabs(A->a[p]);
    for (int i = 0; i < A->n; i++) {
        if (A->ia[i + 1] - A->ia[i] != T->ia[i + 1] - T->ia[i]) return 0;
        for (int p = A->ia[i], q = T->ia[i]; p < A->ia[i + 1]; p++, q++) {
            if (A->ja[p] != T->ja[q]) return 0;
            if (fabs(A->a[p] - T->a[q]) > tol * (fabs(A->a[p]) + 1e-300) && fabs(A->a[p] - T->a[q]) > 1e-14 * amax) return 0;
        }
    }
    return 1;
}

/* triplet builder: row lists */
typedef struct { int *row, *col; double *val; int cnt, cap; } Trip;
static void trip_init(Trip *t, int cap) { t->cap = cap > 16 ? cap : 16; t->cnt = 0; t->row = (int *)malloc(sizeof(int) * t->cap); t->col = (int *)malloc(sizeof(int) * t->cap); t->val = (double *)malloc(sizeof(double) * t->cap); }
static void trip_free(Trip *t) { free(t->row); free(t->col); free(t->val); }
static inline void trip_add(Trip *t, int r, int c, double v) {
    if (t->cnt == t->cap) {
        t->cap *= 2;
        t->row = (int *)realloc(t->row, sizeof(int) * t->cap); t->col = (int *)realloc(t->col, sizeof(int) * t->cap); t->val = (double *)realloc(t->val, sizeof(double) * t->cap);
    }
    t->row[t->cnt] = r; t->col[t->cnt] = c; t->val[t->cnt] = v; t->cnt++;
}
/* sum duplicates; output rows sorted by column */
static void trip_to_csr(const Trip *t, int n, CSR *M) {
    int *cnt = (int *)calloc(n + 1, sizeof(int));
    for (int k = 0; k < t->cnt; k++) cnt[t->col[k] + 1]++;
    for (int i = 0; i < n; i++) cnt[i + 1] += cnt[i];
    /* sort by column, then by row, so that each row comes out sorted (two-pass radix) */
    int *by = (int *)malloc(sizeof(int) * (t->cnt > 0 ? t->cnt : 1));
    for (int k = 0; k < t->cnt; k++) by[cnt[t->col[k]]++] = k;
    free(cnt);
    int *rcnt = (int *)calloc(n + 1, sizeof(int));
    for (int k = 0; k < t->cnt; k++) rcnt[t->row[k] + 1]++;
    for (int i = 0; i < n; i++) rcnt[i + 1] += rcnt[i];
    csr_alloc(M, n, n, t->cnt);
    int *pos = (int *)malloc(sizeof(int) * (n + 1));
    memcpy(pos, rcnt, sizeof(int) * (n + 1));
    int *last = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) last[i] = -1;
    M->ia[0] = 0;
    /* write in column order per row; merge when the column repeats */
    for (int s = 0; s < t->cnt; s++) {
        int k = by[s], r = t->row[k], c = t->col[k];
        if (last[r] == c) { M->a[pos[r] - 1] += t->val[k]; }
        else { int q = pos[r]++; M->ja[q] = c; M->a[q] = t->val[k]; last[r] = c; }
    }
    /* compact */
    int w = 0;
    for (int i = 0; i < n; i++) {
        int start = rcnt[i], end = pos[i];
        M->ia[i] = w;
        for (int p = start; p < end; p++) { M->ja[w] = M->ja[p]; M->a[w] = M->a[p]; w++; }
    }
    M->ia[n] = w;
    free(by); free(rcnt); free(pos); free(last);
}

static double get_diag(const CSR *A, int i) {
    for (int p = A->ia[i]; p < A->ia[i + 1]; p++) if (A->ja[p] == i) return A->a[p];
    return 0.0;
}

/* Choose F: greedy independent set in priority order. Pivot test: |a_kk| >= PIV * sum |a_kj|. Weak edges do not block when theta > 0. */
static void choose_F(const CSR *A, const CSR *T, const double *diag, int *mark, double piv, double theta, int deg_prio) {
    int n = A->n;
    double *key = (double *)malloc(sizeof(double) * n);
    int *ord = (int *)malloc(sizeof(int) * n);
    int maxd = 0;
    for (int i = 0; i < n; i++) { int d = A->ia[i + 1] - A->ia[i] + T->ia[i + 1] - T->ia[i]; if (d > maxd) maxd = d; }
    /* bucket order by degree, random within a bucket */
    int *bcnt = (int *)calloc(maxd + 2, sizeof(int));
    for (int i = 0; i < n; i++) { key[i] = urand(); }
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = rng() % (uint64_t)(i + 1); int s = ord[i]; ord[i] = ord[j]; ord[j] = s; }
    if (deg_prio) {
        int *o2 = (int *)malloc(sizeof(int) * n);
        for (int s = 0; s < n; s++) { int i = ord[s]; int d = A->ia[i + 1] - A->ia[i] + T->ia[i + 1] - T->ia[i]; bcnt[d + 1]++; }
        for (int d = 0; d <= maxd; d++) bcnt[d + 1] += bcnt[d];
        for (int s = 0; s < n; s++) { int i = ord[s]; int d = A->ia[i + 1] - A->ia[i] + T->ia[i + 1] - T->ia[i]; o2[bcnt[d]++] = i; }
        memcpy(ord, o2, sizeof(int) * n);
        free(o2);
    }
    /* mark: 0 undecided, 1 F, 2 C */
    memset(mark, 0, sizeof(int) * n);
    for (int s = 0; s < n; s++) {
        int k = ord[s];
        if (mark[k]) continue;
        double sum = 0, mx = 0;
        for (int p = A->ia[k]; p < A->ia[k + 1]; p++) if (A->ja[p] != k) { double v = fabs(A->a[p]); sum += v; if (v > mx) mx = v; }
        if (fabs(diag[k]) < piv * sum || diag[k] == 0.0) { mark[k] = 2; continue; }
        mark[k] = 1;
        for (int p = A->ia[k]; p < A->ia[k + 1]; p++) { int j = A->ja[p]; if (j != k && !mark[j] && fabs(A->a[p]) >= theta * mx) mark[j] = 2; }
        for (int p = T->ia[k]; p < T->ia[k + 1]; p++) { int j = T->ja[p]; if (j != k && !mark[j] && fabs(T->a[p]) >= theta * mx) mark[j] = 2; }
    }
    free(key); free(ord); free(bcnt);
}

/* one level of reduction. Returns the coarse matrix in *S. */
static inline double scale_of(double d) { return fabs(d); }
static double GUARD = 0.0;
static int LUMP = 1;
static double DROP = 0.0;
static int COUPLE = 1;   /* 0: independent samples; 1: chain for symmetric, transport for general; 2: half-circle for symmetric too */
void ssr_config(double guard, int lump, int couple, double drop) { GUARD = guard; LUMP = lump; COUPLE = couple; DROP = drop; }
static void reduce(Level *L, const CSR *A, CSR *S, const double *e, double *ec, int sym, double piv, double theta, int ns, int deg_prio, int topk) {
    int n = A->n;
    CSR T; transpose(A, &T);
    int *mark = (int *)malloc(sizeof(int) * n);
    double *diag = L->diag;
    for (int i = 0; i < n; i++) diag[i] = get_diag(A, i);
    choose_F(A, &T, diag, mark, piv, theta, deg_prio);
    int nF = 0, nC = 0;
    int *cidx = (int *)malloc(sizeof(int) * n);
    int *fidx = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) { if (mark[i] == 1) fidx[i] = nF++; else cidx[i] = nC++; }
    L->nF = nF; L->nC = nC; L->n = n;
    L->F = (int *)malloc(sizeof(int) * (nF > 0 ? nF : 1));
    L->C = (int *)malloc(sizeof(int) * (nC > 0 ? nC : 1));
    L->dinv = (double *)malloc(sizeof(double) * (nF > 0 ? nF : 1));
    for (int i = 0; i < n; i++) { if (mark[i] == 1) L->F[fidx[i]] = i; else L->C[cidx[i]] = i; }

    /* A_FC, A_CF; F-F entries (weak, when theta > 0) go to the diagonal of the F row, which keeps the row sum */
    {
        int nnzfc = 0, nnzcf = 0;
        for (int i = 0; i < n; i++)
            for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
                int j = A->ja[p]; if (j == i) continue;
                if (mark[i] == 1 && mark[j] != 1) nnzfc++;
                if (mark[i] != 1 && mark[j] == 1) nnzcf++;
            }
        csr_alloc(&L->fc, nF, nC, nnzfc); csr_alloc(&L->cf, nC, nF, nnzcf);
        int a = 0, b = 0, fi = 0, ci = 0;
        L->fc.ia[0] = 0; L->cf.ia[0] = 0;
        for (int i = 0; i < n; i++) {
            if (mark[i] == 1) {
                double d = e[i];                                /* excess form: no cancellation when off-diagonals have one sign */
                for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
                    int j = A->ja[p]; if (j == i) continue;
                    if (mark[j] != 1) { d -= A->a[p]; L->fc.ja[a] = cidx[j]; L->fc.a[a++] = A->a[p]; }
                }
                L->dinv[fi] = 1.0 / d;
                L->fc.ia[++fi] = a;
            } else {
                for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
                    int j = A->ja[p]; if (mark[j] == 1) { L->cf.ja[b] = fidx[j]; L->cf.a[b++] = A->a[p]; }
                }
                L->cf.ia[++ci] = b;
            }
        }
    }

    /* coarse matrix: A_CC plus samples of the Schur update, then the exact row-sum on the diagonal */
    Trip tr; trip_init(&tr, A->ia[n] * 2);
    double *rowsum_exact = (double *)calloc(nC, sizeof(double));  /* exact S * 1 */
    double *rowsum_samp = (double *)calloc(nC, sizeof(double));   /* sampled off-diagonal action on 1 */
    double *dex = (double *)malloc(sizeof(double) * (nC + 1));   /* exact Schur diagonal */
    for (int c = 0; c < nC; c++) dex[c] = diag[L->C[c]];
    for (int i = 0; i < n; i++) {
        if (mark[i] == 1) continue;
        int ci = cidx[i];
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
            int j = A->ja[p];
            if (mark[j] == 1) continue;
            if (j != i) { trip_add(&tr, ci, cidx[j], A->a[p]); rowsum_samp[ci] += A->a[p]; }
        }
        rowsum_exact[ci] = e[i];
    }
    int cap = 256;
    int *oj = (int *)malloc(sizeof(int) * cap), *ij = (int *)malloc(sizeof(int) * cap);
    double *ov = (double *)malloc(sizeof(double) * cap), *iv = (double *)malloc(sizeof(double) * cap), *cum = (double *)malloc(sizeof(double) * cap);
    for (int f = 0; f < nF; f++) {
        int k = L->F[f];
        double dinv = L->dinv[f];
        int no = L->fc.ia[f + 1] - L->fc.ia[f];
        if (no == 0) continue;
        int ni = T.ia[k + 1] - T.ia[k];
        if (no + 2 > cap || ni + 2 > cap) {
            cap = (no > ni ? no : ni) * 2 + 4;
            oj = (int *)realloc(oj, sizeof(int) * cap); ij = (int *)realloc(ij, sizeof(int) * cap);
            ov = (double *)realloc(ov, sizeof(double) * cap); iv = (double *)realloc(iv, sizeof(double) * cap); cum = (double *)realloc(cum, sizeof(double) * cap);
        }
        for (int q = 0; q < no; q++) { int p = L->fc.ia[f] + q; oj[q] = L->fc.ja[p]; ov[q] = L->fc.a[p]; }
        int nin = 0;
        for (int p = T.ia[k]; p < T.ia[k + 1]; p++) { int i = T.ja[p]; if (mark[i] != 1) { ij[nin] = cidx[i]; iv[nin] = T.a[p]; nin++; } }
        /* exact action on the constant */
        for (int q = 0; q < nin; q++) rowsum_exact[ij[q]] -= iv[q] * dinv * e[k];     /* GTH-type update of the excess */
        for (int q = 0; q < nin; q++) for (int c = 0; c < no; c++) if (oj[c] == ij[q]) dex[ij[q]] -= iv[q] * ov[c] * dinv;
        if (nin == 0) continue;
        if (COUPLE && !(sym && COUPLE == 1)) {
            /* Unbiased transport plan on a circle. Rows of the update are intervals of length R_i = |a_ik| Wo / |a_kk|, columns intervals of
               length C_j = |a_kj| Wi / |a_kk|, total L = Wi Wo / |a_kk|. With a uniform random shift the expected overlap of
               interval i and interval j is exactly |a_ik a_kj / a_kk|, and every row total and column total holds exactly in every draw.
               Symmetric matrices use the reflected partition, which makes the plan symmetric. At most nin + no - 1 fill entries. */
            double wo = 0, wi = 0;
            for (int q = 0; q < no; q++) wo += fabs(ov[q]);
            for (int q = 0; q < nin; q++) wi += fabs(iv[q]);
            if (wo == 0 || wi == 0) continue;
            for (int q = no - 1; q > 0; q--) { int r = rng() % (uint64_t)(q + 1); int tj = oj[q]; oj[q] = oj[r]; oj[r] = tj; double tv = ov[q]; ov[q] = ov[r]; ov[r] = tv; }
            if (sym && nin == no) {
                /* Half-circle involution. Intervals of length R_i = sum_{j != i} |U_ij| = |a_ki| (W - |a_ki|) |dinv| in random order on a
                   circle of length L; each point x is paired with x + L/2. The map is its own inverse, so the plan is symmetric; no
                   R_i exceeds L/2, so it has no self-pairs. Every row sum and every diagonal of the Schur update is exact in each draw. */
                double L_ = 0;
                for (int q = 0; q < no; q++) { cum[q] = fabs(ov[q]) * (wo - fabs(ov[q])) * scale_of(dinv); L_ += cum[q]; }
                if (no < 2 || L_ <= 0) continue;
                double half = L_ / 2, x = 0;
                int pa = 0, pb = 0;
                double acc2 = 0;
                while (pb < no - 1 && acc2 + cum[pb] <= half) { acc2 += cum[pb]; pb++; }
                double ra = cum[0], rb = acc2 + cum[pb] - half;
                while (x < half * (1 - 1e-13)) {
                    double piece = ra < rb ? ra : rb;
                    if (piece > half - x) piece = half - x;
                    if (pa != pb && piece > 0) {
                        double v = -piece * (ov[pa] > 0 ? 1 : -1) * (ov[pb] > 0 ? 1 : -1) * (dinv > 0 ? 1 : -1);
                        trip_add(&tr, oj[pa], oj[pb], v); trip_add(&tr, oj[pb], oj[pa], v);
                        rowsum_samp[oj[pa]] += v; rowsum_samp[oj[pb]] += v;
                    }
                    x += piece; ra -= piece; rb -= piece;
                    if (ra <= 1e-14 * L_) { pa = (pa + 1) % no; ra = cum[pa]; }
                    if (rb <= 1e-14 * L_) { pb = (pb + 1) % no; rb = cum[pb]; }
                }
                continue;
            } else {
                for (int q = nin - 1; q > 0; q--) { int r = rng() % (uint64_t)(q + 1); int tj = ij[q]; ij[q] = ij[r]; ij[r] = tj; double tv = iv[q]; iv[q] = iv[r]; iv[r] = tv; }
            }
            double scale = fabs(dinv);
            double total = wi * wo * scale;
            /* start of the first column after a random shift */
            double shift = urand() * total, acc = 0;
            int jp = 0;
            while (jp < no - 1 && acc + fabs(ov[jp]) * wi * scale <= shift) { acc += fabs(ov[jp]) * wi * scale; jp++; }
            double rem_c = acc + fabs(ov[jp]) * wi * scale - shift;
            int guard = nin + no + 2;
            for (int q = 0; q < nin; q++) {
                double rem_r = fabs(iv[q]) * wo * scale;
                while (rem_r > 0 && guard-- > 0) {
                    double piece = rem_r < rem_c ? rem_r : rem_c;
                    if (oj[jp] != ij[q] && piece > 0) {
                        double v = -piece * (iv[q] > 0 ? 1 : -1) * (ov[jp] > 0 ? 1 : -1) * (dinv > 0 ? 1 : -1);
                        trip_add(&tr, ij[q], oj[jp], v); rowsum_samp[ij[q]] += v;
                    }
                    rem_r -= piece; rem_c -= piece;
                    if (rem_c <= 1e-14 * total) { jp = (jp + 1) % no; rem_c = fabs(ov[jp]) * wi * scale; if (rem_r <= 1e-14 * total) rem_r = 0; }
                }
                guard = nin + no + 2;
            }
        } else if (sym) {
            /* Kyng-Sachdeva clique sampling generalized to signed entries: ascending |a_kj|, each vertex joins a later one with
               probability proportional to |a_kl|; weight a_ki sign(a_kl) W_i / a_kk. Unbiased for every pair; symmetric. */
            /* insertion sort, degrees are small */
            for (int a = 1; a < no; a++) {
                int cj = oj[a]; double cv = ov[a]; int b = a - 1;
                while (b >= 0 && fabs(ov[b]) > fabs(cv)) { oj[b + 1] = oj[b]; ov[b + 1] = ov[b]; b--; }
                oj[b + 1] = cj; ov[b + 1] = cv;
            }
            cum[no] = 0;
            for (int a = no - 1; a >= 0; a--) cum[a] = cum[a + 1] + fabs(ov[a]);
            for (int a = 0; a + 1 < no; a++) {
                if (topk) {
                    /* deterministic control: the largest partner only, for the same number of edges */
                    int l = no - 1;
                    double v = -ov[a] * (ov[l] > 0 ? 1 : -1) * cum[a + 1] * dinv;
                    trip_add(&tr, oj[a], oj[l], v); trip_add(&tr, oj[l], oj[a], v);
                    rowsum_samp[oj[a]] += v; rowsum_samp[oj[l]] += v;
                    continue;
                }
                for (int s = 0; s < ns; s++) {
                    double u = urand() * cum[a + 1];
                    int l = no - 1;
                    double acc = 0;
                    for (int c = a + 1; c < no; c++) { acc += fabs(ov[c]); if (u < acc) { l = c; break; } }
                    double v = -ov[a] * (ov[l] > 0 ? 1 : -1) * cum[a + 1] * dinv / ns;
                    trip_add(&tr, oj[a], oj[l], v); trip_add(&tr, oj[l], oj[a], v);
                    rowsum_samp[oj[a]] += v; rowsum_samp[oj[l]] += v;
                }
            }
        } else {
            /* general case: half the samples by row, half by column, each unbiased for the rank-one update */
            double wo = 0, wi = 0;
            for (int q = 0; q < no; q++) wo += fabs(ov[q]);
            for (int q = 0; q < nin; q++) wi += fabs(iv[q]);
            for (int q = 0; q < nin; q++) for (int s = 0; s < ns; s++) {
                double u = urand() * wo, acc = 0; int l = no - 1;
                for (int c = 0; c < no; c++) { acc += fabs(ov[c]); if (u < acc) { l = c; break; } }
                if (oj[l] == ij[q]) continue;                                 /* the diagonal is set from the row sum below */
                double v = -iv[q] * (ov[l] > 0 ? 1 : -1) * wo * dinv / (2 * ns);
                trip_add(&tr, ij[q], oj[l], v); rowsum_samp[ij[q]] += v;
            }
            for (int q = 0; q < no; q++) for (int s = 0; s < ns; s++) {
                double u = urand() * wi, acc = 0; int l = nin - 1;
                for (int c = 0; c < nin; c++) { acc += fabs(iv[c]); if (u < acc) { l = c; break; } }
                if (ij[l] == oj[q]) continue;
                double v = -ov[q] * (iv[l] > 0 ? 1 : -1) * wi * dinv / (2 * ns);
                trip_add(&tr, ij[l], oj[q], v); rowsum_samp[ij[l]] += v;
            }
        }
    }
    for (int c = 0; c < nC; c++) {
        ec[c] = rowsum_exact[c];
        double d = rowsum_exact[c] - rowsum_samp[c];
        if (GUARD > 0 && d * dex[c] < GUARD * dex[c] * dex[c]) d = GUARD * dex[c];
        trip_add(&tr, c, c, d);
    }
    free(dex);
    trip_to_csr(&tr, nC, S);
    trip_free(&tr);
    if (DROP > 0) {
        /* remove entries small against both diagonals; the row sum is kept by rebuilding the diagonal from the excess */
        double *dg = (double *)malloc(sizeof(double) * (nC + 1));
        for (int c = 0; c < nC; c++) dg[c] = get_diag(S, c);
        int w = 0, start = 0;
        for (int c = 0; c < nC; c++) {
            int end = S->ia[c + 1], dpos = -1;
            double off = 0;
            for (int p = start; p < end; p++) {
                int j = S->ja[p];
                if (j == c) { dpos = w; S->ja[w] = j; S->a[w++] = dg[c]; continue; }
                if (S->a[p] * S->a[p] < DROP * DROP * fabs(dg[c] * dg[j])) continue;
                off += S->a[p];
                S->ja[w] = j; S->a[w++] = S->a[p];
            }
            if (dpos >= 0) S->a[dpos] = ec[c] - off;
            start = end;
            S->ia[c + 1] = w;
        }
        free(dg);
    }
    free(oj); free(ij); free(ov); free(iv); free(cum);
    free(rowsum_exact); free(rowsum_samp); free(mark); free(cidx); free(fidx);
    csr_free(&T);
}

/* row-sum-preserving lump of entries with the same sign as the diagonal. Returns 1 and fills B if any entry moved. */
static int lump_positive(const CSR *A, CSR *B) {
    int n = A->n, cnt = 0;
    for (int i = 0; i < n; i++) {
        double d = get_diag(A, i);
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) if (A->ja[p] != i && A->a[p] * d > 0) cnt++;
    }
    if (!cnt) return 0;
    csr_alloc(B, n, n, A->ia[n]);
    int w = 0; B->ia[0] = 0;
    for (int i = 0; i < n; i++) {
        double d = get_diag(A, i), add = 0;
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) if (A->ja[p] != i && A->a[p] * d > 0) add += A->a[p];
        int has = 0;
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
            int j = A->ja[p];
            if (j == i) { B->ja[w] = i; B->a[w++] = d + add; has = 1; }
            else if (A->a[p] * d <= 0) { B->ja[w] = j; B->a[w++] = A->a[p]; }
        }
        (void)has;
        B->ia[i + 1] = w;
    }
    return 1;
}

static void gs(const CSR *A, const double *diag, const double *b, double *x, int backward) {
    int n = A->n;
    if (!backward) {
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int p = A->ia[i]; p < A->ia[i + 1]; p++) s -= A->a[p] * x[A->ja[p]];
            if (diag[i] != 0) x[i] += s / diag[i];
        }
    } else {
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int p = A->ia[i]; p < A->ia[i + 1]; p++) s -= A->a[p] * x[A->ja[p]];
            if (diag[i] != 0) x[i] += s / diag[i];
        }
    }
}

static void lu_factor(double *a, int *piv, int n) {
    for (int k = 0; k < n; k++) {
        int m = k; double mx = fabs(a[k * n + k]);
        for (int i = k + 1; i < n; i++) if (fabs(a[i * n + k]) > mx) { mx = fabs(a[i * n + k]); m = i; }
        piv[k] = m;
        if (m != k) for (int j = 0; j < n; j++) { double t = a[k * n + j]; a[k * n + j] = a[m * n + j]; a[m * n + j] = t; }
        double d = a[k * n + k]; if (d == 0) { d = 1e-300; a[k * n + k] = d; }
        for (int i = k + 1; i < n; i++) {
            double f = a[i * n + k] / d; a[i * n + k] = f;
            if (f != 0) for (int j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j];
        }
    }
}
static void lu_solve(const double *a, const int *piv, int n, double *x) {
    for (int k = 0; k < n; k++) { int m = piv[k]; if (m != k) { double t = x[k]; x[k] = x[m]; x[m] = t; } }
    for (int i = 0; i < n; i++) { double s = x[i]; for (int j = 0; j < i; j++) s -= a[i * n + j] * x[j]; x[i] = s; }
    for (int i = n - 1; i >= 0; i--) { double s = x[i]; for (int j = i + 1; j < n; j++) s -= a[i * n + j] * x[j]; x[i] = s / a[i * n + i]; }
}

static void cycle(SSR *H, int l, const double *r, double *x) {
    if (l == H->nlev) {
        memcpy(x, r, sizeof(double) * H->cn);
        lu_solve(H->lu, H->piv, H->cn, x);
        return;
    }
    Level *L = &H->lev[l];
    int n = L->n;
    int smooth = H->nu > 0 && (l % H->every) == 0;
    const double *rr = r;
    memset(x, 0, sizeof(double) * n);
    if (smooth) {
        for (int s = 0; s < H->nu; s++) gs(&L->A, L->diag, r, x, 0);
        for (int i = 0; i < n; i++) {
            double t = r[i];
            for (int p = L->A.ia[i]; p < L->A.ia[i + 1]; p++) t -= L->A.a[p] * x[L->A.ja[p]];
            L->wk[i] = t;
        }
        rr = L->wk;
    }
    for (int f = 0; f < L->nF; f++) L->rF[f] = rr[L->F[f]] * L->dinv[f];
    for (int c = 0; c < L->nC; c++) {
        double t = rr[L->C[c]];
        for (int p = L->cf.ia[c]; p < L->cf.ia[c + 1]; p++) t -= L->cf.a[p] * L->rF[L->cf.ja[p]];
        L->rC[c] = t;
    }
    cycle(H, l + 1, L->rC, L->yC);
    for (int c = 0; c < L->nC; c++) x[L->C[c]] += L->yC[c];
    for (int f = 0; f < L->nF; f++) {
        double t = 0;
        for (int p = L->fc.ia[f]; p < L->fc.ia[f + 1]; p++) t += L->fc.a[p] * L->yC[L->fc.ja[p]];
        x[L->F[f]] += L->rF[f] - t * L->dinv[f];
    }
    if (smooth) for (int s = 0; s < H->nu; s++) gs(&L->A, L->diag, r, x, 1);
}

SSR *ssr_setup(int n, const int *ia, const int *ja, const double *a, const double *Tv, const double *Yv, int nu, int every, int ns, double piv, double theta, int deg_prio, int coarse, int topk, uint64_t seed, int force_general) {
    SSR *H = (SSR *)calloc(1, sizeof(SSR));
    rng_s = seed * 2654435761u + 88172645463325252ull;
    H->N = n; H->nu = nu; H->every = every > 0 ? every : 1;
    H->T = (double *)malloc(sizeof(double) * n); H->Y = (double *)malloc(sizeof(double) * n);
    H->rs = (double *)malloc(sizeof(double) * n); H->zs = (double *)malloc(sizeof(double) * n);
    memcpy(H->T, Tv, sizeof(double) * n); memcpy(H->Y, Yv, sizeof(double) * n);
    /* scaled copy */
    CSR A; csr_alloc(&A, n, n, ia[n]);
    memcpy(A.ia, ia, sizeof(int) * (n + 1)); memcpy(A.ja, ja, sizeof(int) * ia[n]);
    for (int i = 0; i < n; i++) for (int p = ia[i]; p < ia[i + 1]; p++) A.a[p] = Yv[i] * a[p] * Tv[ja[p]];
    CSR T0; transpose(&A, &T0);
    int sym = force_general ? 0 : is_symmetric(&A, &T0, 1e-8);
    csr_free(&T0);
    H->nnz0 = A.ia[n];
    /* row excess of the scaled matrix in double-double: y_i sum_j a_ij t_j */
    double *e = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double hi = 0, lo = 0;
        for (int p = ia[i]; p < ia[i + 1]; p++) {
            double v = a[p] * Tv[ja[p]], er = fma(a[p], Tv[ja[p]], -v);
            double s = hi + v, bb = s - hi, ts = (hi - (s - bb)) + (v - bb);
            hi = s; lo += ts + er;
        }
        e[i] = Yv[i] * (hi + lo);
    }
    int capl = 64; H->lev = (Level *)calloc(capl, sizeof(Level));
    H->nnz_sum = 0;
    CSR cur = A;
    while (cur.n > coarse) {
        if (H->nlev == capl) { capl *= 2; H->lev = (Level *)realloc(H->lev, sizeof(Level) * capl); }
        Level *L = &H->lev[H->nlev];
        memset(L, 0, sizeof(Level));
        L->diag = (double *)malloc(sizeof(double) * cur.n);
        L->wk = (double *)malloc(sizeof(double) * cur.n);
        CSR S;
        CSR Bar;
        int lumped = LUMP ? lump_positive(&cur, &Bar) : 0;
        double *ec = (double *)malloc(sizeof(double) * (cur.n + 1));
        reduce(L, lumped ? &Bar : &cur, &S, e, ec, sym, piv, theta, ns, deg_prio, topk);
        free(e); e = ec;
        if (lumped) { for (int i = 0; i < cur.n; i++) L->diag[i] = get_diag(&cur, i); csr_free(&Bar); }
        L->A = cur;
        H->nnz_sum += cur.ia[cur.n];
        L->rF = (double *)malloc(sizeof(double) * (L->nF > 0 ? L->nF : 1));
        L->rC = (double *)malloc(sizeof(double) * (L->nC > 0 ? L->nC : 1));
        L->yC = (double *)malloc(sizeof(double) * (L->nC > 0 ? L->nC : 1));
        H->nlev++;
        cur = S;
        if (L->nF == 0) break;   /* no acceptable pivots: fall through to direct if small enough */
    }
    free(e);
    H->cn = cur.n;
    H->nnz_sum += cur.ia[cur.n];
    if (cur.n > 6000) { H->cn = -cur.n; return H; }
    H->lu = (double *)calloc((size_t)cur.n * cur.n + 1, sizeof(double));
    H->piv = (int *)malloc(sizeof(int) * (cur.n + 1));
    for (int i = 0; i < cur.n; i++) for (int p = cur.ia[i]; p < cur.ia[i + 1]; p++) H->lu[(size_t)i * cur.n + cur.ja[p]] += cur.a[p];
    lu_factor(H->lu, H->piv, cur.n);
    csr_free(&cur);
    return H;
}

void ssr_apply(SSR *H, const double *r, double *z) {
    for (int i = 0; i < H->N; i++) H->rs[i] = H->Y[i] * r[i];
    cycle(H, 0, H->rs, H->zs);
    for (int i = 0; i < H->N; i++) z[i] = H->T[i] * H->zs[i];
}

void ssr_stats(SSR *H, double *out) {
    out[0] = H->nlev; out[1] = H->nnz_sum / H->nnz0; out[2] = H->cn;
    double tf = 0;
    for (int l = 0; l < H->nlev; l++) tf += H->lev[l].fc.ia[H->lev[l].nF] + H->lev[l].cf.ia[H->lev[l].nC];
    out[3] = tf / H->nnz0;
}
int ssr_level_info(SSR *H, int l, int *out) { Level *L = &H->lev[l]; out[0] = L->n; out[1] = L->nF; out[2] = L->A.ia[L->n]; return 0; }

void ssr_free(SSR *H) {
    for (int l = 0; l < H->nlev; l++) {
        Level *L = &H->lev[l];
        free(L->F); free(L->C); free(L->dinv); csr_free(&L->fc); csr_free(&L->cf); csr_free(&L->A); free(L->diag); free(L->rF); free(L->rC); free(L->yC); free(L->wk);
    }
    free(H->lev); free(H->lu); free(H->piv); free(H->T); free(H->Y); free(H->rs); free(H->zs); free(H);
}
