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
    int nb, *bptr, *boff;    /* cluster pivots: blocks of the F ordering and their dense inverses; nb = 0 on scalar levels */
    double *binv, *tF, *wF;
    int *dpos;               /* position of the diagonal in each row of A (rows sorted), for the split pre-smoothing */
} Level;

typedef struct {
    int nlev, N;
    Level *lev;
    int cn;                  /* coarsest direct solve */
    double *lu; int *piv;
    double *T, *Y;           /* scaling on the finest level: A^ = diag(Y) A diag(T) */
    double *rs, *zs;
    int nu, every, smode;
    double nnz_sum, nnz0;
    double setup_t[4];
} SSR;

static uint64_t rng_s;
static inline uint64_t rng(void) { uint64_t x = rng_s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; rng_s = x; return x; }
static inline double urand(void) { return (rng() >> 11) * (1.0 / 9007199254740992.0); }
static double SFILT = 0.0;  /* scalar symmetric path: reroute coarse entries below SFILT sqrt(d_i d_j) */
static int MISDYN = 0;      /* 1: dynamic minimum-degree greedy independent set (larger F) */
static double DOM = 0.0;    /* > 0: when at least 5 % of rows have |a_kk| >= DOM sum|a_kj|, the level eliminates exactly those rows */
void ssr_config5(double sfilt, int misdyn) { SFILT = sfilt; MISDYN = misdyn; }
static double NFILT = 0.0;  /* nonsymmetric path: reroute weak entries a->b through a->c->b */
void ssr_config6(double dom) { DOM = dom; }
static int STR2 = 0;        /* 1: an edge is strong when it is strong for either of its two rows */
void ssr_config7(double nfilt) { NFILT = nfilt; }
static int SMODE = 0;       /* 0: pre and post sweeps; 1: pre only; 2: post only */
void ssr_config8(int str2) { STR2 = str2; }
void ssr_config9(int smode) { SMODE = smode; }
static int EXFIX = 1;       /* 1: F rows without C neighbours still update the excess of the rows that point to them (0: old behaviour, a bug) */
void ssr_config10(int exfix) { EXFIX = exfix; }
static double FFMAX = 0.0;  /* > 0: an F row whose weak F-F couplings exceed FFMAX |a_kk| goes back to C */
void ssr_config11(double ffmax) { FFMAX = ffmax; }
static int LOCALORD = 0;    /* > 0: F selection visits rows in random order within blocks of this many consecutive rows */
void ssr_config12(int localord) { LOCALORD = localord; }
static int DROPMIN = 0;     /* 1: DROP removes |a_ij| < DROP min(|d_i|, |d_j|) (small for both rows); 0: |a_ij| < DROP sqrt(|d_i d_j|) */
void ssr_config13(int dropmin) { DROPMIN = dropmin; }

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
static int FASTCSR = 1;
static int cmp_int_dbl(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }
/* row buckets, duplicates merged through a column marker, then each row sorted */
static void trip_to_csr_fast(const Trip *t, int n, CSR *M) {
    int *rc = (int *)calloc(n + 1, sizeof(int));
    for (int k = 0; k < t->cnt; k++) rc[t->row[k] + 1]++;
    for (int i = 0; i < n; i++) rc[i + 1] += rc[i];
    int *jj = (int *)malloc(sizeof(int) * (t->cnt > 0 ? t->cnt : 1));
    double *vv = (double *)malloc(sizeof(double) * (t->cnt > 0 ? t->cnt : 1));
    int *pos = (int *)malloc(sizeof(int) * (n + 1));
    memcpy(pos, rc, sizeof(int) * (n + 1));
    for (int k = 0; k < t->cnt; k++) { int q = pos[t->row[k]]++; jj[q] = t->col[k]; vv[q] = t->val[k]; }
    int *mk = (int *)malloc(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) mk[i] = -1;
    csr_alloc(M, n, n, t->cnt);
    int w = 0;
    typedef struct { int j; double v; } JV;
    JV *buf = NULL; int bcap = 0;
    M->ia[0] = 0;
    for (int i = 0; i < n; i++) {
        int start = w;
        for (int q = rc[i]; q < rc[i + 1]; q++) {
            int c = jj[q];
            if (mk[c] >= start) M->a[mk[c]] += vv[q];
            else { mk[c] = w; M->ja[w] = c; M->a[w] = vv[q]; w++; }
        }
        int len = w - start;
        if (len > 1) {
            int sorted = 1;
            for (int q = start + 1; q < w && sorted; q++) if (M->ja[q] < M->ja[q - 1]) sorted = 0;
            if (!sorted) {
                if (len <= 24) {
                    for (int a = start + 1; a < w; a++) { int cj = M->ja[a]; double cv = M->a[a]; int b = a - 1; while (b >= start && M->ja[b] > cj) { M->ja[b + 1] = M->ja[b]; M->a[b + 1] = M->a[b]; b--; } M->ja[b + 1] = cj; M->a[b + 1] = cv; }
                } else {
                    if (len > bcap) { bcap = 2 * len; buf = (JV *)realloc(buf, sizeof(JV) * bcap); }
                    for (int q = 0; q < len; q++) { buf[q].j = M->ja[start + q]; buf[q].v = M->a[start + q]; }
                    qsort(buf, len, sizeof(JV), cmp_int_dbl);
                    for (int q = 0; q < len; q++) { M->ja[start + q] = buf[q].j; M->a[start + q] = buf[q].v; }
                }
            }
        }
        M->ia[i + 1] = w;
    }
    free(buf); free(rc); free(jj); free(vv); free(pos); free(mk);
}
/* sum duplicates; output rows sorted by column */
static void trip_to_csr(const Trip *t, int n, CSR *M) {
    if (FASTCSR) { trip_to_csr_fast(t, n, M); return; }
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
    if (LOCALORD > 0) {
        /* random order inside consecutive blocks of LOCALORD rows, blocks in natural order: the same kind of greedy set, local memory access */
        for (int b0 = 0; b0 < n; b0 += LOCALORD) {
            int b1 = b0 + LOCALORD < n ? b0 + LOCALORD : n;
            for (int i = b1 - 1; i > b0; i--) { int j = b0 + (int)(rng() % (uint64_t)(i - b0 + 1)); int s = ord[i]; ord[i] = ord[j]; ord[j] = s; }
        }
    } else
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
    if (DOM > 0) {
        /* strongly dominant rows: their mutual couplings are below 1/DOM of the diagonal, so eliminating them together and lumping
           those couplings (excess form) is accurate; the whole set goes in one level */
        int cnt = 0;
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int p = A->ia[k]; p < A->ia[k + 1]; p++) if (A->ja[p] != k) sum += fabs(A->a[p]);
            mark[k] = (fabs(diag[k]) >= DOM * sum && diag[k] != 0) ? 1 : 2;
            cnt += mark[k] == 1;
        }
        if (cnt >= 0.05 * n) { free(key); free(ord); free(bcnt); return; }
        memset(mark, 0, sizeof(int) * n);
    }
    if (MISDYN) {
        /* greedy by the current number of undecided neighbours, lazy binary heap keyed by (degree, random) */
        int *deg = (int *)malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) deg[i] = A->ia[i + 1] - A->ia[i] + T->ia[i + 1] - T->ia[i];
        int hcap = 2 * n + 16, hn = 0;
        int64_t *hk = (int64_t *)malloc(sizeof(int64_t) * hcap);
        #define HPUSH(KEY) do { if (hn == hcap) { hcap *= 2; hk = (int64_t *)realloc(hk, sizeof(int64_t) * hcap); } int64_t kk_ = (KEY); int c_ = hn++; while (c_ > 0) { int p_ = (c_ - 1) / 2; if (hk[p_] <= kk_) break; hk[c_] = hk[p_]; c_ = p_; } hk[c_] = kk_; } while (0)
        #define KEYOF(I) (((int64_t)deg[I] << 52) | ((int64_t)(rng() & 0xFFFFF) << 32) | (int64_t)(I))
        for (int i = 0; i < n; i++) HPUSH(KEYOF(i));
        while (hn > 0) {
            int64_t top = hk[0]; int64_t last = hk[--hn]; int c_ = 0;
            for (;;) { int l_ = 2 * c_ + 1; if (l_ >= hn) break; if (l_ + 1 < hn && hk[l_ + 1] < hk[l_]) l_++; if (hk[l_] >= last) break; hk[c_] = hk[l_]; c_ = l_; }
            if (hn > 0) hk[c_] = last;
            int k = (int)(top & 0xFFFFFFFF), dk = (int)(top >> 52);
            if (mark[k] || dk != deg[k]) continue;
            double sum = 0, mx = 0;
            for (int p = A->ia[k]; p < A->ia[k + 1]; p++) if (A->ja[p] != k) { double v = fabs(A->a[p]); sum += v; if (v > mx) mx = v; }
            if (fabs(diag[k]) < piv * sum || diag[k] == 0.0) { mark[k] = 2; }
            else {
                mark[k] = 1;
                for (int pass = 0; pass < 2; pass++) {
                    const CSR *M = pass ? T : A;
                    for (int p = M->ia[k]; p < M->ia[k + 1]; p++) { int j = M->ja[p]; if (j != k && !mark[j] && fabs(M->a[p]) >= theta * mx) mark[j] = 2; }
                }
            }
            /* neighbours of every newly decided vertex lose an undecided neighbour */
            for (int pass = 0; pass < 2; pass++) {
                const CSR *M = pass ? T : A;
                for (int p = M->ia[k]; p < M->ia[k + 1]; p++) {
                    int j = M->ja[p];
                    if (j == k) continue;
                    if (!mark[j]) { deg[j]--; HPUSH(KEYOF(j)); }
                    else if (mark[j] == 2 && mark[k] == 1) {
                        for (int q2 = 0; q2 < 2; q2++) {
                            const CSR *M2 = q2 ? T : A;
                            for (int pp = M2->ia[j]; pp < M2->ia[j + 1]; pp++) { int l = M2->ja[pp]; if (!mark[l]) { deg[l]--; HPUSH(KEYOF(l)); } }
                        }
                    }
                }
            }
        }
        #undef HPUSH
        #undef KEYOF
        free(deg); free(hk); free(key); free(ord); free(bcnt);
        return;
    }
    double *rmx = NULL;
    if (STR2) {
        rmx = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) for (int p = A->ia[i]; p < A->ia[i + 1]; p++) if (A->ja[p] != i && fabs(A->a[p]) > rmx[i]) rmx[i] = fabs(A->a[p]);
    }
    for (int s = 0; s < n; s++) {
        int k = ord[s];
        if (mark[k]) continue;
        double sum = 0, mx = 0;
        for (int p = A->ia[k]; p < A->ia[k + 1]; p++) if (A->ja[p] != k) { double v = fabs(A->a[p]); sum += v; if (v > mx) mx = v; }
        if (fabs(diag[k]) < piv * sum || diag[k] == 0.0) { mark[k] = 2; continue; }
        mark[k] = 1;
        /* A row k holds a_kj (strong for row k against mx); T row k holds a_jk (strong for row j against rmx[j]) */
        for (int p = A->ia[k]; p < A->ia[k + 1]; p++) { int j = A->ja[p]; if (j != k && !mark[j] && (fabs(A->a[p]) >= theta * mx || (rmx && fabs(A->a[p]) >= theta * rmx[j]))) mark[j] = 2; }
        for (int p = T->ia[k]; p < T->ia[k + 1]; p++) { int j = T->ja[p]; if (j != k && !mark[j] && (fabs(T->a[p]) >= theta * mx || (rmx && fabs(T->a[p]) >= theta * rmx[j]))) mark[j] = 2; }
    }
    if (FFMAX > 0) {
        /* the pivot of an F row lumps its weak F-F couplings; when they are a large part of the diagonal (hubs with many small edges)
           the pivot underestimates a_kk and the elimination amplifies. Demote such rows to C; that only lowers the F-F weight of others. */
        for (int s = 0; s < n; s++) {
            int k = ord[s];
            if (mark[k] != 1) continue;
            double ff = 0;
            for (int p = A->ia[k]; p < A->ia[k + 1]; p++) { int j = A->ja[p]; if (j != k && mark[j] == 1) ff += fabs(A->a[p]); }
            if (ff > FFMAX * fabs(diag[k])) mark[k] = 2;
        }
    }
    free(rmx);
    free(key); free(ord); free(bcnt);
}

static void filter_reroute(CSR *S, double tau);
static void filter_reroute_ns(CSR *S, double tau);
/* one level of reduction. Returns the coarse matrix in *S. */
static inline double scale_of(double d) { return fabs(d); }
static double GUARD = 0.0;
static int LUMP = 1;
static double DROP = 0.0;
static double POS_TAU = 0.0;
static int KEEP = 4;
static int FFROW = 0;
static double BETA_SET = 1.0;
static int COUPLE = 1;   /* 0: independent samples; 1: chain for symmetric, transport for general; 2: half-circle for symmetric too */
static int BLOCK = 8;       /* LUMP 4: largest cluster */
static double TAUF = 0.0;   /* LUMP 4: update entries below TAUF sqrt(d_i d_j) are dropped per cluster */
static double FILT = 0.0;   /* LUMP 4: same test on the assembled coarse matrix */
static int DMODE = 0;       /* compensation of a dropped entry: 0 scaled diagonal (Ajiz-Jennings), 1 sign aware */
static double MAXCX = 60.0; /* give up when the hierarchy exceeds this operator complexity */
void ssr_config(double guard, int lump, int couple, double drop, double pos_tau, double beta, int keep, int ffrow) { BETA_SET = beta; KEEP = keep; FFROW = ffrow; GUARD = guard; LUMP = lump; COUPLE = couple; DROP = drop; POS_TAU = pos_tau; }
static int NTV = 0;         /* LUMP 4: test vectors kept exact by the truncated rows; the first is the constant, the others relaxed from random */
static int TVSW = 4;        /* symmetric sweeps that produce the relaxed test vectors */
static int TVR = 0;         /* symmetric sweeps on the test vectors at each coarse level */
void ssr_config2(int block, double tauf, double filt, int dmode, double maxcx) { BLOCK = block; TAUF = tauf; FILT = filt; DMODE = dmode; MAXCX = maxcx; }
void ssr_config3(int ntv, int tvsw, int tvr) { NTV = ntv; TVSW = tvsw; TVR = tvr; }
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
                if (sym && LUMP == 3) d = diag[i];
                for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
                    int j = A->ja[p]; if (j == i) continue;
                    if (mark[j] != 1) { if (!(sym && LUMP == 3)) d -= A->a[p]; L->fc.ja[a] = cidx[j]; L->fc.a[a++] = A->a[p]; }
                    else if (sym && LUMP == 3) d += FFROW ? A->a[p] : fabs(A->a[p]);      /* weak F-F edge: row-sum lump, or positive semidefinite removal */
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
        if (no == 0 && !EXFIX) continue;   /* EXFIX 0 reproduces the old bug: the excess update below was skipped for F rows without C neighbours */
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
        if (GUARD > 0) for (int q = 0; q < nin; q++) for (int c = 0; c < no; c++) if (oj[c] == ij[q]) dex[ij[q]] -= iv[q] * ov[c] * dinv;
        if (nin == 0 || no == 0) continue;
        if (COUPLE && !sym) {
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
        } else if (sym && LUMP == 3) {
            /* Tismenetsky-Kaporin second-order dropping. With v = A_Ck split into kept and dropped parts, the update -(vk+vd)(vk+vd)^T/d keeps every
               term with a kept factor and discards only -vd vd^T/d, including its diagonal. The discarded matrix is positive semidefinite, so the
               Schur complement stays positive definite for any sign pattern, and the error is second order in vd. */
            for (int a = 1; a < no; a++) {
                int cj = oj[a]; double cv = ov[a]; int b = a - 1;
                while (b >= 0 && fabs(ov[b]) < fabs(cv)) { oj[b + 1] = oj[b]; ov[b + 1] = ov[b]; b--; }
                oj[b + 1] = cj; ov[b + 1] = cv;
            }
            int nk = no < KEEP ? no : KEEP;
            for (int a = 0; a < nk; a++) {
                dex[oj[a]] -= ov[a] * ov[a] * dinv;
                for (int b = a + 1; b < no; b++) {
                    double v = -ov[a] * ov[b] * dinv;
                    trip_add(&tr, oj[a], oj[b], v); trip_add(&tr, oj[b], oj[a], v);
                }
            }
        } else if (sym && LUMP == 2) {
            /* Keep the signed structure. Entries opposite in sign to the pivot form the M part: chain-sample their fill, no cancellation there.
               Any fill touching a same-sign entry is kept exactly unless it is small against both diagonals; dropped mass goes to the diagonal
               through the excess. So the coarse operator carries the positive couplings instead of lumping them away. */
            int nn = 0;
            for (int a = 0; a < no; a++) if (ov[a] * dinv < 0) {
                int cj = oj[a]; double cv = ov[a]; oj[a] = oj[nn]; ov[a] = ov[nn]; oj[nn] = cj; ov[nn] = cv; nn++;
            }
            for (int a = 1; a < nn; a++) {
                int cj = oj[a]; double cv = ov[a]; int b = a - 1;
                while (b >= 0 && fabs(ov[b]) > fabs(cv)) { oj[b + 1] = oj[b]; ov[b + 1] = ov[b]; b--; }
                oj[b + 1] = cj; ov[b + 1] = cv;
            }
            cum[nn] = 0;
            for (int a = nn - 1; a >= 0; a--) cum[a] = cum[a + 1] + fabs(ov[a]);
            for (int a = 0; a + 1 < nn; a++) {
                double u = urand() * cum[a + 1], acc = 0;
                int l = nn - 1;
                for (int c = a + 1; c < nn; c++) { acc += fabs(ov[c]); if (u < acc) { l = c; break; } }
                double v = -ov[a] * (ov[l] > 0 ? 1 : -1) * cum[a + 1] * dinv;
                trip_add(&tr, oj[a], oj[l], v); trip_add(&tr, oj[l], oj[a], v);
                rowsum_samp[oj[a]] += v; rowsum_samp[oj[l]] += v;
            }
            for (int a = 0; a < no; a++) for (int b = (a + 1 > nn ? a + 1 : nn); b < no; b++) {
                double v = -ov[a] * ov[b] * dinv;
                if (v * v < POS_TAU * POS_TAU * fabs(diag[L->C[oj[a]]] * diag[L->C[oj[b]]])) continue;
                trip_add(&tr, oj[a], oj[b], v); trip_add(&tr, oj[b], oj[a], v);
                rowsum_samp[oj[a]] += v; rowsum_samp[oj[b]] += v;
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
        if (sym && LUMP == 3) d = dex[c];
        if (GUARD > 0 && d * dex[c] < GUARD * dex[c] * dex[c]) d = GUARD * dex[c];
        trip_add(&tr, c, c, d);
    }
    free(dex);
    trip_to_csr(&tr, nC, S);
    trip_free(&tr);
    if (sym && SFILT > 0 && LUMP == 1) filter_reroute(S, SFILT);
    if (!sym && NFILT > 0) filter_reroute_ns(S, NFILT);
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
                if (DROPMIN ? fabs(S->a[p]) < DROP * fmin(fabs(dg[c]), fabs(dg[j])) : S->a[p] * S->a[p] < DROP * DROP * fabs(dg[c] * dg[j])) continue;
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

/* Remove the symmetric entry e at (a, b) by adding a positive semidefinite rank-one term |e| u u^T, u = s E_a - sign(e) E_b / s.
   DMODE 0: s^2 = sqrt(d_a / d_b) gives the scale-invariant Ajiz-Jennings shift. DMODE 1: when e > 0 take s = 1, so u annihilates the constant and
   the action on the constant does not change; negative entries still use the scaled shift, because no s can annihilate the constant there. */
static inline int csr_find(const CSR *M, int i, int j) {
    int lo = M->ia[i], hi = M->ia[i + 1] - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (M->ja[m] == j) return m; if (M->ja[m] < j) lo = m + 1; else hi = m - 1; }
    return -1;
}

/* Non-Galerkin sparsification that keeps the constant and definiteness. A weak positive entry e goes to both diagonals: S + e L_ab. A weak negative entry of size w is the
   edge w L_ab; it is rerouted through the common neighbour c with the largest min(|s_ac|, |s_bc|) as rho w (L_ac + L_bc). For rho = 2 the difference
   w (2 L_ac + 2 L_bc - L_ab) is positive semidefinite by the triangle inequality, for any sign pattern of S. An edge with no common neighbour stays. */
static double RHO = 2.0;
void ssr_config4(double rho) { RHO = rho; }
static void filter_reroute(CSR *S, double tau) {
    int n = S->n, nnz = S->ia[n];
    double *dg = (double *)malloc(sizeof(double) * (n + 1));
    int *dpos = (int *)malloc(sizeof(int) * (n + 1));
    char *cut = (char *)calloc(nnz + 1, 1);
    double *add = (double *)calloc(nnz + 1, sizeof(double));
    for (int i = 0; i < n; i++) { dpos[i] = csr_find(S, i, i); dg[i] = dpos[i] >= 0 ? S->a[dpos[i]] : 0; }
    for (int i = 0; i < n; i++) for (int p = S->ia[i]; p < S->ia[i + 1]; p++) { int j = S->ja[p]; if (j != i && S->a[p] * S->a[p] < tau * tau * fabs(dg[i] * dg[j])) cut[p] = 1; }
    for (int i = 0; i < n; i++) for (int p = S->ia[i]; p < S->ia[i + 1]; p++) {
        int j = S->ja[p];
        if (!cut[p] || j <= i) continue;
        double e = S->a[p];
        int mir = csr_find(S, j, i);
        if (mir < 0) { cut[p] = 0; continue; }
        if (e >= 0) { add[dpos[i]] += e; add[dpos[j]] += e; continue; }
        /* merge the two sorted rows */
        int a = S->ia[i], b = S->ia[j], best = -1, pa = -1, pb = -1; double bv = 0;
        while (a < S->ia[i + 1] && b < S->ia[j + 1]) {
            int ca = S->ja[a], cb = S->ja[b];
            if (ca < cb) a++; else if (cb < ca) b++;
            else {
                if (ca != i && ca != j && !cut[a] && !cut[b]) { double v = fabs(S->a[a]) < fabs(S->a[b]) ? fabs(S->a[a]) : fabs(S->a[b]); if (v > bv) { bv = v; best = ca; pa = a; pb = b; } }
                a++; b++;
            }
        }
        if (best < 0) { cut[p] = 0; cut[mir] = 0; continue; }
        double w = -e, q = RHO * w;
        int ma = csr_find(S, best, i), mb = csr_find(S, best, j);
        if (ma < 0 || mb < 0) { cut[p] = 0; cut[mir] = 0; continue; }
        add[pa] -= q; add[ma] -= q; add[pb] -= q; add[mb] -= q;
        add[dpos[i]] += q - w; add[dpos[j]] += q - w; add[dpos[best]] += 2 * q;
    }
    int w = 0, start = 0;
    for (int i = 0; i < n; i++) {
        int end = S->ia[i + 1];
        for (int p = start; p < end; p++) if (!cut[p]) { S->ja[w] = S->ja[p]; S->a[w++] = S->a[p] + add[p]; }
        start = end;
        S->ia[i + 1] = w;
    }
    free(dg); free(dpos); free(cut); free(add);
}

/* Nonsymmetric counterpart. A weak entry e = s_ab moves to the directed path a -> c -> b: s_ac += e, s_cb += e, s_cc -= e. Row a, row c, column c
   and column b keep their sums, so every row sum and every column sum of S is unchanged, and for an M-matrix the signs and the diagonal
   dominance by rows and by columns are kept. c maximises min(|s_ac|, |s_cb|) over existing entries; without such c the entry stays. */
static void filter_reroute_ns(CSR *S, double tau) {
    int n = S->n, nnz = S->ia[n];
    double *dg = (double *)malloc(sizeof(double) * (n + 1));
    int *dpos = (int *)malloc(sizeof(int) * (n + 1));
    char *cut = (char *)calloc(nnz + 1, 1);
    double *add = (double *)calloc(nnz + 1, sizeof(double));
    for (int i = 0; i < n; i++) { dpos[i] = csr_find(S, i, i); dg[i] = dpos[i] >= 0 ? S->a[dpos[i]] : 0; }
    for (int a = 0; a < n; a++) for (int p = S->ia[a]; p < S->ia[a + 1]; p++) {
        int b = S->ja[p];
        if (b == a || S->a[p] * S->a[p] >= tau * tau * fabs(dg[a] * dg[b])) continue;
        int best = -1, pac = -1, pcb = -1; double bv = 0;
        for (int q = S->ia[a]; q < S->ia[a + 1]; q++) {
            int c = S->ja[q];
            if (c == a || c == b || cut[q]) continue;
            if (fabs(S->a[q]) <= bv) continue;
            int r = csr_find(S, c, b);
            if (r < 0 || cut[r]) continue;
            double v = fabs(S->a[q]) < fabs(S->a[r]) ? fabs(S->a[q]) : fabs(S->a[r]);
            if (v > bv) { bv = v; best = c; pac = q; pcb = r; }
        }
        if (best < 0) continue;
        double e = S->a[p] + add[p];
        cut[p] = 1;
        add[pac] += e; add[pcb] += e; add[dpos[best]] -= e;
    }
    int w = 0, start = 0;
    for (int i = 0; i < n; i++) {
        int end = S->ia[i + 1];
        for (int p = start; p < end; p++) if (!cut[p]) { S->ja[w] = S->ja[p]; S->a[w++] = S->a[p] + add[p]; }
        start = end;
        S->ia[i + 1] = w;
    }
    free(dg); free(dpos); free(cut); free(add);
}

static inline void comp_drop(double e, double da, double db, double *add_a, double *add_b) {
    double g = fabs(e);
    if (DMODE == 1 && e > 0) { *add_a = g; *add_b = g; return; }
    double s = (da > 0 && db > 0) ? sqrt(da / db) : 1.0;
    *add_a = g * s; *add_b = g / s;
}

/* Symmetric positive definite levels only. Clusters of up to BLOCK unknowns, mutually decoupled, grown greedily along the strongest relative couplings.
   Each cluster has an exact dense pivot. With V = A_{C,cluster} and D = L L^T, the Schur update is W^T W with W = L^{-1} V^T. For each row of W the
   KEEP largest entries are first-order; products of two dropped entries are discarded with their diagonal (Tismenetsky-Kaporin), which only adds a
   positive semidefinite matrix. */
/* With P_F = -L^-T W' for a truncated W', the Galerkin operator P^T A P equals A_CC - W^T W' - W'^T W + W'^T W'. Taking W' = W on the kept entries and 0 elsewhere
   gives exactly the second-order dropped update, so that dropping is Galerkin coarsening with truncated ideal interpolation, and x^T S' x >= x^T S x always.
   Hence W' may be corrected freely without losing definiteness: NTV > 0 moves the kept entries by the least change, weighted by |w|, that keeps w'.t = w.t for each test vector. */
static void reduce_block(Level *L, const CSR *A, CSR *S, double *ec, double piv, double theta, int deg_prio, const double *tv, double *tvc) {
    int n = A->n, bmax = BLOCK;
    double *diag = L->diag;
    for (int i = 0; i < n; i++) diag[i] = get_diag(A, i);
    int *mark = (int *)calloc(n, sizeof(int));
    int *ord = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = rng() % (uint64_t)(i + 1); int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    if (deg_prio) {
        int maxd = 0;
        for (int i = 0; i < n; i++) { int d = A->ia[i + 1] - A->ia[i]; if (d > maxd) maxd = d; }
        int *bc = (int *)calloc(maxd + 2, sizeof(int)), *o2 = (int *)malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) bc[A->ia[i + 1] - A->ia[i] + 1]++;
        for (int d = 0; d <= maxd; d++) bc[d + 1] += bc[d];
        for (int s = 0; s < n; s++) { int i = ord[s]; o2[bc[A->ia[i + 1] - A->ia[i]]++] = i; }
        memcpy(ord, o2, sizeof(int) * n);
        free(bc); free(o2);
    }
    int *F = (int *)malloc(sizeof(int) * (n + 1)), *bptr = (int *)malloc(sizeof(int) * (n + 2));
    int nF = 0, nb = 0;
    bptr[0] = 0;
    int *clpos = (int *)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) clpos[i] = -1;
    double *gain = (double *)calloc(n, sizeof(double));
    int capc = 256, *cand = (int *)malloc(sizeof(int) * capc);
    double *Lc = (double *)malloc(sizeof(double) * bmax * bmax), *rowv = (double *)malloc(sizeof(double) * bmax);
    for (int s = 0; s < n; s++) {
        int k = ord[s];
        if (mark[k]) continue;
        if (diag[k] <= 0) { mark[k] = 2; continue; }
        int fs = nF, clen = 1, nc = 0;
        F[nF++] = k; mark[k] = 1; clpos[k] = 0; Lc[0] = sqrt(diag[k]);
        int add = k;
        for (;;) {
            /* neighbours of the newest member become candidates */
            double mx = 0;
            for (int p = A->ia[add]; p < A->ia[add + 1]; p++) if (A->ja[p] != add && fabs(A->a[p]) > mx) mx = fabs(A->a[p]);
            for (int p = A->ia[add]; p < A->ia[add + 1]; p++) {
                int j = A->ja[p];
                if (j == add || mark[j] || fabs(A->a[p]) < theta * mx) continue;
                double g = A->a[p] * A->a[p] / fabs(diag[add] * diag[j] + 1e-300);
                if (gain[j] == 0) { if (nc == capc) { capc *= 2; cand = (int *)realloc(cand, sizeof(int) * capc); } cand[nc++] = j; }
                gain[j] += g;
            }
            if (clen == bmax) break;
            /* strongest candidate that keeps the dense pivot well conditioned */
            int best = -1, bi = -1;
            for (int q = 0; q < nc; q++) { int j = cand[q]; if (!mark[j] && (best < 0 || gain[j] > gain[best])) { best = j; bi = q; } }
            if (best < 0) break;
            for (int i = 0; i < clen; i++) rowv[i] = 0;
            for (int p = A->ia[best]; p < A->ia[best + 1]; p++) { int c = clpos[A->ja[p]]; if (c >= 0) rowv[c] = A->a[p]; }
            double sum = 0;
            for (int i = 0; i < clen; i++) {
                double t = rowv[i];
                for (int q = 0; q < i; q++) t -= Lc[i * bmax + q] * rowv[q];
                rowv[i] = t / Lc[i * bmax + i];
                sum += rowv[i] * rowv[i];
            }
            double pv = diag[best] - sum;
            cand[bi] = cand[--nc];
            if (diag[best] <= 0 || pv < piv * diag[best]) { mark[best] = 2; gain[best] = 0; continue; }
            for (int i = 0; i < clen; i++) Lc[clen * bmax + i] = rowv[i];
            Lc[clen * bmax + clen] = sqrt(pv);
            clpos[best] = clen; clen++;
            F[nF++] = best; mark[best] = 1; gain[best] = 0;
            add = best;
        }
        /* strong neighbours of the cluster are C */
        for (int q = fs; q < nF; q++) {
            int m = F[q];
            double mx = 0;
            for (int p = A->ia[m]; p < A->ia[m + 1]; p++) if (A->ja[p] != m && fabs(A->a[p]) > mx) mx = fabs(A->a[p]);
            for (int p = A->ia[m]; p < A->ia[m + 1]; p++) { int j = A->ja[p]; if (j != m && !mark[j] && fabs(A->a[p]) >= theta * mx) mark[j] = 2; }
            clpos[m] = -1;
        }
        for (int q = 0; q < nc; q++) gain[cand[q]] = 0;
        bptr[++nb] = nF;
    }
    free(ord); free(gain); free(cand);

    int nC = n - nF;
    int *C = (int *)malloc(sizeof(int) * (nC + 1)), *cidx = (int *)malloc(sizeof(int) * n), *fidx = (int *)malloc(sizeof(int) * n);
    { int w = 0; for (int i = 0; i < n; i++) if (mark[i] != 1) { cidx[i] = w; C[w++] = i; } }
    for (int f = 0; f < nF; f++) fidx[F[f]] = f;
    L->n = n; L->nF = nF; L->nC = nC; L->F = F; L->C = C; L->nb = nb; L->bptr = bptr;
    L->dinv = (double *)malloc(sizeof(double));
    L->boff = (int *)malloc(sizeof(int) * (nb + 1));
    { int o = 0; for (int k = 0; k < nb; k++) { L->boff[k] = o; int b = bptr[k + 1] - bptr[k]; o += b * b; } L->boff[nb] = o; L->binv = (double *)malloc(sizeof(double) * (o + 1)); }
    L->tF = (double *)malloc(sizeof(double) * (nF + 1)); L->wF = (double *)malloc(sizeof(double) * (nF + 1));

    /* A_FC in F order, A_CF in C order */
    {
        int nnzfc = 0, nnzcf = 0;
        for (int i = 0; i < n; i++) for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
            int j = A->ja[p]; if (j == i) continue;
            if (mark[i] == 1 && mark[j] != 1) nnzfc++;
            if (mark[i] != 1 && mark[j] == 1) nnzcf++;
        }
        csr_alloc(&L->fc, nF, nC, nnzfc); csr_alloc(&L->cf, nC, nF, nnzcf);
        int a = 0; L->fc.ia[0] = 0;
        for (int f = 0; f < nF; f++) {
            int i = F[f];
            for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int j = A->ja[p]; if (mark[j] != 1) { L->fc.ja[a] = cidx[j]; L->fc.a[a++] = A->a[p]; } }
            L->fc.ia[f + 1] = a;
        }
        int b = 0; L->cf.ia[0] = 0;
        for (int c = 0; c < nC; c++) {
            int i = C[c];
            for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int j = A->ja[p]; if (mark[j] == 1) { L->cf.ja[b] = fidx[j]; L->cf.a[b++] = A->a[p]; } }
            L->cf.ia[c + 1] = b;
        }
    }

    /* coarse matrix */
    Trip tr; trip_init(&tr, A->ia[n] * 2);
    double *dex = (double *)malloc(sizeof(double) * (nC + 1)), *d0 = (double *)malloc(sizeof(double) * (nC + 1));
    for (int c = 0; c < nC; c++) { dex[c] = d0[c] = diag[C[c]]; ec[c] = 0; }
    for (int c = 0; c < nC; c++) { int i = C[c]; for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int j = A->ja[p]; if (j != i && mark[j] != 1) trip_add(&tr, c, cidx[j], A->a[p]); } }
    int *lmap = (int *)malloc(sizeof(int) * (nC + 1));
    for (int c = 0; c < nC; c++) lmap[c] = -1;
    int capm = 256;
    int *bd = (int *)malloc(sizeof(int) * capm), *isk = (int *)calloc(capm, sizeof(int)), *kl = (int *)malloc(sizeof(int) * (KEEP + 1));
    double *W = (double *)malloc(sizeof(double) * bmax * capm), *U = (double *)malloc(sizeof(double) * capm * capm), *Ud = (double *)malloc(sizeof(double) * capm);
    double *D = (double *)malloc(sizeof(double) * bmax * bmax), *Li = (double *)malloc(sizeof(double) * bmax * bmax);
    int ntv = NTV, wcap = capm;
    double *wp = (double *)malloc(sizeof(double) * wcap);
    double G[64], rr[8];
    for (int j = 0; j < ntv; j++) for (int c = 0; c < nC; c++) tvc[j * nC + c] = tv[j * n + C[c]];
    for (int k = 0; k < nb; k++) {
        int fs = bptr[k], b = bptr[k + 1] - fs, m = 0;
        for (int i = 0; i < b; i++) clpos[F[fs + i]] = i;
        for (int i = 0; i < b * b; i++) D[i] = 0;
        for (int i = 0; i < b; i++) {
            int r = F[fs + i];
            for (int p = A->ia[r]; p < A->ia[r + 1]; p++) {
                int j = A->ja[p], q = clpos[j];
                if (q >= 0) D[i * b + q] = A->a[p];
                else if (mark[j] == 1) D[i * b + i] += FFROW ? A->a[p] : fabs(A->a[p]);    /* weak edge to another cluster: row-sum lump, or removed positive semidefinitely */
                else if (lmap[cidx[j]] < 0) {
                    if (m == capm) {
                        capm *= 2;
                        bd = (int *)realloc(bd, sizeof(int) * capm); free(isk); isk = (int *)calloc(capm, sizeof(int));
                        W = (double *)realloc(W, sizeof(double) * bmax * capm);
                        free(U); U = (double *)malloc(sizeof(double) * capm * capm); Ud = (double *)realloc(Ud, sizeof(double) * capm);
                    }
                    lmap[cidx[j]] = m; bd[m++] = cidx[j];
                }
            }
        }
        for (int i = 0; i < b * m; i++) W[i] = 0;
        for (int i = 0; i < b; i++) {
            int r = F[fs + i];
            for (int p = A->ia[r]; p < A->ia[r + 1]; p++) { int j = A->ja[p]; if (mark[j] != 1) W[i * m + lmap[cidx[j]]] = A->a[p]; }
        }
        /* dense Cholesky of the pivot; W = L^-1 V^T */
        for (int i = 0; i < b; i++) {
            for (int q = 0; q <= i; q++) {
                double t = D[i * b + q];
                for (int l = 0; l < q; l++) t -= D[i * b + l] * D[q * b + l];
                if (q == i) D[i * b + i] = sqrt(t > 0 ? t : 1e-14 * fabs(diag[F[fs + i]]) + 1e-300);
                else D[i * b + q] = t / D[q * b + q];
            }
            for (int c = 0; c < m; c++) {
                double t = W[i * m + c];
                for (int l = 0; l < i; l++) t -= D[i * b + l] * W[l * m + c];
                W[i * m + c] = t / D[i * b + i];
            }
        }
        /* explicit inverse */
        for (int c = 0; c < b; c++) for (int i = 0; i < b; i++) {
            double t = i == c ? 1.0 : 0.0;
            for (int l = 0; l < i; l++) t -= D[i * b + l] * Li[l * b + c];
            Li[i * b + c] = t / D[i * b + i];
        }
        double *M = L->binv + L->boff[k];
        for (int i = 0; i < b; i++) for (int j = 0; j < b; j++) { double t = 0; for (int l = 0; l < b; l++) t += Li[l * b + i] * Li[l * b + j]; M[i * b + j] = t; }
        /* update */
        for (int i = 0; i < m * m; i++) U[i] = 0;
        for (int i = 0; i < m; i++) Ud[i] = 0;
        if (m > wcap) { wcap = capm; wp = (double *)realloc(wp, sizeof(double) * wcap); }
        for (int r = 0; r < b; r++) {
            const double *w = W + r * m;
            int nk = 0;
            if (m <= KEEP) { for (int c = 0; c < m; c++) kl[nk++] = c; }
            else for (int c = 0; c < m; c++) {
                double v = fabs(w[c]);
                if (nk == KEEP && v <= fabs(w[kl[nk - 1]])) continue;
                int q = nk < KEEP ? nk++ : KEEP - 1;
                while (q > 0 && fabs(w[kl[q - 1]]) < v) { kl[q] = kl[q - 1]; q--; }
                kl[q] = c;
            }
            for (int q = 0; q < nk; q++) isk[kl[q]] = 1;
            for (int q = 0; q < nk; q++) wp[kl[q]] = w[kl[q]];
            int pp = ntv < nk ? ntv : nk;
            if (pp > 0 && m > nk) {
                /* residual of the dropped part on each test vector, then the weighted least change on the kept entries */
                double wmax = 0;
                for (int q = 0; q < nk; q++) if (fabs(w[kl[q]]) > wmax) wmax = fabs(w[kl[q]]);
                for (int j = 0; j < pp; j++) {
                    double t = 0;
                    const double *tj = tvc + j * nC;
                    for (int c = 0; c < m; c++) if (!isk[c]) t += w[c] * tj[bd[c]];
                    rr[j] = t;
                }
                double tr = 0;
                for (int i = 0; i < pp; i++) for (int j = 0; j <= i; j++) {
                    double t = 0;
                    for (int q = 0; q < nk; q++) { int a = kl[q]; t += (fabs(w[a]) + 1e-3 * wmax) * tvc[i * nC + bd[a]] * tvc[j * nC + bd[a]]; }
                    G[i * 8 + j] = t;
                    if (i == j) tr += t;
                }
                for (int i = 0; i < pp; i++) G[i * 8 + i] += 1e-10 * tr + 1e-300;
                int okc = 1;
                for (int i = 0; i < pp && okc; i++) {
                    for (int j = 0; j <= i; j++) {
                        double t = G[i * 8 + j];
                        for (int l = 0; l < j; l++) t -= G[i * 8 + l] * G[j * 8 + l];
                        if (j == i) { if (t <= 0) { okc = 0; break; } G[i * 8 + i] = sqrt(t); }
                        else G[i * 8 + j] = t / G[j * 8 + j];
                    }
                }
                if (okc) {
                    for (int i = 0; i < pp; i++) { double t = rr[i]; for (int l = 0; l < i; l++) t -= G[i * 8 + l] * rr[l]; rr[i] = t / G[i * 8 + i]; }
                    for (int i = pp - 1; i >= 0; i--) { double t = rr[i]; for (int l = i + 1; l < pp; l++) t -= G[l * 8 + i] * rr[l]; rr[i] = t / G[i * 8 + i]; }
                    for (int q = 0; q < nk; q++) {
                        int a = kl[q]; double t = 0;
                        for (int j = 0; j < pp; j++) t += rr[j] * tvc[j * nC + bd[a]];
                        wp[a] += (fabs(w[a]) + 1e-3 * wmax) * t;
                    }
                }
            }
            for (int q = 0; q < nk; q++) {
                int a = kl[q]; double wa = w[a], pa = wp[a];
                Ud[a] += 2 * wa * pa - pa * pa;
                for (int c = 0; c < m; c++) {
                    if (isk[c]) { if (c > a) U[a * m + c] += wa * wp[c] + pa * w[c] - pa * wp[c]; }
                    else { int lo = a < c ? a : c, hi = a < c ? c : a; U[lo * m + hi] += pa * w[c]; }
                }
            }
            for (int q = 0; q < nk; q++) isk[kl[q]] = 0;
        }
        for (int a = 0; a < m; a++) {
            dex[bd[a]] -= Ud[a];
            for (int c = a + 1; c < m; c++) {
                double v = -U[a * m + c];
                if (v == 0) continue;
                int ca = bd[a], cc = bd[c];
                if (TAUF > 0 && v * v < TAUF * TAUF * fabs(d0[ca] * d0[cc])) {
                    double x, y; comp_drop(v, d0[ca], d0[cc], &x, &y);
                    dex[ca] += x; dex[cc] += y;
                    continue;
                }
                trip_add(&tr, ca, cc, v); trip_add(&tr, cc, ca, v);
            }
        }
        for (int a = 0; a < m; a++) lmap[bd[a]] = -1;
        for (int i = 0; i < b; i++) clpos[F[fs + i]] = -1;
    }
    for (int c = 0; c < nC; c++) trip_add(&tr, c, c, dex[c]);
    trip_to_csr(&tr, nC, S);
    trip_free(&tr);
    if (FILT > 0 && DMODE == 2) filter_reroute(S, FILT);
    else if (FILT > 0) {
        double *dg = (double *)malloc(sizeof(double) * (nC + 1)), *add = (double *)calloc(nC + 1, sizeof(double));
        for (int c = 0; c < nC; c++) dg[c] = get_diag(S, c);
        int w = 0, start = 0;
        for (int c = 0; c < nC; c++) {
            int end = S->ia[c + 1];
            for (int p = start; p < end; p++) {
                int j = S->ja[p];
                if (j != c && S->a[p] * S->a[p] < FILT * FILT * fabs(dg[c] * dg[j])) {
                    double x, y; comp_drop(S->a[p], dg[c], dg[j], &x, &y);
                    add[c] += x;     /* the mirror entry adds the other share when its row is visited */
                    continue;
                }
                S->ja[w] = j; S->a[w++] = S->a[p];
            }
            start = end;
            S->ia[c + 1] = w;
        }
        for (int c = 0; c < nC; c++) for (int p = S->ia[c]; p < S->ia[c + 1]; p++) if (S->ja[p] == c) S->a[p] += add[c];
        free(dg); free(add);
    }
    free(wp);
    free(D); free(Li); free(W); free(U); free(Ud); free(bd); free(isk); free(kl); free(lmap); free(dex); free(d0);
    free(Lc); free(rowv); free(clpos); free(mark); free(cidx); free(fidx);
}

/* row-sum-preserving lump of entries with the same sign as the diagonal. Returns 1 and fills B if any entry moved. */
static double BETA = 0.0;   /* fraction of each same-sign entry kept in the hierarchy; the rest is lumped */
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
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) if (A->ja[p] != i && A->a[p] * d > 0) add += (1 - BETA) * A->a[p];
        int has = 0;
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
            int j = A->ja[p];
            if (j == i) { B->ja[w] = i; B->a[w++] = d + add; has = 1; }
            else if (A->a[p] * d <= 0) { B->ja[w] = j; B->a[w++] = A->a[p]; }
            else if (BETA > 0) { B->ja[w] = j; B->a[w++] = BETA * A->a[p]; }
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

static inline void block_apply(const Level *L, const double *in, double *out) {
    for (int k = 0; k < L->nb; k++) {
        int s = L->bptr[k], b = L->bptr[k + 1] - s;
        const double *M = L->binv + L->boff[k];
        if (b == 1) { out[s] = M[0] * in[s]; continue; }
        for (int i = 0; i < b; i++) { double t = 0; for (int j = 0; j < b; j++) t += M[i * b + j] * in[s + j]; out[s + i] = t; }
    }
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
    int pre = smooth && H->smode != 2, post = smooth && H->smode != 1;
    if (pre && H->nu == 1 && L->dpos) {
        /* forward sweep from zero touches only the lower part; the residual left behind is -U x */
        const int *ia = L->A.ia, *ja = L->A.ja; const double *a = L->A.a; const int *dp = L->dpos;
        for (int i = 0; i < n; i++) {
            double t = r[i];
            for (int p = ia[i]; p < dp[i]; p++) t -= a[p] * x[ja[p]];
            x[i] = L->diag[i] != 0 ? t / L->diag[i] : 0;
        }
        for (int i = 0; i < n; i++) {
            double t = 0;
            for (int p = dp[i] + 1; p < ia[i + 1]; p++) t -= a[p] * x[ja[p]];
            L->wk[i] = t;
        }
        rr = L->wk;
    } else if (pre) {
        for (int s = 0; s < H->nu; s++) gs(&L->A, L->diag, r, x, 0);
        for (int i = 0; i < n; i++) {
            double t = r[i];
            for (int p = L->A.ia[i]; p < L->A.ia[i + 1]; p++) t -= L->A.a[p] * x[L->A.ja[p]];
            L->wk[i] = t;
        }
        rr = L->wk;
    }
    if (L->nb) { for (int f = 0; f < L->nF; f++) L->tF[f] = rr[L->F[f]]; block_apply(L, L->tF, L->rF); }
    else for (int f = 0; f < L->nF; f++) L->rF[f] = rr[L->F[f]] * L->dinv[f];
    for (int c = 0; c < L->nC; c++) {
        double t = rr[L->C[c]];
        for (int p = L->cf.ia[c]; p < L->cf.ia[c + 1]; p++) t -= L->cf.a[p] * L->rF[L->cf.ja[p]];
        L->rC[c] = t;
    }
    cycle(H, l + 1, L->rC, L->yC);
    for (int c = 0; c < L->nC; c++) x[L->C[c]] += L->yC[c];
    if (L->nb) {
        for (int f = 0; f < L->nF; f++) {
            double t = 0;
            for (int p = L->fc.ia[f]; p < L->fc.ia[f + 1]; p++) t += L->fc.a[p] * L->yC[L->fc.ja[p]];
            L->tF[f] = t;
        }
        block_apply(L, L->tF, L->wF);
        for (int f = 0; f < L->nF; f++) x[L->F[f]] += L->rF[f] - L->wF[f];
    } else for (int f = 0; f < L->nF; f++) {
        double t = 0;
        for (int p = L->fc.ia[f]; p < L->fc.ia[f + 1]; p++) t += L->fc.a[p] * L->yC[L->fc.ja[p]];
        x[L->F[f]] += L->rF[f] - t * L->dinv[f];
    }
    if (post) for (int s = 0; s < H->nu; s++) gs(&L->A, L->diag, r, x, 1);
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
    /* SMODE 3: post-smoothing only on symmetric M-matrices (no off-diagonal with the sign of the diagonal), pre and post otherwise */
    H->smode = SMODE;
    if (SMODE == 3) {
        int mm = sym;
        for (int i = 0; i < n && mm; i++) { double d = 0; for (int p = ia[i]; p < ia[i + 1]; p++) if (A.ja[p] == i) d = A.a[p]; for (int p = ia[i]; p < ia[i + 1]; p++) if (A.ja[p] != i && A.a[p] * d > 0) { mm = 0; break; } }
        H->smode = mm ? 2 : 0;
    }
    /* cluster pivots only where the matrix is not an M-matrix; there chain sampling needs no lump and stays as it is */
    int blockmode = 0;
    if (LUMP == 4) {
        if (sym) for (int i = 0; i < n && !blockmode; i++) { double d = 0; for (int p = ia[i]; p < ia[i + 1]; p++) if (A.ja[p] == i) d = A.a[p]; for (int p = ia[i]; p < ia[i + 1]; p++) if (A.ja[p] != i && A.a[p] * d > 0) { blockmode = 1; break; } }
        LUMP = 1;
    }
    double *tv = NULL;
    if (blockmode && NTV > 0) {
        if (NTV > 8) NTV = 8;
        tv = (double *)malloc(sizeof(double) * (size_t)NTV * n);
        double *dg = (double *)malloc(sizeof(double) * n), *zb = (double *)calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) dg[i] = get_diag(&A, i);
        for (int i = 0; i < n; i++) tv[i] = 1.0;
        for (int j = 1; j < NTV; j++) {
            double *x = tv + (size_t)j * n;
            for (int i = 0; i < n; i++) x[i] = urand() - 0.5;
            for (int sw = 0; sw < TVSW; sw++) { gs(&A, dg, zb, x, 0); gs(&A, dg, zb, x, 1); }
            double mx = 0;
            for (int i = 0; i < n; i++) if (fabs(x[i]) > mx) mx = fabs(x[i]);
            if (mx > 0) for (int i = 0; i < n; i++) x[i] /= mx;
        }
        free(dg); free(zb);
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
        BETA = (LUMP == 2 && sym && H->nlev == 0) ? BETA_SET : 0.0;
        int lumped = !blockmode && (LUMP == 1 || (LUMP == 2 && !sym) || (LUMP == 2 && H->nlev == 0 && sym)) ? lump_positive(&cur, &Bar) : 0;
        double *ec = (double *)malloc(sizeof(double) * (cur.n + 1));
        double *tvc = (blockmode && NTV > 0) ? (double *)malloc(sizeof(double) * (size_t)NTV * cur.n) : NULL;
        if (blockmode) reduce_block(L, &cur, &S, ec, piv, 0.0, deg_prio, tv, tvc);   /* THETA stays with the scalar path: weak F-F edges failed here */
        else reduce(L, lumped ? &Bar : &cur, &S, e, ec, sym, piv, theta, ns, deg_prio, topk);
        free(e); e = ec;
        if (tvc) {
            free(tv); tv = tvc;
            if (TVR > 0) {
                double *dg = (double *)malloc(sizeof(double) * (S.n + 1)), *zb = (double *)calloc(S.n + 1, sizeof(double));
                for (int i = 0; i < S.n; i++) dg[i] = get_diag(&S, i);
                for (int j = 1; j < NTV; j++) {
                    double *x = tv + (size_t)j * S.n, mx = 0;
                    for (int sw = 0; sw < TVR; sw++) { gs(&S, dg, zb, x, 0); gs(&S, dg, zb, x, 1); }
                    for (int i = 0; i < S.n; i++) if (fabs(x[i]) > mx) mx = fabs(x[i]);
                    if (mx > 0) for (int i = 0; i < S.n; i++) x[i] /= mx;
                }
                free(dg); free(zb);
            }
        }
        if (lumped) { for (int i = 0; i < cur.n; i++) L->diag[i] = get_diag(&cur, i); csr_free(&Bar); }
        L->A = cur;
        {
            /* split smoothing needs sorted rows with an explicit diagonal equal to diag */
            int ok = 1;
            L->dpos = (int *)malloc(sizeof(int) * (cur.n + 1));
            for (int i = 0; i < cur.n && ok; i++) {
                int dpi = -1;
                for (int p = cur.ia[i]; p < cur.ia[i + 1]; p++) { if (p > cur.ia[i] && cur.ja[p] <= cur.ja[p - 1]) { ok = 0; break; } if (cur.ja[p] == i) dpi = p; }
                if (dpi < 0 || cur.a[dpi] != L->diag[i]) ok = 0;
                L->dpos[i] = dpi;
            }
            if (!ok) { free(L->dpos); L->dpos = NULL; }
        }
        H->nnz_sum += cur.ia[cur.n];
        L->rF = (double *)malloc(sizeof(double) * (L->nF > 0 ? L->nF : 1));
        L->rC = (double *)malloc(sizeof(double) * (L->nC > 0 ? L->nC : 1));
        L->yC = (double *)malloc(sizeof(double) * (L->nC > 0 ? L->nC : 1));
        H->nlev++;
        cur = S;
        if (L->nF == 0) break;   /* no acceptable pivots: fall through to direct if small enough */
        if (blockmode && H->nnz_sum + cur.ia[cur.n] > MAXCX * H->nnz0) { H->cn = -cur.n; csr_free(&cur); free(e); return H; }
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

/* Transpose of the cycle, for a hierarchy built on B = A^T: applies M_B^T, an approximation of A^-1.
   With Lf = D + L_B, Ub = D + U_B and K the elimination-plus-coarse part, the cycle is M = Ub^-1 + (I - Ub^-1 B) N, N = Lf^-1 + K (I - B Lf^-1),
   so M^T = Ub^-T + N^T (I - B^T Ub^-T), N^T = Lf^-T + (I - Lf^-T B^T) K^T. All transposed products use the rows of B by scattering. NU = 1 only. */
static void lu_solve_T(const double *a, const int *piv, int n, double *x) {
    for (int i = 0; i < n; i++) { double s = x[i]; for (int j = 0; j < i; j++) s -= a[j * n + i] * x[j]; x[i] = s / a[i * n + i]; }
    for (int i = n - 1; i >= 0; i--) { double s = x[i]; for (int j = i + 1; j < n; j++) s -= a[j * n + i] * x[j]; x[i] = s; }
    for (int k = n - 1; k >= 0; k--) { int m = piv[k]; if (m != k) { double t = x[k]; x[k] = x[m]; x[m] = t; } }
}
/* x = (D + U)^-T r : forward, scattering the strictly upper part of each row */
static void solve_UbT(const CSR *A, const double *dg, const double *r, double *x, double *w) {
    int n = A->n; memcpy(w, r, sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        x[i] = dg[i] != 0 ? w[i] / dg[i] : 0;
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int j = A->ja[p]; if (j > i) w[j] -= A->a[p] * x[i]; }
    }
}
/* x = (D + L)^-T r : backward, scattering the strictly lower part of each row */
static void solve_LfT(const CSR *A, const double *dg, const double *r, double *x, double *w) {
    int n = A->n; memcpy(w, r, sizeof(double) * n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = dg[i] != 0 ? w[i] / dg[i] : 0;
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) { int j = A->ja[p]; if (j < i) w[j] -= A->a[p] * x[i]; }
    }
}
/* y = r - A^T x */
static void resid_T(const CSR *A, const double *r, const double *x, double *y) {
    int n = A->n; memcpy(y, r, sizeof(double) * n);
    for (int i = 0; i < n; i++) { double xi = x[i]; if (xi == 0) continue; for (int p = A->ia[i]; p < A->ia[i + 1]; p++) y[A->ja[p]] -= A->a[p] * xi; }
}
static void block_apply_T(const Level *L, const double *in, double *out) {
    for (int k = 0; k < L->nb; k++) {
        int s = L->bptr[k], b = L->bptr[k + 1] - s;
        const double *M = L->binv + L->boff[k];
        for (int i = 0; i < b; i++) { double t = 0; for (int j = 0; j < b; j++) t += M[j * b + i] * in[s + j]; out[s + i] = t; }
    }
}
/* K^T s into out (length n) */
static void cycle_T(SSR *H, int l, const double *r, double *x);
static void kT(SSR *H, int l, const double *s, double *out) {
    Level *L = &H->lev[l];
    if (L->nb) { for (int f = 0; f < L->nF; f++) L->tF[f] = s[L->F[f]]; block_apply_T(L, L->tF, L->rF); }
    else for (int f = 0; f < L->nF; f++) L->rF[f] = s[L->F[f]] * L->dinv[f];
    for (int c = 0; c < L->nC; c++) L->rC[c] = s[L->C[c]];
    /* rC -= FC^T rF : fc rows are F, columns C */
    for (int f = 0; f < L->nF; f++) { double v = L->rF[f]; if (v == 0) continue; for (int p = L->fc.ia[f]; p < L->fc.ia[f + 1]; p++) L->rC[L->fc.ja[p]] -= L->fc.a[p] * v; }
    cycle_T(H, l + 1, L->rC, L->yC);
    for (int c = 0; c < L->nC; c++) out[L->C[c]] = L->yC[c];
    /* out_F = rF - Dinv^T CF^T y : cf rows are C, columns F */
    double *t = L->nb ? L->tF : NULL;
    if (L->nb) { memset(t, 0, sizeof(double) * L->nF); for (int c = 0; c < L->nC; c++) { double v = L->yC[c]; for (int p = L->cf.ia[c]; p < L->cf.ia[c + 1]; p++) t[L->cf.ja[p]] += L->cf.a[p] * v; } block_apply_T(L, t, L->wF); for (int f = 0; f < L->nF; f++) out[L->F[f]] = L->rF[f] - L->wF[f]; }
    else {
        for (int f = 0; f < L->nF; f++) out[L->F[f]] = 0;
        for (int c = 0; c < L->nC; c++) { double v = L->yC[c]; for (int p = L->cf.ia[c]; p < L->cf.ia[c + 1]; p++) out[L->F[L->cf.ja[p]]] += L->cf.a[p] * v; }
        for (int f = 0; f < L->nF; f++) out[L->F[f]] = L->rF[f] - out[L->F[f]] * L->dinv[f];
    }
}
static double **TBUF = NULL; static int TBUFN = 0;
static void cycle_T(SSR *H, int l, const double *r, double *x) {
    if (l == H->nlev) { memcpy(x, r, sizeof(double) * H->cn); lu_solve_T(H->lu, H->piv, H->cn, x); return; }
    Level *L = &H->lev[l];
    int n = L->n;
    int smooth = H->nu > 0 && (l % H->every) == 0;
    if (!smooth) { kT(H, l, r, x); return; }
    double *a = TBUF[4 * l], *s = TBUF[4 * l + 1], *c = TBUF[4 * l + 2], *w = TBUF[4 * l + 3];
    if (H->smode != 1) { solve_UbT(&L->A, L->diag, r, a, w); resid_T(&L->A, r, a, s); }   /* a = Ub^-T r, s = r - B^T a */
    else { memset(a, 0, sizeof(double) * n); memcpy(s, r, sizeof(double) * n); }
    kT(H, l, s, c);                               /* c = K^T s */
    if (H->smode != 2) { resid_T(&L->A, s, c, w); solve_LfT(&L->A, L->diag, w, x, s); }  /* x = Lf^-T (s - B^T c) */
    else memset(x, 0, sizeof(double) * n);
    for (int i = 0; i < n; i++) x[i] += c[i] + a[i];
}
void ssr_apply_T(SSR *H, const double *r, double *z) {
    if (TBUFN < H->nlev) {
        TBUF = (double **)realloc(TBUF, sizeof(double *) * 4 * H->nlev);
        for (int i = 4 * TBUFN; i < 4 * H->nlev; i++) TBUF[i] = NULL;
        TBUFN = H->nlev;
    }
    for (int l = 0; l < H->nlev; l++) for (int q = 0; q < 4; q++) { free(TBUF[4 * l + q]); TBUF[4 * l + q] = (double *)malloc(sizeof(double) * (H->lev[l].n + 1)); }
    for (int i = 0; i < H->N; i++) H->rs[i] = H->T[i] * r[i];
    cycle_T(H, 0, H->rs, H->zs);
    for (int i = 0; i < H->N; i++) z[i] = H->Y[i] * H->zs[i];
}

void ssr_apply(SSR *H, const double *r, double *z) {
    for (int i = 0; i < H->N; i++) H->rs[i] = H->Y[i] * r[i];
    cycle(H, 0, H->rs, H->zs);
    for (int i = 0; i < H->N; i++) z[i] = H->T[i] * H->zs[i];
}

/* debugging: sizes of level l (n, nF, nC, nnz), then its matrix, F and C lists and F pivots inverses */
void ssr_dbg_info(SSR *H, int l, int *out) { Level *L = &H->lev[l]; out[0] = L->n; out[1] = L->nF; out[2] = L->nC; out[3] = L->A.ia[L->A.n]; out[4] = L->nb; }
void ssr_level_get(SSR *H, int l, int *ia, int *ja, double *a, int *F, int *Cc, double *dinv) {
    Level *L = &H->lev[l];
    memcpy(ia, L->A.ia, sizeof(int) * (L->n + 1)); memcpy(ja, L->A.ja, sizeof(int) * L->A.ia[L->n]); memcpy(a, L->A.a, sizeof(double) * L->A.ia[L->n]);
    memcpy(F, L->F, sizeof(int) * L->nF); memcpy(Cc, L->C, sizeof(int) * L->nC);
    if (!L->nb) memcpy(dinv, L->dinv, sizeof(double) * L->nF);
}
void ssr_scaling(SSR *H, double *T, double *Y) { memcpy(T, H->T, sizeof(double) * H->N); memcpy(Y, H->Y, sizeof(double) * H->N); }

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
        free(L->bptr); free(L->boff); free(L->binv); free(L->tF); free(L->wF); free(L->dpos);
    }
    free(H->lev); free(H->lu); free(H->piv); free(H->T); free(H->Y); free(H->rs); free(H->zs); free(H);
}
