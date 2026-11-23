/* FluxMG cycle in C: same algorithm as flux_mg.py, which still does the setup.
   Build: clang -O3 -march=native -shared -fPIC kernel.c -o libfluxmg.dylib
   Do not use -ffast-math: it may reorder the flux differences. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int F, nz, ny, nx, n;              /* n = F nz ny nx */
    int rz, ry, rx;                    /* coarsening ratio per axis */
    double *kz, *ky, *kx;              /* (F, nz-1, ny, nx), (F, nz, ny-1, nx), (F, nz, ny, nx-1) */
    double *C, *Dinv;                  /* (nz, ny, nx, F, F) */
    double *scale;                     /* (F, nz, ny, nx) */
    int nlines;
    int *line_axis, *line_field;
    unsigned char **line_mask;         /* per line entry: 2 masks, size of the perpendicular plane */
    unsigned char **line_seg;          /* per line entry: cells to solve, size of the grid */
    double *w1, *w2, *w3, *w4, *w5, *w6, *w7, *w8, *w9, *w10; /* work vectors */
    int rK, rnkeep; long *rkeep, *rlab; double *rwt, *rea, *red, *rcoef; /* region vectors of this level */
} Level;

typedef struct {
    int nlev, cycle, nu_pre, nu_post;  /* cycle: 1 V, 2 W, 3 K */
    double k_thresh;
    Level *lev;
    int bn; double *ba, *bd;           /* coarsest level GTH factors */
    int K; int nkeep; long *keep; long *lab; double *ea, *ed; double *coef; double *wt; /* deflation */
    double *q, *r1, *z, *r2;
    long setup_ops;
} Hier;

#define IDX(l, z, y, x) ((((long)(z)) * (l)->ny + (y)) * (l)->nx + (x))

/* out = A x in flux form */
void flux_apply(const Level *l, const double *x, double *out) {
    const int F = l->F, nz = l->nz, ny = l->ny, nx = l->nx;
    const long N = (long)nz * ny * nx, nxy = (long)ny * nx;
    if (F == 1) {
        for (long i = 0; i < N; i++) out[i] = l->C[i] * x[i];
    } else {
        for (long i = 0; i < N; i++) {
            const double *Ci = l->C + i * F * F;
            for (int f = 0; f < F; f++) {
                double s = 0.0;
                for (int g = 0; g < F; g++) s += Ci[f * F + g] * x[g * N + i];
                out[f * N + i] = s;
            }
        }
    }
    for (int f = 0; f < F; f++) {
        const double *xf = x + f * N;
        double *of = out + f * N;
        const double *kz = l->kz + f * (long)(nz - 1) * nxy;
        const double *ky = l->ky + f * (long)nz * (ny - 1) * nx;
        const double *kx = l->kx + f * (long)nz * ny * (nx - 1);
        for (long i = 0; i < (long)(nz - 1) * nxy; i++) {
            double fl = kz[i] * (xf[i] - xf[i + nxy]);
            of[i] += fl; of[i + nxy] -= fl;
        }
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny - 1; y++) {
                long base = IDX(l, z, y, 0), kb = ((long)z * (ny - 1) + y) * nx;
                for (int xx = 0; xx < nx; xx++) {
                    double fl = ky[kb + xx] * (xf[base + xx] - xf[base + xx + nx]);
                    of[base + xx] += fl; of[base + xx + nx] -= fl;
                }
            }
            for (int y = 0; y < ny; y++) {
                long base = IDX(l, z, y, 0), kb = ((long)z * ny + y) * (nx - 1);
                for (int xx = 0; xx < nx - 1; xx++) {
                    double fl = kx[kb + xx] * (xf[base + xx] - xf[base + xx + 1]);
                    of[base + xx] += fl; of[base + xx + 1] -= fl;
                }
            }
        }
    }
}

/* one colour of the block red-black sweep */
static void point_sweep(const Level *l, double *x, const double *rhs, int color) {
    const int F = l->F, nz = l->nz, ny = l->ny, nx = l->nx;
    const long N = (long)nz * ny * nx, nxy = (long)ny * nx;
    const long kzs = (long)(nz - 1) * nxy, kys = (long)nz * (ny - 1) * nx, kxs = (long)nz * ny * (nx - 1);
    double b[64];
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) {
            int start = (z + y + color) & 1;
            long base = IDX(l, z, y, 0);
            long kyb0 = (long)z * (ny - 1), kxb = ((long)z * ny + y) * (nx - 1);
            for (int xx = start; xx < nx; xx += 2) {
                long i = base + xx;
                for (int f = 0; f < F; f++) {
                    const double *xf = x + f * N;
                    const double *kz = l->kz + f * kzs, *ky = l->ky + f * kys, *kx = l->kx + f * kxs;
                    double s = rhs[f * N + i];
                    if (xx > 0) s += kx[kxb + xx - 1] * xf[i - 1];
                    if (xx < nx - 1) s += kx[kxb + xx] * xf[i + 1];
                    if (y > 0) s += ky[(kyb0 + y - 1) * nx + xx] * xf[i - nx];
                    if (y < ny - 1) s += ky[(kyb0 + y) * nx + xx] * xf[i + nx];
                    if (z > 0) s += kz[i - nxy] * xf[i - nxy];
                    if (z < nz - 1) s += kz[i] * xf[i + nxy];
                    b[f] = s;
                }
                if (F == 1) { x[i] = l->Dinv[i] * b[0]; continue; }
                const double *Di = l->Dinv + i * F * F;
                for (int f = 0; f < F; f++) {
                    double s = 0.0;
                    for (int g = 0; g < F; g++) s += Di[f * F + g] * b[g];
                    x[f * N + i] = s;
                }
            }
        }
}

/* exact solve along the marked segments of selected lines of one field; elimination uses additions only */
static void line_sweep(const Level *l, double *x, const double *rhs, int field, int axis, const unsigned char *mask, const unsigned char *seg) {
    const int F = l->F, nz = l->nz, ny = l->ny, nx = l->nx;
    const long N = (long)nz * ny * nx, nxy = (long)ny * nx;
    const long kzs = (long)(nz - 1) * nxy, kys = (long)nz * (ny - 1) * nx, kxs = (long)nz * ny * (nx - 1);
    double *xf = x + field * N;
    const double *kz = l->kz + field * kzs, *ky = l->ky + field * kys, *kx = l->kx + field * kxs;
    const int n1 = axis == 0 ? ny : nz, n2 = axis == 2 ? ny : nx, n = axis == 0 ? nz : axis == 1 ? ny : nx;
    double *b = malloc(sizeof(double) * n * 3), *d = b + n, *u = b + 2 * n;
#define CELL(t, Z, Y, X) int Z = axis == 0 ? (t) : p, Y = axis == 0 ? p : axis == 1 ? (t) : q, X = axis == 2 ? (t) : q
#define ALONG(Z, Y, X, I) (axis == 0 ? kz[I] : axis == 1 ? ky[((long)(Z) * (ny - 1) + (Y)) * nx + (X)] : kx[((long)(Z) * ny + (Y)) * (nx - 1) + (X)])
    for (int p = 0; p < n1; p++)
        for (int q = 0; q < n2; q++) {
            if (!mask[(long)p * n2 + q]) continue;
            int t = 0;
            while (t < n) {
                { CELL(t, z, y, xx); if (!seg[IDX(l, z, y, xx)]) { t++; continue; } }
                int t0 = t;
                for (;;) {
                    if (t + 1 >= n) break;
                    CELL(t + 1, z, y, xx);
                    if (!seg[IDX(l, z, y, xx)]) break;
                    t++;
                }
                int t1 = t;
                t++;
                double qv = 0.0;
                for (int s0 = t0; s0 <= t1; s0++) {
                    CELL(s0, z, y, xx);
                    long i = IDX(l, z, y, xx);
                    double s = l->C[i * F * F + field * F + field], rr = rhs[field * N + i];
                    for (int g = 0; g < F; g++)
                        if (g != field) rr -= l->C[i * F * F + field * F + g] * x[g * N + i];
                    if (axis != 0) {
                        if (z > 0) { double k = kz[i - nxy]; rr += k * xf[i - nxy]; s += k; }
                        if (z < nz - 1) { double k = kz[i]; rr += k * xf[i + nxy]; s += k; }
                    }
                    if (axis != 1) {
                        long kb = (long)z * (ny - 1);
                        if (y > 0) { double k = ky[(kb + y - 1) * nx + xx]; rr += k * xf[i - nx]; s += k; }
                        if (y < ny - 1) { double k = ky[(kb + y) * nx + xx]; rr += k * xf[i + nx]; s += k; }
                    }
                    if (axis != 2) {
                        long kb = ((long)z * ny + y) * (nx - 1);
                        if (xx > 0) { double k = kx[kb + xx - 1]; rr += k * xf[i - 1]; s += k; }
                        if (xx < nx - 1) { double k = kx[kb + xx]; rr += k * xf[i + 1]; s += k; }
                    }
                    if (s0 == t0 && t0 > 0) {                /* neighbours beyond the segment stay fixed */
                        CELL(t0 - 1, zb, yb, xb);
                        long ib = IDX(l, zb, yb, xb);
                        double k = ALONG(zb, yb, xb, ib);
                        rr += k * xf[ib]; s += k;
                    }
                    if (s0 == t1 && t1 < n - 1) {
                        CELL(t1 + 1, za, ya, xa);
                        long ia = IDX(l, za, ya, xa);
                        double k = ALONG(z, y, xx, i);
                        rr += k * xf[ia]; s += k;
                    }
                    int m = s0 - t0;
                    u[m] = s0 == t1 ? 0.0 : ALONG(z, y, xx, i);
                    if (m == 0) { qv = s; b[0] = rr; }
                    else {
                        double lw = u[m - 1];
                        b[m] = rr + lw * b[m - 1] / d[m - 1];
                        qv = s + lw * (qv / d[m - 1]);
                    }
                    d[m] = qv + u[m];
                    if (d[m] < 1e-300) d[m] = 1e-300;
                }
                double nxt = 0.0;
                for (int s0 = t1; s0 >= t0; s0--) {
                    CELL(s0, z, y, xx);
                    int m = s0 - t0;
                    nxt = (b[m] + u[m] * nxt) / d[m];
                    xf[IDX(l, z, y, xx)] = nxt;
                }
            }
        }
#undef CELL
#undef ALONG
    free(b);
}

static void smooth(const Level *l, double *x, const double *rhs, int forward) {
    point_sweep(l, x, rhs, forward ? 0 : 1);
    point_sweep(l, x, rhs, forward ? 1 : 0);
    for (int e = 0; e < l->nlines; e++) {
        int j = forward ? e : l->nlines - 1 - e;
        line_sweep(l, x, rhs, l->line_field[j], l->line_axis[j], l->line_mask[2 * j + (forward ? 0 : 1)], l->line_seg[j]);
        line_sweep(l, x, rhs, l->line_field[j], l->line_axis[j], l->line_mask[2 * j + (forward ? 1 : 0)], l->line_seg[j]);
    }
}

static void restrict_to(const Level *l, const Level *c, const double *r, double *out) {
    const int F = l->F;
    const long N = (long)l->nz * l->ny * l->nx, Nc = (long)c->nz * c->ny * c->nx;
    memset(out, 0, sizeof(double) * F * Nc);
    for (int f = 0; f < F; f++) {
        const double *rf = r + f * N;
        double *of = out + f * Nc;
        long i = 0;
        for (int z = 0; z < l->nz; z++) {
            int cz = z / l->rz;
            for (int y = 0; y < l->ny; y++) {
                long base = ((long)cz * c->ny + y / l->ry) * c->nx;
                if (l->rx == 2) for (int x = 0; x < l->nx; x++) of[base + x / 2] += rf[i++];
                else for (int x = 0; x < l->nx; x++) of[base + x] += rf[i++];
            }
        }
    }
}

static void prolong_add(const Level *l, const Level *c, const double *e, double *x) {
    const int F = l->F;
    const long N = (long)l->nz * l->ny * l->nx, Nc = (long)c->nz * c->ny * c->nx;
    for (int f = 0; f < F; f++) {
        const double *ef = e + f * Nc;
        double *xf = x + f * N;
        long i = 0;
        for (int z = 0; z < l->nz; z++) {
            int cz = z / l->rz;
            for (int y = 0; y < l->ny; y++) {
                long base = ((long)cz * c->ny + y / l->ry) * c->nx;
                if (l->rx == 2) for (int xx = 0; xx < l->nx; xx++) xf[i++] += ef[base + xx / 2];
                else for (int xx = 0; xx < l->nx; xx++) xf[i++] += ef[base + xx];
            }
        }
    }
}

static void gth_solve(int n, const double *a, const double *d, double *x) {
    for (int k = 0; k < n; k++) {
        double xk = x[k];
        const double *col = a + k;
        for (int i = k + 1; i < n; i++) x[i] += col[(long)i * n] * xk;
    }
    for (int k = n - 1; k >= 0; k--) {
        if (d[k] <= 0) { x[k] = 0.0; continue; }
        double s = x[k];
        const double *row = a + (long)k * n;
        for (int j = k + 1; j < n; j++) s += row[j] * x[j];
        x[k] = s / d[k];
    }
}

static double sip(const Level *l, const double *a, const double *b) {
    double s = 0.0;
    for (long i = 0; i < l->n; i++) s += l->scale[i] * a[i] * b[i];
    return s;
}

static void coarse_solve(Hier *h, int lv, const double *r, double *x);

static void level_Q(const Level *l, const double *r, double *z) {
    double *coef = l->rcoef;
    memset(coef, 0, sizeof(double) * l->rK);
    for (int t = 0; t < l->rnkeep; t++) coef[l->rlab[t]] += l->rwt[t] * r[l->rkeep[t]];
    gth_solve(l->rK, l->rea, l->red, coef);
    memset(z, 0, sizeof(double) * l->n);
    for (int t = 0; t < l->rnkeep; t++) z[l->rkeep[t]] += l->rwt[t] * coef[l->rlab[t]];
}

/* x = one cycle on level lv applied to f (x starts from zero) */
static void cycle(Hier *h, int lv, const double *f, double *x) {
    Level *l = &h->lev[lv];
    if (lv == h->nlev - 1) {
        /* the coarsest ordering is (z y x f): flat cell index times F plus field */
        long N = l->n / l->F;
        double *b = l->w1;
        for (long i = 0; i < N; i++) for (int g = 0; g < l->F; g++) b[i * l->F + g] = f[g * N + i];
        gth_solve(h->bn, h->ba, h->bd, b);
        for (long i = 0; i < N; i++) for (int g = 0; g < l->F; g++) x[g * N + i] = b[i * l->F + g];
        return;
    }
    Level *c = &h->lev[lv + 1];
    memset(x, 0, sizeof(double) * l->n);
    for (int s = 0; s < h->nu_pre; s++) smooth(l, x, f, 1);
    flux_apply(l, x, l->w1);
    for (long i = 0; i < l->n; i++) l->w1[i] = f[i] - l->w1[i];
    restrict_to(l, c, l->w1, c->w2);
    coarse_solve(h, lv + 1, c->w2, c->w3);
    prolong_add(l, c, c->w3, x);
    for (int s = 0; s < h->nu_post; s++) smooth(l, x, f, 0);
}

static void coarse_inner(Hier *h, int lv, const double *r, double *x);

static void coarse_solve(Hier *h, int lv, const double *r, double *x) {
    const Level *l = &h->lev[lv];
    coarse_inner(h, lv, r, x);
    if (l->rK) {
        flux_apply(l, x, l->w9);
        for (long i = 0; i < l->n; i++) l->w9[i] = r[i] - l->w9[i];
        level_Q(l, l->w9, l->w10);
        for (long i = 0; i < l->n; i++) x[i] += l->w10[i];
    }
}

static void coarse_inner(Hier *h, int lv, const double *r, double *x) {
    Level *l = &h->lev[lv];
    const long n = l->n;
    if (lv == h->nlev - 1 || h->cycle == 1) { cycle(h, lv, r, x); return; }
    double *c1 = l->w4, *v1 = l->w5, *r1 = l->w6, *c2 = l->w7, *v2 = l->w8;
    if (h->cycle == 2) {
        cycle(h, lv, r, x);
        flux_apply(l, x, v1);
        for (long i = 0; i < n; i++) r1[i] = r[i] - v1[i];
        cycle(h, lv, r1, c2);
        for (long i = 0; i < n; i++) x[i] += c2[i];
        return;
    }
    cycle(h, lv, r, c1);
    flux_apply(l, c1, v1);
    double v11 = sip(l, v1, v1);
    if (!(v11 > 0)) { memcpy(x, c1, sizeof(double) * n); return; }
    double a1 = sip(l, v1, r) / v11;
    for (long i = 0; i < n; i++) { x[i] = a1 * c1[i]; r1[i] = r[i] - a1 * v1[i]; }
    if (sip(l, r1, r1) <= h->k_thresh * h->k_thresh * sip(l, r, r)) return;
    cycle(h, lv, r1, c2);
    flux_apply(l, c2, v2);
    double b = sip(l, v2, v1) / v11;
    for (long i = 0; i < n; i++) { v2[i] -= b * v1[i]; c2[i] -= b * c1[i]; }
    double v22 = sip(l, v2, v2);
    if (v22 > 0) {
        double a2 = sip(l, v2, r1) / v22;
        for (long i = 0; i < n; i++) x[i] += a2 * c2[i];
    }
}

static void Q_apply(Hier *h, const double *r, double *z) {
    double *coef = h->coef;
    memset(coef, 0, sizeof(double) * h->K);
    for (int t = 0; t < h->nkeep; t++) coef[h->lab[t]] += h->wt[t] * r[h->keep[t]];
    gth_solve(h->K, h->ea, h->ed, coef);
    memset(z, 0, sizeof(double) * h->lev[0].n);
    for (int t = 0; t < h->nkeep; t++) z[h->keep[t]] += h->wt[t] * coef[h->lab[t]];
}

/* preconditioner: balanced deflation around one cycle */
void flux_precond(Hier *h, const double *r, double *out) {
    Level *l = &h->lev[0];
    const long n = l->n;
    if (!h->K) { cycle(h, 0, r, out); return; }
    Q_apply(h, r, h->q);
    flux_apply(l, h->q, h->r1);
    for (long i = 0; i < n; i++) h->r1[i] = r[i] - h->r1[i];
    cycle(h, 0, h->r1, h->z);
    flux_apply(l, h->z, h->r2);
    for (long i = 0; i < n; i++) h->r2[i] = h->r1[i] - h->r2[i];
    Q_apply(h, h->r2, out);
    for (long i = 0; i < n; i++) out[i] += h->q[i] + h->z[i];
}

void flux_op(Hier *h, const double *x, double *y) { flux_apply(&h->lev[0], x, y); }

int sizeof_level(void) { return (int)sizeof(Level); }
int sizeof_hier(void) { return (int)sizeof(Hier); }

/* One pass over a CSR matrix, field-blocked, each block 7-point: faces, local couplings and the row excess of each diagonal
   block, summed in double-double so that 1e17 entries cancelling to 1e2 keep their digits. Returns 0, or the row of a bad entry + 1. */
long flux_parse(int F, int nz, int ny, int nx, const int *indptr, const int *indices, const double *data,
                double *kz, double *ky, double *kx, double *C) {
    const long N = (long)nz * ny * nx, nxy = (long)ny * nx, total = F * N;
    const long kzs = (long)(nz - 1) * nxy, kys = (long)nz * (ny - 1) * nx, kxs = (long)nz * ny * (nx - 1);
    memset(kz, 0, sizeof(double) * F * kzs);
    memset(ky, 0, sizeof(double) * F * kys);
    memset(kx, 0, sizeof(double) * F * kxs);
    memset(C, 0, sizeof(double) * N * F * F);
    for (long row = 0; row < total; row++) {
        const int f = (int)(row / N);
        const long i = row - f * N;
        const int x = (int)(i % nx), y = (int)((i / nx) % ny), z = (int)(i / nxy);
        double s = 0.0, e = 0.0;
        for (int p = indptr[row]; p < indptr[row + 1]; p++) {
            const long col = indices[p];
            const int g = (int)(col / N);
            const long j = col - g * N;
            const double v = data[p];
            if (g != f) {
                if (j != i) return row + 1;
                C[i * F * F + f * F + g] = v;
                continue;
            }
            double t = s + v, bb = t - s;                       /* two_sum */
            e += (s - (t - bb)) + (v - bb);
            s = t;
            const long off = j - i;
            if (off <= 0) continue;
            if (nx > 1 && off == 1) {
                if (x == nx - 1) return row + 1;
                kx[f * kxs + ((long)z * ny + y) * (nx - 1) + x] = -v;
            } else if (ny > 1 && off == nx) {
                if (y == ny - 1) return row + 1;
                ky[f * kys + ((long)z * (ny - 1) + y) * nx + x] = -v;
            } else if (nz > 1 && off == nxy) {
                kz[f * kzs + i] = -v;
            } else {
                return row + 1;
            }
        }
        C[i * F * F + f * F + f] = s + e;
    }
    return 0;
}
