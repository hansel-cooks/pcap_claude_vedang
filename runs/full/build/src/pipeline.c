#include "pipeline.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#endif

double wtime(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#elif defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

void params_default(Params *p)
{
    p->tiles = 8;
    p->clip = 3.0f;
    p->beta = 0.5f;
    p->c = 0.02f;
    p->thr = 0.1f;
    p->nsig = 4;
    p->sig[0] = 1.f;
    p->sig[1] = 2.f;
    p->sig[2] = 3.f;
    p->sig[3] = 4.f;
    p->nt = 1;
    p->sched = 0;
    p->clahe_atomic = 0;
    p->scale_par = 0;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ------------------------------------------------------- Stage 1: green --- */

void stage_green(const unsigned char *rgb, unsigned char *g, size_t n, int nt)
{
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long long i = 0; i < (long long)n; i++)
        g[i] = rgb[3 * i + 1];
}

/* ------------------------------------------------------- Stage 2: CLAHE --- */

static void clahe_lut(const unsigned int *hist_in, int npix, float clip, unsigned char *lut)
{
    unsigned int hist[256];
    memcpy(hist, hist_in, sizeof(hist));
    unsigned int limit = (unsigned int)(clip * (float)npix / 256.0f);
    if (limit < 1)
        limit = 1;
    unsigned int excess = 0;
    for (int i = 0; i < 256; i++)
        if (hist[i] > limit) {
            excess += hist[i] - limit;
            hist[i] = limit;
        }
    unsigned int add = excess / 256, rem = excess % 256;
    for (int i = 0; i < 256; i++)
        hist[i] += add;
    if (rem) {
        unsigned int step = 256 / rem;
        if (step < 1)
            step = 1;
        for (unsigned int i = 0, n = 0; i < 256 && n < rem; i += step, n++)
            hist[i]++;
    }
    unsigned int cum = 0;
    for (int i = 0; i < 256; i++) {
        cum += hist[i];
        float v = (float)cum * 255.0f / (float)npix;
        lut[i] = (unsigned char)(v > 255.f ? 255 : (int)(v + 0.5f));
    }
}

void stage_clahe(const unsigned char *g, unsigned char *eq, int w, int h, const Params *p)
{
    const int T = p->tiles, nt = p->nt;
    const int tw = (w + T - 1) / T, th = (h + T - 1) / T;
    unsigned char *lut = (unsigned char *)malloc((size_t)T * T * 256);
    unsigned int *hist = (unsigned int *)calloc((size_t)T * T * 256, sizeof(unsigned int));

    if (p->clahe_atomic) {
        /* Ablation: pixel-parallel histogram, atomic increments. */
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int y = 0; y < h; y++) {
            int ty = y / th;
            for (int x = 0; x < w; x++) {
                unsigned int *hp = &hist[((size_t)ty * T + x / tw) * 256 + g[(size_t)y * w + x]];
#pragma omp atomic
                (*hp)++;
            }
        }
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int t = 0; t < T * T; t++) {
            int tx = t % T, ty = t / T;
            int x1 = (tx + 1) * tw > w ? w : (tx + 1) * tw, y1 = (ty + 1) * th > h ? h : (ty + 1) * th;
            int npix = (x1 - tx * tw) * (y1 - ty * th);
            clahe_lut(&hist[(size_t)t * 256], npix, p->clip, &lut[(size_t)t * 256]);
        }
    } else {
        /* One thread owns a tile: private histogram, no contention. */
        if (p->sched == 0) {
#pragma omp parallel for num_threads(nt) schedule(static)
            for (int t = 0; t < T * T; t++) {
                int tx = t % T, ty = t / T;
                int x0 = tx * tw, y0 = ty * th;
                int x1 = x0 + tw > w ? w : x0 + tw, y1 = y0 + th > h ? h : y0 + th;
                unsigned int *hp = &hist[(size_t)t * 256];
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        hp[g[(size_t)y * w + x]]++;
                clahe_lut(hp, (x1 - x0) * (y1 - y0), p->clip, &lut[(size_t)t * 256]);
            }
        } else {
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
            for (int t = 0; t < T * T; t++) {
                int tx = t % T, ty = t / T;
                int x0 = tx * tw, y0 = ty * th;
                int x1 = x0 + tw > w ? w : x0 + tw, y1 = y0 + th > h ? h : y0 + th;
                unsigned int *hp = &hist[(size_t)t * 256];
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        hp[g[(size_t)y * w + x]]++;
                clahe_lut(hp, (x1 - x0) * (y1 - y0), p->clip, &lut[(size_t)t * 256]);
            }
        }
    }

    /* Bilinear blend of the four nearest tile lookup tables. */
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int y = 0; y < h; y++) {
        float fy = ((float)y + 0.5f) / (float)th - 0.5f;
        int ty0 = (int)floorf(fy);
        float ay = fy - (float)ty0;
        int ty1 = clampi(ty0 + 1, 0, T - 1);
        ty0 = clampi(ty0, 0, T - 1);
        for (int x = 0; x < w; x++) {
            float fx = ((float)x + 0.5f) / (float)tw - 0.5f;
            int tx0 = (int)floorf(fx);
            float ax = fx - (float)tx0;
            int tx1 = clampi(tx0 + 1, 0, T - 1);
            tx0 = clampi(tx0, 0, T - 1);
            int v = g[(size_t)y * w + x];
            float a = lut[((size_t)ty0 * T + tx0) * 256 + v];
            float b = lut[((size_t)ty0 * T + tx1) * 256 + v];
            float c = lut[((size_t)ty1 * T + tx0) * 256 + v];
            float d = lut[((size_t)ty1 * T + tx1) * 256 + v];
            float top = a + (b - a) * ax, bot = c + (d - c) * ax;
            eq[(size_t)y * w + x] = (unsigned char)(top + (bot - top) * ay + 0.5f);
        }
    }
    free(lut);
    free(hist);
}

/* ------------------------------------------------- Stage 3: Frangi filter --- */

static void make_kernels(float sigma, int r, float *k0, float *k1, float *k2)
{
    double s2 = (double)sigma * sigma, sum0 = 0, sum1 = 0, sum2 = 0, mean2 = 0;
    for (int i = -r; i <= r; i++) {
        double x = i, g = exp(-x * x / (2 * s2));
        k0[i + r] = (float)g;
        k1[i + r] = (float)(-x / s2 * g);
        k2[i + r] = (float)((x * x / (s2 * s2) - 1.0 / s2) * g);
        sum0 += g;
        mean2 += k2[i + r];
    }
    mean2 /= (2 * r + 1);
    for (int i = -r; i <= r; i++) {
        double x = i;
        k0[i + r] = (float)(k0[i + r] / sum0);
        k2[i + r] = (float)(k2[i + r] - mean2);
        sum1 += x * k1[i + r];
        sum2 += 0.5 * x * x * k2[i + r];
    }
    /* sum1 = -1 for an ideal first derivative kernel; sum2 = 1 for the second. */
    for (int i = 0; i < 2 * r + 1; i++) {
        k1[i] = (float)(k1[i] / -sum1);
        k2[i] = (float)(k2[i] / sum2);
    }
}

int work_alloc(Work *W, int w, int h, const Params *p)
{
    memset(W, 0, sizeof(*W));
    size_t n = (size_t)w * h;
    W->w = w;
    W->h = h;
    W->g = (unsigned char *)malloc(n);
    W->eq = (unsigned char *)malloc(n);
    W->mask = (unsigned char *)malloc(n);
    W->tmp = (unsigned char *)malloc(n);
    W->V = (float *)malloc(n * sizeof(float));
    W->fin = (float *)malloc(n * sizeof(float));
    W->nslots = p->scale_par ? p->nsig : 1;
    for (int s = 0; s < W->nslots; s++)
        for (int b = 0; b < 7; b++) {
            if (b == 6 && !p->scale_par)
                continue;
            W->buf[s][b] = (float *)malloc(n * sizeof(float));
            if (!W->buf[s][b])
                return -1;
        }
    for (int s = 0; s < p->nsig; s++) {
        int r = (int)ceilf(3.0f * p->sig[s]);
        W->r[s] = r;
        for (int j = 0; j < 3; j++)
            W->k[s][j] = (float *)malloc((2 * r + 1) * sizeof(float));
        make_kernels(p->sig[s], r, W->k[s][0], W->k[s][1], W->k[s][2]);
    }
    return (W->g && W->eq && W->mask && W->tmp && W->V && W->fin) ? 0 : -1;
}

void work_free(Work *W)
{
    free(W->g);
    free(W->eq);
    free(W->mask);
    free(W->tmp);
    free(W->V);
    free(W->fin);
    for (int s = 0; s < MAX_SCALES; s++) {
        for (int b = 0; b < 7; b++)
            free(W->buf[s][b]);
        for (int j = 0; j < 3; j++)
            free(W->k[s][j]);
    }
    memset(W, 0, sizeof(*W));
}

/* Three row passes (g'' , g , g') in one sweep over each padded row. */
static void conv_rows(const float *src, float *A, float *B, float *C, int w, int h, const float *k0,
                      const float *k1, const float *k2, int r, int nt)
{
#pragma omp parallel num_threads(nt)
    {
        float *pad = (float *)malloc((size_t)(w + 2 * r) * sizeof(float));
#pragma omp for schedule(static)
        for (int y = 0; y < h; y++) {
            const float *row = src + (size_t)y * w;
            for (int i = 0; i < r; i++) {
                pad[i] = row[0];
                pad[r + w + i] = row[w - 1];
            }
            memcpy(pad + r, row, (size_t)w * sizeof(float));
            float *a = A + (size_t)y * w, *b = B + (size_t)y * w, *c = C + (size_t)y * w;
            for (int x = 0; x < w; x++) {
                float sa = 0.f, sb = 0.f, sc = 0.f;
                const float *q = pad + x + r;
                for (int j = -r; j <= r; j++) {
                    float f = q[-j];
                    sa += k2[j + r] * f;
                    sb += k0[j + r] * f;
                    sc += k1[j + r] * f;
                }
                a[x] = sa;
                b[x] = sb;
                c[x] = sc;
            }
        }
        free(pad);
    }
}

/* Column passes: Ixx = g0*A, Iyy = g2*B, Ixy = g1*C (all along y). */
static void conv_cols(const float *A, const float *B, const float *C, float *Ixx, float *Iyy, float *Ixy,
                      int w, int h, const float *k0, const float *k1, const float *k2, int r, int nt)
{
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int y = 0; y < h; y++) {
        float *xx = Ixx + (size_t)y * w, *yy = Iyy + (size_t)y * w, *xy = Ixy + (size_t)y * w;
        for (int x = 0; x < w; x++)
            xx[x] = yy[x] = xy[x] = 0.f;
        for (int j = -r; j <= r; j++) {
            int sy = clampi(y - j, 0, h - 1);
            const float *a = A + (size_t)sy * w, *b = B + (size_t)sy * w, *c = C + (size_t)sy * w;
            float w0 = k0[j + r], w1 = k1[j + r], w2 = k2[j + r];
            for (int x = 0; x < w; x++) {
                xx[x] += w0 * a[x];
                yy[x] += w2 * b[x];
                xy[x] += w1 * c[x];
            }
        }
    }
}

static void vesselness(const float *Ixx, const float *Iyy, const float *Ixy, float *V, size_t n, float sigma,
                       float beta, float c, int accumulate, int nt)
{
    const float s2 = sigma * sigma, inv_b = 1.0f / (2.0f * beta * beta), inv_c = 1.0f / (2.0f * c * c);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        float a = s2 * Ixx[i], b = s2 * Ixy[i], d = s2 * Iyy[i];
        float mean = 0.5f * (a + d), rad = sqrtf(0.25f * (a - d) * (a - d) + b * b);
        float e1 = mean - rad, e2 = mean + rad;
        float l1, l2; /* |l1| <= |l2| */
        if (fabsf(e1) <= fabsf(e2)) {
            l1 = e1;
            l2 = e2;
        } else {
            l1 = e2;
            l2 = e1;
        }
        float v = 0.f;
        if (l2 > 0.f) {
            float rb = fabsf(l1) / l2, S2 = l1 * l1 + l2 * l2;
            v = expf(-rb * rb * inv_b) * (1.0f - expf(-S2 * inv_c));
        }
        V[i] = (accumulate && V[i] > v) ? V[i] : v;
    }
}

void stage_frangi(const unsigned char *eq, Work *W, const Params *p, StageT *t)
{
    const int w = W->w, h = W->h;
    const size_t n = (size_t)w * h;
    const int nt = p->nt;
    float *fin = W->fin;

    double t0 = wtime();
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long long i = 0; i < (long long)n; i++)
        fin[i] = (float)eq[i] * (1.0f / 255.0f);
    double tg = wtime() - t0, tv = 0.0;

    if (!p->scale_par) {
        for (int s = 0; s < p->nsig; s++) {
            float *A = W->buf[0][0], *B = W->buf[0][1], *C = W->buf[0][2];
            float *Ixx = W->buf[0][3], *Iyy = W->buf[0][4], *Ixy = W->buf[0][5];
            t0 = wtime();
            conv_rows(fin, A, B, C, w, h, W->k[s][0], W->k[s][1], W->k[s][2], W->r[s], nt);
            conv_cols(A, B, C, Ixx, Iyy, Ixy, w, h, W->k[s][0], W->k[s][1], W->k[s][2], W->r[s], nt);
            double t1 = wtime();
            vesselness(Ixx, Iyy, Ixy, W->V, n, p->sig[s], p->beta, p->c, s > 0, nt);
            double t2 = wtime();
            tg += t1 - t0;
            tv += t2 - t1;
        }
    } else {
        /* scales run concurrently; each uses its own buffers and one thread */
        t0 = wtime();
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1)
        for (int s = 0; s < p->nsig; s++) {
            float *A = W->buf[s][0], *B = W->buf[s][1], *C = W->buf[s][2];
            float *Ixx = W->buf[s][3], *Iyy = W->buf[s][4], *Ixy = W->buf[s][5];
            conv_rows(fin, A, B, C, w, h, W->k[s][0], W->k[s][1], W->k[s][2], W->r[s], 1);
            conv_cols(A, B, C, Ixx, Iyy, Ixy, w, h, W->k[s][0], W->k[s][1], W->k[s][2], W->r[s], 1);
            vesselness(Ixx, Iyy, Ixy, W->buf[s][6], n, p->sig[s], p->beta, p->c, 0, 1);
        }
#pragma omp parallel for num_threads(nt) schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            float m = W->buf[0][6][i];
            for (int s = 1; s < p->nsig; s++)
                m = W->buf[s][6][i] > m ? W->buf[s][6][i] : m;
            W->V[i] = m;
        }
        tg += wtime() - t0;
        tv = 0.0;
    }
    if (t) {
        t->gauss = tg;
        t->vess = tv;
    }
}

/* ----------------------------------------- Stage 4: threshold and morphology --- */

void stage_threshold(const float *V, unsigned char *mask, const unsigned char *fov_er, size_t n, float thr, int nt)
{
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long long i = 0; i < (long long)n; i++)
        mask[i] = (unsigned char)((V[i] > thr) & fov_er[i]);
}

/* 3x3 binary erosion (AND) / dilation (OR) on 0/1 masks. Clamping the window at the
   border only repeats pixels already inside it, so it equals ignoring outside pixels. */
#define MORPH3_BODY(OP)                                                                            \
    for (int y = 0; y < h; y++) {                                                                  \
        const unsigned char *a = src + (size_t)(y > 0 ? y - 1 : 0) * w;                            \
        const unsigned char *b = src + (size_t)y * w;                                              \
        const unsigned char *c = src + (size_t)(y < h - 1 ? y + 1 : h - 1) * w;                    \
        unsigned char *d = dst + (size_t)y * w;                                                    \
        unsigned char col0 = a[0] OP b[0] OP c[0];                                                 \
        unsigned char col1 = w > 1 ? (unsigned char)(a[1] OP b[1] OP c[1]) : col0;                 \
        d[0] = col0 OP col1;                                                                       \
        for (int x = 1; x < w - 1; x++)                                                            \
            d[x] = (a[x - 1] OP b[x - 1] OP c[x - 1]) OP (a[x] OP b[x] OP c[x]) OP                 \
                   (a[x + 1] OP b[x + 1] OP c[x + 1]);                                             \
        if (w > 1)                                                                                 \
            d[w - 1] = (a[w - 2] OP b[w - 2] OP c[w - 2]) OP (a[w - 1] OP b[w - 1] OP c[w - 1]);   \
    }

static void erode3(const unsigned char *src, unsigned char *dst, int w, int h, int nt)
{
#pragma omp parallel for num_threads(nt) schedule(static)
    MORPH3_BODY(&)
}

static void dilate3(const unsigned char *src, unsigned char *dst, int w, int h, int nt)
{
#pragma omp parallel for num_threads(nt) schedule(static)
    MORPH3_BODY(|)
}

static void morph3(const unsigned char *src, unsigned char *dst, int w, int h, int erode, int nt)
{
    if (erode)
        erode3(src, dst, w, h, nt);
    else
        dilate3(src, dst, w, h, nt);
}

/* opening (erode, dilate) then closing (dilate, erode); result back in mask */
void stage_morph(unsigned char *mask, unsigned char *tmp, int w, int h, int nt)
{
    morph3(mask, tmp, w, h, 1, nt);
    morph3(tmp, mask, w, h, 0, nt);
    morph3(mask, tmp, w, h, 0, nt);
    morph3(tmp, mask, w, h, 1, nt);
}

void run_pipeline(const unsigned char *rgb, const unsigned char *fov_er, Work *W, const Params *p, StageT *t)
{
    const int w = W->w, h = W->h;
    const size_t n = (size_t)w * h;
    StageT st;
    memset(&st, 0, sizeof(st));
    double t0 = wtime(), t1;

    stage_green(rgb, W->g, n, p->nt);
    t1 = wtime();
    st.green = t1 - t0;
    t0 = t1;
    stage_clahe(W->g, W->eq, w, h, p);
    t1 = wtime();
    st.clahe = t1 - t0;
    stage_frangi(W->eq, W, p, &st);
    t0 = wtime();
    stage_threshold(W->V, W->mask, fov_er, n, p->thr, p->nt);
    t1 = wtime();
    st.thresh = t1 - t0;
    stage_morph(W->mask, W->tmp, w, h, p->nt);
    st.morph = wtime() - t1;
    st.total = st.green + st.clahe + st.gauss + st.vess + st.thresh + st.morph;
    if (t)
        *t = st;
}

/* ---------------------------------------------------- untimed helpers --- */

void fov_erode(const unsigned char *fov, unsigned char *out, int w, int h, int radius)
{
    unsigned char *tmp = (unsigned char *)malloc((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char v = fov[(size_t)y * w + x] ? 1 : 0;
            for (int dx = -radius; dx <= radius && v; dx++) {
                int xx = x + dx;
                if (xx >= 0 && xx < w && !fov[(size_t)y * w + xx])
                    v = 0;
            }
            tmp[(size_t)y * w + x] = v;
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char v = tmp[(size_t)y * w + x];
            for (int dy = -radius; dy <= radius && v; dy++) {
                int yy = y + dy;
                if (yy >= 0 && yy < h && !tmp[(size_t)yy * w + x])
                    v = 0;
            }
            out[(size_t)y * w + x] = v;
        }
    free(tmp);
}

unsigned char *resample_rgb(const unsigned char *src, int w, int h, int nw, int nh)
{
    unsigned char *dst = (unsigned char *)malloc((size_t)nw * nh * 3);
    float sx = (float)w / (float)nw, sy = (float)h / (float)nh;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < nh; y++) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float ay = fy - (float)y0;
        int y1 = clampi(y0 + 1, 0, h - 1);
        y0 = clampi(y0, 0, h - 1);
        for (int x = 0; x < nw; x++) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float ax = fx - (float)x0;
            int x1 = clampi(x0 + 1, 0, w - 1);
            x0 = clampi(x0, 0, w - 1);
            for (int c = 0; c < 3; c++) {
                float a = src[((size_t)y0 * w + x0) * 3 + c], b = src[((size_t)y0 * w + x1) * 3 + c];
                float cc = src[((size_t)y1 * w + x0) * 3 + c], d = src[((size_t)y1 * w + x1) * 3 + c];
                float top = a + (b - a) * ax, bot = cc + (d - cc) * ax;
                dst[((size_t)y * nw + x) * 3 + c] = (unsigned char)(top + (bot - top) * ay + 0.5f);
            }
        }
    }
    return dst;
}

unsigned char *resample_nearest(const unsigned char *src, int w, int h, int nw, int nh)
{
    unsigned char *dst = (unsigned char *)malloc((size_t)nw * nh);
    for (int y = 0; y < nh; y++) {
        int sy = clampi((int)(((float)y + 0.5f) * (float)h / (float)nh), 0, h - 1);
        for (int x = 0; x < nw; x++) {
            int sx = clampi((int)(((float)x + 0.5f) * (float)w / (float)nw), 0, w - 1);
            dst[(size_t)y * nw + x] = src[(size_t)sy * w + sx];
        }
    }
    return dst;
}

Conf confusion(const unsigned char *pred, const unsigned char *gt, const unsigned char *fov, size_t n, int nt)
{
    long long tp = 0, fp = 0, tn = 0, fn = 0;
#pragma omp parallel for num_threads(nt) reduction(+ : tp, fp, tn, fn) schedule(static)
    for (long long i = 0; i < (long long)n; i++) {
        if (!fov[i])
            continue;
        int p = pred[i] != 0, g = gt[i] != 0;
        tp += p && g;
        fp += p && !g;
        tn += !p && !g;
        fn += !p && g;
    }
    Conf c = {tp, fp, tn, fn};
    return c;
}

double conf_dice(Conf c)
{
    double d = 2.0 * c.tp + c.fp + c.fn;
    return d > 0 ? 2.0 * c.tp / d : 0.0;
}
double conf_sens(Conf c) { return (c.tp + c.fn) > 0 ? (double)c.tp / (c.tp + c.fn) : 0.0; }
double conf_spec(Conf c) { return (c.tn + c.fp) > 0 ? (double)c.tn / (c.tn + c.fp) : 0.0; }
double conf_acc(Conf c)
{
    long long t = c.tp + c.tn + c.fp + c.fn;
    return t > 0 ? (double)(c.tp + c.tn) / (double)t : 0.0;
}
