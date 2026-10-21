/*
 * frangi: serial / OpenMP multi-scale Frangi pipeline for retinal vessel segmentation.
 *
 *   frangi info
 *   frangi tune    <drive_root> <out_params> [--nt N]
 *   frangi eval    <kind> <root> <params> <out_csv> [--nt N] [--masks DIR]
 *   frangi bench   <kind> <root> <params> <out_csv> [--nt LIST] [--scales LIST] [--images N]
 *                  [--warm W] [--reps R] [--variant base|dynamic|atomic|scalepar] [--append]
 *   frangi dump    <kind> <root> <params> <out_dir> [--nt N] [--images N]
 *   frangi compare <dirA> <dirB>
 *
 * kind: drive-train | drive-test | stare
 */
#include "io.h"
#include "pipeline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#define BUILD "omp"
#else
#define BUILD "serial"
#endif

typedef struct {
    char name[64];
    ImgU8 rgb;
    unsigned char *fov, *gt;
} Sample;

static const char *STARE_IDS[20] = {"0001", "0002", "0003", "0004", "0005", "0044", "0077",
                                    "0081", "0082", "0139", "0162", "0163", "0235", "0236",
                                    "0239", "0240", "0255", "0291", "0319", "0324"};

static int fov_erode_radius = 8;

/* ------------------------------------------------------------ datasets --- */

static unsigned char *binarise(ImgU8 *g)
{
    size_t n = (size_t)g->w * g->h;
    unsigned char *b = (unsigned char *)malloc(n);
    for (size_t i = 0; i < n; i++)
        b[i] = g->d[i] > 127;
    img_free(g);
    return b;
}

static unsigned char *stare_fov(const ImgU8 *rgb)
{
    size_t n = (size_t)rgb->w * rgb->h;
    unsigned char *f = (unsigned char *)malloc(n), *t = (unsigned char *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        const unsigned char *p = rgb->d + 3 * i;
        unsigned char m = p[0] > p[1] ? p[0] : p[1];
        m = m > p[2] ? m : p[2];
        t[i] = m > 35;
    }
    /* close small holes then remove specks: one 3x3 opening is enough for STARE */
    fov_erode(t, f, rgb->w, rgb->h, 1);
    for (int y = 0; y < rgb->h; y++)
        for (int x = 0; x < rgb->w; x++) {
            unsigned char v = 0;
            for (int dy = -1; dy <= 1 && !v; dy++)
                for (int dx = -1; dx <= 1 && !v; dx++) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < rgb->h && xx >= 0 && xx < rgb->w)
                        v = f[(size_t)yy * rgb->w + xx];
                }
            t[(size_t)y * rgb->w + x] = v;
        }
    free(f);
    return t;
}

static int load_dataset(const char *kind, const char *root, int maxn, Sample **out)
{
    int n = 20;
    if (maxn > 0 && maxn < n)
        n = maxn;
    Sample *S = (Sample *)calloc((size_t)n, sizeof(Sample));
    char p1[1024], p2[1024], p3[1024];
    for (int i = 0; i < n; i++) {
        ImgU8 m, g;
        if (!strcmp(kind, "drive-train") || !strcmp(kind, "drive-test")) {
            int train = kind[6] == 't' && kind[7] == 'r';
            int id = train ? 21 + i : 1 + i;
            const char *sub = train ? "training" : "test";
            snprintf(S[i].name, sizeof(S[i].name), "%02d_%s", id, sub);
            snprintf(p1, sizeof p1, "%s/%s/images/%02d_%s.tif", root, sub, id, sub);
            snprintf(p2, sizeof p2, "%s/%s/mask/%02d_%s_mask.gif", root, sub, id, sub);
            snprintf(p3, sizeof p3, "%s/%s/1st_manual/%02d_manual1.gif", root, sub, id);
            if (load_rgb(p1, &S[i].rgb) || load_gray(p2, &m)) {
                fprintf(stderr, "cannot load %s or %s\n", p1, p2);
                return -1;
            }
            S[i].fov = binarise(&m);
            S[i].gt = load_gray(p3, &g) ? NULL : binarise(&g);
        } else if (!strcmp(kind, "stare")) {
            snprintf(S[i].name, sizeof(S[i].name), "im%s", STARE_IDS[i]);
            snprintf(p1, sizeof p1, "%s/images/im%s.ppm", root, STARE_IDS[i]);
            snprintf(p3, sizeof p3, "%s/labels/im%s.ah.ppm", root, STARE_IDS[i]);
            if (load_rgb(p1, &S[i].rgb)) {
                fprintf(stderr, "cannot load %s\n", p1);
                return -1;
            }
            S[i].fov = stare_fov(&S[i].rgb);
            S[i].gt = load_gray(p3, &g) ? NULL : binarise(&g);
        } else {
            fprintf(stderr, "unknown dataset kind %s\n", kind);
            return -1;
        }
    }
    *out = S;
    return n;
}

static void free_dataset(Sample *S, int n)
{
    for (int i = 0; i < n; i++) {
        img_free(&S[i].rgb);
        free(S[i].fov);
        free(S[i].gt);
    }
    free(S);
}

/* -------------------------------------------------------------- params --- */

static int load_params(const char *path, Params *p)
{
    params_default(p);
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char key[64];
    double v;
    while (fscanf(f, "%63s %lf", key, &v) == 2) {
        if (!strcmp(key, "clip")) p->clip = (float)v;
        else if (!strcmp(key, "beta")) p->beta = (float)v;
        else if (!strcmp(key, "c")) p->c = (float)v;
        else if (!strcmp(key, "thr")) p->thr = (float)v;
        else if (!strcmp(key, "tiles")) p->tiles = (int)v;
        else if (!strcmp(key, "fov_erode")) fov_erode_radius = (int)v;
    }
    fclose(f);
    return 0;
}

static void save_params(const char *path, const Params *p)
{
    FILE *f = fopen(path, "w");
    fprintf(f, "clip %.6g\nbeta %.6g\nc %.6g\nthr %.6g\ntiles %d\nfov_erode %d\n", p->clip, p->beta, p->c,
            p->thr, p->tiles, fov_erode_radius);
    fclose(f);
}

static const char *opt(int argc, char **argv, const char *name, const char *def)
{
    for (int i = 0; i < argc - 1; i++)
        if (!strcmp(argv[i], name))
            return argv[i + 1];
    return def;
}
static int flag(int argc, char **argv, const char *name)
{
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], name))
            return 1;
    return 0;
}
static int parse_list(const char *s, int *v, int maxn)
{
    int n = 0;
    while (*s && n < maxn) {
        v[n++] = atoi(s);
        while (*s && *s != ',')
            s++;
        if (*s == ',')
            s++;
    }
    return n;
}

static int default_nt(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static unsigned char *eroded_fov(const Sample *s)
{
    unsigned char *e = (unsigned char *)malloc((size_t)s->rgb.w * s->rgb.h);
    fov_erode(s->fov, e, s->rgb.w, s->rgb.h, fov_erode_radius);
    return e;
}

/* ---------------------------------------------------------------- info --- */

static int cmd_info(void)
{
    printf("build=%s\n", BUILD);
#ifdef __VERSION__
    printf("compiler=%s\n", __VERSION__);
#endif
#ifdef _OPENMP
    printf("openmp=%d\nnum_procs=%d\nmax_threads=%d\n", _OPENMP, omp_get_num_procs(), omp_get_max_threads());
#endif
    return 0;
}

/* ---------------------------------------------------------------- tune --- */

static int cmd_tune(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    const char *root = argv[2], *outp = argv[3];
    int nt = atoi(opt(argc, argv, "--nt", "0"));
    if (nt <= 0)
        nt = default_nt();
    Sample *S;
    int n = load_dataset("drive-train", root, 0, &S);
    if (n <= 0)
        return 1;
    const float clips[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float betas[] = {0.5f, 1.0f};
    const float cs[] = {0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f};
    const int erodes[] = {4, 8, 12};
    enum { NTHR = 40, NER = 3 };
    float thrs[NTHR];
    for (int i = 0; i < NTHR; i++)
        thrs[i] = (float)(1e-3 * pow(900.0, (double)i / (NTHR - 1))); /* 0.001 .. 0.9 */

    unsigned char **fe = (unsigned char **)malloc((size_t)n * NER * sizeof(*fe));
    for (int i = 0; i < n; i++)
        for (int e = 0; e < NER; e++) {
            fe[i * NER + e] = (unsigned char *)malloc((size_t)S[i].rgb.w * S[i].rgb.h);
            fov_erode(S[i].fov, fe[i * NER + e], S[i].rgb.w, S[i].rgb.h, erodes[e]);
        }

    char gpath[1100];
    snprintf(gpath, sizeof gpath, "%s.grid.csv", outp);
    FILE *g = fopen(gpath, "w");
    fprintf(g, "clip,beta,c,fov_erode,thr,mean_dice\n");

    Params best;
    params_default(&best);
    int best_er = fov_erode_radius;
    double best_d = -1;
    for (size_t ic = 0; ic < sizeof clips / sizeof *clips; ic++)
        for (size_t ib = 0; ib < sizeof betas / sizeof *betas; ib++)
            for (size_t ik = 0; ik < sizeof cs / sizeof *cs; ik++) {
                Params p;
                params_default(&p);
                p.clip = clips[ic];
                p.beta = betas[ib];
                p.c = cs[ik];
                p.nt = nt;
                double dsum[NER][NTHR];
                memset(dsum, 0, sizeof dsum);
                for (int i = 0; i < n; i++) {
                    Work W;
                    work_alloc(&W, S[i].rgb.w, S[i].rgb.h, &p);
                    size_t np = (size_t)W.w * W.h;
                    stage_green(S[i].rgb.d, W.g, np, nt);
                    stage_clahe(W.g, W.eq, W.w, W.h, &p);
                    stage_frangi(W.eq, &W, &p, NULL);
                    for (int e = 0; e < NER; e++)
                        for (int t = 0; t < NTHR; t++) {
                            stage_threshold(W.V, W.mask, fe[i * NER + e], np, thrs[t], nt);
                            stage_morph(W.mask, W.tmp, W.w, W.h, nt);
                            dsum[e][t] += conf_dice(confusion(W.mask, S[i].gt, S[i].fov, np, nt));
                        }
                    work_free(&W);
                }
                for (int e = 0; e < NER; e++)
                    for (int t = 0; t < NTHR; t++) {
                        double md = dsum[e][t] / n;
                        fprintf(g, "%g,%g,%g,%d,%g,%.6f\n", p.clip, p.beta, p.c, erodes[e], thrs[t], md);
                        if (md > best_d) {
                            best_d = md;
                            best = p;
                            best.thr = thrs[t];
                            best_er = erodes[e];
                        }
                    }
                fflush(g);
                fprintf(stderr, "clip=%g beta=%g c=%g best-so-far dice=%.4f\n", p.clip, p.beta, p.c, best_d);
            }
    fclose(g);
    fov_erode_radius = best_er;
    save_params(outp, &best);
    printf("best mean Dice (training) = %.4f  clip=%g beta=%g c=%g thr=%g fov_erode=%d\n", best_d, best.clip,
           best.beta, best.c, best.thr, best_er);
    for (int i = 0; i < n * NER; i++)
        free(fe[i]);
    free(fe);
    free_dataset(S, n);
    return 0;
}

/* ---------------------------------------------------------------- eval --- */

static int cmd_eval(int argc, char **argv)
{
    if (argc < 6)
        return 2;
    const char *kind = argv[2], *root = argv[3], *pp = argv[4], *outc = argv[5];
    const char *mdir = opt(argc, argv, "--masks", NULL);
    Params p;
    if (load_params(pp, &p))
        return 1;
    p.nt = atoi(opt(argc, argv, "--nt", "0"));
    if (p.nt <= 0)
        p.nt = default_nt();
    Sample *S;
    int n = load_dataset(kind, root, atoi(opt(argc, argv, "--images", "0")), &S);
    if (n <= 0)
        return 1;
    FILE *f = fopen(outc, "w");
    fprintf(f, "dataset,image,tp,fp,tn,fn,dice,sens,spec,acc\n");
    double sd = 0, ss = 0, sp = 0, sa = 0;
    Conf tot = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        Work W;
        work_alloc(&W, S[i].rgb.w, S[i].rgb.h, &p);
        unsigned char *fe = eroded_fov(&S[i]);
        run_pipeline(S[i].rgb.d, fe, &W, &p, NULL);
        size_t np = (size_t)W.w * W.h;
        Conf c = confusion(W.mask, S[i].gt, S[i].fov, np, p.nt);
        fprintf(f, "%s,%s,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f\n", kind, S[i].name, c.tp, c.fp, c.tn, c.fn,
                conf_dice(c), conf_sens(c), conf_spec(c), conf_acc(c));
        sd += conf_dice(c);
        ss += conf_sens(c);
        sp += conf_spec(c);
        sa += conf_acc(c);
        tot.tp += c.tp;
        tot.fp += c.fp;
        tot.tn += c.tn;
        tot.fn += c.fn;
        if (mdir) {
            char mp[1100];
            for (size_t k = 0; k < np; k++)
                W.tmp[k] = W.mask[k] ? 255 : 0;
            snprintf(mp, sizeof mp, "%s/%s_mask.png", mdir, S[i].name);
            write_png_gray(mp, W.tmp, W.w, W.h);
            float vmax = 0;
            for (size_t k = 0; k < np; k++)
                vmax = W.V[k] > vmax ? W.V[k] : vmax;
            for (size_t k = 0; k < np; k++)
                W.tmp[k] = (unsigned char)(vmax > 0 ? 255.f * sqrtf(W.V[k] / vmax) : 0);
            snprintf(mp, sizeof mp, "%s/%s_vesselness.png", mdir, S[i].name);
            write_png_gray(mp, W.tmp, W.w, W.h);
        }
        free(fe);
        work_free(&W);
    }
    fprintf(f, "%s,MEAN,,,,,%.6f,%.6f,%.6f,%.6f\n", kind, sd / n, ss / n, sp / n, sa / n);
    fprintf(f, "%s,POOLED,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f\n", kind, tot.tp, tot.fp, tot.tn, tot.fn,
            conf_dice(tot), conf_sens(tot), conf_spec(tot), conf_acc(tot));
    fclose(f);
    printf("%s: mean Dice=%.4f sens=%.4f spec=%.4f acc=%.4f (n=%d)\n", kind, sd / n, ss / n, sp / n, sa / n, n);
    free_dataset(S, n);
    return 0;
}

/* --------------------------------------------------------------- bench --- */

static int cmd_bench(int argc, char **argv)
{
    if (argc < 6)
        return 2;
    const char *kind = argv[2], *root = argv[3], *pp = argv[4], *outc = argv[5];
    const char *variant = opt(argc, argv, "--variant", "base");
    int ntl[32], scl[8];
    char defnt[16];
    snprintf(defnt, sizeof defnt, "%d", default_nt());
    int nnt = parse_list(opt(argc, argv, "--nt", defnt), ntl, 32);
    int nsc = parse_list(opt(argc, argv, "--scales", "1"), scl, 8);
    int nimg = atoi(opt(argc, argv, "--images", "0"));
    int warm = atoi(opt(argc, argv, "--warm", "5"));
    int reps = atoi(opt(argc, argv, "--reps", "30"));
    Params p;
    if (load_params(pp, &p))
        return 1;
    if (!strcmp(variant, "dynamic"))
        p.sched = 1;
    else if (!strcmp(variant, "atomic"))
        p.clahe_atomic = 1;
    else if (!strcmp(variant, "scalepar"))
        p.scale_par = 1;
    Sample *S;
    int n = load_dataset(kind, root, nimg, &S);
    if (n <= 0)
        return 1;
    int append = flag(argc, argv, "--append");
    FILE *f = fopen(outc, append ? "a" : "w");
    if (!append)
        fprintf(f, "build,variant,dataset,image,scale,width,height,threads,rep,green,clahe,gauss,vess,thresh,"
                   "morph,total\n");
    for (int si = 0; si < nsc; si++) {
        int sc = scl[si];
        for (int i = 0; i < n; i++) {
            int w = S[i].rgb.w * sc, h = S[i].rgb.h * sc;
            unsigned char *rgb = sc == 1 ? S[i].rgb.d : resample_rgb(S[i].rgb.d, S[i].rgb.w, S[i].rgb.h, w, h);
            unsigned char *fov = sc == 1 ? S[i].fov : resample_nearest(S[i].fov, S[i].rgb.w, S[i].rgb.h, w, h);
            unsigned char *fe = (unsigned char *)malloc((size_t)w * h);
            fov_erode(fov, fe, w, h, fov_erode_radius * sc);
            for (int ti = 0; ti < nnt; ti++) {
                p.nt = ntl[ti];
#ifndef _OPENMP
                if (p.nt != 1)
                    continue;
#endif
                Work W;
                if (work_alloc(&W, w, h, &p)) {
                    fprintf(stderr, "out of memory at %dx%d\n", w, h);
                    return 1;
                }
                for (int r = 0; r < warm + reps; r++) {
                    StageT t;
                    run_pipeline(rgb, fe, &W, &p, &t);
                    if (r < warm)
                        continue;
                    fprintf(f, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", BUILD, variant, kind,
                            S[i].name, sc, w, h, p.nt, r - warm, t.green, t.clahe, t.gauss, t.vess, t.thresh,
                            t.morph, t.total);
                }
                fflush(f);
                work_free(&W);
                fprintf(stderr, "[%s/%s] %s x%d nt=%d done\n", BUILD, variant, S[i].name, sc, p.nt);
            }
            free(fe);
            if (sc != 1) {
                free(rgb);
                free(fov);
            }
        }
    }
    fclose(f);
    free_dataset(S, n);
    return 0;
}

/* ------------------------------------------------------- dump / compare --- */

static int cmd_dump(int argc, char **argv)
{
    if (argc < 6)
        return 2;
    const char *kind = argv[2], *root = argv[3], *pp = argv[4], *dir = argv[5];
    Params p;
    if (load_params(pp, &p))
        return 1;
    p.nt = atoi(opt(argc, argv, "--nt", "0"));
    if (p.nt <= 0)
        p.nt = default_nt();
    const char *variant = opt(argc, argv, "--variant", "base");
    if (!strcmp(variant, "dynamic"))
        p.sched = 1;
    else if (!strcmp(variant, "atomic"))
        p.clahe_atomic = 1;
    else if (!strcmp(variant, "scalepar"))
        p.scale_par = 1;
    Sample *S;
    int n = load_dataset(kind, root, atoi(opt(argc, argv, "--images", "0")), &S);
    if (n <= 0)
        return 1;
    char path[1100];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE *man = fopen(path, "w");
    if (!man) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        Work W;
        work_alloc(&W, S[i].rgb.w, S[i].rgb.h, &p);
        unsigned char *fe = eroded_fov(&S[i]);
        run_pipeline(S[i].rgb.d, fe, &W, &p, NULL);
        size_t np = (size_t)W.w * W.h;
        fprintf(man, "%s %d %d\n", S[i].name, W.w, W.h);
        snprintf(path, sizeof path, "%s/%s.eq.u8", dir, S[i].name);
        FILE *o = fopen(path, "wb");
        fwrite(W.eq, 1, np, o);
        fclose(o);
        snprintf(path, sizeof path, "%s/%s.V.f32", dir, S[i].name);
        o = fopen(path, "wb");
        fwrite(W.V, sizeof(float), np, o);
        fclose(o);
        snprintf(path, sizeof path, "%s/%s.mask.u8", dir, S[i].name);
        o = fopen(path, "wb");
        fwrite(W.mask, 1, np, o);
        fclose(o);
        free(fe);
        work_free(&W);
    }
    fclose(man);
    free_dataset(S, n);
    return 0;
}

static void *slurp(const char *path, size_t want)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    void *b = malloc(want);
    size_t got = fread(b, 1, want, f);
    fclose(f);
    if (got != want) {
        free(b);
        return NULL;
    }
    return b;
}

static int cmd_compare(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    const char *A = argv[2], *B = argv[3];
    char path[1100], pb[1100], name[64];
    int w, h;
    snprintf(path, sizeof path, "%s/manifest.txt", A);
    FILE *man = fopen(path, "r");
    if (!man)
        return 1;
    long long npx = 0, eq_same = 0, m_same = 0, v_bits = 0;
    double vmax = 0;
    printf("image,eq_identical,mask_agreement,V_bit_identical,V_max_abs_diff\n");
    while (fscanf(man, "%63s %d %d", name, &w, &h) == 3) {
        size_t n = (size_t)w * h;
        snprintf(path, sizeof path, "%s/%s.eq.u8", A, name);
        snprintf(pb, sizeof pb, "%s/%s.eq.u8", B, name);
        unsigned char *ea = (unsigned char *)slurp(path, n), *eb = (unsigned char *)slurp(pb, n);
        snprintf(path, sizeof path, "%s/%s.mask.u8", A, name);
        snprintf(pb, sizeof pb, "%s/%s.mask.u8", B, name);
        unsigned char *ma = (unsigned char *)slurp(path, n), *mb = (unsigned char *)slurp(pb, n);
        snprintf(path, sizeof path, "%s/%s.V.f32", A, name);
        snprintf(pb, sizeof pb, "%s/%s.V.f32", B, name);
        float *va = (float *)slurp(path, n * 4), *vb = (float *)slurp(pb, n * 4);
        if (!ea || !eb || !ma || !mb || !va || !vb) {
            fprintf(stderr, "missing files for %s\n", name);
            return 1;
        }
        long long es = 0, ms = 0, vs = 0;
        double md = 0;
        for (size_t i = 0; i < n; i++) {
            es += ea[i] == eb[i];
            ms += ma[i] == mb[i];
            vs += memcmp(&va[i], &vb[i], 4) == 0;
            double d = fabs((double)va[i] - vb[i]);
            md = d > md ? d : md;
        }
        printf("%s,%.6f,%.6f,%.6f,%.3e\n", name, (double)es / n, (double)ms / n, (double)vs / n, md);
        npx += (long long)n;
        eq_same += es;
        m_same += ms;
        v_bits += vs;
        vmax = md > vmax ? md : vmax;
        free(ea);
        free(eb);
        free(ma);
        free(mb);
        free(va);
        free(vb);
    }
    fclose(man);
    printf("ALL,%.6f,%.6f,%.6f,%.3e\n", (double)eq_same / npx, (double)m_same / npx, (double)v_bits / npx, vmax);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: frangi info|tune|eval|bench|dump|compare ...\n");
        return 2;
    }
    const char *c = argv[1];
    int rc;
    if (!strcmp(c, "info"))
        rc = cmd_info();
    else if (!strcmp(c, "tune"))
        rc = cmd_tune(argc, argv);
    else if (!strcmp(c, "eval"))
        rc = cmd_eval(argc, argv);
    else if (!strcmp(c, "bench"))
        rc = cmd_bench(argc, argv);
    else if (!strcmp(c, "dump"))
        rc = cmd_dump(argc, argv);
    else if (!strcmp(c, "compare"))
        rc = cmd_compare(argc, argv);
    else if (!strcmp(c, "green") && argc >= 4) {
        /* decoder check: write the green channel of an image as PGM */
        ImgU8 im;
        rc = load_rgb(argv[2], &im) ? 1 : 0;
        if (!rc) {
            unsigned char *g = (unsigned char *)malloc((size_t)im.w * im.h);
            stage_green(im.d, g, (size_t)im.w * im.h, 1);
            rc = write_pgm(argv[3], g, im.w, im.h) ? 1 : 0;
            free(g);
            img_free(&im);
        }
    }
    else
        rc = 2;
    if (rc == 2)
        fprintf(stderr, "bad arguments; see the header of main.c\n");
    return rc;
}
