#ifndef PIPELINE_H
#define PIPELINE_H

#include <stddef.h>

#define MAX_SCALES 4

typedef struct {
    int tiles;        /* CLAHE tile grid (tiles x tiles) */
    float clip;       /* CLAHE clip limit, multiple of the uniform bin count */
    float beta, c;    /* Frangi parameters */
    float thr;        /* global vesselness threshold */
    int nsig;
    float sig[MAX_SCALES];
    int nt;           /* number of OpenMP threads (1 = serial path) */
    int sched;        /* 0 static, 1 dynamic for the CLAHE tile loop */
    int clahe_atomic; /* 1 = pixel-parallel histogram with atomics (ablation) */
    int scale_par;    /* 1 = scales processed in parallel (outer loop), 0 = inside */
} Params;

typedef struct {
    double green, clahe, gauss, vess, thresh, morph, total;
} StageT;

typedef struct {
    int w, h;
    int nslots;
    unsigned char *g, *eq, *mask, *tmp;
    float *V, *fin; /* vesselness, float copy of the CLAHE output */
    float *buf[MAX_SCALES][7]; /* A,B,C (row passes), Ixx,Iyy,Ixy, Vs */
    float *k[MAX_SCALES][3];   /* g0,g1,g2 kernels */
    int r[MAX_SCALES];
} Work;

void params_default(Params *p);
double wtime(void);

int work_alloc(Work *W, int w, int h, const Params *p);
void work_free(Work *W);

/* Full pipeline. rgb: w*h*3, fov_er: eroded field-of-view (0/1), out: W->mask. */
void run_pipeline(const unsigned char *rgb, const unsigned char *fov_er, Work *W, const Params *p, StageT *t);

/* Individual stages (exposed so the harness can time and test them). */
void stage_green(const unsigned char *rgb, unsigned char *g, size_t n, int nt);
void stage_clahe(const unsigned char *g, unsigned char *eq, int w, int h, const Params *p);
void stage_frangi(const unsigned char *eq, Work *W, const Params *p, StageT *t);
void stage_threshold(const float *V, unsigned char *mask, const unsigned char *fov_er, size_t n, float thr, int nt);
void stage_morph(unsigned char *mask, unsigned char *tmp, int w, int h, int nt);

/* Helpers used outside the timed path. */
void fov_erode(const unsigned char *fov, unsigned char *out, int w, int h, int radius);
unsigned char *resample_rgb(const unsigned char *src, int w, int h, int nw, int nh);
unsigned char *resample_nearest(const unsigned char *src, int w, int h, int nw, int nh);

typedef struct {
    long long tp, fp, tn, fn;
} Conf;
Conf confusion(const unsigned char *pred, const unsigned char *gt, const unsigned char *fov, size_t n, int nt);
double conf_dice(Conf c);
double conf_sens(Conf c);
double conf_spec(Conf c);
double conf_acc(Conf c);

#endif
