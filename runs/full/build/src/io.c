#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_GIF
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#define STBI_ONLY_JPEG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

ImgU8 img_alloc(int w, int h, int c)
{
    ImgU8 im;
    im.w = w;
    im.h = h;
    im.c = c;
    im.d = (unsigned char *)malloc((size_t)w * h * c);
    return im;
}

void img_free(ImgU8 *im)
{
    free(im->d);
    im->d = NULL;
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static int has_ext(const char *path, const char *ext)
{
    size_t lp = strlen(path), le = strlen(ext);
    if (lp < le)
        return 0;
    for (size_t i = 0; i < le; i++) {
        char a = path[lp - le + i], b = ext[i];
        if (a >= 'A' && a <= 'Z')
            a = (char)(a + 32);
        if (a != b)
            return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- TIFF --- */

static unsigned rd16(const unsigned char *p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

/* TIFF LZW (MSB-first codes, 9..12 bits, "early change"). Returns bytes written or -1. */
static long lzw_decode(const unsigned char *src, size_t n, unsigned char *dst, size_t cap)
{
    unsigned short pre[4096];
    unsigned char suf[4096];
    unsigned char stk[4100];
    int next = 258, bits = 9, old = -1;
    uint32_t buf = 0;
    int nb = 0;
    size_t ip = 0, op = 0;

    /* Stop once the strip is full: some encoders write the final EOI with the next
       code width, so the trailing bits cannot be trusted (libtiff behaves the same). */
    while (op < cap) {
        while (nb < bits && ip < n) {
            buf = (buf << 8) | src[ip++];
            nb += 8;
        }
        if (nb < bits)
            break;
        int code = (int)((buf >> (nb - bits)) & ((1u << bits) - 1));
        nb -= bits;
        if (code == 257)
            break;
        if (code == 256) {
            next = 258;
            bits = 9;
            old = -1;
            continue;
        }
        if (old == -1) {
            if (code > 255 || op >= cap)
                return -1;
            dst[op++] = (unsigned char)code;
            old = code;
            continue;
        }
        int c = (code >= next) ? old : code;
        int sp = 0;
        if (code > next)
            return -1;
        while (c >= 258) {
            stk[sp++] = suf[c];
            c = pre[c];
        }
        stk[sp++] = (unsigned char)c;
        unsigned char first = stk[sp - 1];
        if (op + (size_t)sp + (code >= next ? 1 : 0) > cap)
            return -1;
        while (sp > 0)
            dst[op++] = stk[--sp];
        if (code >= next)
            dst[op++] = first;
        if (next < 4096) {
            pre[next] = (unsigned short)old;
            suf[next] = first;
            next++;
            if (next == 511)
                bits = 10;
            else if (next == 1023)
                bits = 11;
            else if (next == 2047)
                bits = 12;
        }
        old = code;
    }
    return (long)op;
}

static int tiff_tag_values(const unsigned char *b, size_t len, const unsigned char *e, unsigned *out, unsigned maxn)
{
    unsigned type = rd16(e + 2), cnt = rd32(e + 4);
    unsigned sz = (type == 3) ? 2 : (type == 4 ? 4 : 1);
    const unsigned char *p;
    if ((size_t)sz * cnt <= 4)
        p = e + 8;
    else {
        size_t off = rd32(e + 8);
        if (off + (size_t)sz * cnt > len)
            return -1;
        p = b + off;
    }
    unsigned n = cnt < maxn ? cnt : maxn;
    for (unsigned i = 0; i < n; i++)
        out[i] = (sz == 2) ? rd16(p + 2 * i) : (sz == 4 ? rd32(p + 4 * i) : p[i]);
    return (int)n;
}

static int load_tiff_rgb(const char *path, ImgU8 *out)
{
    size_t len;
    unsigned char *b = read_file(path, &len);
    if (!b)
        return -1;
    if (len < 8 || b[0] != 'I' || b[1] != 'I' || rd16(b + 2) != 42) {
        free(b);
        return -2;
    }
    size_t ifd = rd32(b + 4);
    unsigned nent = rd16(b + ifd);
    unsigned w = 0, h = 0, comp = 1, spp = 1, rps = 0, pred = 1, planar = 1;
    unsigned *offs = NULL, *cnts = NULL;
    unsigned nstrips = 0;
    for (unsigned i = 0; i < nent; i++) {
        const unsigned char *e = b + ifd + 2 + 12 * i;
        unsigned tag = rd16(e), v[1];
        switch (tag) {
        case 256: tiff_tag_values(b, len, e, &w, 1); break;
        case 257: tiff_tag_values(b, len, e, &h, 1); break;
        case 259: tiff_tag_values(b, len, e, &comp, 1); break;
        case 277: tiff_tag_values(b, len, e, &spp, 1); break;
        case 278: tiff_tag_values(b, len, e, &rps, 1); break;
        case 284: tiff_tag_values(b, len, e, &planar, 1); break;
        case 317: tiff_tag_values(b, len, e, &pred, 1); break;
        case 273:
            nstrips = rd32(e + 4);
            offs = (unsigned *)malloc(nstrips * sizeof(unsigned));
            tiff_tag_values(b, len, e, offs, nstrips);
            break;
        case 279:
            cnts = (unsigned *)malloc(rd32(e + 4) * sizeof(unsigned));
            tiff_tag_values(b, len, e, cnts, rd32(e + 4));
            break;
        default: (void)v; break;
        }
    }
    int rc = 0;
    if (!w || !h || spp != 3 || planar != 1 || !offs || !cnts || (comp != 5 && comp != 1)) {
        rc = -3;
        goto done;
    }
    if (rps == 0 || rps > h)
        rps = h;
    *out = img_alloc((int)w, (int)h, 3);
    size_t rowbytes = (size_t)w * 3;
    for (unsigned s = 0; s < nstrips; s++) {
        unsigned y0 = s * rps;
        if (y0 >= h)
            break;
        unsigned rows = (y0 + rps > h) ? h - y0 : rps;
        unsigned char *dst = out->d + (size_t)y0 * rowbytes;
        size_t want = rows * rowbytes;
        if (offs[s] + (size_t)cnts[s] > len) {
            rc = -4;
            break;
        }
        if (comp == 5) {
            long got = lzw_decode(b + offs[s], cnts[s], dst, want);
            if (got != (long)want) {
                rc = -5;
                break;
            }
        } else {
            memcpy(dst, b + offs[s], want);
        }
        if (pred == 2) {
            for (unsigned r = 0; r < rows; r++) {
                unsigned char *row = dst + r * rowbytes;
                for (size_t i = 3; i < rowbytes; i++)
                    row[i] = (unsigned char)(row[i] + row[i - 3]);
            }
        }
    }
    if (rc)
        img_free(out);
done:
    free(offs);
    free(cnts);
    free(b);
    return rc;
}

/* ----------------------------------------------------------------- PNM --- */

static int pnm_int(const unsigned char *b, size_t len, size_t *p)
{
    while (*p < len) {
        if (b[*p] == '#') {
            while (*p < len && b[*p] != '\n')
                (*p)++;
        } else if (b[*p] == ' ' || b[*p] == '\n' || b[*p] == '\r' || b[*p] == '\t')
            (*p)++;
        else
            break;
    }
    int v = 0;
    while (*p < len && b[*p] >= '0' && b[*p] <= '9')
        v = v * 10 + (b[(*p)++] - '0');
    return v;
}

static int load_pnm(const char *path, ImgU8 *out, int want_c)
{
    size_t len;
    unsigned char *b = read_file(path, &len);
    if (!b)
        return -1;
    if (len < 3 || b[0] != 'P' || (b[1] != '5' && b[1] != '6')) {
        free(b);
        return -2;
    }
    int c = (b[1] == '6') ? 3 : 1;
    size_t p = 2;
    int w = pnm_int(b, len, &p), h = pnm_int(b, len, &p), mx = pnm_int(b, len, &p);
    p++;
    if (mx != 255 || (size_t)w * h * c > len - p) {
        free(b);
        return -3;
    }
    *out = img_alloc(w, h, want_c);
    size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; i++) {
        if (want_c == c) {
            for (int k = 0; k < c; k++)
                out->d[i * c + k] = b[p + i * c + k];
        } else if (want_c == 1) {
            out->d[i] = (unsigned char)((b[p + i * 3] + b[p + i * 3 + 1] + b[p + i * 3 + 2]) / 3);
        } else {
            for (int k = 0; k < 3; k++)
                out->d[i * 3 + k] = b[p + i];
        }
    }
    free(b);
    return 0;
}

/* ------------------------------------------------------------- generic --- */

static int load_stb(const char *path, ImgU8 *out, int want_c)
{
    int w, h, n;
    unsigned char *d = stbi_load(path, &w, &h, &n, want_c);
    if (!d)
        return -1;
    out->w = w;
    out->h = h;
    out->c = want_c;
    out->d = d;
    return 0;
}

int load_rgb(const char *path, ImgU8 *out)
{
    if (has_ext(path, ".tif") || has_ext(path, ".tiff"))
        return load_tiff_rgb(path, out);
    if (has_ext(path, ".ppm") || has_ext(path, ".pgm"))
        return load_pnm(path, out, 3);
    return load_stb(path, out, 3);
}

int load_gray(const char *path, ImgU8 *out)
{
    if (has_ext(path, ".ppm") || has_ext(path, ".pgm"))
        return load_pnm(path, out, 1);
    return load_stb(path, out, 1);
}

int write_pgm(const char *path, const unsigned char *d, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(d, 1, (size_t)w * h, f);
    fclose(f);
    return 0;
}

int write_png_gray(const char *path, const unsigned char *d, int w, int h)
{
    return stbi_write_png(path, w, h, 1, d, w) ? 0 : -1;
}
