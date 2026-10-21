#ifndef IO_H
#define IO_H

#include <stddef.h>

typedef struct {
    int w, h, c;
    unsigned char *d;
} ImgU8;

/* RGB (c=3) loader: .tif/.tiff (LZW), .ppm/.pgm, everything else through stb_image. */
int load_rgb(const char *path, ImgU8 *out);
/* Single-channel loader (masks, manual segmentations): .gif/.png/.pgm/.ppm. */
int load_gray(const char *path, ImgU8 *out);
void img_free(ImgU8 *im);
ImgU8 img_alloc(int w, int h, int c);

int write_pgm(const char *path, const unsigned char *d, int w, int h);
int write_png_gray(const char *path, const unsigned char *d, int w, int h);

#endif
