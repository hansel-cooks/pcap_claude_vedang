# Serial / OpenMP multi-scale Frangi pipeline (DRIVE, STARE)

C implementation of green channel -> CLAHE -> multi-scale Frangi (sigma 1..4) -> threshold ->
3x3 opening + closing, with every stage parallelised by OpenMP. One source tree builds two binaries:

| binary | build | role |
|---|---|---|
| `bin/frangi_serial` | no `-fopenmp` (pragmas ignored) | golden reference and timing baseline |
| `bin/frangi_omp` | `-fopenmp` | parallel version |

```
make OPT="-O3 -march=native"                       # both binaries
make OPT="-O3 -march=native" FPFLAGS=-ffp-contract=off BIN=bin_nofma   # exact-equality build
```

## Commands

```
frangi info
frangi tune    <drive_root> <out_params>                      # grid search on DRIVE training only
frangi eval    <kind> <root> <params> <out.csv> [--masks DIR] # Dice / sens / spec / acc inside FOV
frangi bench   <kind> <root> <params> <out.csv> --nt 1,2,4,8 [--scales 1,2,4,8] [--variant base|dynamic|atomic|scalepar]
frangi dump    <kind> <root> <params> <dir> --nt N [--variant V]   # CLAHE, V and mask for equivalence checks
frangi compare <dirA> <dirB>                                    # identical bytes / bits, max |dV|
frangi green   <image> <out.pgm>                                # decoder check
```

`kind` is `drive-train`, `drive-test` or `stare`. DRIVE layout: `<root>/{training,test}/{images,mask,1st_manual}`.
STARE layout: `<root>/images/imNNNN.ppm`, `<root>/labels/imNNNN.ah.ppm` (Hoover labels).

## Files

- `src/io.c` - TIFF (LZW) reader written for DRIVE, PPM reader, stb_image for GIF masks, PNG writer.
- `src/pipeline.c` - the five stages, serial and OpenMP paths in the same loops.
- `src/main.c` - tuning, evaluation, benchmark harness, dump/compare.
- `scripts/run_kaggle.py` - drives every experiment on the Kaggle CPU VM (Python only orchestrates).
- `scripts/check.py` - Pillow decoder check and scikit-image Frangi cross-check.
- `scripts/analyze.py` - statistics, LaTeX tables and figures from the CSVs.

## Data sources

- DRIVE (Staal et al., 2004): user-supplied copy; test-set manual segmentations taken from the
  Kaggle dataset `zionfuo/drive2004` (images verified byte-identical by MD5).
- STARE (Hoover et al., 2000): https://cecas.clemson.edu/~ahoover/stare/probing/index.html
  (`stare-images.tar`, `labels-ah.tar`).
