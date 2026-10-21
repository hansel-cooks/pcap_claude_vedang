"""Reference checks (Python used only for verification, never in the measured pipeline).

1. TIFF decoder: green channel written by the C decoder vs Pillow.
2. Frangi: C vesselness (serial build) vs scikit-image frangi() on the same CLAHE output.

usage: check.py <drive_root> <params.txt> <serial_dump_dir> <out_dir>
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.filters import frangi

drive, params_path, dump, out = sys.argv[1:5]
params = dict(l.split() for l in open(params_path) if l.strip())
beta, c = float(params["beta"]), float(params["c"])

rows = ["file,pixels,identical,max_abs_diff"]
for pgm in sorted(glob.glob(os.path.join(out, "decode", "*_green.pgm"))):
    base = os.path.basename(pgm).replace("_green.pgm", "")
    sub = "training" if "training" in base else "test"
    ref = np.asarray(Image.open(os.path.join(drive, sub, "images", base + ".tif")).convert("RGB"))[:, :, 1]
    mine = np.asarray(Image.open(pgm))
    rows.append("%s,%d,%d,%d" % (base, ref.size, int((ref == mine).sum()),
                                 int(np.abs(ref.astype(int) - mine.astype(int)).max())))
open(os.path.join(out, "check_decoder.csv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))

rows = ["image,pearson,spearman,mask_dice_vs_skimage"]
thr = float(params["thr"])
for line in open(os.path.join(dump, "manifest.txt")).read().split("\n")[:4]:
    if not line.strip():
        continue
    name, w, h = line.split()
    w, h = int(w), int(h)
    eq = np.fromfile(os.path.join(dump, name + ".eq.u8"), np.uint8).reshape(h, w) / 255.0
    V = np.fromfile(os.path.join(dump, name + ".V.f32"), np.float32).reshape(h, w)
    sk = frangi(eq, sigmas=[1, 2, 3, 4], beta=beta, gamma=c, black_ridges=True, mode="nearest")
    idx = name.split("_")[0]
    fov = np.asarray(Image.open(os.path.join(drive, "test", "mask", idx + "_test_mask.gif")).convert("L")) > 127
    a, b = V[fov].astype(np.float64), sk[fov].astype(np.float64)
    ma, mb = a > thr, b > thr
    dice = 2.0 * (ma & mb).sum() / max(1, ma.sum() + mb.sum())
    rows.append("%s,%.4f,%.4f,%.4f" % (name, pearsonr(a, b)[0], spearmanr(a, b)[0], dice))
open(os.path.join(out, "check_skimage.csv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
