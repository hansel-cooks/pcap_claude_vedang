"""Generates notebooks/PCAP_OpenMP_Frangi_Demo.ipynb (run: python make_notebook.py <out.ipynb>)."""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hansel-cooks/pcap_claude_vedang/blob/main/notebooks/PCAP_OpenMP_Frangi_Demo.ipynb)

# Serial vs OpenMP: a multi-scale Frangi pipeline for retinal vessel segmentation

**PCAP mini project, Manipal Institute of Technology** (Devik Nangia, Snehal Tyagi, Srijan Bera, Prathamesh Patil)

This notebook builds the project's C code, runs it on the **DRIVE** and **STARE** fundus datasets, and shows

1. what each of the five pipeline stages does to an image (before and after),
2. how accurate the segmentation is against expert annotations,
3. that the OpenMP version produces **bit-identical** output to the serial version,
4. how long **every stage** takes at **every thread count**, measured live on this Colab machine,
5. speedup, efficiency, Karp-Flatt serial fraction, throughput, thread placement, design variants and image-size scaling,
6. the same measurements from the paper's Kaggle run, for comparison.

| Stage | What it does | Character |
|---|---|---|
| 1. Green channel | take G from RGB (vessels have most contrast there) | trivial, memory-bound |
| 2. CLAHE | 8x8 tiles, clipped histograms, bilinear blend of tile lookup tables | histograms |
| 3a. Gaussian derivatives | 6 separable 1-D passes per scale, sigma = 1, 2, 3, 4 | memory-bound convolution, ~70% of time |
| 3b. Eigen + vesselness | closed-form Hessian eigenvalues, Frangi formula, max over scales | arithmetic (sqrt, exp) |
| 4. Threshold | global threshold inside the eroded field of view | trivial |
| 5. Morphology | 3x3 opening then closing | byte-wide, low work |

All computation is in C (`openmp_frangi/src`); Python here only drives the binaries and draws the figures.

**Runtime:** use the default CPU runtime. A free Colab VM usually has 2 logical CPUs (1 physical core with Hyper-Threading), so expect smaller speedups than the paper's 2-core Kaggle machine; section 9 shows both.
""")

md(r"""
## 1. Get the code and data

The repository (code, DRIVE and STARE data, and the paper's Kaggle results) is cloned from GitHub. No account
or token is needed. **Runtime > Run all** runs the whole notebook in about 10 minutes on a free CPU runtime.
""")

code(r"""
import os, subprocess
REPO = "hansel-cooks/pcap_claude_vedang"
ROOT = "/content/pcap_claude_vedang"
if not os.path.exists(ROOT):
    r = subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{REPO}.git", ROOT],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
print(sorted(os.listdir(ROOT)))
""")

code(r"""
import glob, json, re, shutil, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from IPython.display import display, Markdown

CODE  = f"{ROOT}/openmp_frangi"
DRIVE = f"{ROOT}/data/drive"
STARE = f"{ROOT}/data/stare"
BIN   = f"{CODE}/bin"
OUT   = "/content/demo"
KAGGLE = f"{ROOT}/runs/full/results"      # the paper's measurements
os.makedirs(OUT, exist_ok=True)

STAGES = ["green", "clahe", "gauss", "vess", "thresh", "morph"]
LABEL = {"green": "Green channel", "clahe": "CLAHE", "gauss": "Gaussian derivatives",
         "vess": "Eigen + vesselness", "thresh": "Threshold", "morph": "Morphology", "total": "Total"}
COL = {"green": "#2a78d6", "clahe": "#eb6834", "gauss": "#1baf7a", "vess": "#eda100",
       "thresh": "#e87ba4", "morph": "#008300", "total": "#0b0b0b"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

def run(cmd, env=None, quiet=False):
    e = dict(os.environ)
    if env: e.update(env)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=e)
    if not quiet or r.returncode:
        print("$", cmd, f"  [{time.time()-t0:.1f}s]")
        print((r.stdout + r.stderr)[-2500:])
    r.check_returncode()
    return r.stdout

def show(paths, titles, ncol=4, size=3.2, cmap="gray"):
    n = len(paths); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow * 1.03))
    for ax in np.ravel(axs): ax.axis("off")
    for ax, p, t in zip(np.ravel(axs), paths, titles):
        img = np.asarray(Image.open(p)) if isinstance(p, str) else p
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(t, fontsize=9)
    plt.tight_layout(); plt.show()
""")

md(r"""
## 2. The machine we are running on

OpenMP speedup is bounded by **physical cores**; Hyper-Threading adds a second logical CPU per core that shares
the core's execution units and caches. `lscpu -e` shows which logical CPUs share a core.
""")

code(r"""
print(run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|L1d|L2|L3'", quiet=True))
print(run("lscpu -e", quiet=True))
print(run("free -h | head -2; gcc --version | head -1", quiet=True))
NP = os.cpu_count()
print("logical CPUs visible to OpenMP:", NP)
""")

md(r"""
## 3. Build: one source tree, two binaries

`frangi_serial` is compiled **without** `-fopenmp`, so every `#pragma omp` is ignored and the same loops run on one
thread. `frangi_omp` is the parallel build. Differences in time are therefore due to parallel execution, not to
different code.
""")

code(r"""
run(f"cd {CODE} && make clean >/dev/null; make OPT='-O3 -march=native'")
run(f"{BIN}/frangi_omp info")
pragmas = run(f"grep -n '#pragma omp' {CODE}/src/pipeline.c", quiet=True)
print(f"{len(pragmas.splitlines())} OpenMP directives in pipeline.c:\n")
print(pragmas)
""")

md(r"""
### Parameters

The pipeline parameters (CLAHE clip factor, Frangi beta and c, FOV erosion, threshold) were chosen by a grid
search over 5,760 settings on the 20 DRIVE **training** images only (`frangi tune`, run on Kaggle). They are loaded
from the paper's run; re-tuning here would take about 15 minutes on a 2-CPU VM (set `RETUNE = True`).
""")

code(r"""
RETUNE = False
PARAMS = f"{OUT}/params.txt"
if RETUNE or not os.path.exists(f"{KAGGLE}/params.txt"):
    run(f"{BIN}/frangi_omp tune {DRIVE} {PARAMS} --nt {NP}")
else:
    shutil.copy(f"{KAGGLE}/params.txt", PARAMS)
print(open(PARAMS).read())
""")

md(r"""
## 3b. The C code: serial and OpenMP are the same loops

There is **one** implementation. Each stage is a plain C loop with an OpenMP directive above it:

* `frangi_serial` is compiled **without** `-fopenmp`: the compiler ignores every `#pragma omp` line and the loop runs on one thread.
* `frangi_omp` is compiled **with** `-fopenmp`: the same pragma splits the loop's iterations across `nt` threads.

Every output element is written by exactly one thread, in the serial summation order, which is why the two builds
give bit-identical results (section 7). The cells below print the real source from `openmp_frangi/src`.
""")

code(r"""
from IPython.display import Code
SRC = f"{CODE}/src"

def c_function(name, file="pipeline.c"):
    # return (first_line, text) of the C function or macro called `name`
    lines = open(f"{SRC}/{file}").read().splitlines()
    if name.startswith("#define"):
        i = next(k for k, l in enumerate(lines) if l.startswith(name))
        j = i
        while lines[j].rstrip().endswith("\\"): j += 1
        return i + 1, "\n".join(lines[i:j + 1])
    pat = re.compile(r"^[A-Za-z_][\w \*]*\b" + re.escape(name) + r"\(")
    i = next(k for k, l in enumerate(lines) if pat.match(l))
    depth, j, opened = 0, i, False
    while True:
        depth += lines[j].count("{") - lines[j].count("}")
        opened = opened or "{" in lines[j]
        if opened and depth == 0: break
        j += 1
    return i + 1, "\n".join(lines[i:j + 1])

def show_c(*names, file="pipeline.c"):
    for n in names:
        start, text = c_function(n, file)
        display(Markdown(f"`src/{file}`, line {start}: **{n}**"))
        display(Code(text, language="c"))

print("Build commands (from the Makefile):")
print(run(f"grep -A2 'frangi_serial:\\|frangi_omp:' {CODE}/Makefile | grep CC", quiet=True))
show_c("wtime")   # the timer: omp_get_wtime in the OpenMP build, a monotonic clock in the serial build
""")

md(r"""
### Stage 1: green channel
One `parallel for` over all pixels; each thread copies a contiguous block of pixels.
""")
code(r"""show_c("stage_green")""")

md(r"""
### Stage 2: CLAHE
The base design gives each of the 64 tiles to one thread (`schedule(static)`), so every histogram is private and no
synchronisation is needed. The two ablations are in the same function: `dynamic` scheduling of the tile loop, and a
pixel-parallel version that updates shared histograms with `#pragma omp atomic`. The bilinear blend at the end is a
`parallel for` over rows.
""")
code(r"""show_c("clahe_lut", "stage_clahe")""")

md(r"""
### Stage 3a: Gaussian derivatives (about 72% of the run time)
Row pass: `#pragma omp parallel` creates the threads once, each allocates its own padded row buffer, and
`#pragma omp for` shares out the rows. Column pass: a `parallel for` over output rows; the inner `x` loop is
vectorised by the compiler. Three derivative images are produced per sweep, so six 1-D passes per scale.
""")
code(r"""show_c("conv_rows", "conv_cols")""")

md(r"""
### Stage 3b: Hessian eigenvalues and vesselness
A `parallel for` over pixels: closed-form eigenvalues, the Frangi formula and the running maximum over scales.
""")
code(r"""show_c("vesselness")""")

md(r"""
### Stage 3: the scale loop
In the base design the four scales run one after another and each pass is parallel inside. The `scalepar` variant
instead runs the four scales concurrently (`schedule(dynamic,1)`), one thread per scale, then combines them.
""")
code(r"""show_c("stage_frangi")""")

md(r"""
### Stages 4 and 5: threshold and morphology
Threshold is a `parallel for` over pixels. Opening and closing are four 3x3 erosion/dilation passes, each a
`parallel for` over rows (the loop body is a macro shared by erosion and dilation).
""")
code(r"""show_c("stage_threshold", "#define MORPH3_BODY", "erode3", "dilate3", "stage_morph")""")

md(r"""
### Metrics and the whole pipeline
The confusion counts use an OpenMP `reduction`. `run_pipeline` chains the five stages and times each one.
""")
code(r"""show_c("confusion", "run_pipeline")""")

md(r"""
## 3c. Compile and run it on one image

The two commands below are exactly what the Makefile does. Then both binaries process the first DRIVE test image:
the serial build once, the OpenMP build at every thread count, with the time of each stage printed.
""")

code(r"""
%cd {CODE}
!gcc -std=gnu99 -O3 -march=native -Wno-unknown-pragmas -o bin/frangi_serial src/main.c src/pipeline.c src/io.c -lm
!gcc -std=gnu99 -O3 -march=native -fopenmp              -o bin/frangi_omp    src/main.c src/pipeline.c src/io.c -lm
!ls -l bin
""")

code(r"""
ONE = f"{OUT}/one"; os.makedirs(ONE, exist_ok=True)
for f in glob.glob(f"{ONE}/*.csv"): os.remove(f)
run(f"{BIN}/frangi_serial bench drive-test {DRIVE} {PARAMS} {ONE}/serial.csv --nt 1 --warm 2 --reps 10 --images 1", quiet=True)
run(f"{BIN}/frangi_omp    bench drive-test {DRIVE} {PARAMS} {ONE}/omp.csv --nt {','.join(map(str, sorted({1, 2, NP})))} --warm 2 --reps 10 --images 1", quiet=True)
one = pd.concat([pd.read_csv(f"{ONE}/serial.csv"), pd.read_csv(f"{ONE}/omp.csv")])
one["config"] = np.where(one.build == "serial", "serial", "OpenMP, " + one.threads.astype(str) + " thr")
t = one.groupby("config", sort=False)[STAGES + ["total"]].median() * 1e3
t.columns = [LABEL[c] for c in t.columns]
display(Markdown("**Image 01_test, median of 10 runs, milliseconds per stage**"))
display(t.style.format("{:.2f}"))
print(run(f"{BIN}/frangi_omp eval drive-test {DRIVE} {PARAMS} {ONE}/acc.csv --images 1 --masks {ONE}", quiet=True))
show([np.asarray(Image.open(f"{DRIVE}/test/images/01_test.tif").convert("RGB")), f"{ONE}/01_test_vesselness.png",
      f"{ONE}/01_test_mask.png", np.asarray(Image.open(f"{DRIVE}/test/1st_manual/01_manual1.gif").convert("L"))],
     ["input", "vesselness V (C output)", "segmentation (C output)", "expert annotation"], ncol=4, size=3.6)
""")

md(r"""
### Try it yourself
Change `THREADS_TO_TRY`, `IMAGE_COUNT` or the dataset (`drive-test` or `stare`) and rerun the cell. To change the
algorithm, edit `src/pipeline.c` (for example remove a `#pragma omp` line), then rerun the compile cell above.
""")

code(r"""
THREADS_TO_TRY = "1,2"
IMAGE_COUNT = 1
DATASET, ROOTDIR = "drive-test", DRIVE        # or: "stare", STARE
print(run(f"{BIN}/frangi_omp bench {DATASET} {ROOTDIR} {PARAMS} {ONE}/try.csv --nt {THREADS_TO_TRY} --warm 1 --reps 5 --images {IMAGE_COUNT}", quiet=True))
d = pd.read_csv(f"{ONE}/try.csv")
print((d.groupby("threads")[STAGES + ["total"]].median() * 1e3).round(2).rename(columns=LABEL).to_string())
""")

md(r"""
## 4. The datasets

**DRIVE** (Staal et al., 2004): 40 colour fundus images, 565 x 584 pixels, 20 training and 20 test, each with a
field-of-view (FOV) mask and an expert's manual vessel segmentation.
**STARE** (Hoover et al., 2000): 20 images, 700 x 605, with hand labels; used as a cross-dataset check with the
DRIVE parameters unchanged.
""")

code(r"""
rows = []
for name, imgs, gts in [("DRIVE train", f"{DRIVE}/training/images/*.tif", f"{DRIVE}/training/1st_manual/*.gif"),
                        ("DRIVE test", f"{DRIVE}/test/images/*.tif", f"{DRIVE}/test/1st_manual/*.gif"),
                        ("STARE", f"{STARE}/images/*.ppm", f"{STARE}/labels/*.ppm")]:
    I, G = sorted(glob.glob(imgs)), sorted(glob.glob(gts))
    im = Image.open(I[0]); vess = np.mean([np.asarray(Image.open(g).convert("L")) > 127 for g in G[:5]])
    rows.append({"set": name, "images": len(I), "labels": len(G), "size": f"{im.size[0]} x {im.size[1]}",
                 "vessel pixels (first 5 labels)": f"{100*vess:.1f}%"})
display(pd.DataFrame(rows))

ex = [f"{DRIVE}/test/images/01_test.tif", f"{DRIVE}/test/mask/01_test_mask.gif", f"{DRIVE}/test/1st_manual/01_manual1.gif",
      f"{STARE}/images/im0077.ppm", f"{STARE}/labels/im0077.ah.ppm"]
show([np.asarray(Image.open(p).convert("RGB" if i in (0, 3) else "L")) for i, p in enumerate(ex)],
     ["DRIVE 01_test (input)", "FOV mask", "manual segmentation", "STARE im0077 (input)", "Hoover labels"], ncol=5, size=3)
""")

md(r"""
## 5. Stage by stage: before and after

`frangi stages` writes the output of every stage. The Frangi filter is shown at each scale separately and as the
maximum over scales: small sigma picks up capillaries, large sigma the main vessels.
""")

code(r"""
SD = f"{OUT}/stages"; os.makedirs(SD, exist_ok=True)
run(f"{BIN}/frangi_omp stages drive-test {DRIVE} {PARAMS} {SD} --images 2")
run(f"{BIN}/frangi_omp stages stare {STARE} {PARAMS} {SD} --images 7")
def stage_grid(name, rgb):
    p = lambda s: f"{SD}/{name}_{s}.png"
    show([np.asarray(Image.open(rgb).convert("RGB")), p("1_green"), p("2_clahe"), p("3_vesselness_sigma1"),
          p("3_vesselness_sigma2"), p("3_vesselness_sigma3"), p("3_vesselness_sigma4"), p("3_vesselness_max"),
          p("4_threshold"), p("5_final")],
         ["input RGB", "1. green channel", "2. CLAHE", "3. Frangi, sigma=1", "3. Frangi, sigma=2", "3. Frangi, sigma=3",
          "3. Frangi, sigma=4", "3. max over scales", "4. threshold (inside FOV)", "5. after opening + closing"], ncol=5, size=3)
stage_grid("01_test", f"{DRIVE}/test/images/01_test.tif")
""")

md(r"""
### CLAHE: contrast before and after

CLAHE equalises each 8x8 tile's histogram with a clip limit, so vessel contrast becomes similar across the image
(compare the dark periphery and the bright optic disc).
""")

code(r"""
g = np.asarray(Image.open(f"{SD}/01_test_1_green.png")); c = np.asarray(Image.open(f"{SD}/01_test_2_clahe.png"))
fov = np.asarray(Image.open(f"{DRIVE}/test/mask/01_test_mask.gif").convert("L")) > 127
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
axs[0].imshow(g, cmap="gray"); axs[0].set_title("green channel (before)"); axs[0].axis("off")
axs[1].imshow(c, cmap="gray"); axs[1].set_title("CLAHE (after)"); axs[1].axis("off")
axs[2].hist(g[fov], 256, range=(0, 255), color="#2a78d6", alpha=.75, label="before")
axs[2].hist(c[fov], 256, range=(0, 255), color="#eb6834", alpha=.6, label="after")
axs[2].set_title("histogram inside FOV"); axs[2].legend(frameon=False)
y0, x0 = 250, 330
axs[3].imshow(np.hstack([g[y0:y0+120, x0:x0+120], c[y0:y0+120, x0:x0+120]]), cmap="gray")
axs[3].set_title("zoom: before | after"); axs[3].axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
### Morphology: what opening and closing remove

Red pixels are removed by the clean-up (isolated speckle), green pixels are added (one-pixel gaps bridged).
""")

code(r"""
t = np.asarray(Image.open(f"{SD}/01_test_4_threshold.png")) > 127
f = np.asarray(Image.open(f"{SD}/01_test_5_final.png")) > 127
diff = np.ones(t.shape + (3,)); diff[t & f] = (.15, .15, .15); diff[t & ~f] = (.89, .29, .28); diff[~t & f] = (0, .51, 0)
y0, x0 = 180, 300
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(t[y0:y0+200, x0:x0+200], cmap="gray"); axs[0].set_title("4. after threshold (zoom)")
axs[1].imshow(f[y0:y0+200, x0:x0+200], cmap="gray"); axs[1].set_title("5. after opening + closing (zoom)")
axs[2].imshow(diff); axs[2].set_title(f"whole image: removed {int((t&~f).sum())} px (red), added {int((~t&f).sum())} px (green)")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()
""")

code(r"""
stage_grid("im0077", f"{STARE}/images/im0077.ppm")   # same parameters on STARE
""")

md(r"""
## 6. Accuracy against the expert annotations

Dice = 2TP / (2TP + FP + FN), counted inside the FOV. In the overlays, black = vessel found correctly, orange =
false positive, blue = missed vessel.
""")

code(r"""
MD = f"{OUT}/masks"; os.makedirs(MD, exist_ok=True)
acc = {}
for kind, root in [("drive-test", DRIVE), ("stare", STARE)]:
    print(run(f"{BIN}/frangi_omp eval {kind} {root} {PARAMS} {OUT}/acc_{kind}.csv --masks {MD}", quiet=True))
    d = pd.read_csv(f"{OUT}/acc_{kind}.csv"); acc[kind] = d[~d.image.isin(["MEAN", "POOLED"])]
fig, axs = plt.subplots(1, 2, figsize=(15, 3.6), sharey=True)
for ax, (kind, d) in zip(axs, acc.items()):
    ax.bar(d.image.str.replace("_test", ""), d.dice, color="#2a78d6", width=.7)
    ax.axhline(d.dice.mean(), color="#0b0b0b", lw=1, ls="--"); ax.text(len(d) - .5, d.dice.mean() + .01, f"mean {d.dice.mean():.3f}", ha="right")
    ax.set_title(f"{kind}: Dice per image"); ax.tick_params(axis="x", rotation=90); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()
summary = pd.DataFrame([{"set": k, "Dice": f"{d.dice.mean():.4f} ± {d.dice.std():.4f}", "sensitivity": round(d.sens.mean(), 4),
                         "specificity": round(d.spec.mean(), 4), "accuracy": round(d.acc.mean(), 4)} for k, d in acc.items()])
lit = pd.DataFrame([{"set": "LightVesselNet (75K params), DRIVE", "Dice": "0.8070", "sensitivity": 0.8189},
                    {"set": "LFA-Net (0.11M params), DRIVE", "Dice": "0.8328"},
                    {"set": "LVS-Net (0.71M params), DRIVE", "Dice": "0.8644"}])
display(pd.concat([summary, lit], ignore_index=True).fillna("-"))
""")

code(r"""
def overlay(kind, name, gt_path):
    pm = np.asarray(Image.open(f"{MD}/{name}_mask.png")) > 127
    gt = np.asarray(Image.open(gt_path).convert("L")) > 127
    o = np.ones(gt.shape + (3,)); o[pm & gt] = (.04, .04, .04); o[pm & ~gt] = (.92, .41, .2); o[~pm & gt] = (.16, .47, .84)
    return o
d = acc["drive-test"].sort_values("dice"); best, worst = d.iloc[-1].image, d.iloc[0].image
imgs, titles = [], []
for nm in [best, worst]:
    i = nm.split("_")[0]
    imgs += [np.asarray(Image.open(f"{DRIVE}/test/images/{nm}.tif").convert("RGB")), np.asarray(Image.open(f"{MD}/{nm}_vesselness.png")),
             overlay("drive-test", nm, f"{DRIVE}/test/1st_manual/{i}_manual1.gif")]
    titles += [f"{nm} (Dice {d.set_index('image').dice[nm]:.3f})", "vesselness V", "TP black / FP orange / FN blue"]
show(imgs, titles, ncol=3, size=4.2)
""")

md(r"""
## 7. Correctness: serial and OpenMP give identical output

Every OpenMP configuration (four designs, each at every thread count) writes the CLAHE output, the vesselness map
V (float32) and the mask, which are compared byte for byte with the serial build. Each output element is computed
by exactly one thread, in the serial summation order, so OpenMP changes *when* values are computed but never
*how*. For contrast, the same comparison is made between the serial build and a serial build with floating-point
contraction (FMA) disabled. That one change in the compiler's arithmetic *does* alter V.
""")

code(r"""
run(f"cd {CODE} && make OPT='-O3 -march=native' FPFLAGS=-ffp-contract=off BIN=bin_nofma >/dev/null", quiet=True)
VD = f"{OUT}/verify"; NV = 5
def dump(binary, d, extra=""):
    os.makedirs(d, exist_ok=True)
    run(f"{binary} dump drive-test {DRIVE} {PARAMS} {d} --images {NV} {extra}", quiet=True)
def compare(a, b):
    last = run(f"{BIN}/frangi_serial compare {a} {b}", quiet=True).strip().splitlines()[-1].split(",")
    return float(last[1]), float(last[2]), float(last[3]), float(last[4])
dump(f"{BIN}/frangi_serial", f"{VD}/serial")
dump(f"{CODE}/bin_nofma/frangi_serial", f"{VD}/serial_nofma")
configs = [(v, nt) for v in ["base", "dynamic", "atomic", "scalepar"] for nt in sorted({1, 2, 4, 2 * NP})]
bad = []
for v, nt in configs:
    d = f"{VD}/{v}_{nt}"; dump(f"{BIN}/frangi_omp", d, f"--nt {nt} --variant {v}")
    r = compare(f"{VD}/serial", d)
    if r[:3] != (1.0, 1.0, 1.0): bad.append((v, nt, r))
fma = compare(f"{VD}/serial", f"{VD}/serial_nofma")
npx = NV * 565 * 584
display(Markdown(
    f"**Serial vs OpenMP:** {len(configs) - len(bad)} of {len(configs)} configurations are bit-identical "
    f"({len(configs) * npx / 1e6:.1f} million pixel values compared for each of CLAHE, V and the mask).  \n"
    f"**Serial FMA vs serial no-FMA:** only {100 * fma[2]:.1f}% of V values are bit-identical "
    f"(max |dV| = {fma[3]:.1e}), yet {100 * fma[1]:.2f}% of mask pixels agree."))
if bad: display(pd.DataFrame(bad, columns=["variant", "threads", "result"]))

name = "01_test"
rd = lambda d: np.fromfile(f"{VD}/{d}/{name}.V.f32", np.float32).reshape(584, 565)
V0, Vomp, Vnf = rd("serial"), rd(f"base_{max(n for _, n in configs)}"), rd("serial_nofma")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
axs[0].imshow(np.sqrt(V0 / V0.max()), cmap="gray"); axs[0].set_title("V, serial build")
vmax = np.abs(Vnf - V0).max()
for ax, D, t in [(axs[1], np.abs(Vomp - V0), f"|V(OpenMP, p={max(n for _, n in configs)}) - V(serial)|"),
                 (axs[2], np.abs(Vnf - V0), "|V(serial, no FMA) - V(serial)|")]:
    im = ax.imshow(D, cmap="magma", vmin=0, vmax=vmax); ax.set_title(f"{t}\nmax = {D.max():.1e}")
for a in axs: a.axis("off")
fig.colorbar(im, ax=axs[1:], fraction=.025, pad=.02, label="|dV|")
plt.show()
""")

md(r"""
## 8. Execution time of every stage at every thread count (live, on this VM)

The C code times each stage with `omp_get_wtime` (file I/O and decoding excluded). To keep Colab's background
activity and slow clock drift from favouring one build, the serial and OpenMP runs are **interleaved** over several
rounds, and the **median** of all runs is reported. Raise `ROUNDS`, `REPS` or `NIMG` for tighter numbers.

* **speedup** S(p) = T_serial / T(p), against the pragma-free serial build
* **efficiency** E = S / p
* **Karp-Flatt serial fraction** e = (1/S - 1/p) / (1 - 1/p), shown only where S > 1; a rising e means overhead, not a fixed serial part, limits scaling
""")

code(r"""
NIMG, WARM, REPS, ROUNDS = 5, 2, 4, 3
THREADS = sorted({1, 2, 4, NP, 2 * NP})
TL = ",".join(map(str, THREADS))
topo = run("lscpu -p=CORE,SOCKET | grep -v '^#'", quiet=True).split()
PHYS = len(set(topo))
quota = run("cat /sys/fs/cgroup/cpu.max 2>/dev/null || true", quiet=True).split()
CPU_QUOTA = float(quota[0]) / float(quota[1]) if len(quota) == 2 and quota[0] != "max" else None
for f in ["b_serial.csv", "b_omp.csv"]:
    if os.path.exists(f"{OUT}/{f}"): os.remove(f"{OUT}/{f}")
for r in range(ROUNDS):   # interleave the builds
    run(f"{BIN}/frangi_serial bench drive-test {DRIVE} {PARAMS} {OUT}/b_serial.csv --nt 1 --warm {WARM} --reps {REPS} --images {NIMG} --append", quiet=True)
    run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {OUT}/b_omp.csv --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG} --append", quiet=True)
ser, omp = pd.read_csv(f"{OUT}/b_serial.csv"), pd.read_csv(f"{OUT}/b_omp.csv")
display(Markdown(
    f"**This VM:** {PHYS} physical core(s), {NP} logical CPUs"
    + (f", CPU quota {CPU_QUOTA:g}" if CPU_QUOTA else "") + f". {len(ser)} serial and {len(omp)} OpenMP timed runs.  \n"
    + (f"With a single physical core the two logical CPUs share one set of execution units, so the best possible "
       f"speedup is only Hyper-Threading's gain (typically 0-15%), and threads beyond {NP} only add switching overhead. "
       f"Expect speedups near 1 here; section 9 shows the same experiment on a 2-core machine."
       if PHYS == 1 else f"The best possible speedup is about {PHYS}x, plus a little from Hyper-Threading.")))
""")

code(r"""
def kf(S, p): return (1 / S - 1 / p) / (1 - 1 / p) if p > 1 and S > 1 else np.nan

def stage_table(ser, omp, stat="median"):
    cols = STAGES + ["total"]
    t = pd.DataFrame({"serial": getattr(ser[cols], stat)()})
    for p, g in omp.groupby("threads"): t[p] = getattr(g[cols], stat)()
    return t * 1e3   # ms

def report(ser, omp, title, phys, ncpu, stat="median"):
    T = stage_table(ser, omp, stat); P = sorted(omp.threads.unique())
    S = T[P].rdiv(T["serial"], axis=0)
    display(Markdown(f"### {title}: {stat} time per image (ms)"))
    Tn = T.rename(index=LABEL); Tn.columns = ["serial"] + [f"p={p}" for p in P]
    display(Tn.style.format("{:.2f}"))
    whole = pd.DataFrame({"threads": P, "time (ms)": T.loc["total", P].values, "speedup": S.loc["total"].values,
                          "efficiency": S.loc["total"].values / np.array(P),
                          "Karp-Flatt e": [kf(S.loc["total", p], p) for p in P],
                          "images/s": 1e3 / T.loc["total", P].values})
    display(whole.style.format({"time (ms)": "{:.1f}", "speedup": "{:.2f}", "efficiency": "{:.2f}",
                                "Karp-Flatt e": lambda v: "n/a (no speedup)" if pd.isna(v) else f"{v:.3f}", "images/s": "{:.1f}"}))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = ["serial"] + [f"{p} thread" + ("s" if p > 1 else "") for p in P]
    vals = [T["serial"]] + [T[p] for p in P]
    left = np.zeros(len(rows))
    for s in STAGES:
        w = np.array([v[s] for v in vals])
        a0.barh(rows, w, left=left, color=COL[s], label=LABEL[s], edgecolor="white", height=.62); left += w
    for i, x in enumerate(left):
        a0.text(x + left.max() * .01, i, f"{x:.0f} ms", va="center", fontsize=9)
    a0.invert_yaxis(); a0.set_xlim(0, left.max() * 1.15); a0.set_xlabel("time per image (ms)")
    a0.set_title("where the time goes", loc="left"); a0.grid(axis="y", visible=False)
    a0.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.14))

    ideal = [min(p, phys) for p in P]
    a1.axhline(1, color="#52514e", lw=1)
    a1.plot(P, ideal, "--", color="#52514e", lw=1.2, label=f"ideal on {phys} physical core" + ("s" if phys > 1 else ""))
    if max(P) > ncpu:
        a1.axvspan(ncpu * 1.06, max(P) * 1.3, color="#e4e3df", alpha=.6, lw=0)
        a1.text(max(P), 0.08, "oversubscribed", ha="right", fontsize=8, color="#52514e")
    for s in ["clahe", "gauss", "vess", "morph"]:
        a1.plot(P, S.loc[s].values, "o-", color=COL[s], lw=1.4, ms=4, label=LABEL[s])
    a1.plot(P, S.loc["total"].values, "o-", color=COL["total"], lw=2.8, ms=6, label="whole pipeline")
    for p, v in zip(P, S.loc["total"].values):
        a1.annotate(f"{v:.2f}x", (p, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, weight="bold")
    a1.set_xscale("log", base=2); a1.set_xticks(P); a1.set_xticklabels(P); a1.set_xlim(min(P) / 1.3, max(P) * 1.3)
    a1.set_ylim(0, max(phys + .6, S.values.max() * 1.2)); a1.set_xlabel("OpenMP threads"); a1.set_ylabel("speedup vs serial")
    a1.set_title("speedup by stage", loc="left")
    a1.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.14))
    plt.tight_layout(); plt.show()

    from matplotlib.colors import TwoSlopeNorm
    fig, (h0, h1) = plt.subplots(1, 2, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    hi = max(S.values.max(), 1.05); lo = min(S.values.min(), .95)
    im = h0.imshow(S.values, cmap="RdBu", norm=TwoSlopeNorm(vmin=lo, vcenter=1, vmax=hi), aspect="auto")
    h0.set_xticks(range(len(P))); h0.set_xticklabels([f"p={p}" for p in P])
    h0.set_yticks(range(len(S))); h0.set_yticklabels([LABEL[i] for i in S.index])
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = S.values[i, j]; strong = abs(v - 1) > .45 * max(hi - 1, 1 - lo)
            h0.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if strong else "#0b0b0b")
    h0.grid(False); h0.set_title("speedup per stage (blue = faster, red = slower than serial)", loc="left")
    fig.colorbar(im, ax=h0, fraction=.03, pad=.02)
    tp = [1e3 / T.loc["total", "serial"]] + list(1e3 / T.loc["total", P].values)
    bars = h1.bar(["serial"] + [f"p={p}" for p in P], tp, color=["#52514e"] + ["#2a78d6"] * len(P), width=.6)
    for b, v in zip(bars, tp): h1.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    h1.set_ylim(0, max(tp) * 1.18); h1.set_ylabel("images per second"); h1.set_title("throughput", loc="left")
    plt.tight_layout(); plt.show()
    return T, S

T_live, S_live = report(ser, omp, "This Colab VM", PHYS, NP)
""")

md(r"""
### Thread placement: where do the threads land?

With `OMP_PLACES=threads OMP_PROC_BIND=close`, libgomp lists the hardware threads core by core, so two threads go
onto the **two hyperthreads of one core**. `spread` over `cores` puts them on different physical cores. On a VM with a
single physical core every placement uses the same core, so the three lines should coincide here; on the paper's
2-core machine close binding at p=2 gave 1.08x against 1.98x for spread (section 9).
""")

code(r"""
place = {"close, threads": "OMP_PLACES=threads OMP_PROC_BIND=close",
         "spread, cores": "OMP_PLACES=cores OMP_PROC_BIND=spread",
         "no binding": "OMP_PROC_BIND=false"}
rows = []
for i, (lab, envs) in enumerate(place.items()):
    f = f"{OUT}/b_place_{i}.csv"
    if os.path.exists(f): os.remove(f)
    run(f"env -u OMP_PLACES -u OMP_PROC_BIND {envs} {BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {f} --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG}", quiet=True)
    for p, g in pd.read_csv(f).groupby("threads"):
        rows.append({"placement": lab, "threads": p, "speedup": ser.total.median() / g.total.median()})
pl = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.axhline(1, color="#52514e", lw=1)
ax.plot(THREADS, [min(p, PHYS) for p in THREADS], "--", color="#52514e", lw=1.2, label=f"ideal on {PHYS} core(s)")
for (lab, g), c in zip(pl.groupby("placement", sort=False), ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.plot(g.threads, g.speedup, "o-", color=c, label=lab, lw=2)
ax.set_xscale("log", base=2); ax.set_xticks(THREADS); ax.set_xticklabels(THREADS); ax.set_ylim(0, max(PHYS + .6, pl.speedup.max() * 1.2))
ax.set_xlabel("OpenMP threads"); ax.set_ylabel("speedup vs serial"); ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.16))
plt.tight_layout(); plt.show()
t = pl.pivot(index="placement", columns="threads", values="speedup"); t.columns = [f"p={c}" for c in t.columns]
display(t.style.format("{:.2f}"))
""")

md(r"""
### Alternative OpenMP designs

* **dynamic**: CLAHE tile loop with `schedule(dynamic,1)` instead of `static`
* **atomic**: CLAHE histograms built pixel-parallel with `#pragma omp atomic` instead of one thread per tile
* **scalepar**: the four Frangi scales run concurrently (one thread each) instead of parallelising inside each scale

Bars show the variant's stage time divided by the base design's; above 1 means the variant is slower.
""")

code(r"""
rows = []
for v, stage in [("dynamic", "clahe"), ("atomic", "clahe"), ("scalepar", "frangi")]:
    f = f"{OUT}/b_{v}.csv"
    if os.path.exists(f): os.remove(f)
    run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {f} --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG} --variant {v}", quiet=True)
    d = pd.read_csv(f)
    pick = (lambda x: x.gauss + x.vess) if stage == "frangi" else (lambda x: x[stage])
    for p, g in d.groupby("threads"):
        b = omp[omp.threads == p]
        rows.append({"variant": v, "stage": "Frangi (gauss + vess)" if stage == "frangi" else "CLAHE", "threads": p,
                     "base (ms)": pick(b).median() * 1e3, "variant (ms)": pick(g).median() * 1e3})
va = pd.DataFrame(rows); va["variant / base"] = va["variant (ms)"] / va["base (ms)"]
fig, ax = plt.subplots(figsize=(9, 4)); w = .26
for i, (v, c) in enumerate(zip(["dynamic", "atomic", "scalepar"], ["#2a78d6", "#eb6834", "#1baf7a"])):
    g = va[va.variant == v]
    bars = ax.bar(np.arange(len(g)) + (i - 1) * w, g["variant / base"], w, color=c, label=v)
    for b, val in zip(bars, g["variant / base"]): ax.text(b.get_x() + w / 2, val + .02, f"{val:.2f}", ha="center", fontsize=8)
ax.axhline(1, color="#0b0b0b", lw=1); ax.set_xticks(range(len(THREADS))); ax.set_xticklabels([f"p={p}" for p in THREADS])
ax.set_ylim(0, va["variant / base"].max() * 1.2); ax.set_ylabel("time relative to base design")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12)); plt.tight_layout(); plt.show()
display(va.style.format({"base (ms)": "{:.2f}", "variant (ms)": "{:.2f}", "variant / base": "{:.2f}"}))
""")

md(r"""
### Image size: does the parallel gain hold for larger images?

The image is bilinearly upsampled to 2x and 4x linear size (1.3 and 5.3 megapixels) before timing; set `BIG = True`
to add 8x (21 megapixels, about a minute per serial run here).
""")

code(r"""
BIG = False
scales = [1, 2, 4] + ([8] if BIG else [])
for f in ["s_ser.csv", "s_omp.csv"]:
    if os.path.exists(f"{OUT}/{f}"): os.remove(f"{OUT}/{f}")
for sc in scales:
    r = 5 if sc <= 2 else 2
    run(f"{BIN}/frangi_serial bench drive-test {DRIVE} {PARAMS} {OUT}/s_ser.csv --nt 1 --scales {sc} --warm 1 --reps {r} --images 1 --append", quiet=True)
    run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {OUT}/s_omp.csv --nt {TL} --scales {sc} --warm 1 --reps {r} --images 1 --append", quiet=True)
ss, so = pd.read_csv(f"{OUT}/s_ser.csv"), pd.read_csv(f"{OUT}/s_omp.csv")
first = ss.groupby("scale")[["width", "height"]].first()
mp = (first.width * first.height / 1e6).values
ts = ss.groupby("scale").total.median().values
fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 4.2))
a0.loglog(mp, ts * 1e3, "o-", color="#0b0b0b", lw=2.4, label="serial")
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
for (p, g), c in zip(so.groupby("threads"), cols):
    tp = g.groupby("scale").total.median().values
    a0.loglog(mp, tp * 1e3, "o-", color=c, label=f"p={p}")
    a1.semilogx(mp, ts / tp, "o-", color=c, label=f"p={p}", lw=2)
a0.set_xlabel("megapixels"); a0.set_ylabel("ms per image"); a0.set_title("time grows with image size", loc="left")
a1.axhline(1, color="#52514e", lw=1); a1.set_ylim(0, max(PHYS + .6, (ts / so.groupby(["threads", "scale"]).total.median().unstack().values).max() * 1.2))
a1.set_xlabel("megapixels"); a1.set_ylabel("speedup vs serial"); a1.set_title("speedup vs image size", loc="left")
for a in (a0, a1):
    a.set_xticks(mp); a.set_xticklabels([f"{m:.2f}" for m in mp]); a.minorticks_off()
    a.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16))
plt.tight_layout(); plt.show()
c1 = np.asarray(Image.open(f"{DRIVE}/test/images/01_test.tif").convert("RGB"))[250:310, 330:390]
show([c1, np.asarray(Image.fromarray(c1).resize((240, 240), Image.BILINEAR))], ["native crop (60 x 60 px)", "same crop at 4x (240 x 240 px)"], ncol=2, size=3.5)
""")

md(r"""
## 9. The paper's measurements (Kaggle CPU: 2 physical cores, 4 hyperthreads)

The same harness, run on all 20 DRIVE test images with 5 warm-up and 30 timed runs each, on a Kaggle CPU VM
(Intel Xeon @ 2.20 GHz, 2 cores x 2 threads). These CSVs are in `runs/full/results` of the repository.
""")

code(r"""
if os.path.exists(f"{KAGGLE}/bench_threads_omp.csv"):
    print(open(f"{KAGGLE}/env_lscpu_e.txt").read() if os.path.exists(f"{KAGGLE}/env_lscpu_e.txt") else "")
    kser, komp = pd.read_csv(f"{KAGGLE}/bench_threads_serial.csv"), pd.read_csv(f"{KAGGLE}/bench_threads_omp.csv")
    T_k, S_k = report(kser, komp, "Kaggle (paper, 2 cores x 2 threads)", 2, 4, stat="mean")
    figs = sorted(glob.glob(f"{ROOT}/runs/full/analysis/figures/*.png"))
    if figs: show(figs, [os.path.basename(f) for f in figs], ncol=3, size=4.5, cmap=None)
else:
    print("Kaggle results not found in the repository.")
""")

md(r"""
## 10. What the numbers say

* **The Gaussian-derivative stage dominates** (about 70% of serial time); the eigen/vesselness stage is next. Optimising
  or parallelising anything else barely moves the total.
* **OpenMP is limited by physical cores.** Two threads on two hyperthreads of one core give little; two threads on two
  cores give nearly 2x; the second hyperthread per core adds only a few percent; oversubscription adds nothing.
* **Design matters per stage:** one thread per CLAHE tile beats atomic histograms; parallelising inside each Frangi
  scale beats running the four scales concurrently, because the sigma = 4 kernels are about 4x longer than sigma = 1.
* **Correctness is exact:** every thread count and design variant reproduces the serial output bit for bit.
* **Accuracy:** a training-free classical pipeline, below lightweight deep networks on Dice but with parameters that
  transfer to STARE unchanged.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[1], len(cells), "cells")
