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

The repository is private. Add a GitHub token with read access as a Colab secret named `GITHUB_TOKEN`
(key icon in the left sidebar), or paste it when prompted. The token is used for the clone only and is removed
from the git remote afterwards.
""")

code(r"""
import os, subprocess, getpass
REPO = "hansel-cooks/pcap_claude_vedang"
ROOT = "/content/pcap_claude_vedang"
if not os.path.exists(ROOT):
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
    if not tok:
        tok = getpass.getpass("GitHub token (leave empty if the repo is public): ").strip()
    url = f"https://x-access-token:{tok}@github.com/{REPO}.git" if tok else f"https://github.com/{REPO}.git"
    r = subprocess.run(["git", "clone", "--depth", "1", url, ROOT], capture_output=True, text=True)
    print(r.stdout, r.stderr.replace(tok, "***") if tok else r.stderr)
    subprocess.run(["git", "-C", ROOT, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"])
    del tok
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

For each thread count the OpenMP build writes the CLAHE output, the vesselness map V (float32) and the mask; they are
compared byte by byte with the serial build. Every output element is computed by exactly one thread in the same
order, so even floating-point results match bit for bit.
""")

code(r"""
VD = f"{OUT}/verify"; NV = 5
ref = f"{VD}/serial"; os.makedirs(ref, exist_ok=True)
run(f"{BIN}/frangi_serial dump drive-test {DRIVE} {PARAMS} {ref} --images {NV}", quiet=True)
rows = []
for variant in ["base", "dynamic", "atomic", "scalepar"]:
    for nt in sorted({1, 2, 4, 2 * NP}):
        dd = f"{VD}/{variant}_{nt}"; os.makedirs(dd, exist_ok=True)
        run(f"{BIN}/frangi_omp dump drive-test {DRIVE} {PARAMS} {dd} --nt {nt} --variant {variant} --images {NV}", quiet=True)
        last = run(f"{BIN}/frangi_serial compare {ref} {dd}", quiet=True).strip().splitlines()[-1].split(",")
        rows.append({"variant": variant, "threads": nt, "CLAHE identical": float(last[1]), "mask identical": float(last[2]),
                     "V bit-identical": float(last[3]), "max |dV|": float(last[4])})
display(pd.DataFrame(rows))
""")

md(r"""
## 8. Execution time of every stage at every thread count (live, on this VM)

Each configuration runs a few untimed warm-up iterations, then timed iterations per image, with per-stage timers
inside the C code (`omp_get_wtime`). File I/O and decoding are excluded. Increase `REPS`/`NIMG` for tighter numbers.

* **speedup** S(p) = T_serial / T(p), against the pragma-free serial build
* **efficiency** E = S / p
* **Karp-Flatt serial fraction** e = (1/S - 1/p) / (1 - 1/p): if e grows with p, overhead (not a fixed serial part) limits scaling
""")

code(r"""
NIMG, WARM, REPS = 5, 2, 10
THREADS = sorted({1, 2, 4, NP, 2 * NP})
TL = ",".join(map(str, THREADS))
BIND = {"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"}
run(f"{BIN}/frangi_serial bench drive-test {DRIVE} {PARAMS} {OUT}/b_serial.csv --nt 1 --warm {WARM} --reps {REPS} --images {NIMG}", quiet=True)
run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {OUT}/b_omp.csv --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG}", env=BIND, quiet=True)
ser, omp = pd.read_csv(f"{OUT}/b_serial.csv"), pd.read_csv(f"{OUT}/b_omp.csv")
print(len(ser), "serial and", len(omp), "OpenMP timed runs")
""")

code(r"""
def kf(S, p): return (1 / S - 1 / p) / (1 - 1 / p) if p > 1 else np.nan

def stage_table(ser, omp):
    cols = STAGES + ["total"]
    t = pd.DataFrame({"serial": ser[cols].mean()})
    for p, g in omp.groupby("threads"): t[f"p={p}"] = g[cols].mean()
    return t * 1e3   # ms

def report(ser, omp, title):
    T = stage_table(ser, omp); P = sorted(omp.threads.unique())
    S = T.iloc[:, 1:].rdiv(T["serial"], axis=0); S.columns = P
    display(Markdown(f"### {title}: mean time per image (ms)"))
    display(T.rename(index=LABEL).style.format("{:.2f}").background_gradient(axis=None, cmap="Blues"))
    whole = pd.DataFrame({"threads": P, "time (ms)": [T.loc["total", f"p={p}"] for p in P],
                          "speedup": S.loc["total"].values, "efficiency": S.loc["total"].values / np.array(P),
                          "Karp-Flatt e": [kf(S.loc["total", p], p) for p in P],
                          "images/s": [1e3 / T.loc["total", f"p={p}"] for p in P]})
    display(whole.style.format({"time (ms)": "{:.1f}", "speedup": "{:.2f}", "efficiency": "{:.2f}", "Karp-Flatt e": "{:.3f}", "images/s": "{:.1f}"}))

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    labels = ["serial"] + [str(p) for p in P]; bottom = np.zeros(len(labels))
    for s in STAGES:
        h = T.loc[s].values; axs[0].bar(labels, h, bottom=bottom, color=COL[s], label=LABEL[s], edgecolor="white"); bottom += h
    for i, b in enumerate(bottom): axs[0].text(i, b * 1.01, f"{b:.0f}", ha="center", va="bottom", fontsize=9)
    axs[0].set_title("time per image by stage"); axs[0].set_xlabel("threads"); axs[0].set_ylabel("ms"); axs[0].legend(fontsize=8, frameon=False)
    axs[1].plot(P, P, "--", color="#52514e", lw=1, label="ideal")
    for s in ["clahe", "gauss", "vess", "morph", "total"]:
        axs[1].plot(P, S.loc[s].values, "o-", color=COL[s], lw=2.4 if s == "total" else 1.6, label=LABEL[s])
    axs[1].set_xscale("log", base=2); axs[1].set_xticks(P); axs[1].set_xticklabels(P); axs[1].set_ylim(0, max(4.2, S.values.max() * 1.15))
    axs[1].set_title("speedup vs serial"); axs[1].set_xlabel("threads"); axs[1].legend(fontsize=8, frameon=False)
    im = axs[2].imshow(S.values, cmap="Blues", aspect="auto", vmin=0)
    axs[2].set_xticks(range(len(P))); axs[2].set_xticklabels(P); axs[2].set_yticks(range(len(S))); axs[2].set_yticklabels([LABEL[i] for i in S.index])
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            axs[2].text(j, i, f"{S.values[i, j]:.2f}", ha="center", va="center", color="white" if S.values[i, j] > S.values.max() * .6 else "#0b0b0b", fontsize=9)
    axs[2].set_title("speedup heatmap (stage x threads)"); axs[2].set_xlabel("threads"); axs[2].grid(False)
    plt.tight_layout(); plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    for s in ["clahe", "gauss", "vess", "morph", "total"]:
        axs[0].plot(P[1:], [kf(S.loc[s, p], p) for p in P[1:]], "o-", color=COL[s], label=LABEL[s])
    axs[0].set_xscale("log", base=2); axs[0].set_xticks(P[1:]); axs[0].set_xticklabels(P[1:]); axs[0].set_title("Karp-Flatt serial fraction"); axs[0].legend(fontsize=8, frameon=False)
    axs[1].bar(labels, [1e3 / T.loc["total", "serial"]] + [1e3 / T.loc["total", f"p={p}"] for p in P], color="#2a78d6")
    axs[1].set_title("throughput (images / second)"); axs[1].set_xlabel("threads")
    plt.tight_layout(); plt.show()
    return T, S

T_live, S_live = report(ser, omp, "This Colab VM")
""")

md(r"""
### Thread placement: where do the threads land?

With `OMP_PLACES=threads OMP_PROC_BIND=close`, libgomp lists the hardware threads core by core, so two threads go
onto the **two hyperthreads of one core**. `spread` over `cores` puts them on different physical cores. On a VM with a
single physical core all placements behave the same.
""")

code(r"""
place = {"close, threads": BIND, "spread, cores": {"OMP_PLACES": "cores", "OMP_PROC_BIND": "spread"}, "no binding": {"OMP_PROC_BIND": "false"}}
rows = []
for lab, env in place.items():
    f = f"{OUT}/b_place_{lab.replace(', ', '_')}.csv"
    e = {k: v for k, v in env.items()}
    run(f"env -u OMP_PLACES -u OMP_PROC_BIND {' '.join(k + '=' + v for k, v in e.items())} {BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {f} --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG}", quiet=True)
    d = pd.read_csv(f)
    for p, g in d.groupby("threads"): rows.append({"placement": lab, "threads": p, "speedup": ser.total.mean() / g.total.mean()})
pl = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for (lab, g), c in zip(pl.groupby("placement", sort=False), ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.plot(g.threads, g.speedup, "o-", color=c, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(THREADS); ax.set_xticklabels(THREADS); ax.set_xlabel("threads"); ax.set_ylabel("speedup vs serial"); ax.legend(frameon=False)
plt.show(); display(pl.pivot(index="placement", columns="threads", values="speedup").round(2))
""")

md(r"""
### Alternative OpenMP designs

* **dynamic**: CLAHE tile loop with `schedule(dynamic,1)` instead of `static`
* **atomic**: CLAHE histograms built pixel-parallel with `#pragma omp atomic` instead of one thread per tile
* **scalepar**: the four Frangi scales run concurrently (one thread each) instead of parallelising inside each scale
""")

code(r"""
rows = []
for v, stage in [("dynamic", "clahe"), ("atomic", "clahe"), ("scalepar", "frangi")]:
    run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {OUT}/b_{v}.csv --nt {TL} --warm {WARM} --reps {REPS} --images {NIMG} --variant {v}", env=BIND, quiet=True)
    d = pd.read_csv(f"{OUT}/b_{v}.csv")
    for p, g in d.groupby("threads"):
        b = omp[omp.threads == p]
        f = (lambda x: x.gauss + x.vess) if stage == "frangi" else (lambda x: x[stage])
        rows.append({"variant": v, "stage": stage, "threads": p, "base (ms)": f(b).mean() * 1e3, "variant (ms)": f(g).mean() * 1e3})
va = pd.DataFrame(rows); va["ratio (variant / base)"] = va["variant (ms)"] / va["base (ms)"]
display(va.style.format({"base (ms)": "{:.2f}", "variant (ms)": "{:.2f}", "ratio (variant / base)": "{:.2f}"}))
fig, ax = plt.subplots(figsize=(8, 3.6)); w = .25
for i, (v, c) in enumerate(zip(["dynamic", "atomic", "scalepar"], ["#2a78d6", "#eb6834", "#1baf7a"])):
    g = va[va.variant == v]; ax.bar(np.arange(len(g)) + (i - 1) * w, g["ratio (variant / base)"], w, color=c, label=v)
ax.axhline(1, color="#0b0b0b", lw=1); ax.set_xticks(range(len(THREADS))); ax.set_xticklabels(THREADS); ax.set_xlabel("threads")
ax.set_ylabel("time relative to base design"); ax.legend(frameon=False); plt.show()
""")

md(r"""
### Image size: does the parallel gain hold for larger images?

The image is bilinearly upsampled to 2x and 4x linear size (1.3 and 5.3 megapixels) before timing; set `BIG = True`
to add 8x (21 megapixels, about a minute per serial run here).
""")

code(r"""
BIG = False
scales = [1, 2, 4] + ([8] if BIG else [])
run(f"rm -f {OUT}/s_ser.csv {OUT}/s_omp.csv")
for sc in scales:
    r = 5 if sc <= 2 else 2
    run(f"{BIN}/frangi_serial bench drive-test {DRIVE} {PARAMS} {OUT}/s_ser.csv --nt 1 --scales {sc} --warm 1 --reps {r} --images 1 --append", quiet=True)
    run(f"{BIN}/frangi_omp bench drive-test {DRIVE} {PARAMS} {OUT}/s_omp.csv --nt {TL} --scales {sc} --warm 1 --reps {r} --images 1 --append", env=BIND, quiet=True)
ss, so = pd.read_csv(f"{OUT}/s_ser.csv"), pd.read_csv(f"{OUT}/s_omp.csv")
mp = ss.groupby("scale").apply(lambda g: g.width.iloc[0] * g.height.iloc[0] / 1e6)
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
axs[0].loglog(mp.values, ss.groupby("scale").total.mean().values * 1e3, "o-", color="#0b0b0b", label="serial")
for p, g in so.groupby("threads"):
    axs[0].loglog(mp.values, g.groupby("scale").total.mean().values * 1e3, "o-", label=f"p={p}")
    axs[1].semilogx(mp.values, ss.groupby("scale").total.mean().values / g.groupby("scale").total.mean().values, "o-", label=f"p={p}")
axs[0].set_xlabel("megapixels"); axs[0].set_ylabel("ms per image"); axs[0].legend(frameon=False); axs[0].set_title("time vs image size")
axs[1].set_xlabel("megapixels"); axs[1].set_ylabel("speedup vs serial"); axs[1].legend(frameon=False); axs[1].set_title("speedup vs image size")
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
    T_k, S_k = report(kser, komp, "Kaggle (paper)")
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
