"""Turn the Kaggle CSVs into tables (CSV + LaTeX) and figures for the paper.

usage: analyze.py <results_dir> <out_dir> <drive_root>
Python is used here only for statistics and plotting; every number comes from the C runs.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT, DRIVE = sys.argv[1:4]
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

STAGES = ["green", "clahe", "gauss", "vess", "thresh", "morph"]
LABEL = {"green": "Green channel", "clahe": "CLAHE", "gauss": "Gaussian derivatives",
         "vess": "Eigen + vesselness", "thresh": "Threshold", "morph": "Morphology", "total": "Total"}
# reference categorical palette, fixed slot order (validated light-mode set)
COL = {"green": "#2a78d6", "clahe": "#eb6834", "gauss": "#1baf7a", "vess": "#eda100",
       "thresh": "#e87ba4", "morph": "#008300", "total": "#0b0b0b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLW = 3.4  # IEEE column width in inches


def rd(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save_tex(df, name, caption_cols=None, fmt=None):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        cols = list(df.columns)
        f.write("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(caption_cols or cols) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if fmt and c in fmt and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                    cells.append(fmt[c].format(v))
                else:
                    cells.append("--" if (isinstance(v, float) and pd.isna(v)) else str(v))
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def per_image(df, keys):
    """mean over repetitions for each image and configuration"""
    cols = STAGES + ["total"]
    return df.groupby(keys + ["image"])[cols].mean().reset_index()


def kf(S, p):
    return (1.0 / S - 1.0 / p) / (1.0 - 1.0 / p) if p > 1 else np.nan


summary = {}

# ------------------------------------------------------------------ environment
env = {}
for f in ["env_misc.txt", "env_build.txt", "env_lscpu.txt"]:
    p = os.path.join(RES, f)
    if os.path.exists(p):
        env[f] = open(p).read()
with open(os.path.join(TAB, "environment.txt"), "w") as f:
    for k, v in env.items():
        f.write("==== " + k + "\n" + v + "\n")

# ------------------------------------------------------------------ E1 thread sweep
ser, omp = rd("bench_threads_serial.csv"), rd("bench_threads_omp.csv")
if ser is not None and omp is not None:
    s_img = per_image(ser, ["threads"])
    o_img = per_image(omp, ["threads"])
    base = s_img.set_index("image")
    rows, stage_rows = [], []
    T_ser = ser[STAGES + ["total"]].mean()
    o1 = omp[omp.threads == 1][STAGES + ["total"]].mean()
    # serial stage breakdown
    br = pd.DataFrame({"Stage": [LABEL[s] for s in STAGES + ["total"]],
                       "Mean (ms)": [T_ser[s] * 1e3 for s in STAGES + ["total"]],
                       "Std (ms)": [ser[s].std() * 1e3 for s in STAGES + ["total"]],
                       "Median (ms)": [ser[s].median() * 1e3 for s in STAGES + ["total"]],
                       "Share (%)": [100 * T_ser[s] / T_ser["total"] for s in STAGES + ["total"]]})
    save_tex(br, "serial_breakdown", fmt={"Mean (ms)": "{:.2f}", "Std (ms)": "{:.2f}", "Median (ms)": "{:.2f}",
                                          "Share (%)": "{:.1f}"})
    for p, g in omp.groupby("threads"):
        Tp = g[STAGES + ["total"]].mean()
        gi = o_img[o_img.threads == p].set_index("image")
        sp_img = base["total"] / gi["total"]
        S = T_ser["total"] / Tp["total"]
        rows.append({"Threads": p, "Mean (ms)": Tp["total"] * 1e3, "Std (ms)": g["total"].std() * 1e3,
                     "Median (ms)": g["total"].median() * 1e3,
                     "Speedup": S, "Speedup sd": sp_img.std(), "Rel. OMP-1": o1["total"] / Tp["total"],
                     "Efficiency": S / p, "Karp-Flatt": kf(S, p), "Images/s": 1.0 / Tp["total"]})
        for s in STAGES + ["total"]:
            Ss = T_ser[s] / Tp[s]
            stage_rows.append({"stage": s, "threads": p, "ms": Tp[s] * 1e3, "speedup": Ss,
                               "efficiency": Ss / p, "karp_flatt": kf(Ss, p)})
    th = pd.DataFrame(rows)
    st = pd.DataFrame(stage_rows)
    save_tex(th, "thread_sweep", fmt={"Mean (ms)": "{:.1f}", "Std (ms)": "{:.1f}", "Median (ms)": "{:.1f}",
                                      "Speedup": "{:.2f}", "Speedup sd": "{:.2f}", "Rel. OMP-1": "{:.2f}",
                                      "Efficiency": "{:.2f}", "Karp-Flatt": "{:.3f}", "Images/s": "{:.2f}"})
    st.to_csv(os.path.join(TAB, "stage_speedup_long.csv"), index=False)
    piv = st.pivot(index="stage", columns="threads", values="speedup").reindex(STAGES + ["total"])
    piv.index = [LABEL[s] for s in piv.index]
    piv = piv.reset_index().rename(columns={"index": "Stage"})
    piv.columns = ["Stage"] + ["$p=%d$" % c for c in piv.columns[1:]]
    save_tex(piv, "stage_speedup", fmt={c: "{:.2f}" for c in piv.columns[1:]})
    pk = st.pivot(index="stage", columns="threads", values="karp_flatt").reindex(STAGES + ["total"])
    pk.index = [LABEL[s] for s in pk.index]
    pk = pk.reset_index().rename(columns={"index": "Stage"})
    pk.columns = ["Stage"] + ["$p=%d$" % c for c in pk.columns[1:]]
    save_tex(pk, "stage_karpflatt", fmt={c: "{:.3f}" for c in pk.columns[1:]})
    summary["thread_sweep"] = th
    summary["serial_ms"] = T_ser["total"] * 1e3
    summary["omp1_ms"] = o1["total"] * 1e3
    P = sorted(omp.threads.unique())

    # Fig 1: stacked stage times vs threads (serial first)
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    labels = ["serial"] + [str(p) for p in P]
    vals = [T_ser] + [omp[omp.threads == p][STAGES].mean() for p in P]
    bottom = np.zeros(len(labels))
    for s in STAGES:
        h = np.array([v[s] * 1e3 for v in vals])
        ax.bar(labels, h, bottom=bottom, color=COL[s], width=0.62, label=LABEL[s], edgecolor="white", linewidth=1)
        bottom += h
    for i, b in enumerate(bottom):
        ax.text(i, b + bottom.max() * 0.015, "%.0f" % b, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Time per image (ms)")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.set_ylim(0, bottom.max() * 1.15)
    fig.savefig(os.path.join(FIG, "fig_stage_breakdown.pdf"))
    fig.savefig(os.path.join(FIG, "fig_stage_breakdown.png"))
    plt.close(fig)

    # Fig 2: speedup vs threads per stage (+ ideal)
    fig, ax = plt.subplots(figsize=(COLW, 2.4))
    ax.plot(P, P, ls="--", color=INK2, lw=1, label="Ideal")
    for s in ["clahe", "gauss", "vess", "morph", "total"]:
        d = st[st.stage == s].sort_values("threads")
        ax.plot(d.threads, d.speedup, marker="o", ms=4, color=COL[s], label=LABEL[s],
                lw=2.2 if s == "total" else 1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(P)
    ax.set_xticklabels([str(p) for p in P])
    ax.set_xlabel("OpenMP threads $p$")
    ax.set_ylabel("Speedup vs serial")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(os.path.join(FIG, "fig_speedup.pdf"))
    fig.savefig(os.path.join(FIG, "fig_speedup.png"))
    plt.close(fig)

    # Fig 3: Karp-Flatt
    fig, ax = plt.subplots(figsize=(COLW, 2.2))
    for s in ["clahe", "gauss", "vess", "morph", "total"]:
        d = st[(st.stage == s) & (st.threads > 1)].sort_values("threads")
        ax.plot(d.threads, d.karp_flatt, marker="o", ms=4, color=COL[s], label=LABEL[s],
                lw=2.2 if s == "total" else 1.6)
    ax.set_xscale("log", base=2)
    Pk = [p for p in P if p > 1]
    ax.set_xticks(Pk)
    ax.set_xticklabels([str(p) for p in Pk])
    ax.set_xlabel("OpenMP threads $p$")
    ax.set_ylabel("Karp-Flatt serial fraction $e$")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(os.path.join(FIG, "fig_karpflatt.pdf"))
    fig.savefig(os.path.join(FIG, "fig_karpflatt.png"))
    plt.close(fig)

# ------------------------------------------------------------------ E2 variants
rows = []
if omp is not None:
    nimg = None
    for v in ["dynamic", "atomic", "scalepar"]:
        d = rd("bench_variant_%s.csv" % v)
        if d is None:
            continue
        imgs = sorted(d.image.unique())
        b = omp[omp.image.isin(imgs)]
        for p, g in d.groupby("threads"):
            bb = b[b.threads == p]
            stage = {"dynamic": "clahe", "atomic": "clahe", "scalepar": None}[v]
            if stage:
                rows.append({"Variant": v, "Threads": p, "Stage": LABEL[stage],
                             "Base (ms)": bb[stage].mean() * 1e3, "Variant (ms)": g[stage].mean() * 1e3,
                             "Ratio": g[stage].mean() / bb[stage].mean()})
            else:
                rows.append({"Variant": v, "Threads": p, "Stage": "Frangi (gauss+vess)",
                             "Base (ms)": (bb.gauss + bb.vess).mean() * 1e3,
                             "Variant (ms)": (g.gauss + g.vess).mean() * 1e3,
                             "Ratio": (g.gauss + g.vess).mean() / (bb.gauss + bb.vess).mean()})
    if rows:
        va = pd.DataFrame(rows)
        save_tex(va, "variants", fmt={"Base (ms)": "{:.2f}", "Variant (ms)": "{:.2f}", "Ratio": "{:.2f}"})
        summary["variants"] = va

# ------------------------------------------------------------------ E3 resolution scaling
ss, so = rd("bench_scaling_serial.csv"), rd("bench_scaling_omp.csv")
if ss is not None and so is not None:
    rows = []
    for sc, g in so.groupby("scale"):
        gs = ss[ss.scale == sc]
        Ts = gs.total.mean()
        mp = gs.width.iloc[0] * gs.height.iloc[0] / 1e6
        for p, gp in g.groupby("threads"):
            Tp = gp.total.mean()
            rows.append({"Scale": "%dx" % sc, "MP": mp, "Threads": p, "Serial (ms)": Ts * 1e3,
                         "OpenMP (ms)": Tp * 1e3, "Speedup": Ts / Tp, "Efficiency": Ts / Tp / p,
                         "MP/s": mp / Tp})
    sct = pd.DataFrame(rows)
    sct.to_csv(os.path.join(TAB, "scaling_long.csv"), index=False)
    Pmax = sct.Threads.max()
    best = sct.loc[sct.groupby("Scale").Speedup.idxmax()].sort_values("MP")
    save_tex(best[["Scale", "MP", "Threads", "Serial (ms)", "OpenMP (ms)", "Speedup", "Efficiency", "MP/s"]],
             "scaling", fmt={"MP": "{:.2f}", "Serial (ms)": "{:.0f}", "OpenMP (ms)": "{:.0f}",
                             "Speedup": "{:.2f}", "Efficiency": "{:.2f}", "MP/s": "{:.2f}"})
    summary["scaling"] = sct
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    tcol = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for i, (p, g) in enumerate(sct[sct.Threads > 1].groupby("Threads")):
        g = g.sort_values("MP")
        ax.plot(g.MP, g.Speedup, marker="o", ms=4, color=tcol[i % len(tcol)], label="$p=%d$" % p)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Image size (megapixels)")
    ax.set_ylabel("Speedup vs serial")
    mps = sorted(sct.MP.unique())
    ax.set_xticks(mps)
    ax.set_xticklabels(["%.2f" % m for m in mps])
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig_scaling.pdf"))
    fig.savefig(os.path.join(FIG, "fig_scaling.png"))
    plt.close(fig)

# ------------------------------------------------------------------ E4 STARE timing
s2, o2 = rd("bench_stare_serial.csv"), rd("bench_stare_omp.csv")
if s2 is not None and o2 is not None:
    Ts = s2.total.mean()
    rows = [{"Threads": p, "Mean (ms)": g.total.mean() * 1e3, "Speedup": Ts / g.total.mean(),
             "Efficiency": Ts / g.total.mean() / p, "Karp-Flatt": kf(Ts / g.total.mean(), p)}
            for p, g in o2.groupby("threads")]
    stt = pd.DataFrame(rows)
    save_tex(stt, "stare_threads", fmt={"Mean (ms)": "{:.1f}", "Speedup": "{:.2f}", "Efficiency": "{:.2f}",
                                        "Karp-Flatt": "{:.3f}"})
    summary["stare_serial_ms"] = Ts * 1e3

# ------------------------------------------------------------------ accuracy
rows = []
for kind, label in [("drive-test", "DRIVE test (20)"), ("drive-train", "DRIVE train (20, tuning set)"),
                    ("stare", "STARE (20, cross-dataset)")]:
    d = rd("acc_%s.csv" % kind)
    if d is None:
        continue
    per = d[~d.image.isin(["MEAN", "POOLED"])]
    rows.append({"Set": label, "Dice": "%.4f $\\pm$ %.4f" % (per.dice.mean(), per.dice.std()),
                 "Sensitivity": "%.4f" % per.sens.mean(), "Specificity": "%.4f" % per.spec.mean(),
                 "Accuracy": "%.4f" % per.acc.mean()})
    summary["acc_" + kind] = per
if rows:
    save_tex(pd.DataFrame(rows), "accuracy")

# ------------------------------------------------------------------ correctness
rows = []
for f in sorted(glob.glob(os.path.join(RES, "verify_*.csv"))):
    d = pd.read_csv(f)
    a = d[d.image == "ALL"].iloc[0]
    tag = os.path.basename(f)[7:-4]
    rows.append({"Build/variant/threads": tag.replace("_", " "), "CLAHE identical": "%.4f" % a.eq_identical,
                 "Mask agreement": "%.6f" % a.mask_agreement, "V bit-identical": "%.6f" % a.V_bit_identical,
                 "Max |dV|": "%.1e" % a.V_max_abs_diff})
if rows:
    save_tex(pd.DataFrame(rows), "correctness")
    summary["correctness"] = pd.DataFrame(rows)

for f in ["check_decoder.csv", "check_skimage.csv"]:
    p = os.path.join(RES, f)
    if os.path.exists(p):
        save_tex(pd.read_csv(p), f[:-4])

# ------------------------------------------------------------------ qualitative figure
try:
    from PIL import Image
    ids = ["01", "02"]
    fig, axs = plt.subplots(len(ids), 3, figsize=(COLW * 2, 2.1 * len(ids)))
    for r, i in enumerate(ids):
        rgb = np.asarray(Image.open(os.path.join(DRIVE, "test", "images", i + "_test.tif")).convert("RGB"))
        gt = np.asarray(Image.open(os.path.join(DRIVE, "test", "1st_manual", i + "_manual1.gif")).convert("L")) > 127
        vm = np.asarray(Image.open(os.path.join(RES, "masks", i + "_test_vesselness.png")))
        pm = np.asarray(Image.open(os.path.join(RES, "masks", i + "_test_mask.png"))) > 127
        comp = np.ones(gt.shape + (3,))
        comp[pm & gt] = (0.04, 0.04, 0.04)          # true positive: ink
        comp[pm & ~gt] = (0.92, 0.41, 0.20)         # false positive: orange
        comp[~pm & gt] = (0.16, 0.47, 0.84)         # false negative: blue
        for c, (img, t) in enumerate([(rgb, "Input (%s_test)" % i), (vm, "Vesselness $V$"),
                                      (comp, "TP black, FP orange, FN blue")]):
            axs[r, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axs[r, c].set_title(t)
            axs[r, c].axis("off")
    fig.savefig(os.path.join(FIG, "fig_qualitative.pdf"))
    fig.savefig(os.path.join(FIG, "fig_qualitative.png"), dpi=200)
    plt.close(fig)
except Exception as e:  # figure is optional
    print("qualitative figure skipped:", e)

with open(os.path.join(TAB, "summary.txt"), "w") as f:
    for k, v in summary.items():
        f.write("==== %s\n%s\n\n" % (k, v.to_string() if hasattr(v, "to_string") else v))
print(open(os.path.join(TAB, "summary.txt")).read())
