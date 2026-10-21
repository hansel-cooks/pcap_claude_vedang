"""Kaggle CPU driver: builds the C code and runs every experiment.

All computation happens in the C binaries (bin/frangi_serial, bin/frangi_omp).
This file only orchestrates, because Kaggle kernels start from Python.
Outputs land in /kaggle/working/results.
"""
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "results")
LOG = os.path.join(OUT, "run.log")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, out=None, check=True):
    log("$ " + cmd)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    text = r.stdout + r.stderr
    if out:
        with open(os.path.join(OUT, out), "w") as f:
            f.write(r.stdout)
    with open(LOG, "a") as f:
        f.write(text[-4000:] + "\n")
    log("  -> rc=%d in %.1fs" % (r.returncode, time.time() - t0))
    if check and r.returncode != 0:
        print(text)
        raise SystemExit("command failed: " + cmd)
    return r.stdout


def find_dir(name, must_contain):
    for root, dirs, _ in os.walk("/kaggle/input"):
        if os.path.basename(root) == name and all(d in dirs for d in must_contain):
            return root
    raise SystemExit("cannot find %s in /kaggle/input" % name)


def find_file(name):
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return root
    raise SystemExit("cannot find %s" % name)


MODE = os.environ.get("FRANGI_MODE", "full")  # "full" or "quick"
DRIVE = find_dir("drive", ["training", "test"])
STARE = find_dir("stare", ["images", "labels"])
SRC = find_file("Makefile")
BUILD = os.path.join(WORK, "build")
if os.path.exists(BUILD):
    shutil.rmtree(BUILD)
shutil.copytree(SRC, BUILD)
os.chdir(BUILD)
B = os.path.join(BUILD, "bin")
P = os.path.join(OUT, "params.txt")

# ------------------------------------------------------------ environment
sh("lscpu", out="env_lscpu.txt")
sh("lscpu -e", out="env_lscpu_e.txt", check=False)
sh("nproc; nproc --all; free -h; uname -a; gcc --version; cat /proc/cpuinfo | grep 'model name' | head -1",
   out="env_misc.txt")
sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true",
   out="env_cgroup.txt", check=False)

# ------------------------------------------------------------ builds
OPT = "-O3 -march=native"
sh('make OPT="%s" BIN=bin' % OPT)
sh('make OPT="%s" FPFLAGS=-ffp-contract=off BIN=bin_nofma' % OPT)
sh(B + "/frangi_omp info", out="env_build.txt")
sh(B + "/frangi_serial info", out="env_build_serial.txt")
NP = int(subprocess.run("nproc", shell=True, capture_output=True, text=True).stdout.strip())
THREADS = sorted(set([1, 2, 4, NP, 2 * NP]))
TL = ",".join(str(t) for t in THREADS)
log("threads to sweep: " + TL)
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "threads"

# ------------------------------------------------------------ decoder check
os.makedirs(os.path.join(OUT, "decode"), exist_ok=True)
for name in ["training/images/21_training.tif", "training/images/32_training.tif", "test/images/01_test.tif"]:
    base = os.path.basename(name).replace(".tif", "")
    sh("%s/frangi_serial green %s/%s %s/decode/%s_green.pgm" % (B, DRIVE, name, OUT, base))

# ------------------------------------------------------------ tuning (training set only)
if MODE == "quick":
    shutil.copy(os.path.join(BUILD, "scripts", "params_quick.txt"), P)
else:
    sh("%s/frangi_omp tune %s %s --nt %d" % (B, DRIVE, P, NP), out="tune_stdout.txt")
log(open(P).read())

# ------------------------------------------------------------ accuracy
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
for kind, root in [("drive-test", DRIVE), ("drive-train", DRIVE), ("stare", STARE)]:
    sh("%s/frangi_serial eval %s %s %s %s/acc_%s.csv --masks %s/masks" % (B, kind, root, P, OUT, kind, OUT))

# ------------------------------------------------------------ correctness: serial vs OpenMP
dumps = "/tmp/dumps"  # kept out of /kaggle/working so it is not saved as output
nimg = "4" if MODE == "quick" else "20"
for tag, bindir in [("fma", B), ("nofma", os.path.join(BUILD, "bin_nofma"))]:
    ref = os.path.join(dumps, tag, "serial")
    os.makedirs(ref, exist_ok=True)
    sh("%s/frangi_serial dump drive-test %s %s %s --images %s" % (bindir, DRIVE, P, ref, nimg))
    for variant in ["base", "dynamic", "atomic", "scalepar"]:
        for nt in THREADS:
            d = os.path.join(dumps, tag, "%s_%d" % (variant, nt))
            os.makedirs(d, exist_ok=True)
            sh("%s/frangi_omp dump drive-test %s %s %s --nt %d --variant %s --images %s"
               % (bindir, DRIVE, P, d, nt, variant, nimg))
            sh("%s/frangi_serial compare %s %s" % (bindir, ref, d),
               out="verify_%s_%s_%d.csv" % (tag, variant, nt))
# serial FMA build vs serial no-FMA build (does contraction change the result at all?)
sh("%s/frangi_serial compare %s %s" % (B, os.path.join(dumps, "fma", "serial"), os.path.join(dumps, "nofma", "serial")),
   out="verify_fma_vs_nofma_serial.csv")
# STARE equivalence at the largest thread count
ref = os.path.join(dumps, "stare_serial")
d = os.path.join(dumps, "stare_omp")
os.makedirs(ref, exist_ok=True)
os.makedirs(d, exist_ok=True)
sh("%s/frangi_serial dump stare %s %s %s --images %s" % (B, STARE, P, ref, nimg))
sh("%s/frangi_omp dump stare %s %s %s --nt %d --images %s" % (B, STARE, P, d, NP, nimg))
sh("%s/frangi_serial compare %s %s" % (B, ref, d), out="verify_stare_%d.csv" % NP)

# ------------------------------------------------------------ timing
if MODE == "quick":
    W, R, NI = 1, 3, 2
else:
    W, R, NI = 5, 30, 20
T = lambda f: os.path.join(OUT, f)

# E1: thread sweep, native DRIVE test images, all stages
sh("%s/frangi_serial bench drive-test %s %s %s --nt 1 --warm %d --reps %d --images %d"
   % (B, DRIVE, P, T("bench_threads_serial.csv"), W, R, NI))
sh("%s/frangi_omp bench drive-test %s %s %s --nt %s --warm %d --reps %d --images %d"
   % (B, DRIVE, P, T("bench_threads_omp.csv"), TL, W, R, NI))

# E2: design variants (CLAHE schedule, atomic histogram, scale-parallel Frangi)
nv = 2 if MODE == "quick" else 5
for variant in ["dynamic", "atomic", "scalepar"]:
    sh("%s/frangi_omp bench drive-test %s %s %s --nt %s --warm %d --reps %d --images %d --variant %s"
       % (B, DRIVE, P, T("bench_variant_%s.csv" % variant), TL, W, R, nv, variant))

# E3: resolution scaling 1x..8x
ns = 1 if MODE == "quick" else 3
for sc, w, r in [(1, 3, 10), (2, 2, 10), (4, 1, 5), (8, 1, 3)]:
    if MODE == "quick":
        w, r = 1, 2
    sh("%s/frangi_serial bench drive-test %s %s %s --nt 1 --scales %d --warm %d --reps %d --images %d --append"
       % (B, DRIVE, P, T("bench_scaling_serial.csv"), sc, w, r, ns))
    sh("%s/frangi_omp bench drive-test %s %s %s --nt %s --scales %d --warm %d --reps %d --images %d --append"
       % (B, DRIVE, P, T("bench_scaling_omp.csv"), TL, sc, w, r, ns))

# E4: STARE thread sweep (cross-dataset timing)
rs = 3 if MODE == "quick" else 10
sh("%s/frangi_serial bench stare %s %s %s --nt 1 --warm 2 --reps %d --images %d"
   % (B, STARE, P, T("bench_stare_serial.csv"), rs, NI))
sh("%s/frangi_omp bench stare %s %s %s --nt %s --warm 2 --reps %d --images %d"
   % (B, STARE, P, T("bench_stare_omp.csv"), TL, rs, NI))

# E5: thread placement. Default above is OMP_PLACES=threads + OMP_PROC_BIND=close
# (consecutive logical CPUs); here one thread per physical core first, and no binding.
for tag, envs in [("spread_cores", "OMP_PLACES=cores OMP_PROC_BIND=spread"), ("unbound", "OMP_PROC_BIND=false")]:
    sh("env -u OMP_PLACES -u OMP_PROC_BIND %s %s/frangi_omp bench drive-test %s %s %s --nt %s --warm %d --reps %d --images %d"
       % (envs, B, DRIVE, P, T("bench_affinity_%s.csv" % tag), TL, W, R, nv))

# ------------------------------------------------------------ reference cross-check (Python, check only)
sh("%s %s/scripts/check.py %s %s %s %s" % (sys.executable, BUILD, DRIVE, P, os.path.join(dumps, "fma", "serial"),
                                            OUT), out="check_stdout.txt", check=False)
log("ALL DONE")
