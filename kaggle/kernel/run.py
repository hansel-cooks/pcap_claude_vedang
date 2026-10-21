# Launcher: runs the driver shipped in the frangi-omp-src dataset.
import os, runpy
os.environ["FRANGI_MODE"] = "full"
for root, _, files in os.walk("/kaggle/input"):
    if "run_kaggle.py" in files:
        runpy.run_path(os.path.join(root, "run_kaggle.py"), run_name="__main__")
        break
else:
    raise SystemExit("run_kaggle.py not found")
