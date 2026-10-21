# Serial vs OpenMP multi-scale Frangi pipeline for retinal vessel segmentation

PCAP mini project (Manipal Institute of Technology): green channel -> CLAHE -> multi-scale Frangi ->
threshold -> morphological clean-up, written in C, parallelised with OpenMP, profiled stage by stage on the
DRIVE and STARE datasets. All timing runs are done on a Kaggle CPU VM (2 physical cores, 4 hyperthreads).

| Folder | Contents |
|---|---|
| `openmp_frangi/` | C source (`src/`), `Makefile`, Kaggle driver and analysis scripts (`scripts/`). See its README. |
| `paper/` | Paper in IEEE (`ieee/`) and Springer LNCS (`springer/`) formats, shared `sections/`, `refs.bib`; compiled PDFs `PCAP_OpenMP_Frangi_IEEE.pdf` and `PCAP_OpenMP_Frangi_Springer.pdf`. |
| `notebooks/` | `PCAP_OpenMP_Frangi_Demo.ipynb`: Google Colab demo (stage-by-stage before/after images, accuracy, bit-exactness, live per-stage timings for every thread count, placement, design variants, image-size scaling, and the paper's Kaggle results). Needs a `GITHUB_TOKEN` Colab secret while the repo is private. |
| `data/drive/` | DRIVE training + test images, FOV masks and first-observer manual segmentations. |
| `data/stare/` | STARE 20 images and Hoover's hand labels. |
| `runs/pilot/` | Two-image pilot run used to debug the harness (not reported in the paper). |
| `runs/full/` | Full Kaggle run: CSV logs, masks, analysis tables and figures (added when the run finishes). |
| `kaggle/` | Kaggle dataset and kernel metadata used to run the experiments. |
| `docs/` | Project synopsis and literature survey (Submission 1). |
| `.claude/skills/`, `third_party/academic-research-skills/` | Academic Research Skills (Imbad0202, CC BY-NC 4.0) used for the paper workflow. |

## Reproduce

```
cd openmp_frangi
make OPT="-O3 -march=native"
bin/frangi_omp tune ../data/drive params.txt
bin/frangi_serial eval drive-test ../data/drive params.txt acc.csv
bin/frangi_omp bench drive-test ../data/drive params.txt bench.csv --nt 1,2,4,8
```

On Kaggle, `scripts/run_kaggle.py` runs every experiment in the paper and writes the CSVs to
`/kaggle/working/results`; `scripts/analyze.py` turns them into the paper's tables and figures.

## Data sources and licences

- DRIVE: Staal et al., IEEE TMI 2004. Test-set manual segmentations from the Kaggle dataset
  `zionfuo/drive2004` (images verified byte-identical to the original copy).
- STARE: Hoover et al., IEEE TMI 2000, https://cecas.clemson.edu/~ahoover/stare/probing/index.html
- Both datasets are for research use under their providers' terms; this repository is private for that reason.
