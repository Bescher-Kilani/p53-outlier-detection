# p53 Mutants — Outlier Detection

## Problem Statement

Cells divide constantly, and during this process DNA can get damaged (e.g.
by chemicals, replication errors, or UV radiation). The protein p53 acts as
a checkpoint that senses this damage and decides whether a cell should be
repaired or eliminated (apoptosis). If p53 itself is mutated, it can become
inactive and lose this ability — damaged cells then survive and keep
dividing instead of being repaired or removed, eventually leading to
cancer. p53 mutations are involved in roughly 50% of human cancers.

This project applies outlier detection to a dataset of structure-based
features computed for ~31,000 p53 mutants, to explore which mutations
disrupt this protective function. The target label is used for
verification only — never for training or tuning — per the task rules.

## Repo structure

```
p53mutant/
├── setup.sh                   # one-time environment setup
├── requirements.txt
├── split_data.py              # cleans the raw data, creates train/test split
├── p53_od_pipeline.py         # outlier detection pipeline (the deliverable)
├── eval_internal_test_auc.py  # our own check of the final prediction (not part of the pipeline)
├── compute_correlations.py    # Pearson/Spearman/MI via BLAS matmul, run on the KIT server
└── notbooks/
    └── data_expolartion.ipynb # exploration & correlation/PCA plots only,
                                # not required to run the pipeline
```

Raw data (`train_data.data`, `train_gt.csv`, ~3.3 GB) is gitignored — see
"Reproducing the pipeline" below. So are the files it produces
(`train_set_p53mutant.parquet`, `test_set_p53mutant.parquet`,
`train_labels_p53mutant.csv`, `train_data_clean.parquet`,
`pearson_corr.parquet`, `spearman_corr.parquet`, `mi_corr.parquet`).

## Setup

```bash
git clone https://github.com/Bescher-Kilani/p53-outlier-detection.git
cd p53mutant
bash setup.sh
source venv/bin/activate
```

## Reproducing the pipeline

1. Place `train_data.data` and `train_gt.csv` (from ILIAS) in this folder.
2. Clean the data and create the stratified train/test split:
   ```bash
   python split_data.py
   ```
   Produces `train_set_p53mutant.parquet`, `test_set_p53mutant.parquet`,
   `train_labels_p53mutant.csv`, and
   `_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv` (kept aside for our
   own verification only — never read by the pipeline itself).

   **Why a stratified split?** Only 61 of 30,754 mutants are active
   (~503:1 imbalance). A purely random split could leave one side with
   very few or zero active cases, making any AUC computed on it
   unreliable. `train_test_split(..., stratify=labels)` keeps both sides
   proportional (49 active in train, 12 in test).

   **Why two separate label files?** `train_labels_p53mutant.csv` is the
   only one the pipeline ever reads, and only for a post-hoc AUC check
   after fitting — never for `.fit()` itself (OCC rule, see Problem
   Statement above). `_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv`
   holds back the other 12 active cases on purpose, so we can score our
   own final prediction afterwards without ever letting these labels
   reach the pipeline — simulating the secret hold-out set the course
   staff will grade us on.
3. Run the outlier detection pipeline:
   ```bash
   python p53_od_pipeline.py --method isolation_forest --reduce pca
   ```
   Produces `outputs/team2_prediction.csv`. Run with `--help` to see all
   options (`--method isolation_forest|lof|gmd_lof`, `--reduce
   none|pca|random_projection|truncated_svd|isomap|umap`, `--sweep` to
   compare several `--n-components` values for one method in a single run).
4. Check the prediction against the internal test labels (our own
   sanity check, separate from the pipeline):
   ```bash
   python eval_internal_test_auc.py
   ```

## Notebook (exploration only)

`notbooks/data_expolartion.ipynb` documents the original exploration:
structure inspection, missing-value analysis, the cleaning logic later
ported into `split_data.py`, and the correlation/PCA analysis
(`compute_correlations.py` runs the heavy Pearson/Spearman/MI computation
on the KIT server; results are copied back and plotted in the notebook).
Not required to reproduce the pipeline above.

## Week 1 — key findings

- Raw: 31,021 rows × 5,408 features (2D electrostatic/surface block,
  3D distance-based block) + 1 empty parsing artifact column
- Cleaned: 30,754 rows (dropped 137 fully-empty, 124 missing the 2D block,
  5 duplicates)
- Size: 3.2 GB CSV → 665.5 MB in-RAM (float32) → 281.4 MB Parquet
- Correlation: all 3 methods (Pearson, Spearman, MI) confirm one strong
  block (~index 2400–4800); Spearman/MI narrow it to a tighter hot zone
  (~index 3190–4690); MI uniquely flags extra long-range pairs (e.g.
  3299–4383); Pearson/Spearman also reveal a negatively correlated region
  outside the block (~index 4500–5300)
- PCA: 147 of 5,408 components capture 95% variance (not standardized,
  to preserve outlier-driven variance)
- Labels (verification only): 30,693 inactive / 61 active (~503:1)

## Week 2 — key findings

- Pipeline rebuilt as two standalone scripts (`split_data.py`,
  `p53_od_pipeline.py`) instead of notebook cells, mirroring the
  cleaning/pipeline split from the Earthquake task
- OCC discipline maintained throughout: labels never enter `.fit()`,
  used only for the stratified split itself and the post-hoc AUC check
- Six dimensionality reduction methods compared on the same raw
  5,408-feature input (fairness), scored with Isolation Forest,
  n_components=147 unless noted. Full-train AUC = fit on all train,
  score train (post-hoc check). CV AUC = 3-fold **stratified** CV on
  train (labels never used for fitting); mean +/- std across folds.
  CV + norm = same CV with StandardScaler fit on each train fold only
  (`--scale`, default in pipeline). CV + norm + SMOTE = scale then SMOTE on
  train fold only (uses labels on train split). UMAP CV uses k=200 only.

  | Reduction                | AUC (full train) | AUC (3-fold stratified CV) | AUC (CV + norm)   | AUC (CV + norm + SMOTE) |
  |--------------------------|------------------|----------------------------|-------------------|-------------------------|
  | PCA                      | 0.6733           | 0.6734 +/- 0.0411          | 0.7291 +/- 0.0543 | 0.5200 +/- 0.0673       |
  | TruncatedSVD             | 0.6657           | 0.6733 +/- 0.0333          | 0.7285 +/- 0.0504 | 0.5076 +/- 0.0624       |
  | none (raw)               | 0.5902           | 0.5926 +/- 0.0624          | 0.5926 +/- 0.0624 | 0.4256 +/- 0.0706       |
  | Isomap                   | 0.5649           | 0.6008 +/- 0.0542          | 0.6851 +/- 0.0807 (1469s) | —                       |
  | Random Projection        | 0.5232           | 0.5232 +/- 0.0464          | 0.6689 +/- 0.0249 | 0.5952 +/- 0.0590       |
  | UMAP (k=200)             | 0.4013           | 0.4179 +/- 0.0314          | —                 | —                       |

  full-train UMAP from best of k=5..300 sweep (k=200 gave 0.4013 in
  that sweep); CV column uses k=200 only.

  Linear, variance-preserving methods (PCA, SVD) outperform manifold
  methods (Isomap, UMAP — local-neighborhood-preserving) and Random
  Projection (distance-preserving but data-blind) for this task

- **mRMR + IFS** feature selection (max 147 features, then Isolation Forest
  on selected subset; mRMR/IFS use labels on train only):

  | Method                   | AUC (full train) | AUC (3-fold stratified CV) | AUC (CV + norm)   | AUC (CV + norm + SMOTE) |
  |--------------------------|------------------|----------------------------|-------------------|-------------------------|
  | mRMR + IFS (max 147)     | 0.8181 (575s)    | 0.7511 +/- 0.0466 (1024s)  | 0.7499 +/- 0.0576 (1015s) | 0.3474 +/- 0.0746 (2261s) |

  mRMR ranks features (MI relevance − correlation redundancy); IFS picks
  subset size by best train-fold IF AUC among {25, 50, …, 147}. Runtime
  in parentheses per column (seconds, single run on 24,603 train rows).

- GMD (subspace search) + LOF on reduced features (147 dims). Full-train
  AUC = fit on all train, score train (post-hoc). CV columns = 3-fold
  stratified CV on train. Runtime in parentheses per CV cell.

  | Reduction               | Subspaces found | Runtime (full train) | AUC (full train) | AUC (3-fold stratified CV) | AUC (CV + norm) | AUC (CV + norm + SMOTE) |
  |--------------------------|-----------------|----------------------|------------------|----------------------------|-----------------|-------------------------|
  | none (raw, 5,408 dims)   | —               | >36h, never finished | —                | —                          | —               | —                       |
  | PCA (147)                | 147             | 577.1s               | 0.6078           | 0.6106 +/- 0.0612 (1396s) | 0.5920 +/- 0.0947 (1223s) | 0.8888 +/- 0.0135 (2196s) |
  | Random Projection (147)  | 147             | 361.3s               | 0.6276           | 0.6313 +/- 0.0135 (1017s) | 0.5949 +/- 0.0615 (1297s) | 0.8930 +/- 0.0389 (2618s) |

  GMD's pairwise subspace search is O(d²) and intractable on the raw
  5,408 dimensions — feasible only after a prior dimensionality
  reduction step. Requires `gmd` (Linux / MSVC on Windows).

- Current best (3-fold stratified CV on train, our selection metric):

  | Rank | Config | AUC (CV) | OCC note |
  |------|--------|----------|----------|
  | 1 | mRMR + IFS + Isolation Forest | **0.7511 ± 0.0466** | labels used for feature selection only |
  | 2 | Isolation Forest + PCA (147) + StandardScaler | **0.7291 ± 0.0543** | fully unsupervised fit |
  | 3 | GMD + LOF + Random Projection (147) + SMOTE | 0.8930 ± 0.0389 | labels via SMOTE; treat as exploratory |

  **Reproduce internal CV scores** (train only, no test prediction):

  ```bash
  # Best overall (mRMR + IFS, no scaling — matches 0.7511 CV)
  python benchmark_mrmr_ifs.py --step cv

  # Best fully unsupervised (PCA + norm — matches 0.7291 CV)
  python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 147 --scale
  # (CV only, no test prediction: python benchmark_week2_cv.py --only pca)

  # GMD + LOF + Random Projection + SMOTE (matches 0.8930 CV; needs gmd)
  python benchmark_gmd_lof_cv.py --only rp_smote
  ```

  **Produce `outputs/team2_prediction.csv`** (fit on full train, then score test):

  ```bash
  # Recommended deliverable: best CV under strict OCC (no labels in fit)
  python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 147 --scale

  # Best CV overall (uses labels for mRMR/IFS on train only)
  python p53_od_pipeline.py --method isolation_forest --reduce none --no-scale --feature-select mrmr_ifs

  # GMD exploratory (labels via SMOTE; requires gmd)
  python p53_od_pipeline.py --method gmd_lof --reduce random_projection --n-components 147 --scale --smote
  ```

  Each `p53_od_pipeline.py` run prints 3-fold stratified CV on train, then fits
  on all train and writes test scores. Check with `python eval_internal_test_auc.py`.