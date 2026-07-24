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
├── mrmr_ifs.py                # mRMR+IFS feature selector (Week 2, exploratory, see below)
├── pca_2d_3d_blocks.py        # separate PCA on 2D/3D feature blocks (Week 3, see below)
├── pca_2d_3d_blocks_cv.py     # CV check for the block-PCA representation (Week 3)
├── benchmark_deep_svdd_week4.py # Deep SVDD comparison (Week 4, see below)
├── vgan_lib/                  # bundled V-GAN source (Week 5, see below), not a PyPI package
│   ├── vgan.py
│   └── models/
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

**Week 5 additions:** TabPFN's model weights are gated on HuggingFace —
before first use, accept the license at
https://huggingface.co/Prior-Labs/tabpfn_3 and run `hf auth login` (or set
`HF_TOKEN`). V-GAN's `torch-two-sample` dependency isn't on PyPI and needs
`pip install --no-build-isolation "git+https://github.com/josipd/torch-two-sample.git"`
(same numpy-not-visible-during-build issue as the `gmd` workaround below).

**Week 6 additions:** hyperparameter search (`--hpo`) needs `optuna`
(added to `requirements.txt`); its `GPSampler` is still marked
experimental upstream (works fine here, just prints an
`ExperimentalWarning` on every run).

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
   options.
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
  |--------------------------|------------------|----------------------------|--------------------|-------------------------|
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
  |--------------------------|------------------|----------------------------|--------------------|-------------------------|
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

## Week 3 — key findings

Focus of the week (per course slides "Method Selection"): how to fairly
compare several OCC methods under extreme imbalance (61 active out of
30,754, ~503:1), and whether ensembling helps. Reference point going in:
Isolation Forest + PCA (147) + StandardScaler, CV AUC 0.7291 (Week 2's
only fully OCC-compliant result) — the label-contaminated mRMR/GMD+SMOTE
results were not used further. Everything below (new eval strategy, new
methods, feature bagging, heterogeneous ensemble) is new work this week.

### 2D/3D block-separated PCA

PCA separately on the 2D (electrostatic/surface) and 3D (distance-based)
blocks, then relate them — instead of the single PCA over all 5,408
mixed columns used elsewhere.

Two-stage (`pca_2d_3d_blocks.py`): PCA per block to 95% variance,
concatenate, PCA again to 95% variance:

| Stage | Raw dimensions | Components for 95% variance |
|-------|---------------|------------------------------|
| 2D block | 4,826 | 145 |
| 3D block | 582 | 148 |
| Concatenated (293) → stage 2 | 293 | **62** |

293→62 compression means the 2D and 3D components strongly co-vary —
consistent with both blocks coming from the same underlying 3D
structural model (surface charge and distance displacement as two views
of the same structural change).

CV check with Isolation Forest on the 62-dim. result
(`pca_2d_3d_blocks_cv.py`):

| Setup | CV AUC |
|-------|--------|
| Baseline: single PCA (147 comp.) + Scaler | 0.7291 ± 0.0543 |
| 2D/3D block-separated PCA + stage 2 (62 comp.) + Scaler | 0.7156 ± 0.0674 |

No improvement — difference within CV noise. There is a measurable
relation between the blocks (293→62 compression), but exploiting it this
way doesn't improve detection. Single-stage 147-component PCA remains
the baseline reduction used throughout.

### Evaluation strategy: repeated random subsampling

3-fold CV leaves only ~16 active cases per validation fold — too few for
a stable AUC estimate (this is explicitly called out on the course
slides). Instead, Week 3 introduces **repeated random subsampling**:
sample 80% of inliers for training, combine the remaining 20% inliers
with an 80% sample of outliers as validation, repeat many times,
average. Unlike our Week 2 CV, this trains **only on inliers** (no known
actives in `.fit()` at all) — the strict OCC framing from the slides.

**25 repeats:** tested 25 vs. 100 for Isolation Forest — 0.7248 ± 0.0227
vs. 0.7292 ± 0.0215, negligible difference for 4x the compute. Kept as
default; per-repeat results stored (not just mean/std) for next week's
Friedman/Conover-Iman test.

New pipeline command:
```bash
python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 147 \
    --eval-strategy repeated_subsampling --n-repeats 25
```
(`--eval-strategy cv`, unchanged from Week 2, remains the default for
single-method runs.)

### Step A — comparing OCC methods across all three families

OD methods are grouped into three families: **locality-based**
(LOF, KNN), **cluster-based** (OCSVM), **subspace-based**
(ABOD, Isolation Forest). Isolation Forest, KNN, OCSVM, ABOD (fast), and
LOF were compared with repeated subsampling (25 repeats) on PCA (147) +
StandardScaler:

| Method | AUC | Family |
|--------|-----|--------|
| Isolation Forest | **0.7248 ± 0.0227** | Subspace-based |
| KNN | 0.6997 ± 0.0205 | Locality-based |
| OCSVM | 0.6860 ± 0.0175 | Cluster-based |
| ABOD (fast) | 0.6494 ± 0.0195 | Subspace-based |
| LOF | 0.5067 ± 0.0227 | Locality-based |


LOF (AUC≈0.51, near-random) vs. KNN (0.70) despite same family, same
`n_neighbors=20` — LOF's local density-ratio computation appears
specifically weak on this PCA-reduced space, not a family-level issue.

**Top-3 carried forward to Steps B/C:** Isolation Forest, KNN, OCSVM.

Reproduce:
```bash
python p53_od_pipeline.py --compare-methods --reduce pca --n-components 147 --n-repeats 25
```
Prints the full 25×5 per-repeat AUC matrix to the terminal (not saved) —
input for next week's Friedman/Conover-Iman test.

### Step B — feature bagging (per method, no method mixing)

Per Top-3 method separately: 10 models on random feature subsets
(uniform size 74–147 of 147 PCA components); each subset's scores are
z-normalized (different subsets can land on different scales even with
the same detector), then combined via Sum, Max, AOM. Single full-train
post-hoc AUC (not nested in the 25-repeat subsampling — too expensive at
this data size); confirmed stable at 100 subsets.

| Method | No bagging | Sum | Max | AOM |
|--------|-----------|-----|-----|-----|
| Isolation Forest | 0.7279 | 0.7335 | **0.7354** | 0.7344 |
| KNN | 0.6816 | **0.6861** | 0.6834 | 0.6852 |
| OCSVM | 0.6891 | 0.6990 | **0.7012** | 0.6945 |

Small but consistent gain across all three methods/aggregations
(+0.5 to +1.2pp). Max wins for IF/OCSVM, Sum for KNN — differences
between aggregations are small overall.

Dilution effect (seen with GMD in Week 2) barely shows here: these
subsets are large and overlapping (≥50% of 147 components), unlike
GMD's small disjoint subspaces, so little signal gets diluted away
regardless of aggregation.

Reproduce:
```bash
python p53_od_pipeline.py --feature-bagging --reduce pca --n-components 147 --n-subsets 10
```

### Step C — heterogeneous ensemble (same feature space, different methods)

Opposite of Step B: Top-3 methods fit on the same full 147-dim. space
(no subspace split — mixing method and subspace would confound which one
causes a result), scores z-normalized, combined via Sum, Max,
rank-averaging (AOM doesn't apply meaningfully with only 3 candidates).

| | AUC |
|---|-----|
| Isolation Forest (alone) | **0.7279** |
| OCSVM (alone) | 0.6891 |
| KNN (alone) | 0.6816 |
| Ensemble (Sum) | 0.7062 |
| Ensemble (Max) | 0.7062 |
| Ensemble (Rank avg) | 0.7048 |

**Negative result:** ensemble (0.7048–0.7062) lands below the best
single method (Isolation Forest, 0.7279) — behaves like a weighted
average, not an improvement. KNN/OCSVM appear to be weaker versions of
the same signal Isolation Forest already captures rather than
complementary information. Dominance check (highest z-score per sample):
roughly even 35/35/30% split across methods — no single method
dominates the combination.

Reproduce:
```bash
python p53_od_pipeline.py --heterogeneous-ensemble --reduce pca --n-components 147
```

### Best result this week

**Isolation Forest + Feature Bagging (Max aggregation): AUC 0.7354**
(full-train post-hoc, Step B) — the highest score across every
configuration tested in Week 3, ahead of any single method alone (Step
A) and ahead of the heterogeneous ensemble (Step C, which underperformed
the best single method). Note this number uses a different evaluation
convention (full-train post-hoc) than Step A's repeated-subsampling
numbers, so it is a "best across setups" statement rather than a single
directly-comparable ranking — see the per-section notes above for how
each number was produced.

## Week 4 — key findings

Follow-up on Week 3: upgrading Steps B and C to give proper
repeated-subsampling statistics (mean ± std) instead of single full-train
point estimates, then statistically testing whether the AUC differences
seen so far are actually significant.

### Steps B/C upgraded to full repeated subsampling

Steps B and C originally used a single full-train post-hoc check (fit
and score on all training data, no held-out split) — fast, but only one
point estimate, no standard deviation, unlike Step A. Both were rebuilt
to use the same 80/20 repeated-subsampling split logic as Step A:
scale+reduce fit on the train split only, evaluated on the held-out
validation split, repeated 25 times.

Cost difference: Step B needs `n_subsets` extra fits inside every single
repeat (one model per feature subset before aggregation), so total fits
per method = n_repeats × n_subsets (750 total for 3 methods). Step C
needs only one fit per method per repeat (no subset loop), so only
n_repeats × 3 = **75** fits total.

Results at n=147 (same value as the already-presented slides), now with
proper mean ± std:

**Step B (repeated subsampling, 750 fits):**

| Method | No bagging | Sum | Max | AOM |
|--------|-----------|-----|-----|-----|
| Isolation Forest | 0.7249 ± 0.0226 | 0.7310 ± 0.0195 | 0.7308 ± 0.0206 | 0.7314 ± 0.0197 |
| KNN | 0.6997 ± 0.0205 | 0.7016 ± 0.0202 | 0.6976 ± 0.0210 | 0.6999 ± 0.0203 |
| OCSVM | 0.6860 ± 0.0175 | 0.6953 ± 0.0186 | 0.6970 ± 0.0177 | 0.6920 ± 0.0180 |

The `no_bagging` column exactly reproduces Step A's baseline numbers,
confirming the new implementation is consistent. The bagging gain is
smaller than the old full-train post-hoc numbers suggested (those were
optimistically biased), but the direction — small, consistent
improvement — holds.

**Step C (repeated subsampling, 75 fits):**

| | AUC |
|---|-----|
| Isolation Forest (alone) | **0.7249 ± 0.0226** |
| KNN (alone) | 0.6997 ± 0.0205 |
| OCSVM (alone) | 0.6860 ± 0.0175 |
| Ensemble (Sum) | 0.7079 ± 0.0194 |
| Ensemble (Max) | 0.7037 ± 0.0172 |
| Ensemble (Rank avg) | 0.7089 ± 0.0197 |

Same negative result as before, now on a paired, per-repeat basis:
Isolation Forest beats the ensemble in nearly every individual repeat,
not just on average — stronger evidence than the single full-train
comparison could give.

Reproduce:
```bash
python p53_od_pipeline.py --feature-bagging --reduce pca --n-components 147 --n-repeats 25 --n-subsets 10
python p53_od_pipeline.py --heterogeneous-ensemble --reduce pca --n-components 147 --n-repeats 25
```

### Statistical testing: Friedman + Conover-Iman

Per the course's "Multiple Comparisons Tests" week: are the AUC
differences between methods in Step A/B actually significant, or could
they be noise across the 25 repeats? Since every repeat scores all
configurations on the *same* subsample (a paired/blocked design, not
independent samples), Friedman (global) + Conover-Iman (pairwise
post-hoc) is the correct test family — not Kruskal-Wallis/Dunn, which
assume independent groups.

New pipeline command, generic over any repeats × configurations AUC
matrix CSV (works on both the 5-method Step A matrix and the 12-config
Step B matrix without changes):
```bash
python p53_od_pipeline.py --compare-methods --reduce pca --n-components 147 --n-repeats 25 \
    --output-csv outputs/auc_matrix_compare_methods.csv
python p53_od_pipeline.py --feature-bagging --reduce pca --n-components 147 --n-repeats 25 --n-subsets 10 \
    --output-csv outputs/auc_feature_bagging_matrix.csv

# Deep SVDD added later (same OCC setup: PCA 147 + scale, inlier-only fit):
python benchmark_deep_svdd_week4.py --n-repeats 25 --n-subsets 10

python p53_od_pipeline.py --stat-test outputs/auc_matrix_compare_methods.csv \
    --output-csv outputs/significance_table_compare_methods.csv
python p53_od_pipeline.py --stat-test outputs/auc_feature_bagging_matrix.csv \
    --output-csv outputs/significance_table_feature_bagging.csv
```
Prints the Friedman chi-square/p-value, then (if significant) the
Conover-Iman pairwise p-values, then a `++`/`+`/`-` significance table
(row significantly better than column at p≤0.05 / p≤0.10 / not
significant) — saved to CSV with `--output-csv`.

**Step A (6 methods, incl. Deep SVDD):** Friedman chi-square = 107.31, p ≈ 0.0000 → H0
rejected. Conover-Iman confirms the Step A ranking is not a repeat-sampling
artifact:

Isolation Forest ≫ KNN ≈ Deep SVDD ≫ OCSVM ≫ ABOD ≫ LOF

| Method | AUC (25 repeats) | Family |
|--------|------------------|--------|
| Isolation Forest | **0.7248 ± 0.0227** | Subspace |
| KNN | 0.6997 ± 0.0205 | Locality |
| **Deep SVDD** | **0.6981 ± 0.0351** | Cluster (neural) |
| OCSVM | 0.6860 ± 0.0175 | Cluster |
| ABOD | 0.6494 ± 0.0195 | Subspace |
| LOF | 0.5067 ± 0.0227 | Locality |

Deep SVDD (PyOD, MLP `[64,32]`, 20 epochs, `preprocessing=False` after
PCA+scale, **inlier-only** `.fit()`) lands between KNN and OCSVM — slightly
below kernel OCSVM's cluster-based cousin on this PCA space, and **not**
significantly different from KNN (pairwise n.s.).

**Step B (16 configs: 4 methods × 4 aggregations, incl. Deep SVDD):** Friedman
chi-square = 303.95, p ≈ 0.0000 → H0 rejected. Conover-Iman pairwise results:
- Within each method, the bagging variants (sum/max/aom) are
  significantly better than `no_bagging` (e.g. Isolation Forest:
  `no_bagging` vs. `sum` p=0.0007, vs. `max` p=0.0043) — Feature Bagging's
  gain from Step B is statistically real, not noise.
- The bagging variants are *not* significantly different from each other
  within a method (e.g. Isolation Forest `sum` vs. `max` p=0.578, vs.
  `aom` p=0.458) — no aggregation choice (sum/max/aom) clearly beats the
  others once bagging is applied.
- Across methods, Isolation Forest (any variant) beats every KNN/OCSVM/Deep
  SVDD variant significantly — the Step A method ranking holds after bagging
  too.
- **Deep SVDD:** `no_bagging` best (**0.6981 ± 0.0351**); feature bagging
  **hurts** (sum 0.6747, max 0.6543, aom 0.6726) — unlike IF/KNN/OCSVM.
  Deep SVDD is not a good candidate for Step B ensembling on this data.

| Method | No bagging | Sum | Max | AOM |
|--------|-----------|-----|-----|-----|
| Isolation Forest | 0.7279 | 0.7335 | **0.7354** | 0.7344 |
| KNN | 0.6816 | **0.6861** | 0.6834 | 0.6852 |
| OCSVM | 0.6891 | 0.6990 | **0.7012** | 0.6945 |
| **Deep SVDD** | **0.6981 ± 0.0351** | 0.6747 ± 0.0235 | 0.6543 ± 0.0276 | 0.6726 ± 0.0231 |

**Significance table, Step A (6 methods):**

|                  | isolation_forest | lof | knn | ocsvm | abod | deep_svdd |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| isolation_forest |     | ++  | ++  | ++  | ++  | ++  |
| lof              | -   |     | -   | -   | -   | -   |
| knn              | -   | ++  |     | ++  | ++  |     |
| ocsvm            | -   | ++  | -   |     | ++  | -   |
| abod             | -   | ++  | -   | -   |     | -   |
| deep_svdd        | -   | ++  |     | ++  | ++  |     |

**Significance table, Step B (4 methods × 4 aggregations; `no`/`sum`/`max`/`aom`):**

|              | IF_no | IF_sum | IF_max | IF_aom | KNN_no | KNN_sum | KNN_max | KNN_aom | OCSVM_no | OCSVM_sum | OCSVM_max | OCSVM_aom | DSVD_no | DSVD_sum | DSVD_max | DSVD_aom |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| IF_no        |     | -   | -   | -   | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  |
| IF_sum       | ++  |     |     | -   | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  |
| IF_max       | ++  | -   |     | -   | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  |
| IF_aom       | ++  |     |     |     | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  | ++  |
| KNN_no       | -   | -   | -   | -   |     | -   | ++  | -   | ++  | ++  |     | ++  |     | ++  | ++  | ++  |
| KNN_sum      | -   | -   | -   | -   | ++  |     | ++  | ++  | ++  | ++  | ++  | ++  |     | ++  | ++  | ++  |
| KNN_max      | -   | -   | -   | -   | -   | -   |     | -   | ++  |     | ++  | ++  | -   | ++  | ++  | +   |
| KNN_aom      | -   | -   | -   | -   |     | -   | ++  |     | ++  | ++  |     | ++  |     | ++  | ++  | ++  |
| OCSVM_no     | -   | -   | -   | -   | -   | -   | -   | -   |     | -   | -   | -   |     | ++  | ++  | ++  |
| OCSVM_sum    | -   | -   | -   | -   | -   | -   | -   | -   | ++  |     | -   | ++  |     | ++  | ++  | ++  |
| OCSVM_max    | -   | -   | -   | -   | -   | -   | -   | -   | ++  | ++  |     | ++  |     | ++  | ++  | ++  |
| OCSVM_aom    | -   | -   | -   | -   | -   | -   | -   | -   | ++  | -   | -   |     |     | ++  | ++  | ++  |
| DSVD_no      | -   | -   | -   | -   |     | -   | +   | -   | ++  |     |     | ++  |     | ++  | ++  | ++  |
| DSVD_sum     | -   | -   | -   | -   | -   | -   | -   | -   |     | -   | -   | -   |     |     | ++  | +   |
| DSVD_max     | -   | -   | -   | -   | -   | -   | -   | -   |     | -   | -   | -   |     | -   |     | -   |
| DSVD_aom     | -   | -   | -   | -   | -   | -   | -   | -   |     | -   | -   | -   |     |     | ++  |     |

Full pairwise results in `outputs/significance_table_compare_methods.csv` and
`outputs/significance_table_feature_bagging.csv`.

### Week 4 best (OCC deliverable)

After comparing six detectors (Step A), feature bagging (Step B), and
Friedman/Conover-Iman on 25 repeated subsamples:

| Rank | Configuration | AUC | Why |
|------|---------------|-----|-----|
| **1 (use this)** | **Isolation Forest + PCA (147) + StandardScaler + feature bagging (AOM)** | **0.7354** | Highest AUC in Step B; significantly beats every other method/aggregation in the significance table; fully OCC (inlier-only `.fit()`); small but real gain from bagging (+0.75 pp vs IF alone) |
| 2 | Isolation Forest + PCA (147) + scale (no bagging) | 0.7248 ± 0.0227 | Best single detector in Step A; simpler, nearly as good |
| 3 | OCSVM + bagging (max) | 0.7012 | Best cluster-based variant; still well below IF |
| — | Deep SVDD + PCA + scale | 0.6981 ± 0.0351 | Comparable to KNN; bagging hurts — not recommended |
| — | Heterogeneous ensemble (IF + KNN + OCSVM) | ~0.705 | Step C: mixing methods **below** IF alone — do not use |

**Pipeline command for the Week 4 winner:**

```bash
python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 147 \
    --feature-bagging --n-repeats 25 --n-subsets 10
```

Use **max** aggregation when reading Step B output (`isolation_forest_max` column in
`outputs/auc_feature_bagging_matrix.csv`). Labels are for evaluation only — never
passed into `.fit()`.

## Week 5 — key findings

Course topic: Outlier Ensembles. Three steps this week: (1) add new OD
methods and re-run the Step A comparison, (2) revisit feature bagging with
two new aggregation variants and compare subspace sources (random vs.
V-GAN-generated), (3) revisit the heterogeneous ensemble with the same new
aggregation variants.

### Step 1 — new OD methods

**TabPFN** (`--method tabpfn`) added: a zero-shot tabular foundation model,
repurposed for unsupervised OD via Prior Labs' `TabPFNUnsupervisedModel`
(estimates per-sample likelihood; low likelihood = anomalous). `fit(X)`
only, no labels — fits the OCC rule as-is, no adaptation needed. Model
weights are gated on HuggingFace (`hf auth login` required once per
machine).


**Result — TabPFN vs. all 6 methods from last week** (25 repeats, PCA-147):

| Method | AUC | 
|--------|-----|
| Isolation Forest | **0.7248 ± 0.0227** |
| TabPFN | 0.7049 ± 0.0307 |
| KNN | 0.6997 ± 0.0205 |
| Deep SVDD | 0.6981 ± 0.0351 |
| OCSVM | 0.6860 ± 0.0175 |
| ABOD | 0.6494 ± 0.0195 |
| LOF | 0.5067 ± 0.0227 |

Friedman chi-square = 117.29, p ≈ 0.0000 → H0 rejected.

**Significance table (row significantly better than column):**

|                  | isolation_forest | lof | knn | ocsvm | abod | deep_svdd | tabpfn |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| isolation_forest |     | ++  | ++  | ++  | ++  | ++  | ++  |
| lof              | --  |     | --  | --  | --  | --  | --  |
| knn              | --  | ++  |     | ++  | ++  |     | --  |
| ocsvm            | --  | ++  | --  |     | ++  | --  | --  |
| abod             | --  | ++  | --  | --  |     | --  | --  |
| deep_svdd        | --  | ++  |     | ++  | ++  |     | --  |
| tabpfn           | --  | ++  | ++  | ++  | ++  | ++  |     |

Isolation Forest significantly beats every other method (incl. TabPFN,
p = 0.0102). TabPFN is a genuine second contender despite zero tuning:
significantly ahead of LOF/OCSVM/ABOD/Deep SVDD, and not significantly
different from KNN.

**Top-3 carried forward to Step 3: Isolation Forest, TabPFN, KNN.**

Reproduce:
```bash
python p53_od_pipeline.py --compare-methods --methods-set week5 --reduce pca --n-components 147 \
    --output-csv outputs/week5_compare.csv
python p53_od_pipeline.py --stat-test outputs/week5_compare.csv \
    --output-csv outputs/week5_significance.csv
```

### Step 2 — feature bagging: new aggregations + subspace source (Isolation Forest only)

Restricted to Isolation Forest only this step — feature bagging costs
`n_repeats x n_subsets` fits, too expensive for TabPFN at 25 repeats. Two
new aggregation variants added alongside sum/max/aom:
- **Pruned:** drop the feature subsets least correlated with the ensemble
  consensus (mean of all z-normalized subset scores) before averaging the
  rest.
- **Damped:** clip each subset's z-normalized scores to [-3, 3] before
  averaging, so one subset with an extreme, badly-calibrated score range
  can't dominate.

Subspace source compared via `--subspace-method`: **random** (unchanged
from Weeks 3/4) vs. **V-GAN with kernel learning** (Cribeiro-Ramallo et al.
2025, "Adversarial Subspace Generation for Outlier Detection in
High-Dimensional Data" — source bundled locally under `vgan_lib/`, no PyPI
package; needs `torch-two-sample`, also not on PyPI, installed with
`--no-build-isolation` same as the `gmd` workaround). One bug fixed in the
bundled source: `float(tensor.to('cpu').numpy())` crashed with
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
on our torch/numpy versions — fixed by switching to `tensor.item()`.

**Results (25 repeats, PCA-147; `no_bagging` = 0.7249 ± 0.0226 baseline,
identical for both subspace sources — confirms consistent splits):**

| Subspace source | Sum | Max | AOM | Pruned | Damped |
|---|---|---|---|---|---|
| Random | 0.7310 ± 0.0195 | 0.7308 ± 0.0206 | **0.7314 ± 0.0197** | 0.7298 ± 0.0198 | 0.7310 ± 0.0195 |
| V-GAN, kernel learning (50 epochs) | 0.7282 ± 0.0206 | 0.7289 ± 0.0212 | **0.7295 ± 0.0209** | 0.7227 ± 0.0200 | 0.7283 ± 0.0206 |

Random: Friedman p = 0.00007, sum/max/aom/pruned/damped all significantly
beat `no_bagging`. Among the five aggregations, aom and damped edge out
pruned specifically (both significantly better, `--`/`++`), while the
rest are not significantly different from each other.

**Random significance table:**

| | no_bagging | sum | max | aom | pruned | damped |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_bagging |  | -- | -- | -- | -- | -- |
| sum | ++ |  |  |  |  |  |
| max | ++ |  |  | - |  |  |
| aom | ++ |  | + |  | ++ |  |
| pruned | ++ |  |  | -- |  | -- |
| damped | ++ |  |  |  | ++ |  |

**V-GAN (kernel learning) significance table:** Friedman p = 0.00035.

| | no_bagging | sum | max | aom | pruned | damped |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_bagging |  |  |  | -- | + |  |
| sum |  |  |  | -- | ++ |  |
| max |  |  |  |  | ++ |  |
| aom | ++ | ++ |  |  | ++ |  |
| pruned | - | -- | -- | -- |  | -- |
| damped |  |  |  |  | ++ |  |

Only `aom` significantly beats `no_bagging` (p = 0.0037); `pruned` is the
only aggregation across the whole Step 2 sweep that lands *below*
`no_bagging`, and here that gap is itself marginally significant
(p = 0.061, `no_bagging` row vs. `pruned` column = `+`). The detector
loss was frozen at an identical value across dozens of epochs during
training — the adversarial game had already stalled, so V-GAN's
kernel-learning subspaces did not translate into a real advantage here.
**Random subspaces remain the recommended choice** — simpler, no extra
dependency, and numerically ahead of V-GAN (kernel learning) on every
aggregation.

Reproduce:
```bash
python p53_od_pipeline.py --feature-bagging --reduce pca --n-components 147 \
    --output-csv outputs/week5_feature_bagging.csv
python p53_od_pipeline.py --feature-bagging --subspace-method vgan_kernel --vgan-epochs 50 \
    --reduce pca --n-components 147 --output-csv outputs/week5_fb_vgan_kernel_50ep_full.csv
python p53_od_pipeline.py --stat-test outputs/week5_feature_bagging.csv \
    --output-csv outputs/week5_feature_bagging_significance.csv
```

### Step 3 — heterogeneous ensemble: Top-3 from Step 1

`isolation_forest`, `tabpfn`, `knn` (Step 1's Top-3) combined on the same
full 147-dim. feature space (no subspace split — opposite of Step 2).
Same aggregation recipe as Step 2, with `aom` swapped back for
**rank-averaging** (as in Weeks 3/4, `aom` doesn't apply meaningfully with
only 3 candidates): sum, max, rank-avg, pruned, damped.

| | AUC |
|---|-----|
| Isolation Forest (alone) | **0.7249 ± 0.0226** |
| TabPFN (alone) | 0.7036 ± 0.0286 |
| KNN (alone) | 0.6997 ± 0.0205 |
| Ensemble (sum) | 0.7144 ± 0.0212 |
| Ensemble (max) | 0.7130 ± 0.0188 |
| Ensemble (rank avg) | 0.7182 ± 0.0210 |
| Ensemble (pruned) | 0.7160 ± 0.0210 |
| Ensemble (damped) | 0.7144 ± 0.0212 |

Friedman chi-square = 72.91, p ≈ 0.0000 → H0 rejected. **Same negative
result as Weeks 3/4's Step C, confirmed again with a stronger Top-3:**
numerically, Isolation Forest alone (0.7249) beats every ensemble
variant (0.7130–0.7182) — mixing in the weaker TabPFN/KNN drags the
combination down rather than adding complementary signal. Testing the
five aggregation variants directly against each other (below) confirms
rank-averaging is the clear winner among them.

**Significance table (aggregation variants only):**

| | sum | max | rank_avg | pruned | damped |
|---|:---:|:---:|:---:|:---:|:---:|
| sum |  |  | -- | -- |  |
| max |  |  | -- |  |  |
| rank_avg | ++ | ++ |  | ++ | ++ |
| pruned | ++ |  | -- |  |  |
| damped |  |  | -- |  |  |

Rank-averaging significantly beats every other aggregation (sum, max,
pruned, damped); pruned is significantly better than sum but not
significantly different from max or damped. Rank-averaging's robustness
to the different score distributions of Isolation Forest/TabPFN/KNN (a
tree ensemble, a likelihood estimate, and a distance-based method — all
on very different native scales) is the most plausible reason it holds
up better than the other four aggregations here.

Reproduce:
```bash
python p53_od_pipeline.py --heterogeneous-ensemble --reduce pca --n-components 147 \
    --output-csv outputs/week5_heterogeneous_ensemble.csv
python p53_od_pipeline.py --stat-test outputs/week5_heterogeneous_ensemble.csv \
    --output-csv outputs/week5_heterogeneous_ensemble_significance.csv
```

### Week 5 best (OCC deliverable)

TabPFN turned out to be a genuine, competitive contender — but does not
beat the simpler Week 4 winner. V-GAN (kernel learning) did not improve
on random subspaces either. The recommended configuration is unchanged
from Week 4:

| Rank | Configuration | AUC | Why |
|------|---------------|-----|-----|
| **1 (use this)** | **Isolation Forest + PCA (147) + StandardScaler + feature bagging (AOM, random subspaces)** | **0.7314 ± 0.0197** | Best result across all of Step 2/3; simpler than V-GAN, no extra dependency |
| 2 | Isolation Forest + PCA (147) + scale (no bagging) | 0.7248 ± 0.0227 | Best single detector; simpler still |
| 3 | TabPFN (zero-shot, no tuning) | 0.7049 ± 0.0307 | Best non-IF single method; genuine second contender for future work |
| — | Heterogeneous ensemble (IF + TabPFN + KNN, any aggregation) | 0.7130–0.7182 | Below IF alone — do not use |

**Pipeline command for the Week 5 recommendation (same as Week 4):**

```bash
python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 147 \
    --feature-bagging --n-repeats 25 --n-subsets 10
```

## Week 6 — key findings

Course topic: Novel Techniques in OD for Large Databases. Structure this
week: (1) hyperparameter optimization (HPO) via Optuna for isolation_forest,
knn, deep_svdd, lunar, and dif, (2) a follow-up on n_components=200 for
PCA, including whether other reduction techniques beat PCA there, and
whether HPO improves Isolation Forest specifically at 200 dims.

### Step 1 — HPO for 5 models (Optuna, GP sampler, 150 trials × 15 repeats/trial, PCA-147)

HPO run for isolation_forest, knn, deep_svdd, lunar, and dif (search space:
3–4 hyperparameters per model, see `HPO_PARAM_SPACES` in
`p53_od_pipeline.py`). DIF's search was too slow to finish within budget
(~5 min/trial) — its best trial found so far
(`representation_dim=45, n_ensemble=75, n_estimators=10`) was used as its
"tuned" configuration.

**Default params:**

| | AUC |
|---|-----|
| Isolation Forest | **0.7248 ± 0.0227** |
| DIF | 0.7056 ± 0.0223 |
| KNN | 0.6997 ± 0.0205 |
| Deep SVDD | 0.6981 ± 0.0351 |
| LUNAR | 0.6736 ± 0.0210 |

**Tuned params (`--hpo`):**

| | AUC |
|---|-----|
| Isolation Forest | **0.7239 ± 0.0209** |
| DIF | 0.7087 ± 0.0228 |
| KNN | 0.7048 ± 0.0199 |
| Deep SVDD | 0.6981 ± 0.0351 |
| LUNAR | 0.6849 ± 0.0211 |

HPO gave a small, consistent gain for KNN/LUNAR/DIF (+0.3 to +1.1 pp),
essentially no change for Deep SVDD (tuning round-tripped back to the
same `[64,32]` architecture already used since Week 4) and no gain for
Isolation Forest (0.7248 → 0.7239, within noise). The ranking is
unchanged either way — Isolation Forest stays the best single method by
a clear margin.

Reproduce:
```bash
python p53_od_pipeline.py --hpo --hpo-model isolation_forest --reduce pca --n-components 147 \
    --hpo-trials 150 --hpo-repeats 15 --hpo-final-repeats 25 --output-csv outputs/hpo_isolation_forest.csv
# repeat for --hpo-model knn / deep_svdd / lunar, or --hpo-model all to run every model in one invocation

python p53_od_pipeline.py --compare-methods --methods-set week6_final --reduce pca --n-components 147 --n-repeats 25 \
    --output-csv outputs/week6_final_default_auc_matrix.csv
python p53_od_pipeline.py --compare-methods --methods-set week6_final --tuned --reduce pca --n-components 147 --n-repeats 25 \
    --output-csv outputs/week6_final_tuned_auc_matrix.csv
```

### Step 2 — reduction method sweep (Isolation Forest, 200 dims, 25 repeats)

Testing n_components=200 for the first time found a clear improvement
over the n=147 used throughout (AUC 0.7393 ± 0.0233 vs. 0.7248 ± 0.0227).
This step checks whether another reduction technique does even better at
the same target dimensionality.

| | AUC |
|---|-----|
| PCA | **0.7393 ± 0.0233** |
| Truncated SVD | 0.7391 ± 0.0213 |
| Random Projection | 0.6828 ± 0.0215 |
| UMAP | 0.5619 ± 0.0505 |

PCA and Truncated SVD are statistically indistinguishable here (expected —
Truncated SVD is PCA without centering). Random Projection and UMAP both
underperform PCA/SVD by a wide margin.

Reproduce:
```bash
python p53_od_pipeline.py --method isolation_forest --reduce-sweep pca,random_projection,truncated_svd,umap \
    --n-components 200 --n-repeats 25 --output-csv outputs/week6_reduce_sweep_auc_matrix.csv
```

### Step 3 — feature bagging, Isolation Forest at 200 dims (default params, 25 repeats, 10 subsets)

| | AUC |
|---|-----|
| No bagging | 0.7392 ± 0.0233 |
| Sum | 0.7429 ± 0.0198 |
| Max | 0.7409 ± 0.0201 |
| AOM | 0.7427 ± 0.0196 |
| Pruned | **0.7438 ± 0.0199** |
| Damped | 0.7429 ± 0.0198 |

Same small, consistent bagging gain (+0.5 to +0.9 pp) as the Week 5
147-dim result — the effect scales cleanly to 200 dims. `pruned` is the
new best aggregation variant, and **0.7438 is the best result of the
whole project so far.**

Reproduce:
```bash
python p53_od_pipeline.py --feature-bagging --reduce pca --n-components 200 --n-repeats 25 \
    --output-csv outputs/week6_feature_bagging_if_pca200.csv
```

### Step 4 — feature bagging, Isolation Forest at 200 dims, with HPO-tuned params

Separate HPO runs (150 trials × 15 repeats each) for Isolation Forest
specifically at 200 dims, run once for PCA and once for Truncated SVD —
since best-by-year hyperparameters can differ per reduction technique
(different feature scale/structure per method), not just per detector.

**PCA-200, tuned (`n_estimators=500, max_samples=0.1, max_features=0.831`):**

| | AUC |
|---|-----|
| No bagging | 0.7372 ± 0.0218 |
| Sum | 0.7369 ± 0.0203 |
| Max | 0.7372 ± 0.0200 |
| AOM | 0.7368 ± 0.0204 |
| Pruned | **0.7375 ± 0.0207** |
| Damped | 0.7369 ± 0.0203 |

**Truncated SVD-200, tuned (`n_estimators=84, max_samples=0.1, max_features=0.363`):**

| | AUC |
|---|-----|
| No bagging | 0.7323 ± 0.0227 |
| Sum | 0.7345 ± 0.0203 |
| Max | 0.7344 ± 0.0213 |
| AOM | **0.7349 ± 0.0204** |
| Pruned | 0.7337 ± 0.0211 |
| Damped | 0.7346 ± 0.0203 |

**Negative result:** neither tuned configuration beats Step 3's
default-params result (0.7438), and the SVD-200 no_bagging baseline
(0.7323) is actually below the equivalent default-params run from Step 2
(0.7391). All configurations here sit within roughly one standard
deviation of each other — the 200-dim Isolation Forest HPO search (15
repeats/trial) most likely landed on evaluation noise rather than a
genuine improvement over well-chosen defaults. Unlike KNN/LUNAR/DIF in
Step 1, HPO did not help Isolation Forest at either dimensionality tested
this week.

Reproduce:
```bash
python p53_od_pipeline.py --feature-bagging --bagging-methods isolation_forest --reduce pca --n-components 200 \
    --n-repeats 25 --detector-params-json '{"n_estimators": 500, "max_samples": 0.1, "max_features": 0.8312426187498287}' \
    --output-csv outputs/week6_feature_bagging_if_pca200_tuned.csv
python p53_od_pipeline.py --feature-bagging --bagging-methods isolation_forest --reduce truncated_svd --n-components 200 \
    --n-repeats 25 --detector-params-json '{"n_estimators": 84, "max_samples": 0.1, "max_features": 0.36321980494541484}' \
    --output-csv outputs/week6_feature_bagging_if_svd200_tuned.csv
```

### Week 6 best (OCC deliverable)

n_components=200 (Step 2) turned out to matter more than HPO (Steps 1
and 4) — dimensionality choice beat hyperparameter tuning as the more
effective lever this week, for Isolation Forest specifically.

| Rank | Configuration | AUC | Why |
|------|---------------|-----|-----|
| **1 (use this)** | **Isolation Forest + PCA (200) + StandardScaler + feature bagging (Pruned, default IF params, random subspaces)** | **0.7438 ± 0.0199** | Best result of the project so far; simple, no HPO needed |
| 2 | Isolation Forest + PCA (200) + scale (no bagging) | 0.7392 ± 0.0233 | Confirms n=200 > n=147 (0.7248) independent of bagging |
| 3 | Isolation Forest + Truncated SVD (200) + feature bagging (AOM, tuned params) | 0.7349 ± 0.0204 | Truncated SVD ties PCA on raw AUC (Step 2) but tuning + bagging together still land below PCA's untuned result |
| — | Isolation Forest + PCA (200), HPO-tuned (no bagging) | 0.7372 ± 0.0218 | HPO gave no measurable gain over defaults at this dimensionality |
| — | DIF / KNN / LUNAR / Deep SVDD (147 dims, tuned) | 0.6849–0.7087 | All below Isolation Forest; DIF and KNN benefited most from HPO among the non-IF models |

**Pipeline command for the Week 6 winner:**

```bash
python p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 200 \
    --feature-bagging --n-repeats 25 --n-subsets 10
```

Use **pruned** aggregation when reading Step 3 output
(`isolation_forest_pruned` column in
`outputs/week6_feature_bagging_if_pca200.csv`). Labels are for evaluation
only — never passed into `.fit()`.

----------------------------------------------------
OMAR,IF YOU SEE THIS; YOU CAN TRY OUT OVERSAMPLING, MODEL THE PROBELM AS KASSIFICATON PROBLEM NOT AS OUTLIER DETECTION PROBLEM OR TRY OUT ANOTHER IDEAS :)