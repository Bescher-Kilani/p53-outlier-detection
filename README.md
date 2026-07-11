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
point estimates.

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
| Isolation Forest | 0.7249 ± 0.0226 | 0.7310 ± 0.0195 | 0.7308 ± 0.0206 | **0.7314 ± 0.0197** |
| KNN | 0.6997 ± 0.0205 | **0.7016 ± 0.0202** | 0.6976 ± 0.0210 | 0.6999 ± 0.0203 |
| OCSVM | 0.6860 ± 0.0175 | 0.6953 ± 0.0186 | **0.6970 ± 0.0177** | 0.6920 ± 0.0180 |

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