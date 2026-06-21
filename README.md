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
├── compute_correlations.py   # Pearson/Spearman via BLAS matmul, run on the KIT server
├── requirements.txt
└── notbooks/
    └── data_expolartion.ipynb
```

Raw data (`train_data.data`, `train_gt.csv`) and large intermediate files
(`train_data_clean.parquet`, `pearson_corr.parquet`, `spearman_corr.parquet`)
are gitignored — see "Reproducing the pipeline" below.

## Reproducing the pipeline

1. Place `train_data.data` and `train_gt.csv` (from ILIAS) in this folder.
2. Run `notbooks/data_expolartion.ipynb` top to bottom through **Cell 12**
   ("Export the cleaned dataframe to Parquet"). This produces
   `train_data_clean.parquet` (cleaning steps: drop fully-empty rows, drop
   rows missing the 2D block, drop exact duplicates, downcast to float32,
   drop the empty trailing column).
3. Copy `train_data_clean.parquet` to the KIT server
   (`server`), then run there:
   ```
   python3 compute_correlations.py
   ```
   This produces `pearson_corr.parquet` and `spearman_corr.parquet`
   (~1.7s each via matrix-multiplication, vs. minutes+ with pandas `.corr()`).
   Run as a standalone script rather than in the notebook itself, because
   the remote Jupyter kernel on the server had dependency issues at the time.
4. Copy both result files back into this folder, then continue running
   the notebook from **Cell 13** onward ("Load and plot the precomputed
   Pearson correlation matrix") through the correlation heatmaps,
   top-pair / Mutual Information analysis, PCA, and hierarchical clustering.

## Notebook overview

Setup → structure inspection → missing-value analysis → cleaning pipeline
→ label alignment (verification only) → Parquet export → correlation
heatmaps → top-pairs + Mutual Information → PCA → hierarchical clustering
→ summary.

## Week 1 — key findings

- Raw: 31,021 rows × 5,408 features (2D electrostatic/surface block,
  3D distance-based block) + 1 empty parsing artifact column
- Cleaned: 30,754 rows (dropped 137 fully-empty, 124 missing the 2D block,
  5 duplicates)
- Size: 3.2 GB CSV → 665.5 MB in-RAM (float32) → 281.4 MB Parquet
- Correlation: mostly weak; one strong block (~index 2400–4800); a few
  long-range pairs (e.g. 3197–4649) confirmed by Mutual Information
- PCA: 147 of 5,408 components capture 95% variance (not standardized,
  to preserve outlier-driven variance)
- Hierarchical clustering: reordering reveals correlated feature groups
  hidden in the original index order, including one inverse block
  (~index 4500–5300)
- Labels (verification only): 30,693 inactive / 61 active (~503:1)