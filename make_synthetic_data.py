"""Generate a small SYNTHETIC dataset in the same raw format as the course data.

The real p53 data (provided by the course supervisors) is not redistributed.
This script creates random data with the same layout so the whole pipeline can
be run end to end:

    python make_synthetic_data.py             # writes synthetic/train_data.data + synthetic/train_gt.csv
    cd synthetic
    python ../split_data.py
    python ../p53_od_pipeline.py --method isolation_forest --reduce pca --n-components 50

Layout (same as the real file):
- 5,408 features: 2D block (4,826 columns) + 3D block (582 columns)
- one empty trailing artifact column ("5409")
- a few fully empty rows, rows missing the 2D block and duplicate rows,
  so the cleaning steps in split_data.py have something to do
- labels "active" / "inactive" with a strong class imbalance

The numbers are random. Results on this data say nothing about the real task.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_2D = 4826
N_3D = 582
N_FEATURES = N_2D + N_3D


def main():
	parser = argparse.ArgumentParser(description="Create synthetic data in the raw p53 format")
	parser.add_argument("--n-rows", type=int, default=1500, help="number of valid rows")
	parser.add_argument("--n-active", type=int, default=30, help="number of 'active' (outlier) rows")
	parser.add_argument("--n-latent", type=int, default=15, help="latent dimensions of the inlier structure")
	parser.add_argument("--outlier-strength", type=float, default=0.05,
	                     help="mean shift of the 'active' rows (smaller = harder to detect)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out-dir", type=str, default="synthetic")
	args = parser.parse_args()

	rng = np.random.default_rng(args.seed)
	n, k = args.n_rows, args.n_latent

	# Inliers: low-rank structure plus noise, so PCA has something to find.
	loadings = rng.normal(size=(k, N_FEATURES)).astype(np.float32)
	latent = rng.normal(size=(n, k)).astype(np.float32)
	X = latent @ loadings * 0.1 + rng.normal(scale=0.05, size=(n, N_FEATURES)).astype(np.float32)

	# Outliers ("active"): shifted along a subset of features.
	active_idx = rng.choice(n, size=args.n_active, replace=False)
	shifted = rng.choice(N_FEATURES, size=300, replace=False)
	X[np.ix_(active_idx, shifted)] += rng.normal(loc=args.outlier_strength, scale=args.outlier_strength / 2, size=(args.n_active, 300)).astype(np.float32)

	labels = np.full(n, "inactive", dtype=object)
	labels[active_idx] = "active"

	df = pd.DataFrame(X, columns=[str(i) for i in range(N_FEATURES)])
	df["5409"] = np.nan  # empty parsing artifact column, as in the real file

	# Messy rows, so the cleaning logic is exercised.
	messy = []
	empty = pd.DataFrame(np.nan, index=range(3), columns=df.columns)
	missing_2d = df.iloc[:3].copy()
	missing_2d.iloc[:, :N_2D] = np.nan
	dups = df.iloc[10:12].copy()
	messy = pd.concat([empty, missing_2d, dups])

	df = pd.concat([df, messy], ignore_index=True)
	all_labels = np.concatenate([labels, ["inactive"] * len(messy)])

	df.index.name = None
	out = Path(args.out_dir)
	out.mkdir(parents=True, exist_ok=True)

	# Raw feature file: first (unnamed) column is the row id, "?" marks missing values.
	df.to_csv(out / "train_data.data", float_format="%.4f", na_rep="?")
	pd.DataFrame({"5408": all_labels}).to_csv(out / "train_gt.csv")

	print(f"Wrote {out / 'train_data.data'} ({len(df)} rows x {N_FEATURES} features)")
	print(f"Wrote {out / 'train_gt.csv'} ({args.n_active} active / {n - args.n_active} inactive valid rows)")
	print("Synthetic random data. Not comparable to the real task.")


if __name__ == "__main__":
	main()
