# pca_2d_3d_blocks.py - separate PCA on the 2D and 3D feature blocks, then
# a second PCA on the concatenation of both results. Exploratory only,
# not part of p53_od_pipeline.py.
#
# NOTE on column split: README says 2D block = cols 0-4825 (4826 cols),
# 3D block = cols 4826-5407 (582 cols). Our own notes flag an unresolved
# 582 vs. 591 discrepancy for the 3D block - if the actual column count
# doesn't match 5408, the split below is wrong and needs re-checking
# against Danziger et al. 2006 before trusting these numbers.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DATA_FILE = "train_set_p53mutant.parquet"
N_2D = 4826  # columns 0-4825
VARIANCE_TARGET = 0.95


def fit_pca_95(X, label, n_cap=1000):
	print("=" * 60)
	print(f"PCA on {label}  (shape: {X.shape})")
	print("=" * 60)

	# n_components=0.95 (variance target) forces svd_solver="full" in sklearn,
	# which does an exact SVD on the whole matrix - too slow at ~4800 cols.
	# Instead: randomized PCA with a fixed cap, then find the 95% cutoff
	# from the cumulative explained variance ourselves.
	n_cap = min(n_cap, X.shape[1] - 1, X.shape[0] - 1)

	# not standardized - same reasoning as the Week 1 full-feature PCA:
	# preserve outlier-driven variance rather than treating all features
	# as equally important
	pca = PCA(n_components=n_cap, svd_solver="randomized", random_state=42)
	Z_full = pca.fit_transform(X)

	cumvar = np.cumsum(pca.explained_variance_ratio_)
	n_needed = int(np.searchsorted(cumvar, VARIANCE_TARGET) + 1)

	if cumvar[-1] < VARIANCE_TARGET:
		print(f"WARNING: only {cumvar[-1]:.2%} variance explained by {n_cap} "
		      f"components (cap too low) - raise n_cap and re-run.")
		n_needed = n_cap

	print(f"Components for {VARIANCE_TARGET:.0%} variance: {n_needed}  "
	      f"(cumulative variance at cap={n_cap}: {cumvar[-1]:.2%})")
	print(f"Explained variance (first 5): {pca.explained_variance_ratio_[:5].round(4)}")
	print("=" * 60 + "\n")

	pca.n_components_ = n_needed  # for the summary print at the end
	Z = Z_full[:, :n_needed]
	return pca, Z


def main():
	print("Loading data...")
	df = pd.read_parquet(DATA_FILE)
	print(f"Loaded shape: {df.shape}")

	if df.shape[1] != 5408:
		print(f"WARNING: expected 5408 feature columns, got {df.shape[1]}. "
		      f"The 2D/3D split below (N_2D={N_2D}) may be wrong - check "
		      f"the 582 vs. 591 discrepancy before trusting these results.\n")

	X_2d = df.iloc[:, :N_2D].values.astype(np.float64)
	X_3d = df.iloc[:, N_2D:].values.astype(np.float64)
	print(f"2D block: {X_2d.shape}   3D block: {X_3d.shape}\n")

	pca_2d, Z_2d = fit_pca_95(X_2d, "2D block")
	pca_3d, Z_3d = fit_pca_95(X_3d, "3D block")

	# second-stage PCA on the concatenation of both PCA outputs
	Z_combined_raw = np.hstack([Z_2d, Z_3d])
	print(f"Concatenated 2D+3D PCA output shape: {Z_combined_raw.shape}\n")

	pca_stage2, Z_stage2 = fit_pca_95(Z_combined_raw, "concatenated 2D+3D PCA output (stage 2)")

	print("=" * 60)
	print("SUMMARY")
	print("=" * 60)
	print(f"2D block:        {X_2d.shape[1]:>5} raw -> {pca_2d.n_components_:>4} components (95% var)")
	print(f"3D block:        {X_3d.shape[1]:>5} raw -> {pca_3d.n_components_:>4} components (95% var)")
	print(f"Concatenated:    {Z_combined_raw.shape[1]:>5} raw -> {pca_stage2.n_components_:>4} components (95% var, stage 2)")
	print("=" * 60)


if __name__ == "__main__":
	main()
