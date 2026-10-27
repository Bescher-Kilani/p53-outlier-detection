# pca_2d_3d_blocks_cv.py - 3-fold stratified CV: Isolation Forest on the
# two-stage 2D/3D-block PCA representation (see pca_2d_3d_blocks.py for the
# component counts this is based on: 2D->145, 3D->148, concat->62).
# No-leakage rule kept: every PCA/scaler is fit on the train fold only.

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_2D = 4826  # columns 0-4825 = 2D block, rest = 3D block
N_COMP_2D = 145   # from pca_2d_3d_blocks.py full-train run (95% var)
N_COMP_3D = 148   # from pca_2d_3d_blocks.py full-train run (95% var)
N_COMP_STAGE2 = 62  # from pca_2d_3d_blocks.py full-train run (95% var)
N_SPLITS = 3


def reduce_2d_3d_stage2(X_train, X_val, scale=True):
	X_train_2d, X_train_3d = X_train[:, :N_2D], X_train[:, N_2D:]
	X_val_2d, X_val_3d = X_val[:, :N_2D], X_val[:, N_2D:]

	if scale:
		scaler_2d = StandardScaler().fit(X_train_2d)
		scaler_3d = StandardScaler().fit(X_train_3d)
		X_train_2d, X_val_2d = scaler_2d.transform(X_train_2d), scaler_2d.transform(X_val_2d)
		X_train_3d, X_val_3d = scaler_3d.transform(X_train_3d), scaler_3d.transform(X_val_3d)

	pca_2d = PCA(n_components=N_COMP_2D, svd_solver="randomized", random_state=42).fit(X_train_2d)
	pca_3d = PCA(n_components=N_COMP_3D, svd_solver="randomized", random_state=42).fit(X_train_3d)

	Z_train = np.hstack([pca_2d.transform(X_train_2d), pca_3d.transform(X_train_3d)])
	Z_val = np.hstack([pca_2d.transform(X_val_2d), pca_3d.transform(X_val_3d)])

	pca_stage2 = PCA(n_components=N_COMP_STAGE2, svd_solver="randomized", random_state=42).fit(Z_train)
	return pca_stage2.transform(Z_train), pca_stage2.transform(Z_val)


def main():
	print("Loading data...")
	X = pd.read_parquet("train_set_p53mutant.parquet").values.astype(np.float64)
	labels = pd.read_csv("train_labels_p53mutant.csv", index_col=0)["label"]
	y = np.asarray(labels == "active", dtype=int)
	print(f"X: {X.shape}   active: {y.sum()}\n")

	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
	aucs = []

	for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
		t0 = time.time()
		X_train, X_val = X[train_idx], X[val_idx]
		y_val = y[val_idx]

		Z_train, Z_val = reduce_2d_3d_stage2(X_train, X_val, scale=True)

		clf = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
		clf.fit(Z_train)
		scores_val = -clf.score_samples(Z_val)

		auc = roc_auc_score(y_val, scores_val)
		aucs.append(auc)
		print(f"fold {fold}/{N_SPLITS}: AUC={auc:.4f}  "
		      f"(n_components final={Z_train.shape[1]}, {time.time()-t0:.1f}s)")

	print("\n" + "=" * 60)
	print("SUMMARY - Isolation Forest + 2D/3D-block PCA (stage 2) + Scaler")
	print("=" * 60)
	print(f"CV AUC: {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1):.4f}")
	print("Baseline for comparison (single PCA, 147 comp, Scaler): 0.7291 +/- 0.0543")
	print("=" * 60)


if __name__ == "__main__":
	main()
