"""
Final train/test AUC for the best OCC and label-using (supervised/SMOTE) models.

Train AUC  = fit on train (OCC: inliers only), score full train, AUC vs train labels
Test AUC   = same fitted model, score held-out test_set, AUC vs
             _internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root, so p53_od_pipeline can be imported
from p53_od_pipeline import (
	anomaly_scores,
	build_pipeline,
	fit_pipeline_train,
	fit_scale_reduce,
	load_data,
	transform_scale_reduce,
)


TEST_LABELS = "_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv"
OUT_CSV = Path("outputs/final_train_test_auc.csv")
N_SUBSETS = 10
SUBSET_MIN_FRAC = 0.5
RANDOM_STATE = 42


def load_test_labels(index):
	labels = pd.read_csv(TEST_LABELS, index_col=0)["label"]
	labels = labels.reindex(index)
	if labels.isna().any():
		raise ValueError("test labels do not align with test_set index")
	return labels


def auc_binary(scores, labels):
	y = np.asarray(labels == "active", dtype=int)
	return float(roc_auc_score(y, np.asarray(scores)))


def fit_bagging_pruned(X_tr_red, n_estimators=200):
	"""Fit IF on random feature subsets; return (detectors, col_masks)."""
	d = X_tr_red.shape[1]
	rng = np.random.default_rng(RANDOM_STATE)
	detectors, col_masks = [], []
	for j in range(N_SUBSETS):
		k = int(rng.integers(int(SUBSET_MIN_FRAC * d), d + 1))
		cols = rng.choice(d, size=k, replace=False)
		det = IsolationForest(
			n_estimators=n_estimators, random_state=RANDOM_STATE + j + 1, n_jobs=-1,
		)
		det.fit(X_tr_red[:, cols])
		detectors.append(det)
		col_masks.append(cols)
	return detectors, col_masks


def score_bagging_pruned(detectors, col_masks, X_score_red):
	"""Z-normalize each subset on the scored set, then pruned average."""
	subset_scores = []
	for det, cols in zip(detectors, col_masks):
		raw = -det.score_samples(X_score_red[:, cols])
		z = (raw - raw.mean()) / (raw.std() + 1e-12)
		subset_scores.append(z)
	mat = np.vstack(subset_scores)
	consensus = mat.mean(axis=0)
	corrs = np.array([np.corrcoef(mat[s], consensus)[0, 1] for s in range(len(detectors))])
	keep = corrs >= np.median(corrs)
	return mat[keep].mean(axis=0)


def eval_if_pca(X_train, train_labels, X_test, test_labels, n_components=200, bagging=False):
	name = f"IF + PCA({n_components}) + scale" + (" + bagging (pruned)" if bagging else "")
	print("\n" + "=" * 60)
	print(name)
	print("=" * 60)

	inlier_mask = train_labels != "active"
	X_fit = X_train.loc[inlier_mask]
	print(f"  OCC fit on {len(X_fit):,} inliers (of {len(X_train):,} train)")

	t0 = time.time()
	X_fit_red, scaler, reducer = fit_scale_reduce(
		X_fit, "pca", n_components, RANDOM_STATE, True, return_fitted=True,
	)
	X_train_red = transform_scale_reduce(X_train, scaler, reducer)
	X_test_red = transform_scale_reduce(X_test, scaler, reducer)

	if bagging:
		detectors, col_masks = fit_bagging_pruned(X_fit_red)
		train_scores = score_bagging_pruned(detectors, col_masks, X_train_red)
		test_scores = score_bagging_pruned(detectors, col_masks, X_test_red)
	else:
		det = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
		det.fit(X_fit_red)
		train_scores = -det.score_samples(X_train_red)
		test_scores = -det.score_samples(X_test_red)

	train_auc = auc_binary(train_scores, train_labels)
	test_auc = auc_binary(test_scores, test_labels)
	elapsed = time.time() - t0
	print(f"  Train AUC={train_auc:.4f}  Test AUC={test_auc:.4f}  ({elapsed:.1f}s)")
	return {
		"setting": "OCC",
		"model": name,
		"train_auc": train_auc,
		"test_auc": test_auc,
		"wall_sec": elapsed,
	}


def eval_gmd_lof_rp_smote(X_train, train_labels, X_test, test_labels, n_components):
	name = f"GMD+LOF + RP({n_components}) + scale + SMOTE"
	print("\n" + "=" * 60)
	print(name)
	print("=" * 60)
	print("  Label-using fit (SMOTE on full train)")

	t0 = time.time()
	pipeline = build_pipeline(
		reduce="random_projection",
		method="gmd_lof",
		n_components=n_components,
		n_estimators=200,
		random_state=RANDOM_STATE,
		max_subspaces=None,
		scale=True,
		feature_select=None,
	)
	fit_pipeline_train(pipeline, X_train, train_labels, smote=True)
	train_scores = anomaly_scores(pipeline, X_train)
	test_scores = anomaly_scores(pipeline, X_test)
	train_auc = auc_binary(train_scores, train_labels)
	test_auc = auc_binary(test_scores, test_labels)
	elapsed = time.time() - t0
	print(f"  Train AUC={train_auc:.4f}  Test AUC={test_auc:.4f}  ({elapsed:.1f}s)")
	return {
		"setting": "supervised / label-using (SMOTE)",
		"model": name,
		"train_auc": train_auc,
		"test_auc": test_auc,
		"wall_sec": elapsed,
	}


def main():
	X_train, X_test, train_labels = load_data()
	test_labels = load_test_labels(X_test.index)
	print(f"Test labels: active={(test_labels == 'active').sum()}, "
	      f"inactive={(test_labels != 'active').sum()}")

	rows = []
	rows.append(eval_if_pca(X_train, train_labels, X_test, test_labels, 200, bagging=False))
	rows.append(eval_if_pca(X_train, train_labels, X_test, test_labels, 200, bagging=True))
	rows.append(eval_if_pca(X_train, train_labels, X_test, test_labels, 147, bagging=True))
	rows.append(eval_gmd_lof_rp_smote(X_train, train_labels, X_test, test_labels, 147))
	rows.append(eval_gmd_lof_rp_smote(X_train, train_labels, X_test, test_labels, 200))

	df = pd.DataFrame(rows)
	OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUT_CSV, index=False)

	print("\n" + "=" * 80)
	print("FINAL TABLE")
	print("=" * 80)
	print(f"{'Setting':<36} {'Model':<48} {'Train AUC':>10} {'Test AUC':>10}")
	print("-" * 80)
	for _, r in df.iterrows():
		print(f"{r['setting']:<36} {r['model']:<48} {r['train_auc']:>10.4f} {r['test_auc']:>10.4f}")
	print("=" * 80)
	print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
	main()
