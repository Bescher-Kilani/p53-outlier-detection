"""CV-only: GMD+LOF + Random Projection + scale + SMOTE (matches Week 2 table column)."""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root, so p53_od_pipeline can be imported
from p53_od_pipeline import (
	_best_mcc_threshold,
	anomaly_scores,
	build_pipeline,
	fit_pipeline_train,
	load_data,
)


def run_cv(n_components, n_splits=3, random_state=42):
	X_train, _, train_labels = load_data()
	y = np.asarray(train_labels == "active", dtype=int)
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	fold_aucs, fold_mccs = [], []
	t0 = time.time()

	class Args:
		pass

	for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), start=1):
		print(f"\n=== fold {fold}/{n_splits}  n_components={n_components} ===", flush=True)
		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		labels_tr = train_labels.iloc[train_idx]
		labels_val = train_labels.iloc[val_idx]

		pipeline = build_pipeline(
			reduce="random_projection",
			method="gmd_lof",
			n_components=n_components,
			n_estimators=200,
			random_state=random_state,
			max_subspaces=None,
			scale=True,
			feature_select=None,
		)
		fit_pipeline_train(pipeline, X_tr, labels_tr, smote=True)
		scores = anomaly_scores(pipeline, X_val)
		y_true = np.asarray(labels_val == "active", dtype=int)
		auc = roc_auc_score(y_true, scores)
		_, mcc = _best_mcc_threshold(y_true, scores)
		fold_aucs.append(auc)
		fold_mccs.append(mcc)
		print(f"  fold {fold}: AUC={auc:.4f}  MCC={mcc:.4f}", flush=True)

	aucs = np.asarray(fold_aucs)
	print("\n" + "=" * 60)
	print(f"GMD+LOF + RP({n_components}) + scale + SMOTE | {n_splits}-fold stratified CV")
	print(f"AUC: {aucs.mean():.4f} +/- {aucs.std(ddof=1):.4f}")
	print(f"folds: {[round(a, 4) for a in aucs]}")
	print(f"wall: {(time.time() - t0) / 60:.1f} min")
	print("=" * 60)
	return float(aucs.mean()), float(aucs.std(ddof=1)), aucs.tolist()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n-components", type=int, required=True)
	parser.add_argument("--cv-folds", type=int, default=3)
	args = parser.parse_args()
	run_cv(args.n_components, n_splits=args.cv_folds)


if __name__ == "__main__":
	main()
