"""
p53 Outlier Detection Pipeline
Team 2 - DS Lab 2026

Pipeline Steps:
1. Load train_set/test_set/train_labels (produced by split_data.py)
2. Fit an unsupervised sklearn Pipeline on train_set features only (no labels)
   - scale step (optional): StandardScaler, fit on train fold / full train only
   - reduction step (optional): PCA, Random Projection, etc.
   - detector step: Isolation Forest, LOF, COF, KNN, OCSVM, Fast-ABOD, or GMD+LOF ensemble
3. Score train_set and test_set
4. Internal metrics check (train_set only), via --eval-strategy:
   - cv (default, unchanged from before): 3-fold stratified CV
   - repeated_subsampling (new, Week 3 Folie 7): 80% of inliers train,
     remaining inliers + 80% of outliers val, repeated --n-repeats times,
     inlier-only fit
   Both report AUC and MCC as mean +/- std.
5. Save test_set scores as team2_prediction.csv

Method comparison (--compare-methods): runs the Week 3 OCC methods in
COMPARE_METHODS through repeated_subsampling (always, regardless of
--eval-strategy), prints the per-repeat AUC matrix to the terminal (no
file written) - this is the input for next week's Friedman/Conover-Iman
test, just copy it from the terminal.


"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import OneClassSVM

from mrmr_ifs import MRMRIFSSelector

COMPARE_METHODS = ["isolation_forest", "lof", "knn", "ocsvm", "abod"]

# Step B (Week 3 plan): Top-3 from Step A, one family each (subspace/locality/cluster)
FEATURE_BAGGING_METHODS = ["isolation_forest", "knn", "ocsvm"]


def load_data():
	print("\n" + "="*60)
	print("LOADING DATA")
	print("="*60)

	X_train = pd.read_parquet("train_set_p53mutant.parquet")
	X_test = pd.read_parquet("test_set_p53mutant.parquet")
	train_labels = pd.read_csv("train_labels_p53mutant.csv", index_col=0)["label"]

	print(f"Train set: {X_train.shape}")
	print(f"Test set:  {X_test.shape}")
	print(f"Train labels: {len(train_labels)} (active: {(train_labels == 'active').sum()})")
	print("="*60 + "\n")

	return X_train, X_test, train_labels


def make_reducer(reduce, n_components, random_state):
	if reduce == "none":
		return "passthrough"

	if reduce == "pca":
		return PCA(n_components=n_components, random_state=random_state)
	if reduce == "random_projection":
		return GaussianRandomProjection(n_components=n_components, random_state=random_state)
	if reduce == "truncated_svd":
		return TruncatedSVD(n_components=n_components, random_state=random_state)
	if reduce == "isomap":
		from sklearn.manifold import Isomap
		return Isomap(n_components=n_components)
	if reduce == "umap":
		import umap  # pip install umap-learn
		return umap.UMAP(n_components=n_components, random_state=random_state)

	raise ValueError(f"Unknown reduction method: {reduce}")


class GMDLOFEnsemble(BaseEstimator):
	# sklearn-compatible final step for Pipeline; sums LOF scores across GMD subspaces.
	def __init__(self, max_subspaces=None, random_state=42):
		self.max_subspaces = max_subspaces
		self.random_state = random_state

	def fit(self, X, y=None):
		from gmd import GMD

		X_arr = np.asarray(X)
		t0 = time.time()
		gmd = GMD()
		gmd.fit(X_arr)
		subspace_items = list(gmd.subspaces_.items())
		self.n_subspaces_found_ = len(subspace_items)
		if self.max_subspaces is not None:
			subspace_items = subspace_items[:self.max_subspaces]
		self.subspace_items_ = subspace_items
		self.lof_models_ = []
		for i, (_, cols) in enumerate(subspace_items):
			lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
			lof.fit(X_arr[:, cols])
			self.lof_models_.append((cols, lof))
			if (i + 1) % 50 == 0 or (i + 1) == len(subspace_items):
				print(f"  Subspace {i+1}/{len(subspace_items)}  ({time.time()-t0:.1f}s elapsed)")
		self.fit_time_ = time.time() - t0
		return self

	def score_samples(self, X):
		X_arr = np.asarray(X)
		scores = np.zeros(len(X_arr))
		for cols, lof in self.lof_models_:
			scores += lof.score_samples(X_arr[:, cols])
		return scores


class PyODAdapter(BaseEstimator):
	# Wraps a PyOD detector so it exposes score_samples() like sklearn does
	# (higher = more normal). PyOD's own convention is the opposite
	# (decision_function: higher = more anomalous), so we flip the sign
	# here once - anomaly_scores() below flips it back for every method
	# uniformly, no special-casing needed downstream.
	def __init__(self, model):
		self.model = model

	def fit(self, X, y=None):
		self.model.fit(np.asarray(X))
		return self

	def score_samples(self, X):
		return -self.model.decision_function(np.asarray(X))


def make_detector(method, n_estimators, random_state, max_subspaces):
	if method == "isolation_forest":
		return IsolationForest(
			n_estimators=n_estimators,
			random_state=random_state,
			n_jobs=-1,
		)
	if method == "lof":
		return LocalOutlierFactor(novelty=True, n_jobs=-1)
	if method == "gmd_lof":
		return GMDLOFEnsemble(max_subspaces=max_subspaces, random_state=random_state)
	if method == "cof":
		from pyod.models.cof import COF
		return PyODAdapter(COF(n_neighbors=20))
	if method == "knn":
		from pyod.models.knn import KNN
		return PyODAdapter(KNN(n_neighbors=20, method="largest", n_jobs=-1))
	if method == "ocsvm":
		# nu = upper bound on the fraction of training points allowed to be
		# support vectors / margin errors, roughly analogous to sklearn's
		# `contamination` elsewhere in this pipeline. 0.05 is a deliberately
		# loose cap, not the true ~0.2% contamination - OCSVM tends to need
		# a nu comfortably above the true outlier fraction to fit a stable
		# boundary at all; worth sweeping later if OCSVM makes the Top-3.
		return OneClassSVM(kernel="rbf", nu=0.05)
	if method == "abod":
		from pyod.models.abod import ABOD
		# method="fast" = neighbor-based approximation (naive ABOD is
		# O(n^3), infeasible at ~20k train rows)
		return PyODAdapter(ABOD(n_neighbors=10, method="fast"))
	raise ValueError(f"Unknown method: {method}")


def fit_scale_reduce(X, reduce, n_components, random_state, scale):
	# Fits scale+reduce once on the full array, separate from build_pipeline
	# (which bundles reduce+detector together) - needed for Step B feature
	# bagging, where the SAME reduced feature space is reused across many
	# detector fits on different column subsets, instead of re-fitting PCA
	# per subset.
	X_arr = np.asarray(X, dtype=np.float64)
	if scale:
		X_arr = StandardScaler().fit_transform(X_arr)
	reducer = make_reducer(reduce, n_components, random_state)
	if reducer == "passthrough":
		return X_arr
	return reducer.fit_transform(X_arr)


def build_pipeline(reduce, method, n_components=147, n_estimators=200,
                   random_state=42, max_subspaces=None, scale=True,
                   feature_select=None):
	steps = []
	if scale:
		steps.append(("scale", StandardScaler()))
	if feature_select == "mrmr_ifs":
		steps.append(("select", MRMRIFSSelector(
			max_features=n_components,
			random_state=random_state,
			n_estimators=min(100, n_estimators),
		)))
		reduce = "none"
	steps.extend([
		("reduce", make_reducer(reduce, n_components, random_state)),
		("detector", make_detector(method, n_estimators, random_state, max_subspaces)),
	])
	return Pipeline(steps)


def _transform_features(pipeline, X):
	prep = pipeline[:-1]
	if not prep.steps or all(step == "passthrough" for _, step in prep.steps):
		return X
	return prep.transform(X)


def anomaly_scores(pipeline, X):
	# sklearn score_samples: higher = more normal; flip for higher = more anomalous.
	X_red = _transform_features(pipeline, X)
	return -pipeline.named_steps["detector"].score_samples(X_red)


def fit_pipeline_train(pipeline, X_train, train_labels=None, smote=False):
	needs_custom = smote or "select" in pipeline.named_steps
	if not needs_custom:
		pipeline.fit(X_train)
		return pipeline

	from imblearn.over_sampling import SMOTE

	X = X_train
	y = None
	if train_labels is not None:
		y = np.asarray(train_labels == "active", dtype=int)

	if "scale" in pipeline.named_steps:
		X = pipeline.named_steps["scale"].fit_transform(X_train)

	if smote:
		n_minority = int(y.sum())
		k_neighbors = min(5, n_minority - 1)
		if k_neighbors < 1:
			raise ValueError(f"Not enough minority samples for SMOTE (n={n_minority})")
		X, y = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(X, y)

	if "select" in pipeline.named_steps:
		selector = pipeline.named_steps["select"]
		selector.fit(X, y)
		X = selector.transform(X)

	reducer = pipeline.named_steps["reduce"]
	X_red = X if reducer == "passthrough" else reducer.fit_transform(X)
	pipeline.named_steps["detector"].fit(X_red)
	return pipeline


def log_reduction(pipeline, input_shape, reduce, n_train=None):
	if reduce == "none":
		return

	reducer = pipeline.named_steps["reduce"]
	print("="*60)
	print(f"DIMENSIONALITY REDUCTION ({reduce}, "
	      f"n_components={getattr(reducer, 'n_components', '?')})")
	print("="*60)
	if reduce == "pca":
		print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
	print(f"Shape: {input_shape} -> ({n_train or input_shape[0]}, {reducer.n_components})")
	print("="*60 + "\n")


_DETECTOR_LABELS = {
	"isolation_forest": "FITTING ISOLATION FOREST",
	"lof": "FITTING LOF (single model, no subspace ensemble)",
	"cof": "FITTING COF (PyOD, n_neighbors=20)",
	"knn": "FITTING KNN (PyOD, n_neighbors=20)",
	"ocsvm": "FITTING ONE-CLASS SVM (sklearn, rbf kernel)",
	"abod": "FITTING FAST-ABOD (PyOD, n_neighbors=10)",
}


def log_detector_fit(pipeline, method, n_train, elapsed):
	print("="*60)
	if method == "gmd_lof":
		detector = pipeline.named_steps["detector"]
		print("FITTING GMD + LOF (subspace ensemble)")
		print(f"GMD found {detector.n_subspaces_found_} subspaces "
		      f"(using {len(detector.subspace_items_)})")
		elapsed = detector.fit_time_
	else:
		print(_DETECTOR_LABELS.get(method, f"FITTING {method.upper()}"))
	print("="*60)
	print(f"Fitted on {n_train:,} samples in {elapsed:.1f}s")
	print("="*60 + "\n")


def fit_pipeline(pipeline, X_train, method, reduce, scale=True, train_labels=None, smote=False):
	input_shape = X_train.shape
	if scale:
		print("="*60)
		print("STANDARDIZING FEATURES (StandardScaler, fit on train only)")
		print("="*60 + "\n")
	if smote:
		print("="*60)
		print("SMOTE on train only (after scaling, before reduction/detector)")
		print("="*60 + "\n")
	t0 = time.time()
	fit_pipeline_train(pipeline, X_train, train_labels, smote=smote)
	elapsed = time.time() - t0
	n_train = None
	if smote and "reduce" in pipeline.named_steps:
		reducer = pipeline.named_steps["reduce"]
		if reducer != "passthrough" and hasattr(reducer, "n_components"):
			n_train = getattr(reducer, "n_samples_", None)
	log_reduction(pipeline, input_shape, reduce, n_train=n_train)
	log_detector_fit(pipeline, method, n_train or len(X_train), elapsed)
	return pipeline


def _mcc_from_counts(tp, tn, fp, fn):
	tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
	denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
	if denom == 0:
		return 0.0
	return (tp * tn - fp * fn) / np.sqrt(denom)


def _best_mcc_threshold(y_true, scores):
	# O(n): try predicting the top-k highest scores as active (k = 0..n).
	# Equivalent to sweeping thresholds with >=; avoids O(n^2) over np.unique(scores).
	y_true = np.asarray(y_true)
	scores = np.asarray(scores)
	order = np.argsort(-scores, kind="mergesort")
	y_sorted = y_true[order]

	n_pos = int(y_true.sum())
	n_neg = len(y_true) - n_pos

	best_mcc = -1.0
	best_thresh = float(np.min(scores)) - 1.0
	tp = 0

	# k=0: predict nobody active
	mcc = _mcc_from_counts(0, n_neg, 0, n_pos)
	if mcc > best_mcc:
		best_mcc = mcc
		best_thresh = float(np.max(scores)) + 1.0

	for k, idx in enumerate(order):
		tp += y_sorted[k]
		fp = (k + 1) - tp
		fn = n_pos - tp
		tn = n_neg - fp
		mcc = _mcc_from_counts(tp, tn, fp, fn)
		if mcc > best_mcc:
			best_mcc = mcc
			best_thresh = float(scores[idx])

	return best_thresh, best_mcc


def _print_confusion_matrix(cm):
	print("Confusion matrix (rows=actual, cols=predicted):")
	print(f"{'':>12} {'inactive':>10} {'active':>10}")
	print(f"{'inactive':>12} {cm[0, 0]:>10} {cm[0, 1]:>10}")
	print(f"{'active':>12} {cm[1, 0]:>10} {cm[1, 1]:>10}")
	print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}")


def evaluate_labeled_scores(scores, labels, title="INTERNAL METRICS CHECK (train_set only, never used for fitting)"):
	print("="*60)
	print(title)
	print("="*60)

	y_true = np.asarray(labels == "active", dtype=int)
	scores = np.asarray(scores)
	auc = roc_auc_score(y_true, scores)
	threshold, mcc = _best_mcc_threshold(y_true, scores)
	y_pred = (scores >= threshold).astype(int)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

	print(f"AUC: {auc:.4f}")
	print(f"MCC: {mcc:.4f}  (threshold={threshold:.6f}, higher score = more anomalous)")
	_print_confusion_matrix(cm)
	print("="*60 + "\n")

	return {"auc": auc, "mcc": mcc, "threshold": threshold, "confusion_matrix": cm}


def evaluate_internal_metrics(train_scores, train_labels):
	return evaluate_labeled_scores(train_scores, train_labels)


def evaluate_full_train_metrics(X_train, train_labels, args):
	pipeline = build_pipeline(
		reduce=args.reduce,
		method=args.method,
		n_components=args.n_components,
		n_estimators=args.n_estimators,
		random_state=args.random_state,
		max_subspaces=args.max_subspaces,
		scale=args.scale,
		feature_select=getattr(args, "feature_select", None),
	)
	fit_pipeline_train(
		pipeline, X_train, train_labels,
		smote=getattr(args, "smote", False),
	)
	scores = anomaly_scores(pipeline, X_train)
	y_true = np.asarray(train_labels == "active", dtype=int)
	auc = roc_auc_score(y_true, scores)
	_, mcc = _best_mcc_threshold(y_true, scores)
	if "select" in pipeline.named_steps:
		sel = pipeline.named_steps["select"]
		print(f"  mRMR+IFS selected {sel.n_features_selected_} features (IFS train AUC={sel.ifs_best_auc_:.4f})")
	return {"auc": auc, "mcc": mcc}


def cross_validate_train_metrics(X_train, train_labels, args, n_splits=3):
	# Unchanged from before - k-fold stratified CV, fits on the full train
	# fold (including known actives, contamination-robust fit assumption).
	y = np.asarray(train_labels == "active", dtype=int)
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
	fold_aucs, fold_mccs = [], []

	for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), start=1):
		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		labels_val = train_labels.iloc[val_idx]
		labels_tr = train_labels.iloc[train_idx]

		pipeline = build_pipeline(
			reduce=args.reduce,
			method=args.method,
			n_components=args.n_components,
			n_estimators=args.n_estimators,
			random_state=args.random_state,
			max_subspaces=args.max_subspaces,
			scale=args.scale,
			feature_select=getattr(args, "feature_select", None),
		)
		fit_pipeline_train(
			pipeline, X_tr, labels_tr,
			smote=getattr(args, "smote", False),
		)
		scores_val = anomaly_scores(pipeline, X_val)

		y_true = np.asarray(labels_val == "active", dtype=int)
		if y_true.sum() == 0 or y_true.sum() == len(y_true):
			print(f"  fold {fold}/{n_splits}: skipped (validation fold has a single class)")
			continue
		fold_aucs.append(roc_auc_score(y_true, scores_val))
		_, mcc = _best_mcc_threshold(y_true, scores_val)
		fold_mccs.append(mcc)
		print(f"  fold {fold}/{n_splits}: AUC={fold_aucs[-1]:.4f}  MCC={fold_mccs[-1]:.4f}")

	if not fold_aucs:
		raise ValueError("CV produced no valid folds (each validation split had only one class)")

	auc_mean, auc_std = np.mean(fold_aucs), np.std(fold_aucs, ddof=1)
	mcc_mean, mcc_std = np.mean(fold_mccs), np.std(fold_mccs, ddof=1)
	return {
		"auc": auc_mean,
		"auc_std": auc_std,
		"mcc": mcc_mean,
		"mcc_std": mcc_std,
		"repeat_aucs": fold_aucs,   # same key name as repeated_subsampling below, so
		"repeat_mccs": fold_mccs,   # downstream code doesn't need to branch on strategy
		"n_repeats": n_splits,
	}


def repeated_random_subsampling_metrics(X_train, train_labels, args, n_repeats=25,
                                         inlier_frac=0.8, outlier_frac=0.8, verbose=True):
	# New, added alongside cross_validate_train_metrics (not a replacement).
	# Slide 7 (Week 3): sample 80% of inliers for training; combine the
	# remaining 20% inliers with a random 80% sample of outliers as the
	# validation split; repeat many times. Training uses ONLY inliers, no
	# labels passed into fit anywhere - pure OCC framing as on that slide.
	# This differs from cross_validate_train_metrics, which fits on the
	# full train fold (including known actives).
	if getattr(args, "feature_select", None) or getattr(args, "smote", False):
		raise ValueError(
			"repeated_subsampling doesn't support --feature-select/--smote "
			"(those need labeled data passed into fit differently) - use "
			"--eval-strategy cv for those instead."
		)

	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx_all = np.where(y == 0)[0]
	outlier_idx_all = np.where(y == 1)[0]

	repeat_aucs, repeat_mccs = [], []

	for i in range(n_repeats):
		rng = np.random.default_rng(args.random_state + i)

		perm_in = rng.permutation(inlier_idx_all)
		n_train_in = int(round(inlier_frac * len(inlier_idx_all)))
		train_idx = perm_in[:n_train_in]
		remaining_in = perm_in[n_train_in:]

		perm_out = rng.permutation(outlier_idx_all)
		n_val_out = int(round(outlier_frac * len(outlier_idx_all)))
		val_out = perm_out[:n_val_out]

		val_idx = np.concatenate([remaining_in, val_out])

		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		y_val = y[val_idx]

		pipeline = build_pipeline(
			reduce=args.reduce,
			method=args.method,
			n_components=args.n_components,
			n_estimators=args.n_estimators,
			random_state=args.random_state,
			max_subspaces=args.max_subspaces,
			scale=args.scale,
			feature_select=None,
		)
		pipeline.fit(X_tr)  # inlier-only fit, no labels passed anywhere
		scores_val = anomaly_scores(pipeline, X_val)

		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			if verbose:
				print(f"  repeat {i+1}/{n_repeats}: skipped (val split has a single class)")
			continue

		auc = roc_auc_score(y_val, scores_val)
		_, mcc = _best_mcc_threshold(y_val, scores_val)
		repeat_aucs.append(auc)
		repeat_mccs.append(mcc)
		if verbose:
			print(f"  repeat {i+1}/{n_repeats}: AUC={auc:.4f}  MCC={mcc:.4f}  "
			      f"(train={len(train_idx)}, val={len(val_idx)}, val_active={int(y_val.sum())})")

	if not repeat_aucs:
		raise ValueError("Repeated subsampling produced no valid repeats")

	auc_mean, auc_std = np.mean(repeat_aucs), np.std(repeat_aucs, ddof=1)
	mcc_mean, mcc_std = np.mean(repeat_mccs), np.std(repeat_mccs, ddof=1)
	return {
		"auc": auc_mean,
		"auc_std": auc_std,
		"mcc": mcc_mean,
		"mcc_std": mcc_std,
		"repeat_aucs": repeat_aucs,
		"repeat_mccs": repeat_mccs,
		"n_repeats": len(repeat_aucs),
	}


def feature_bagging_metrics(X_train, train_labels, args, method, n_subsets=10,
                             subset_min_frac=0.5, verbose=True):
	# Step B (Week 3 plan): SEPARATE per method, no mixing of method and
	# feature subspace (would confound the two effects - see chat notes).
	# Single full-train fit-and-score check (post-hoc AUC, same convention
	# as evaluate_full_train_metrics), NOT nested inside the 25x repeated
	# subsampling from Step A - nesting would mean n_repeats * n_subsets
	# fits per method, too expensive at this data size.
	X_red = fit_scale_reduce(X_train, args.reduce, args.n_components, args.random_state, args.scale)
	d = X_red.shape[1]
	y_true = np.asarray(train_labels == "active", dtype=int)

	# baseline: single model on the full reduced space, no bagging
	base_detector = make_detector(method, args.n_estimators, args.random_state, args.max_subspaces)
	base_detector.fit(X_red)
	base_scores = -base_detector.score_samples(X_red)
	base_auc = roc_auc_score(y_true, base_scores)
	if verbose:
		print(f"  baseline (no bagging, all {d} components): AUC={base_auc:.4f}")

	rng = np.random.default_rng(args.random_state)
	subset_scores = []  # z-normalized score vectors, one per subset

	for i in range(n_subsets):
		k = int(rng.integers(int(subset_min_frac * d), d + 1))
		cols = rng.choice(d, size=k, replace=False)

		detector = make_detector(method, args.n_estimators, args.random_state + i + 1, args.max_subspaces)
		detector.fit(X_red[:, cols])
		scores = -detector.score_samples(X_red[:, cols])

		# z-normalize before aggregation - subsets aren't on the same scale,
		# summing/maxing raw scores would just favor whichever subset
		# happens to produce the largest numbers (see chat notes)
		scores_z = (scores - scores.mean()) / (scores.std() + 1e-12)
		subset_scores.append(scores_z)

		if verbose:
			auc_i = roc_auc_score(y_true, scores)
			print(f"  subset {i+1}/{n_subsets}: k={k}/{d} components, AUC={auc_i:.4f}")

	subset_matrix = np.vstack(subset_scores)  # (n_subsets, n_samples)

	agg_sum = subset_matrix.sum(axis=0)

	agg_max = subset_matrix.max(axis=0)

	# AOM (Average of Maximum): bundle subsets into a handful of groups,
	# take the max within each group, then average across groups - a
	# middle ground between Sum (dilutes a signal only strong in a few
	# subsets) and Max (no dilution but sensitive to single-subset noise)
	n_groups = min(5, n_subsets)
	groups = np.array_split(subset_matrix, n_groups, axis=0)
	agg_aom = np.mean([g.max(axis=0) for g in groups], axis=0)

	return {
		"no_bagging": base_auc,
		"bagging_sum": roc_auc_score(y_true, agg_sum),
		"bagging_max": roc_auc_score(y_true, agg_max),
		"bagging_aom": roc_auc_score(y_true, agg_aom),
	}


def feature_bagging_comparison(X_train, train_labels, args):
	# Runs feature_bagging_metrics separately for each of FEATURE_BAGGING_METHODS
	# (no mixing of method and subspace across methods), prints per-method
	# results plus a summary table (methods x {no_bagging, sum, max, aom}).
	print("="*60)
	print(f"FEATURE BAGGING: {', '.join(FEATURE_BAGGING_METHODS)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_subsets={args.n_subsets})")
	print("="*60 + "\n")

	rows = {}
	for method in FEATURE_BAGGING_METHODS:
		print("#" * 60)
		print(f"# METHOD: {method}")
		print("#" * 60)
		rows[method] = feature_bagging_metrics(
			X_train, train_labels, args, method, n_subsets=args.n_subsets,
		)
		print()

	summary = pd.DataFrame(rows).T
	summary.index.name = "method"

	print("=" * 60)
	print("FEATURE BAGGING SUMMARY (AUC, full-train post-hoc)")
	print("no_bagging = single model on all components (Step A baseline)")
	print("=" * 60)
	print(summary.round(4).to_string())
	print("=" * 60 + "\n")


def print_eval_metrics(metrics, label):
	print("=" * 60)
	print(f"INTERNAL METRICS CHECK ({label})")
	print("=" * 60)
	print(f"AUC: {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}")
	print(f"MCC: {metrics['mcc']:.4f} +/- {metrics['mcc_std']:.4f}")
	print("=" * 60 + "\n")


def save_prediction(test_scores, test_ids, path="outputs/team2_prediction.csv"):
	prediction = pd.DataFrame({"id": test_ids, "score": test_scores})
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	prediction.to_csv(path, index=False)

	print("="*60)
	print("PREDICTION SAVED")
	print("="*60)
	print(f"File: {path}")
	print(f"Rows: {len(prediction):,}")
	print("="*60 + "\n")


def run_pipeline(X_train, X_test, train_labels, test_ids, args, save=True):
	if getattr(args, "eval_strategy", "cv") == "repeated_subsampling":
		eval_metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, args, n_repeats=args.n_repeats,
		)
		print_eval_metrics(
			eval_metrics,
			f"{eval_metrics['n_repeats']}x repeated random subsampling on train, "
			f"inlier-only fit",
		)
	else:
		eval_metrics = cross_validate_train_metrics(
			X_train, train_labels, args, n_splits=args.cv_folds,
		)
		suffix = ", SMOTE on train fold" if getattr(args, "smote", False) else ""
		print_eval_metrics(eval_metrics, f"{args.cv_folds}-fold stratified CV on train{suffix}")

	pipeline = build_pipeline(
		reduce=args.reduce,
		method=args.method,
		n_components=args.n_components,
		n_estimators=args.n_estimators,
		random_state=args.random_state,
		max_subspaces=args.max_subspaces,
		scale=args.scale,
		feature_select=getattr(args, "feature_select", None),
	)
	fit_pipeline(
		pipeline, X_train, args.method, args.reduce,
		scale=args.scale, train_labels=train_labels, smote=getattr(args, "smote", False),
	)
	test_scores = anomaly_scores(pipeline, X_test)

	if save:
		save_prediction(test_scores, test_ids)
	return eval_metrics


def compare_methods(X_train, train_labels, args):
	# Runs COMPARE_METHODS through repeated_random_subsampling_metrics with
	# identical reduce/n_components/scale settings (fair comparison, same
	# feature space for every method), prints the per-repeat AUC matrix to
	# the terminal. Always uses repeated_subsampling regardless of
	# --eval-strategy - this comparison mode is separate from the single-
	# method CV/repeated_subsampling choice above. Nothing is written to
	# disk - copy the printed table for the Friedman/Conover-Iman test
	# next week.
	print("="*60)
	print(f"COMPARING METHODS: {', '.join(COMPARE_METHODS)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats})")
	print("="*60 + "\n")

	results = {}
	for method in COMPARE_METHODS:
		print("#" * 60)
		print(f"# METHOD: {method}")
		print("#" * 60)

		method_args = argparse.Namespace(**{**vars(args), "method": method,
		                                     "feature_select": None, "smote": False})
		metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, method_args, n_repeats=args.n_repeats, verbose=False,
		)
		results[method] = metrics["repeat_aucs"]
		print(f"AUC: {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}  "
		      f"({metrics['n_repeats']} valid repeats)\n")

	min_len = min(len(v) for v in results.values())
	if any(len(v) != min_len for v in results.values()):
		print(f"WARNING: methods had different numbers of valid repeats, "
		      f"truncating all to {min_len} for the matrix.\n")

	matrix = pd.DataFrame({m: v[:min_len] for m, v in results.items()})
	matrix.index.name = "repeat"

	print("=" * 60)
	print(f"AUC MATRIX ({matrix.shape[0]} repeats x {matrix.shape[1]} methods)")
	print("copy this for next week's Friedman/Conover-Iman test")
	print("=" * 60)
	print(matrix.round(4).to_string())
	print()
	print("Mean +/- Std per method:")
	print((matrix.mean().round(4).astype(str) + " +/- " + matrix.std(ddof=1).round(4).astype(str)).to_string())
	print("=" * 60 + "\n")


def main():
	parser = argparse.ArgumentParser(description="p53 Outlier Detection Pipeline")
	parser.add_argument("--method", type=str, default="isolation_forest",
	                     choices=["isolation_forest", "lof", "gmd_lof", "cof", "knn", "ocsvm", "abod"],
	                     help="Outlier detection method")
	parser.add_argument("--compare-methods", action="store_true",
	                     help=f"Run all of {COMPARE_METHODS} through repeated_subsampling "
	                          f"with identical reduce/n_components/scale settings, print the "
	                          f"per-repeat AUC matrix to the terminal (no file written), "
	                          f"then exit. Overrides --method and --eval-strategy.")
	parser.add_argument("--feature-bagging", action="store_true",
	                     help=f"Step B: for each of {FEATURE_BAGGING_METHODS} separately, "
	                          f"fit --n-subsets models on random feature subsets of the "
	                          f"reduced space, compare no_bagging vs sum/max/aom aggregation "
	                          f"(full-train post-hoc AUC, not repeated_subsampling - see "
	                          f"chat notes on cost). Overrides --method/--compare-methods.")
	parser.add_argument("--n-subsets", type=int, default=10,
	                     help="Number of random feature subsets per method for --feature-bagging")
	parser.add_argument("--n-estimators", type=int, default=200,
	                     help="Number of trees in the Isolation Forest")
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--eval-strategy", type=str, default="cv",
	                     choices=["cv", "repeated_subsampling"],
	                     help="cv = stratified k-fold, unchanged from before (default); "
	                          "repeated_subsampling = new Folie 7 Week 3 approach (80%% "
	                          "inliers train, remaining inliers + 80%% outliers val, "
	                          "repeated). Only applies to single --method runs, not "
	                          "--compare-methods (which always uses repeated_subsampling).")
	parser.add_argument("--cv-folds", type=int, default=3,
	                     help="K-fold CV on train for internal AUC/MCC (--eval-strategy cv)")
	parser.add_argument("--n-repeats", type=int, default=25,
	                     help="Repeats for --eval-strategy repeated_subsampling / --compare-methods")
	parser.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True,
	                     help="StandardScaler before reduction (fit on train fold in CV, full train for final model)")
	parser.add_argument("--smote", action=argparse.BooleanOptionalAction, default=False,
	                     help="SMOTE on train fold only (after scale, before reduce/detector; "
	                          "uses labels on train). --eval-strategy cv only.")
	parser.add_argument("--reduce", type=str, default="none",
	                     choices=["none", "pca", "random_projection", "truncated_svd", "isomap", "umap"],
	                     help="Dimensionality reduction before fitting")
	parser.add_argument("--feature-select", type=str, default=None,
	                     choices=["mrmr_ifs"],
	                     help="Supervised feature selection before detector (uses labels on "
	                          "train only). --eval-strategy cv only.")
	parser.add_argument("--n-components", type=int, default=147,
	                     help="Components for --reduce (147 = W1 PCA result, 95%% variance)")
	parser.add_argument("--sample", type=int, default=None,
	                     help="Subsample train_set to this many rows (quick testing)")
	parser.add_argument("--max-subspaces", type=int, default=None,
	                     help="Limit number of GMD subspaces used (quick testing)")
	parser.add_argument("--sweep", type=str, default=None,
	                     help="Comma-separated n_components to compare, e.g. 10,20,50,147,300 "
	                          "(finds each method's own best dimension instead of a fixed one)")
	args = parser.parse_args()

	X_train, X_test, train_labels = load_data()
	test_ids = X_test.index

	if args.sample:
		X_train = X_train.sample(n=args.sample, random_state=args.random_state)
		train_labels = train_labels.loc[X_train.index]
		n_active = (train_labels == "active").sum()
		print(f"Subsampled train_set to {len(X_train):,} rows for quick testing "
		      f"(active: {n_active} - metrics below are NOT meaningful, runtime check only)\n")

	if args.feature_bagging:
		feature_bagging_comparison(X_train, train_labels, args)
		return

	if args.compare_methods:
		compare_methods(X_train, train_labels, args)
		return

	if args.sweep:
		k_values = [int(k) for k in args.sweep.split(",")]
		results = []
		for k in k_values:
			sweep_args = argparse.Namespace(**{**vars(args), "n_components": k})
			metrics = run_pipeline(X_train, X_test, train_labels, test_ids, sweep_args, save=False)
			results.append((k, metrics))

		print("="*60)
		print(f"SWEEP SUMMARY ({args.method} + {args.reduce})")
		print("="*60)
		for k, metrics in results:
			print(f"  n_components={k:<5} "
			      f"AUC={metrics['auc']:.4f}±{metrics['auc_std']:.4f}  "
			      f"MCC={metrics['mcc']:.4f}±{metrics['mcc_std']:.4f}")
		best_k, best_metrics = max(results, key=lambda r: r[1]["auc"])
		print(f"Best (by AUC): n_components={best_k} "
		      f"(AUC={best_metrics['auc']:.4f}±{best_metrics['auc_std']:.4f}, "
		      f"MCC={best_metrics['mcc']:.4f}±{best_metrics['mcc_std']:.4f})")
		print("="*60 + "\n")
		return

	run_pipeline(X_train, X_test, train_labels, test_ids, args, save=True)


if __name__ == "__main__":
	main()