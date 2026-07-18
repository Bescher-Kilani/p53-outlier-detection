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
--eval-strategy), prints the per-repeat AUC matrix, and with --output-csv
also saves it (5 columns: one per method) - this is one of the two
possible inputs for --stat-test below.

COF is excluded from COMPARE_METHODS - PyOD's implementation is too slow
at ~20k train rows per repeat (near-O(n^2) pairwise connectivity), a
single --compare-methods run never finished. --method cof still works
for a one-off single run if needed.

Feature bagging (--feature-bagging) and heterogeneous ensemble
(--heterogeneous-ensemble) now ALSO use full repeated random subsampling
(same 80/20 inlier/outlier split logic as Step A), not a single full-train
post-hoc check - this gives a proper mean +/- std instead of one point
estimate. Cost: --feature-bagging does n_repeats * n_subsets fits PER
METHOD (e.g. 25 * 10 = 250, times 3 methods = 750 total), and
--heterogeneous-ensemble does n_repeats fits per method (25 * 3 = 75
total) - see chat notes on why the two scale so differently (Step B needs
n_subsets extra fits inside every single repeat, Step C doesn't).
--feature-bagging --output-csv saves a 12-column matrix (3 methods x 4
aggregation variants: no_bagging/sum/max/aom) - the other possible input
for --stat-test.

Week 5, Step 1: TabPFN (--method tabpfn, zero-shot foundation model via
tabpfn_extensions) added as a new detector. Gated on HuggingFace - before
first use on the server, accept the license at
https://huggingface.co/Prior-Labs/tabpfn_3 and run `hf auth login` (or
set HF_TOKEN), otherwise fit() raises TabPFNHuggingFaceGatedRepoError.
See COMPARE_METHODS_WEEK5 / --methods-set below to compare tabpfn
against all 6 methods from last week in one run. Cost/OOM fixes and
throttling params are documented on TabPFNDetector itself.

Statistical testing (--stat-test): Week 4. Runs a Friedman test (global:
do all configurations in the given matrix differ?) followed by, if
significant, a Conover-Iman post-hoc test (pairwise: which configurations
differ from which?) on ANY repeats x configurations AUC matrix CSV - so
the same command works on the --compare-methods matrix (5 methods) and
the --feature-bagging matrix (12 method+aggregation variants), or any
other matrix with the same repeats-as-rows/configs-as-columns shape.
Prints a ++/+/- significance table (row significantly better than
column); save it with --output-csv.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
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

COMPARE_METHODS = ["isolation_forest", "lof", "knn", "ocsvm", "abod", "deep_svdd"]

# Week 5, Step 1: TabPFN compared against all 6 methods from last week
# (isolation_forest, lof, knn, ocsvm, abod, deep_svdd) - reuses COMPARE_METHODS
# rather than a separate hand-picked baseline, so this doubles as the
# "other OD methods" comparison (KNN, OCSVM, etc. are already in there).
# Two commands for Step 1:
#   --compare-methods                     (last week's default, unchanged)
#   --compare-methods --methods-set week5 (same 6 + tabpfn)
COMPARE_METHODS_WEEK5 = COMPARE_METHODS + ["tabpfn"]

# Step B: Top-3 from Step A + Deep SVDD (cluster-based, same slot as OCSVM for Week 4)
FEATURE_BAGGING_METHODS = ["isolation_forest", "knn", "ocsvm", "deep_svdd"]


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


class DeepSVDDDetector(BaseEstimator):
	# PyOD DeepSVDD needs n_features at init; infer from fit data so feature
	# bagging works on column subsets. preprocessing=False — scale+PCA already
	# applied upstream. Trained on inlier-only rows in repeated_subsampling (OCC).
	def __init__(self, random_state=42, epochs=20, batch_size=512):
		self.random_state = random_state
		self.epochs = epochs
		self.batch_size = batch_size
		self.model_ = None

	def fit(self, X, y=None):
		from pyod.models.deep_svdd import DeepSVDD

		X_arr = np.asarray(X, dtype=np.float32)
		self.model_ = DeepSVDD(
			n_features=X_arr.shape[1],
			epochs=self.epochs,
			batch_size=self.batch_size,
			hidden_neurons=[64, 32],
			preprocessing=False,
			verbose=0,
			random_state=self.random_state,
			contamination=0.05,
		)
		self.model_.fit(X_arr)
		return self

	def score_samples(self, X):
		return -self.model_.decision_function(np.asarray(X, dtype=np.float32))


class TabPFNDetector(BaseEstimator):
	# Zero-shot foundation model via Prior Labs' TabPFNUnsupervisedModel:
	# estimates per-sample likelihood, low likelihood = anomalous. fit(X)
	# only, no labels - fits the OCC rule as-is.
	#
	# TabPFN v2 limits: <=500 features, <=10_000 context rows. PCA-147 is
	# already under the feature limit; max_context below subsamples rows
	# if the train fold is larger.
	#
	# COST: outliers() does one forward pass per feature per permutation
	# (no built-in way to limit features). At d=147/n_permutations=10
	# that's ~1470 passes per score_samples() call - hung 10+ min and
	# OOM'd on an idle RTX 4090 on the server. Fixed via three levers:
	# n_permutations 10->2, max_context 3000->1000, max_features 147->30
	# (takes the first N columns - relies on PCA ordering by explained
	# variance, don't rely on this with --reduce != pca). ~25x fewer
	# passes combined. Still consider running with few --n-repeats
	# separately rather than inside --compare-methods.
	def __init__(self, random_state=42, max_context=1000, n_permutations=2, max_features=30):
		self.random_state = random_state
		self.max_context = max_context
		self.n_permutations = n_permutations
		self.max_features = max_features
		self.model_ = None

	def fit(self, X, y=None):
		from tabpfn import TabPFNClassifier, TabPFNRegressor
		from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

		X_arr = np.asarray(X, dtype=np.float32)
		if self.max_features is not None and X_arr.shape[1] > self.max_features:
			X_arr = X_arr[:, :self.max_features]
		if len(X_arr) > self.max_context:
			rng = np.random.default_rng(self.random_state)
			idx = rng.choice(len(X_arr), size=self.max_context, replace=False)
			X_arr = X_arr[idx]

		self.model_ = TabPFNUnsupervisedModel(
			tabpfn_clf=TabPFNClassifier(random_state=self.random_state),
			tabpfn_reg=TabPFNRegressor(random_state=self.random_state),
		)
		self.model_.fit(X_arr)
		return self

	def score_samples(self, X):
		import torch

		X_arr = np.asarray(X, dtype=np.float32)
		if self.max_features is not None and X_arr.shape[1] > self.max_features:
			X_arr = X_arr[:, :self.max_features]
		X_t = torch.tensor(X_arr)
		scores = self.model_.outliers(X_t, n_permutations=self.n_permutations)
		scores = scores.detach().cpu().numpy()
		# outliers() returns log(AM(densities)) - lower = more anomalous,
		# which already matches sklearn's score_samples convention, so no
		# sign flip here (unlike every other detector in this file).
		#
		# Non-finite values: extremely atypical points can underflow the
		# log-density to -inf/nan, which crashes roc_auc_score. Remap to
		# just outside this batch's own finite range (preserves ranking;
		# -inf/nan -> most anomalous, +inf -> most normal) instead of a
		# fixed constant.
		finite = scores[np.isfinite(scores)]
		if finite.size:
			lo, hi = finite.min(), finite.max()
		else:
			lo, hi = -1.0, 1.0
		pad = max(1.0, hi - lo)
		scores = np.nan_to_num(scores, nan=lo - pad, posinf=hi + pad, neginf=lo - pad)
		return scores


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
	if method == "deep_svdd":
		return DeepSVDDDetector(random_state=random_state)
	if method == "tabpfn":
		return TabPFNDetector(random_state=random_state)
	raise ValueError(f"Unknown method: {method}")


def fit_scale_reduce(X, reduce, n_components, random_state, scale, return_fitted=False):
	# Fits scale+reduce once on the full array, separate from build_pipeline
	# (which bundles reduce+detector together). return_fitted=True additionally
	# returns the fitted scaler/reducer so a separate val/test set can be
	# transformed with the exact same fit (no leakage) - used by every
	# repeated-subsampling variant below (Step A/B/C all need this).
	X_arr = np.asarray(X, dtype=np.float64)
	scaler = None
	if scale:
		scaler = StandardScaler().fit(X_arr)
		X_arr = scaler.transform(X_arr)
	reducer = make_reducer(reduce, n_components, random_state)
	if reducer == "passthrough":
		X_red = X_arr
	else:
		X_red = reducer.fit_transform(X_arr)
	if return_fitted:
		return X_red, scaler, reducer
	return X_red


def transform_scale_reduce(X, scaler, reducer):
	# Applies an already-fitted scaler/reducer pair (from fit_scale_reduce
	# with return_fitted=True) to a new array - e.g. a val split that must
	# use the train split's fit, not its own.
	X_arr = np.asarray(X, dtype=np.float64)
	if scaler is not None:
		X_arr = scaler.transform(X_arr)
	if reducer == "passthrough":
		return X_arr
	return reducer.transform(X_arr)


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
	"deep_svdd": "FITTING DEEP SVDD (PyOD, MLP [64,32], epochs=20, inlier-only OCC)",
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


def _repeated_split_indices(y, inlier_idx_all, outlier_idx_all, seed, inlier_frac, outlier_frac):
	# Shared split logic for Step A/B/C repeated subsampling: 80% of inliers
	# for train, remaining inliers + 80% of outliers for val. Factored out
	# so all three steps draw splits the exact same way.
	rng = np.random.default_rng(seed)

	perm_in = rng.permutation(inlier_idx_all)
	n_train_in = int(round(inlier_frac * len(inlier_idx_all)))
	train_idx = perm_in[:n_train_in]
	remaining_in = perm_in[n_train_in:]

	perm_out = rng.permutation(outlier_idx_all)
	n_val_out = int(round(outlier_frac * len(outlier_idx_all)))
	val_out = perm_out[:n_val_out]

	val_idx = np.concatenate([remaining_in, val_out])
	return train_idx, val_idx


def repeated_random_subsampling_metrics(X_train, train_labels, args, n_repeats=25,
                                         inlier_frac=0.8, outlier_frac=0.8, verbose=True):
	# Step A. Folie 7 (Week 3): sample 80% of inliers for training; combine
	# the remaining 20% inliers with a random 80% sample of outliers as the
	# validation split; repeat many times. Training uses ONLY inliers, no
	# labels passed into fit anywhere - pure OCC framing as on that slide.
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
		train_idx, val_idx = _repeated_split_indices(
			y, inlier_idx_all, outlier_idx_all, args.random_state + i, inlier_frac, outlier_frac,
		)

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


def feature_bagging_repeated_metrics(X_train, train_labels, args, method, n_repeats=25,
                                      n_subsets=10, subset_min_frac=0.5,
                                      inlier_frac=0.8, outlier_frac=0.8, verbose=True):
	# Step B, full repeated-subsampling version (see chat notes: this means
	# n_subsets extra fits INSIDE every single repeat, so total fits per
	# method = n_repeats * n_subsets, e.g. 25*10=250, x3 methods = 750).
	# Same 80/20 split logic as Step A; scale+reduce fit on the train split
	# only, applied to val with the same fit (no leakage). Replaces the old
	# single full-train post-hoc check - this version gives a proper
	# mean +/- std across repeats instead of one point estimate.
	if getattr(args, "feature_select", None) or getattr(args, "smote", False):
		raise ValueError("feature bagging repeated subsampling doesn't support --feature-select/--smote")

	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx_all = np.where(y == 0)[0]
	outlier_idx_all = np.where(y == 1)[0]

	repeat_results = {"no_bagging": [], "bagging_sum": [], "bagging_max": [], "bagging_aom": []}

	for i in range(n_repeats):
		train_idx, val_idx = _repeated_split_indices(
			y, inlier_idx_all, outlier_idx_all, args.random_state + i, inlier_frac, outlier_frac,
		)

		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		y_val = y[val_idx]

		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			if verbose:
				print(f"  repeat {i+1}/{n_repeats}: skipped (val split has a single class)")
			continue

		X_tr_red, scaler, reducer = fit_scale_reduce(
			X_tr, args.reduce, args.n_components, args.random_state, args.scale,
			return_fitted=True,
		)
		X_val_red = transform_scale_reduce(X_val, scaler, reducer)
		d = X_tr_red.shape[1]

		# baseline: no bagging, single model on all components, trained on
		# this repeat's train split, scored on this repeat's val split
		base_detector = make_detector(method, args.n_estimators, args.random_state, args.max_subspaces)
		base_detector.fit(X_tr_red)
		base_scores_val = -base_detector.score_samples(X_val_red)
		repeat_results["no_bagging"].append(roc_auc_score(y_val, base_scores_val))

		rng_subsets = np.random.default_rng(args.random_state)
		subset_scores_val = []
		for j in range(n_subsets):
			k = int(rng_subsets.integers(int(subset_min_frac * d), d + 1))
			cols = rng_subsets.choice(d, size=k, replace=False)

			detector = make_detector(method, args.n_estimators, args.random_state + j + 1, args.max_subspaces)
			detector.fit(X_tr_red[:, cols])
			scores_val = -detector.score_samples(X_val_red[:, cols])

			# z-normalize using this repeat's val scores - subsets aren't on
			# the same scale, raw summing/maxing would just favor whichever
			# subset happens to produce the largest numbers
			scores_val_z = (scores_val - scores_val.mean()) / (scores_val.std() + 1e-12)
			subset_scores_val.append(scores_val_z)

		subset_matrix = np.vstack(subset_scores_val)  # (n_subsets, n_val)
		agg_sum = subset_matrix.sum(axis=0)
		agg_max = subset_matrix.max(axis=0)
		n_groups = min(5, n_subsets)
		groups = np.array_split(subset_matrix, n_groups, axis=0)
		agg_aom = np.mean([g.max(axis=0) for g in groups], axis=0)

		repeat_results["bagging_sum"].append(roc_auc_score(y_val, agg_sum))
		repeat_results["bagging_max"].append(roc_auc_score(y_val, agg_max))
		repeat_results["bagging_aom"].append(roc_auc_score(y_val, agg_aom))

		if verbose:
			print(f"  repeat {i+1}/{n_repeats}: no_bagging={repeat_results['no_bagging'][-1]:.4f}  "
			      f"sum={repeat_results['bagging_sum'][-1]:.4f}  "
			      f"max={repeat_results['bagging_max'][-1]:.4f}  "
			      f"aom={repeat_results['bagging_aom'][-1]:.4f}")

	if not repeat_results["no_bagging"]:
		raise ValueError("Feature bagging repeated subsampling produced no valid repeats")

	summary = {}
	for key, vals in repeat_results.items():
		summary[key] = {
			"mean": float(np.mean(vals)),
			"std": float(np.std(vals, ddof=1)),
			"n": len(vals),
			"values": vals,  # raw per-repeat AUCs, needed for the CSV export (12-column matrix)
		}
	return summary


def feature_bagging_comparison(X_train, train_labels, args):
	# Runs feature_bagging_repeated_metrics separately for each of
	# FEATURE_BAGGING_METHODS (no mixing of method and subspace across
	# methods), prints per-method mean +/- std plus a summary table.
	total_fits = args.n_repeats * args.n_subsets * len(FEATURE_BAGGING_METHODS)
	print("="*60)
	print(f"FEATURE BAGGING (repeated subsampling): {', '.join(FEATURE_BAGGING_METHODS)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats}, n_subsets={args.n_subsets})")
	print(f"Total fits: {args.n_repeats} repeats x {args.n_subsets} subsets x "
	      f"{len(FEATURE_BAGGING_METHODS)} methods = {total_fits}")
	print("="*60 + "\n")

	rows = {}
	for method in FEATURE_BAGGING_METHODS:
		print("#" * 60)
		print(f"# METHOD: {method}")
		print("#" * 60)
		rows[method] = feature_bagging_repeated_metrics(
			X_train, train_labels, args, method,
			n_repeats=args.n_repeats, n_subsets=args.n_subsets, verbose=False,
		)
		for key, stats_ in rows[method].items():
			print(f"  {key:<12} {stats_['mean']:.4f} +/- {stats_['std']:.4f}  ({stats_['n']} repeats)")
		print()

	print("=" * 60)
	print("FEATURE BAGGING SUMMARY (mean +/- std over repeats)")
	print("no_bagging = single model on all components (Step A baseline)")
	print("=" * 60)
	header = f"{'method':<18}" + "".join(f"{k:>20}" for k in ["no_bagging", "sum", "max", "aom"])
	print(header)
	for method in FEATURE_BAGGING_METHODS:
		r = rows[method]
		line = f"{method:<18}"
		for key in ["no_bagging", "bagging_sum", "bagging_max", "bagging_aom"]:
			line += f"{r[key]['mean']:>12.4f}+/-{r[key]['std']:.4f}"
		print(line)
	print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		# 12 columns: 3 methods x 4 aggregation variants (no_bagging/sum/max/aom),
		# one row per repeat - for a statistical test comparing all 12
		# method+aggregation configurations against each other (separate
		# question from the --compare-methods 5-method test above).
		variant_keys = ["no_bagging", "bagging_sum", "bagging_max", "bagging_aom"]
		variant_labels = ["no_bagging", "sum", "max", "aom"]
		columns = {}
		for method in FEATURE_BAGGING_METHODS:
			for key, label in zip(variant_keys, variant_labels):
				columns[f"{method}_{label}"] = rows[method][key]["values"]

		min_len = min(len(v) for v in columns.values())
		if any(len(v) != min_len for v in columns.values()):
			print(f"WARNING: configurations had different numbers of valid repeats, "
			      f"truncating all to {min_len} for the matrix.\n")

		matrix = pd.DataFrame({k: v[:min_len] for k, v in columns.items()})
		matrix.index.name = "repeat"

		Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
		matrix.to_csv(args.output_csv)
		print(f"Saved feature bagging AUC matrix to {args.output_csv} "
		      f"({matrix.shape[0]} rows x {matrix.shape[1]} columns: "
		      f"{len(FEATURE_BAGGING_METHODS)} methods x {len(variant_labels)} aggregations)\n")


def heterogeneous_ensemble_repeated_metrics(X_train, train_labels, args, methods, n_repeats=25,
                                             inlier_frac=0.8, outlier_frac=0.8, verbose=True):
	# Step C, full repeated-subsampling version (see chat notes: only
	# n_repeats * len(methods) fits total, e.g. 25*3=75 - no extra
	# multiplier like Step B, since there's no subset loop here, just one
	# fit per method per repeat). Same 80/20 split logic as Step A/B.
	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx_all = np.where(y == 0)[0]
	outlier_idx_all = np.where(y == 1)[0]

	repeat_results = {m: [] for m in methods}
	repeat_results.update({"ensemble_sum": [], "ensemble_max": [], "ensemble_rank_avg": []})

	for i in range(n_repeats):
		train_idx, val_idx = _repeated_split_indices(
			y, inlier_idx_all, outlier_idx_all, args.random_state + i, inlier_frac, outlier_frac,
		)

		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		y_val = y[val_idx]

		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			if verbose:
				print(f"  repeat {i+1}/{n_repeats}: skipped (val split has a single class)")
			continue

		X_tr_red, scaler, reducer = fit_scale_reduce(
			X_tr, args.reduce, args.n_components, args.random_state, args.scale,
			return_fitted=True,
		)
		X_val_red = transform_scale_reduce(X_val, scaler, reducer)

		method_scores_val = {}
		for method in methods:
			detector = make_detector(method, args.n_estimators, args.random_state, args.max_subspaces)
			detector.fit(X_tr_red)
			scores_val = -detector.score_samples(X_val_red)
			method_scores_val[method] = scores_val
			repeat_results[method].append(roc_auc_score(y_val, scores_val))

		# z-normalize using this repeat's val scores - methods are on
		# completely different scales (IF ~[-0.6,0.6], KNN distances,
		# OCSVM signed margin, ...)
		scores_z = {m: (s - s.mean()) / (s.std() + 1e-12) for m, s in method_scores_val.items()}
		stacked = np.vstack(list(scores_z.values()))

		agg_sum = stacked.sum(axis=0)
		agg_max = stacked.max(axis=0)
		ranks = np.vstack([pd.Series(s).rank().values for s in scores_z.values()])
		agg_rank = ranks.mean(axis=0)

		repeat_results["ensemble_sum"].append(roc_auc_score(y_val, agg_sum))
		repeat_results["ensemble_max"].append(roc_auc_score(y_val, agg_max))
		repeat_results["ensemble_rank_avg"].append(roc_auc_score(y_val, agg_rank))

		if verbose:
			method_str = "  ".join(f"{m}={repeat_results[m][-1]:.4f}" for m in methods)
			print(f"  repeat {i+1}/{n_repeats}: {method_str}  "
			      f"sum={repeat_results['ensemble_sum'][-1]:.4f}  "
			      f"max={repeat_results['ensemble_max'][-1]:.4f}  "
			      f"rank_avg={repeat_results['ensemble_rank_avg'][-1]:.4f}")

	if not repeat_results["ensemble_sum"]:
		raise ValueError("Heterogeneous ensemble repeated subsampling produced no valid repeats")

	summary = {}
	for key, vals in repeat_results.items():
		summary[key] = {
			"mean": float(np.mean(vals)),
			"std": float(np.std(vals, ddof=1)),
			"n": len(vals),
		}
	return summary


def heterogeneous_ensemble_comparison(X_train, train_labels, args):
	# Runs FEATURE_BAGGING_METHODS (same Top-3 as Step B) all on the same
	# repeated-subsampling splits, combines via sum/max/rank-averaging,
	# prints individual method mean +/- std plus the three ensemble variants.
	total_fits = args.n_repeats * len(FEATURE_BAGGING_METHODS)
	print("="*60)
	print(f"HETEROGENEOUS ENSEMBLE (repeated subsampling): {', '.join(FEATURE_BAGGING_METHODS)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats}, same feature space for all methods)")
	print(f"Total fits: {args.n_repeats} repeats x {len(FEATURE_BAGGING_METHODS)} methods = {total_fits}")
	print("="*60 + "\n")

	results = heterogeneous_ensemble_repeated_metrics(
		X_train, train_labels, args, FEATURE_BAGGING_METHODS, n_repeats=args.n_repeats, verbose=True,
	)

	print("\n" + "=" * 60)
	print("HETEROGENEOUS ENSEMBLE SUMMARY (mean +/- std over repeats)")
	print("=" * 60)
	for method in FEATURE_BAGGING_METHODS:
		r = results[method]
		print(f"  {method:<20} {r['mean']:.4f} +/- {r['std']:.4f}")
	print("  " + "-" * 40)
	for key, label in [("ensemble_sum", "ensemble (sum)"), ("ensemble_max", "ensemble (max)"),
	                    ("ensemble_rank_avg", "ensemble (rank avg)")]:
		r = results[key]
		print(f"  {label:<20} {r['mean']:.4f} +/- {r['std']:.4f}")
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
	# Runs a method list through repeated_random_subsampling_metrics with
	# identical reduce/n_components/scale settings (fair comparison, same
	# feature space for every method), prints the per-repeat AUC matrix to
	# the terminal. Always uses repeated_subsampling regardless of
	# --eval-strategy - this comparison mode is separate from the single-
	# method CV/repeated_subsampling choice above. If --output-csv is set,
	# also writes the matrix to disk - this is one of the two direct
	# inputs for --stat-test (Friedman/Conover-Iman, Week 4): 25 rows
	# (repeats/blocks) x N columns (methods, no-bagging AUC only -
	# Feature Bagging variants are a separate question, see
	# feature_bagging_comparison above).
	#
	# Week 5: defaults to the original 6-method COMPARE_METHODS (unchanged,
	# so the Week 4 comparison stays reproducible), but --methods-set lets
	# you point this at COMPARE_METHODS_WEEK5 (same 6 methods + tabpfn) or
	# any other comma-separated subset without touching this function again.
	raw_methods_set = getattr(args, "methods_set", None)
	if raw_methods_set is None:
		methods = COMPARE_METHODS
	elif raw_methods_set == "week5":
		methods = COMPARE_METHODS_WEEK5
	else:
		methods = [m.strip() for m in raw_methods_set.split(",") if m.strip()]

	print("="*60)
	print(f"COMPARING METHODS: {', '.join(methods)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats})")
	print("="*60 + "\n")

	results = {}
	for method in methods:
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
	print("run --stat-test on the saved CSV for the Friedman/Conover-Iman test")
	print("=" * 60)
	print(matrix.round(4).to_string())
	print()
	print("Mean +/- Std per method:")
	print((matrix.mean().round(4).astype(str) + " +/- " + matrix.std(ddof=1).round(4).astype(str)).to_string())
	print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
		matrix.to_csv(args.output_csv)
		print(f"Saved AUC matrix to {args.output_csv} "
		      f"({matrix.shape[0]} rows x {matrix.shape[1]} columns, no header index beyond 'repeat')\n")


def friedman_conover_test(matrix, alpha=0.05):
	# Week 4: Friedman test (global, paired/blocked design - rows=repeats
	# are the blocks, columns=configurations are the groups) followed by,
	# if significant, Conover-Iman post-hoc (pairwise). This is generic
	# over the column set - works the same whether columns are the 5
	# --compare-methods methods or the 12 --feature-bagging configurations.
	# Returns (friedman_stat, friedman_p, conover_pvals_or_None, sig_table_or_None).
	stat, p = stats.friedmanchisquare(*[matrix[c] for c in matrix.columns])

	print("=" * 60)
	print("FRIEDMAN TEST (global: are all configurations equal?)")
	print("=" * 60)
	print(f"chi-square statistic: {stat:.4f}")
	print(f"p-value: {p:.6f}")

	if p > alpha:
		print(f"p > alpha ({alpha}) -> H0 NOT rejected, no significant difference found")
		print("Post-hoc test skipped (omnibus test not significant).")
		print("=" * 60 + "\n")
		return stat, p, None, None
	print(f"p <= alpha ({alpha}) -> H0 rejected, configurations differ significantly")
	print("=" * 60 + "\n")

	pvals = sp.posthoc_conover_friedman(matrix)
	print("=" * 60)
	print("CONOVER-IMAN POST-HOC TEST (pairwise p-values)")
	print("=" * 60)
	print(pvals.round(4).to_string())
	print("=" * 60 + "\n")

	# Significance table convention (Week 4 slides):
	#   "++" : p<=0.05 and mean(row) > mean(column)
	#   "+"  : p<=0.10 and mean(row) > mean(column)
	#   "-"  : mean(row) <= mean(column), not significant either way
	means = matrix.mean()
	configs = matrix.columns
	sig_table = pd.DataFrame("", index=configs, columns=configs)
	for row in configs:
		for col in configs:
			if row == col:
				continue
			p_ij = pvals.loc[row, col]
			row_higher = means[row] > means[col]
			if p_ij <= 0.05 and row_higher:
				sig_table.loc[row, col] = "++"
			elif p_ij <= 0.10 and row_higher:
				sig_table.loc[row, col] = "+"
			elif not row_higher:
				sig_table.loc[row, col] = "-"
			# else: numerically higher but not significant at either
			# threshold - cell stays blank (original Week 3/4 convention,
			# kept on request rather than the "always fill with -" fix
			# discussed in chat).

	print("=" * 60)
	print("SIGNIFICANCE TABLE (row vs column)")
	print("++ : p<=0.05, row significantly better | + : p<=0.10, row better")
	print("-  : row not better (numerically worse or equal)")
	print("=" * 60)
	print(sig_table.to_string())
	print("=" * 60 + "\n")

	return stat, p, pvals, sig_table


def run_statistical_test(args):
	# Entry point for --stat-test: reads any repeats x configurations AUC
	# matrix CSV (from --compare-methods --output-csv or --feature-bagging
	# --output-csv) and runs friedman_conover_test on it. Doesn't need
	# X_train/X_test/train_labels at all, so main() must call this BEFORE
	# load_data() to skip the (slow) parquet loading entirely.
	print("=" * 60)
	print(f"STATISTICAL TEST (Friedman + Conover-Iman): {args.stat_test}")
	print("=" * 60 + "\n")

	matrix = pd.read_csv(args.stat_test, index_col=0)
	print(f"Matrix: {matrix.shape[0]} repeats x {matrix.shape[1]} configurations")
	print(f"Configurations: {', '.join(matrix.columns)}\n")
	print("Mean +/- Std per configuration:")
	print((matrix.mean().round(4).astype(str) + " +/- " + matrix.std(ddof=1).round(4).astype(str)).to_string())
	print()

	_, _, _, sig_table = friedman_conover_test(matrix, alpha=args.alpha)

	if sig_table is not None and getattr(args, "output_csv", None):
		Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
		sig_table.to_csv(args.output_csv)
		print(f"Saved significance table to {args.output_csv}\n")


def main():
	parser = argparse.ArgumentParser(description="p53 Outlier Detection Pipeline")
	parser.add_argument("--method", type=str, default="isolation_forest",
	                     choices=["isolation_forest", "lof", "gmd_lof", "cof", "knn", "ocsvm", "abod",
	                              "deep_svdd", "tabpfn"],
	                     help="Outlier detection method")
	parser.add_argument("--compare-methods", action="store_true",
	                     help=f"Run all of {COMPARE_METHODS} through repeated_subsampling "
	                          f"with identical reduce/n_components/scale settings, print the "
	                          f"per-repeat AUC matrix to the terminal, then exit. "
	                          f"Overrides --method and --eval-strategy. See --methods-set to "
	                          f"compare a different method list instead.")
	parser.add_argument("--methods-set", type=str, default=None,
	                     help="Week 5: which method list --compare-methods should use. "
	                          "'week5' = COMPARE_METHODS_WEEK5 (last week's 6 methods + "
	                          "tabpfn). Otherwise pass a comma-separated list of --method "
	                          "names, e.g. 'isolation_forest,tabpfn'. Default: original "
	                          "6-method COMPARE_METHODS (Week 4 behavior, unchanged).")
	parser.add_argument("--output-csv", type=str, default=None,
	                     help="Save repeats x configurations AUC matrix to this CSV path. With "
	                          "--compare-methods: 5 columns, no-bagging AUC only. With "
	                          "--feature-bagging: 12 columns (3 methods x 4 aggregation "
	                          "variants: no_bagging/sum/max/aom). With --stat-test: saves the "
	                          "resulting ++/+/- significance table instead of an AUC matrix.")
	parser.add_argument("--feature-bagging", action="store_true",
	                     help=f"Step B: for each of {FEATURE_BAGGING_METHODS} separately, "
	                          f"repeated subsampling (--n-repeats) with --n-subsets random "
	                          f"feature subsets fit INSIDE each repeat, compare no_bagging vs "
	                          f"sum/max/aom aggregation. Total fits per method = n_repeats * "
	                          f"n_subsets (e.g. 25*10=250, x3 methods=750). "
	                          f"Overrides --method/--compare-methods.")
	parser.add_argument("--n-subsets", type=int, default=10,
	                     help="Number of random feature subsets per repeat for --feature-bagging")
	parser.add_argument("--heterogeneous-ensemble", action="store_true",
	                     help=f"Step C: {FEATURE_BAGGING_METHODS} on the SAME full feature "
	                          f"space (no subspace split, opposite of --feature-bagging), "
	                          f"repeated subsampling (--n-repeats), scores z-normalized and "
	                          f"combined via sum/max/rank-averaging each repeat. Total fits = "
	                          f"n_repeats * 3 methods (e.g. 25*3=75). "
	                          f"Overrides --method/--compare-methods/--feature-bagging.")
	parser.add_argument("--stat-test", type=str, default=None,
	                     help="Week 4: path to a repeats x configurations AUC matrix CSV "
	                          "(from --compare-methods --output-csv or --feature-bagging "
	                          "--output-csv). Runs a Friedman test, then (if significant) "
	                          "Conover-Iman post-hoc, and prints a ++/+/- significance table. "
	                          "Save the table with --output-csv. Skips data loading entirely - "
	                          "overrides all other modes.")
	parser.add_argument("--alpha", type=float, default=0.05,
	                     help="Significance level for --stat-test's Friedman test")
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
	                     help="Repeats for --eval-strategy repeated_subsampling / "
	                          "--compare-methods / --feature-bagging / --heterogeneous-ensemble")
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

	if args.stat_test:
		# No X_train/X_test/train_labels needed - skip load_data() entirely.
		run_statistical_test(args)
		return

	X_train, X_test, train_labels = load_data()
	test_ids = X_test.index

	if args.sample:
		X_train = X_train.sample(n=args.sample, random_state=args.random_state)
		train_labels = train_labels.loc[X_train.index]
		n_active = (train_labels == "active").sum()
		print(f"Subsampled train_set to {len(X_train):,} rows for quick testing "
		      f"(active: {n_active} - metrics below are NOT meaningful, runtime check only)\n")

	if args.heterogeneous_ensemble:
		heterogeneous_ensemble_comparison(X_train, train_labels, args)
		return

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