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
METHOD, and --heterogeneous-ensemble does n_repeats fits per method - see
chat notes on why the two scale so differently (Step B needs n_subsets
extra fits inside every single repeat, Step C doesn't).

Week 5, Step 1: TabPFN (--method tabpfn, zero-shot foundation model via
tabpfn_extensions) added as a new detector. Gated on HuggingFace - before
first use on the server, accept the license at
https://huggingface.co/Prior-Labs/tabpfn_3 and run `hf auth login` (or
set HF_TOKEN), otherwise fit() raises TabPFNHuggingFaceGatedRepoError.
See COMPARE_METHODS_WEEK5 / --methods-set below to compare tabpfn
against all 6 methods from last week in one run. Cost/OOM fixes and
throttling params are documented on TabPFNDetector itself.

Week 5, Step 2: Feature Bagging restricted to isolation_forest only (Top-3
from Step 1 was isolation_forest/tabpfn/knn, but feature bagging multiplies
cost by n_subsets per repeat - too expensive for tabpfn, and isolation_forest
alone is the cleanest test of whether subspace bagging helps at all before
widening to other methods). Two new aggregation variants added alongside
the existing sum/max/aom: pruned (drop the subsets least correlated with
the ensemble consensus before averaging) and damped (clip z-scores to
[-3,3] before averaging, so one runaway subset can't dominate). Subspace
source is --subspace-method: 'random' (default, unchanged) or 'vgan'
(V-GAN_no_kl-generated subspaces, trained once per repeat - see
VGANSubspaceGenerator; needs vgan_lib/ bundled locally and
torch-two-sample installed, see requirements.txt, neither is on PyPI).

Statistical testing (--stat-test): Week 4. Runs a Friedman test (global:
do all configurations in the given matrix differ?) followed by, if
significant, a Conover-Iman post-hoc test (pairwise: which configurations
differ from which?) on ANY repeats x configurations AUC matrix CSV - so
the same command works on the --compare-methods matrix and the
--feature-bagging matrix, or any other matrix with the same
repeats-as-rows/configs-as-columns shape. Prints a ++/+/- significance
table (row significantly better than column); save it with --output-csv.
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

# Week 6, Step 1: Neural Networks + Graph-based methods evaluated on the
# OCC task. LUNAR/DIF move on to HPO.
COMPARE_METHODS_WEEK6_NEURAL_GRAPH = ["lunar", "dif"]

# Week 5, Step 2: feature bagging restricted to isolation_forest only
# (see module docstring for why).
FEATURE_BAGGING_METHODS = ["isolation_forest"]

# Week 5, Step 3: heterogeneous ensemble uses the Top-3 methods from Step 1
# (isolation_forest, tabpfn, knn) as ensemble members - unlike Step 2, the
# whole point here is combining DIFFERENT methods, so this needs its own
# 3-method list rather than reusing FEATURE_BAGGING_METHODS.
HETEROGENEOUS_ENSEMBLE_METHODS = ["isolation_forest", "tabpfn", "knn"]

# Week 6, Step 3: final 5 models after HPO.
COMPARE_METHODS_WEEK6_FINAL = ["isolation_forest", "knn", "deep_svdd", "lunar", "dif"]

# Best params found by --hpo for each of the 5 (dif: best trial so far,
# search still running at representation_dim=45/n_ensemble=75/n_estimators=10,
# AUC=0.7166). Used by --compare-methods --tuned via build_hpo_detector.
TUNED_PARAMS = {
	"isolation_forest": {
		"n_estimators": 143,
		"max_samples": 0.21404896259450956,
		"max_features": 0.9381550605728436,
		"bootstrap": True,
	},
	"knn": {"n_neighbors": 50, "method": "largest"},
	"deep_svdd": {"hidden_neurons": [64, 32], "epochs": 40, "batch_size": 128},
	"lunar": {"n_neighbours": 9, "n_epochs": 110, "lr": 0.0002912245673897},
	"dif": {"representation_dim": 45, "n_ensemble": 75, "n_estimators": 10},
	"inne": {"n_estimators": 320, "max_samples": 11},
	"loda": {"n_bins": 46, "n_random_cuts": 234},
}

# Week 6, Step 2: HPO via Optuna for isolation_forest, knn, deep_svdd,
# dif, lunar. Each entry is (suggest_fn, resolve_fn): suggest_fn
# suggests trial params, resolve_fn rebuilds the same kwargs from
# study.best_params afterwards (for models with categorical arch options).
_ARCH_OPTIONS_DEEP_SVDD = {"32_16": [32, 16], "64_32": [64, 32], "128_64_32": [128, 64, 32]}


def _suggest_params_isolation_forest(trial):
	return {
		"n_estimators": trial.suggest_int("n_estimators", 50, 500),
		"max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
		"max_features": trial.suggest_float("max_features", 0.3, 1.0),
		"bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
	}


def _suggest_params_knn(trial):
	return {
		"n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
		"method": trial.suggest_categorical("method", ["largest", "mean", "median"]),
	}


def _suggest_params_deep_svdd(trial):
	arch = trial.suggest_categorical("hidden_neurons", list(_ARCH_OPTIONS_DEEP_SVDD))
	return {
		"hidden_neurons": _ARCH_OPTIONS_DEEP_SVDD[arch],
		"epochs": trial.suggest_int("epochs", 10, 60),
		"batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
	}


def _resolve_params_deep_svdd(raw):
	return {
		"hidden_neurons": _ARCH_OPTIONS_DEEP_SVDD[raw["hidden_neurons"]],
		"epochs": raw["epochs"],
		"batch_size": raw["batch_size"],
	}


def _suggest_params_dif(trial):
	return {
		"representation_dim": trial.suggest_int("representation_dim", 10, 50),
		"n_ensemble": trial.suggest_int("n_ensemble", 10, 100),
		"n_estimators": trial.suggest_int("n_estimators", 3, 10),
	}


def _suggest_params_lunar(trial):
	return {
		"n_neighbours": trial.suggest_int("n_neighbours", 3, 20),
		"n_epochs": trial.suggest_int("n_epochs", 50, 300),
		"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
	}


def _suggest_params_inne(trial):
	# PyOD INNE: max_samples='auto' means min(8, n). Float fractions of n
	# (~20k) make distance computations infeasible — keep small ints.
	return {
		"n_estimators": trial.suggest_int("n_estimators", 50, 400),
		"max_samples": trial.suggest_int("max_samples", 8, 64),
	}


def _suggest_params_loda(trial):
	return {
		"n_bins": trial.suggest_int("n_bins", 5, 50),
		"n_random_cuts": trial.suggest_int("n_random_cuts", 50, 500),
	}


def _identity(raw):
	return dict(raw)


HPO_PARAM_SPACES = {
	"isolation_forest": (_suggest_params_isolation_forest, _identity),
	"knn": (_suggest_params_knn, _identity),
	"deep_svdd": (_suggest_params_deep_svdd, _resolve_params_deep_svdd),
	"dif": (_suggest_params_dif, _identity),
	"lunar": (_suggest_params_lunar, _identity),
	"inne": (_suggest_params_inne, _identity),
	"loda": (_suggest_params_loda, _identity),
}


def build_hpo_detector(model, params, random_state):
	if model == "isolation_forest":
		return IsolationForest(random_state=random_state, n_jobs=-1, **params)
	if model == "knn":
		from pyod.models.knn import KNN
		return PyODAdapter(KNN(n_jobs=-1, **params))
	if model == "deep_svdd":
		return DeepSVDDDetector(random_state=random_state, **params)
	if model == "dif":
		from pyod.models.dif import DIF
		return PyODAdapter(DIF(random_state=random_state, **params))
	if model == "lunar":
		from pyod.models.lunar import LUNAR
		return PyODAdapter(LUNAR(random_state=random_state, **params))
	if model == "inne":
		from pyod.models.inne import INNE
		return PyODAdapter(INNE(random_state=random_state, contamination=0.05, **params))
	if model == "loda":
		from pyod.models.loda import LODA
		return PyODAdapter(LODA(contamination=0.05, **params))
	raise ValueError(f"No HPO search space for {model}. Available: {list(HPO_PARAM_SPACES)}")


def run_hpo_single(model, X_train, train_labels, args):
	# Runs HPO for one model; returns (best_value, best_params, final_metrics).
	import optuna

	suggest_fn, resolve_fn = HPO_PARAM_SPACES[model]

	sampler = optuna.samplers.GPSampler(seed=args.random_state) if args.hpo_sampler == "gp" \
		else optuna.samplers.TPESampler(seed=args.random_state)
	study = optuna.create_study(direction="maximize", sampler=sampler)

	hpo_args = argparse.Namespace(**{**vars(args), "method": model,
	                                  "feature_select": None, "smote": False})

	print("=" * 60)
	print(f"HPO: {model}  (sampler={args.hpo_sampler}, trials={args.hpo_trials}, "
	      f"repeats_per_trial={args.hpo_repeats})")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale})")
	print("=" * 60 + "\n")

	def objective(trial):
		trial_t0 = time.time()
		params = suggest_fn(trial)
		factory = lambda: build_hpo_detector(model, params, args.random_state)
		metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, hpo_args, n_repeats=args.hpo_repeats, verbose=False,
			detector_factory=factory,
		)
		elapsed = time.time() - trial_t0
		print(f"  trial {trial.number + 1}/{args.hpo_trials}: AUC={metrics['auc']:.4f}  "
		      f"params={trial.params}  ({elapsed:.1f}s)")
		return metrics["auc"]

	t0 = time.time()
	study.optimize(objective, n_trials=args.hpo_trials)
	total_elapsed = time.time() - t0

	print("\n" + "=" * 60)
	print("HPO RESULT")
	print("=" * 60)
	print(f"Best AUC ({args.hpo_repeats} repeats/trial during search): {study.best_value:.4f}")
	print(f"Best params: {study.best_params}")
	print(f"Total HPO time: {total_elapsed / 60:.1f} min "
	      f"({total_elapsed / max(1, args.hpo_trials):.1f}s/trial avg)")
	print("=" * 60 + "\n")

	final_metrics = None
	if args.hpo_final_repeats:
		print(f"Running final check with best params at {args.hpo_final_repeats} repeats...\n")
		final_params = resolve_fn(study.best_params)
		final_factory = lambda: build_hpo_detector(model, final_params, args.random_state)
		final_metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, hpo_args, n_repeats=args.hpo_final_repeats, verbose=False,
			detector_factory=final_factory,
		)
		print("=" * 60)
		print(f"FINAL CHECK ({args.hpo_final_repeats} repeats)")
		print("=" * 60)
		print(f"AUC: {final_metrics['auc']:.4f} +/- {final_metrics['auc_std']:.4f}")
		print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		out_path = args.output_csv
		if getattr(args, "hpo_model", None) == "all":
			p = Path(args.output_csv)
			out_path = str(p.with_name(f"{p.stem}_{model}{p.suffix}"))
		Path(out_path).parent.mkdir(parents=True, exist_ok=True)
		study.trials_dataframe().to_csv(out_path, index=False)
		print(f"Saved all {len(study.trials)} trials to {out_path}\n")

	return study.best_value, study.best_params, final_metrics


def run_hpo(X_train, train_labels, args):
	if args.hpo_model == "all":
		models = list(HPO_PARAM_SPACES)
		print("=" * 60)
		print(f"HPO (ALL MODELS): {', '.join(models)}")
		print(f"({args.hpo_trials} trials x {args.hpo_repeats} repeats/trial each, "
		      f"then {args.hpo_final_repeats}-repeat final check per model)")
		print("=" * 60 + "\n")

		summary = {}
		t0 = time.time()
		for model in models:
			print("#" * 60)
			print(f"# MODEL: {model}")
			print("#" * 60)
			best_value, best_params, final_metrics = run_hpo_single(model, X_train, train_labels, args)
			summary[model] = (best_value, best_params, final_metrics)
		total_elapsed = time.time() - t0

		print("\n" + "=" * 60)
		print(f"HPO SUMMARY (all {len(models)} models, {total_elapsed / 60:.1f} min total)")
		print("=" * 60)
		for model, (best_value, best_params, final_metrics) in summary.items():
			if final_metrics is not None:
				print(f"  {model:<18} search_AUC={best_value:.4f}  "
				      f"final_AUC={final_metrics['auc']:.4f}+/-{final_metrics['auc_std']:.4f}")
			else:
				print(f"  {model:<18} search_AUC={best_value:.4f}  (no final check)")
			print(f"    best params: {best_params}")
		print("=" * 60 + "\n")
		return

	if args.hpo_model not in HPO_PARAM_SPACES:
		raise ValueError(f"No HPO search space for {args.hpo_model}. Available: {list(HPO_PARAM_SPACES)}")
	run_hpo_single(args.hpo_model, X_train, train_labels, args)


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
	# PyOD DeepSVDD needs n_features at init; infer from fit data.
	# preprocessing=False - scale+PCA already applied upstream.
	def __init__(self, random_state=42, epochs=20, batch_size=512, hidden_neurons=None):
		self.random_state = random_state
		self.epochs = epochs
		self.batch_size = batch_size
		self.hidden_neurons = hidden_neurons if hidden_neurons is not None else [64, 32]
		self.model_ = None

	def fit(self, X, y=None):
		from pyod.models.deep_svdd import DeepSVDD

		X_arr = np.asarray(X, dtype=np.float32)
		self.model_ = DeepSVDD(
			n_features=X_arr.shape[1],
			epochs=self.epochs,
			batch_size=self.batch_size,
			hidden_neurons=self.hidden_neurons,
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
		# just outside each batch's own finite range (preserves ranking;
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


class VGANSubspaceGenerator:
	# Wraps V-GAN (Cribeiro-Ramallo et al. 2025, "Adversarial Subspace
	# Generation for Outlier Detection in High-Dimensional Data") as an
	# alternative subspace source for --feature-bagging, instead of random
	# feature subsets. Uses the no-kernel-learning variant (VGAN_no_kl) -
	# simpler/faster to train than the paper's default kernel-learning
	# VGAN, at some cost to subspace quality.
	#
	# Not pip-installable (no PyPI package, no setup.py in the repo) -
	# source bundled locally under vgan_lib/ (copied from
	# https://github.com/jcribeiro98/V-GAN's src/ folder, same internal
	# structure so its relative imports still work). Also needs
	# torch-two-sample, itself not on PyPI either - installed via git, see
	# requirements.txt. vgan_lib/vgan.py imports torch_two_sample at
	# module level even though VGAN_no_kl itself doesn't use it, so it's
	# a hard dependency to even import the file.
	#
	# epochs=50 here is far below the paper's default (2000) - training
	# V-GAN once per repeat (e.g. 25x in a full --feature-bagging run) at
	# 2000 epochs would be prohibitively slow. 50 is a starting point to
	# check whether V-GAN subspaces beat random ones at all before
	# investing in a longer training budget (--vgan-epochs to sweep).
	def __init__(self, epochs=50, lr=0.001, random_state=42):
		self.epochs = epochs
		self.lr = lr
		self.random_state = random_state
		self.model_ = None

	def fit(self, X):
		from vgan_lib.vgan import VGAN_no_kl

		self.model_ = VGAN_no_kl(epochs=self.epochs, lr=self.lr, seed=self.random_state)
		self.model_.fit(np.asarray(X, dtype=np.float32))
		return self

	def generate_column_masks(self, n_subsets):
		# Returns n_subsets boolean numpy arrays (True = feature included) -
		# same shape/indexing semantics as the integer index arrays the
		# random path builds via rng_subsets.choice(), so X[:, mask] works
		# identically downstream regardless of which path produced it.
		subspaces = self.model_.generate_subspaces(n_subsets)
		return [s.to("cpu").numpy() for s in subspaces]


class VGANKernelSubspaceGenerator:
	# Same idea as VGANSubspaceGenerator, but uses the paper's default
	# kernel-learning variant (VGAN, not VGAN_no_kl) - trains an
	# encoder/decoder alongside the generator (see module docstring in
	# vgan_lib/vgan.py), which the paper reports as the stronger of the
	# two variants for subspace quality, at extra per-epoch cost.
	#
	# VGAN's constructor takes separate lr_G/lr_D rather than a single lr
	# (unlike VGAN_no_kl) - both set to the same --vgan-lr value here to
	# keep the "same setup" comparison simple (chat request: same epochs
	# and lr as the VGAN_no_kl run, not a separate tuning pass).
	def __init__(self, epochs=50, lr=0.001, random_state=42):
		self.epochs = epochs
		self.lr = lr
		self.random_state = random_state
		self.model_ = None

	def fit(self, X):
		from vgan_lib.vgan import VGAN

		self.model_ = VGAN(
			epochs=self.epochs, lr_G=self.lr, lr_D=self.lr, seed=self.random_state,
		)
		self.model_.fit(np.asarray(X, dtype=np.float32))
		return self

	def generate_column_masks(self, n_subsets):
		subspaces = self.model_.generate_subspaces(n_subsets)
		return [s.to("cpu").numpy() for s in subspaces]


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
	if method == "lunar":
		from pyod.models.lunar import LUNAR
		return PyODAdapter(LUNAR(random_state=random_state))
	if method == "ae1svm":
		from pyod.models.ae1svm import AE1SVM
		return PyODAdapter(AE1SVM())
	if method == "dif":
		from pyod.models.dif import DIF
		return PyODAdapter(DIF(random_state=random_state))
	if method == "inne":
		from pyod.models.inne import INNE
		return PyODAdapter(INNE(n_estimators=n_estimators, random_state=random_state, contamination=0.05))
	if method == "loda":
		from pyod.models.loda import LODA
		return PyODAdapter(LODA(n_bins=10, n_random_cuts=100, contamination=0.05))
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
                   feature_select=None, detector_factory=None):
	# detector_factory: optional callable returning a pre-parametrized
	# detector, used by --hpo instead of make_detector(method, ...).
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
	detector = detector_factory() if detector_factory is not None else \
		make_detector(method, n_estimators, random_state, max_subspaces)
	steps.extend([
		("reduce", make_reducer(reduce, n_components, random_state)),
		("detector", detector),
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
	"inne": "FITTING INNE (PyOD Isolation Nearest-Neighbor Ensemble)",
	"loda": "FITTING LODA (PyOD Lightweight On-line Detector of Anomalies)",
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
                                         inlier_frac=0.8, outlier_frac=0.8, verbose=True,
                                         detector_factory=None):
	# Step A. Folie 7 (Week 3): sample 80% of inliers for training; combine
	# the remaining 20% inliers with a random 80% sample of outliers as the
	# validation split; repeat many times. Training uses ONLY inliers, no
	# labels passed into fit anywhere - pure OCC framing as on that slide.
	#
	# detector_factory: used by --hpo to inject a trial's hyperparameters.
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
			detector_factory=detector_factory,
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
                                      inlier_frac=0.8, outlier_frac=0.8, verbose=True,
                                      detector_params=None):
	# Step B, full repeated-subsampling version. Same 80/20 split logic as
	# Step A; scale+reduce fit on the train split only, applied to val with
	# the same fit (no leakage).
	#
	# detector_params: optional dict of tuned hyperparameters (from --hpo),
	# used via build_hpo_detector instead of make_detector's defaults - for
	# both the no_bagging baseline and every subset detector.
	#
	# Week 5, Step 2: two new aggregation variants alongside sum/max/aom:
	#   - pruned: drop the subsets least correlated with the ensemble
	#     consensus (mean of all z-normalized subset scores) before
	#     averaging the rest. No labels used - purely an unsupervised proxy
	#     for "this subset disagrees with the group".
	#   - damped: clip z-normalized scores to [-3, 3] before averaging, so
	#     one subset with an extreme, badly-calibrated score range can't
	#     dominate the sum/average.
	if getattr(args, "feature_select", None) or getattr(args, "smote", False):
		raise ValueError("feature bagging repeated subsampling doesn't support --feature-select/--smote")
	subspace_method = getattr(args, "subspace_method", "random")
	vgan_epochs = getattr(args, "vgan_epochs", 50)
	vgan_lr = getattr(args, "vgan_lr", 0.001)

	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx_all = np.where(y == 0)[0]
	outlier_idx_all = np.where(y == 1)[0]

	repeat_results = {
		"no_bagging": [], "bagging_sum": [], "bagging_max": [], "bagging_aom": [],
		"bagging_pruned": [], "bagging_damped": [],
	}

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
		if detector_params is not None:
			base_detector = build_hpo_detector(method, detector_params, args.random_state)
		else:
			base_detector = make_detector(method, args.n_estimators, args.random_state, args.max_subspaces)
		base_detector.fit(X_tr_red)
		base_scores_val = -base_detector.score_samples(X_val_red)
		repeat_results["no_bagging"].append(roc_auc_score(y_val, base_scores_val))

		if subspace_method == "vgan":
			# Train V-GAN once per repeat on this repeat's train split,
			# then draw n_subsets subspaces from it - one VGAN fit, reused
			# for all n_subsets detector fits in this repeat (matches the
			# random path's cost shape: one rng, n_subsets draws from it).
			# Total cost = n_repeats FULL VGAN trainings (each restarts
			# from scratch, not resumed) - the per-epoch progress printed
			# by VGAN_no_kl.fit() only reflects the CURRENT repeat, not
			# the whole --feature-bagging run (source of confusion on the
			# server, chat notes).
			print(f"  [repeat {i+1}/{n_repeats}] training V-GAN (no kernel learning, "
			      f"{vgan_epochs} epochs) for subspace generation...")
			vgan_gen = VGANSubspaceGenerator(
				epochs=vgan_epochs, lr=vgan_lr, random_state=args.random_state,
			).fit(X_tr_red)
			col_masks = vgan_gen.generate_column_masks(n_subsets)
		elif subspace_method == "vgan_kernel":
			# Same idea, kernel-learning variant (VGANKernelSubspaceGenerator) -
			# same epochs/lr as --subspace-method vgan, per chat request to
			# compare the two variants under an otherwise identical setup.
			print(f"  [repeat {i+1}/{n_repeats}] training V-GAN (kernel learning, "
			      f"{vgan_epochs} epochs) for subspace generation...")
			vgan_gen = VGANKernelSubspaceGenerator(
				epochs=vgan_epochs, lr=vgan_lr, random_state=args.random_state,
			).fit(X_tr_red)
			col_masks = vgan_gen.generate_column_masks(n_subsets)
		else:
			col_masks = None  # drawn inline per-subset below (unchanged random path)

		rng_subsets = np.random.default_rng(args.random_state)
		subset_scores_val = []
		for j in range(n_subsets):
			if col_masks is not None:
				cols = col_masks[j]
			else:
				k = int(rng_subsets.integers(int(subset_min_frac * d), d + 1))
				cols = rng_subsets.choice(d, size=k, replace=False)

			detector = build_hpo_detector(method, detector_params, args.random_state + j + 1) \
				if detector_params is not None else \
				make_detector(method, args.n_estimators, args.random_state + j + 1, args.max_subspaces)
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

		# pruned: keep only the top half of subsets most correlated with
		# the ensemble consensus (mean of all z-normalized subset scores),
		# then average those.
		consensus = subset_matrix.mean(axis=0)
		correlations = np.array([
			np.corrcoef(subset_matrix[s], consensus)[0, 1] for s in range(n_subsets)
		])
		keep_mask = correlations >= np.median(correlations)
		agg_pruned = subset_matrix[keep_mask].mean(axis=0)

		# damped: clip each subset's z-scores to [-3, 3] before averaging.
		damped_matrix = np.clip(subset_matrix, -3, 3)
		agg_damped = damped_matrix.mean(axis=0)

		repeat_results["bagging_sum"].append(roc_auc_score(y_val, agg_sum))
		repeat_results["bagging_max"].append(roc_auc_score(y_val, agg_max))
		repeat_results["bagging_aom"].append(roc_auc_score(y_val, agg_aom))
		repeat_results["bagging_pruned"].append(roc_auc_score(y_val, agg_pruned))
		repeat_results["bagging_damped"].append(roc_auc_score(y_val, agg_damped))

		if verbose:
			print(f"  repeat {i+1}/{n_repeats}: no_bagging={repeat_results['no_bagging'][-1]:.4f}  "
			      f"sum={repeat_results['bagging_sum'][-1]:.4f}  "
			      f"max={repeat_results['bagging_max'][-1]:.4f}  "
			      f"aom={repeat_results['bagging_aom'][-1]:.4f}  "
			      f"pruned={repeat_results['bagging_pruned'][-1]:.4f}  "
			      f"damped={repeat_results['bagging_damped'][-1]:.4f}")

	if not repeat_results["no_bagging"]:
		raise ValueError("Feature bagging repeated subsampling produced no valid repeats")

	summary = {}
	for key, vals in repeat_results.items():
		summary[key] = {
			"mean": float(np.mean(vals)),
			"std": float(np.std(vals, ddof=1)),
			"n": len(vals),
			"values": vals,  # raw per-repeat AUCs, needed for the CSV export
		}
	return summary


def feature_bagging_comparison(X_train, train_labels, args):
	# Runs feature_bagging_repeated_metrics separately for each method in
	# the resolved method list (no mixing of method and subspace across
	# methods), prints per-method mean +/- std plus a summary table.
	#
	# Method list resolution: --bagging-methods overrides the
	# FEATURE_BAGGING_METHODS default (isolation_forest only) with a
	# comma-separated list, to try a different method (or subset) before
	# deciding whether to widen FEATURE_BAGGING_METHODS itself. Default
	# (no flag) stays exactly Week 5 Step 2's behavior.
	raw_bagging_methods = getattr(args, "bagging_methods", None)
	if raw_bagging_methods:
		methods = [m.strip() for m in raw_bagging_methods.split(",") if m.strip()]
	else:
		methods = FEATURE_BAGGING_METHODS

	json_override = getattr(args, "detector_params_json", None)
	use_tuned = getattr(args, "tuned", False)
	if json_override:
		import json as _json
		override_params = _json.loads(json_override)
	elif use_tuned:
		missing = [m for m in methods if m not in TUNED_PARAMS]
		if missing:
			raise ValueError(f"--tuned has no TUNED_PARAMS entry for: {missing}")
		override_params = None  # resolved per-method from TUNED_PARAMS below
	else:
		override_params = None

	total_fits = args.n_repeats * args.n_subsets * len(methods)
	print("="*60)
	print(f"FEATURE BAGGING (repeated subsampling): {', '.join(methods)}" +
	      (" (tuned params)" if (use_tuned or json_override) else ""))
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats}, n_subsets={args.n_subsets})")
	print(f"Total fits: {args.n_repeats} repeats x {args.n_subsets} subsets x "
	      f"{len(methods)} methods = {total_fits}")
	print("="*60 + "\n")

	rows = {}
	for method in methods:
		print("#" * 60)
		print(f"# METHOD: {method}")
		print("#" * 60)
		if json_override:
			params_for_method = override_params
		elif use_tuned:
			params_for_method = TUNED_PARAMS[method]
		else:
			params_for_method = None
		rows[method] = feature_bagging_repeated_metrics(
			X_train, train_labels, args, method,
			n_repeats=args.n_repeats, n_subsets=args.n_subsets, verbose=False,
			detector_params=params_for_method,
		)
		for key, stats_ in rows[method].items():
			print(f"  {key:<15} {stats_['mean']:.4f} +/- {stats_['std']:.4f}  ({stats_['n']} repeats)")
		print()

	print("=" * 60)
	print("FEATURE BAGGING SUMMARY (mean +/- std over repeats)")
	print("no_bagging = single model on all components (Step A baseline)")
	print("=" * 60)
	variant_keys = ["no_bagging", "bagging_sum", "bagging_max", "bagging_aom",
	                 "bagging_pruned", "bagging_damped"]
	variant_labels = ["no_bagging", "sum", "max", "aom", "pruned", "damped"]
	header = f"{'method':<18}" + "".join(f"{k:>16}" for k in variant_labels)
	print(header)
	for method in methods:
		r = rows[method]
		line = f"{method:<18}"
		for key in variant_keys:
			line += f"{r[key]['mean']:>10.4f}+/-{r[key]['std']:.4f}"
		print(line)
	print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		# len(methods) x 6 aggregation variants, one row per
		# repeat - for a statistical test comparing all method+aggregation
		# configurations against each other.
		columns = {}
		for method in methods:
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
		      f"{len(methods)} methods x {len(variant_labels)} aggregations)\n")


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
	repeat_results.update({
		"ensemble_sum": [], "ensemble_max": [], "ensemble_rank_avg": [],
		"ensemble_pruned": [], "ensemble_damped": [],
	})

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
		stacked = np.vstack(list(scores_z.values()))  # (n_methods, n_val)
		n_methods = stacked.shape[0]

		agg_sum = stacked.sum(axis=0)
		agg_max = stacked.max(axis=0)
		ranks = np.vstack([pd.Series(s).rank().values for s in scores_z.values()])
		agg_rank = ranks.mean(axis=0)

		# pruned: keep only the methods most correlated with the ensemble
		# consensus (mean of all z-normalized method scores), then average
		# those - same recipe as feature_bagging_repeated_metrics's pruned,
		# just over methods instead of feature subsets.
		consensus = stacked.mean(axis=0)
		correlations = np.array([
			np.corrcoef(stacked[s], consensus)[0, 1] for s in range(n_methods)
		])
		keep_mask = correlations >= np.median(correlations)
		agg_pruned = stacked[keep_mask].mean(axis=0)

		# damped: clip each method's z-scores to [-3, 3] before averaging.
		damped_matrix = np.clip(stacked, -3, 3)
		agg_damped = damped_matrix.mean(axis=0)

		repeat_results["ensemble_sum"].append(roc_auc_score(y_val, agg_sum))
		repeat_results["ensemble_max"].append(roc_auc_score(y_val, agg_max))
		repeat_results["ensemble_rank_avg"].append(roc_auc_score(y_val, agg_rank))
		repeat_results["ensemble_pruned"].append(roc_auc_score(y_val, agg_pruned))
		repeat_results["ensemble_damped"].append(roc_auc_score(y_val, agg_damped))

		if verbose:
			method_str = "  ".join(f"{m}={repeat_results[m][-1]:.4f}" for m in methods)
			print(f"  repeat {i+1}/{n_repeats}: {method_str}  "
			      f"sum={repeat_results['ensemble_sum'][-1]:.4f}  "
			      f"max={repeat_results['ensemble_max'][-1]:.4f}  "
			      f"rank_avg={repeat_results['ensemble_rank_avg'][-1]:.4f}  "
			      f"pruned={repeat_results['ensemble_pruned'][-1]:.4f}  "
			      f"damped={repeat_results['ensemble_damped'][-1]:.4f}")

	if not repeat_results["ensemble_sum"]:
		raise ValueError("Heterogeneous ensemble repeated subsampling produced no valid repeats")

	summary = {}
	for key, vals in repeat_results.items():
		summary[key] = {
			"mean": float(np.mean(vals)),
			"std": float(np.std(vals, ddof=1)),
			"n": len(vals),
			"values": vals,  # raw per-repeat AUCs, needed for the CSV export
		}
	return summary


def heterogeneous_ensemble_comparison(X_train, train_labels, args):
	# Runs HETEROGENEOUS_ENSEMBLE_METHODS (Top-3 from Step 1: isolation_forest,
	# tabpfn, knn) all on the same repeated-subsampling splits, combines via
	# sum/max/rank_avg/pruned/damped (same 5-variant recipe as Step 2's
	# feature bagging, just over methods instead of feature subsets),
	# prints individual method mean +/- std plus the ensemble variants.
	total_fits = args.n_repeats * len(HETEROGENEOUS_ENSEMBLE_METHODS)
	print("="*60)
	print(f"HETEROGENEOUS ENSEMBLE (repeated subsampling): {', '.join(HETEROGENEOUS_ENSEMBLE_METHODS)}")
	print(f"(reduce={args.reduce}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats}, same feature space for all methods)")
	print(f"Total fits: {args.n_repeats} repeats x {len(HETEROGENEOUS_ENSEMBLE_METHODS)} methods = {total_fits}")
	print("="*60 + "\n")

	results = heterogeneous_ensemble_repeated_metrics(
		X_train, train_labels, args, HETEROGENEOUS_ENSEMBLE_METHODS, n_repeats=args.n_repeats, verbose=True,
	)

	print("\n" + "=" * 60)
	print("HETEROGENEOUS ENSEMBLE SUMMARY (mean +/- std over repeats)")
	print("=" * 60)
	for method in HETEROGENEOUS_ENSEMBLE_METHODS:
		r = results[method]
		print(f"  {method:<20} {r['mean']:.4f} +/- {r['std']:.4f}")
	print("  " + "-" * 40)
	ensemble_variants = [
		("ensemble_sum", "ensemble (sum)"), ("ensemble_max", "ensemble (max)"),
		("ensemble_rank_avg", "ensemble (rank avg)"), ("ensemble_pruned", "ensemble (pruned)"),
		("ensemble_damped", "ensemble (damped)"),
	]
	for key, label in ensemble_variants:
		r = results[key]
		print(f"  {label:<20} {r['mean']:.4f} +/- {r['std']:.4f}")
	print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		# len(HETEROGENEOUS_ENSEMBLE_METHODS) individual methods + 5 ensemble
		# variants, one row per repeat - for --stat-test.
		columns = {m: results[m]["values"] for m in HETEROGENEOUS_ENSEMBLE_METHODS}
		for key, _ in ensemble_variants:
			columns[key] = results[key]["values"]

		min_len = min(len(v) for v in columns.values())
		if any(len(v) != min_len for v in columns.values()):
			print(f"WARNING: configurations had different numbers of valid repeats, "
			      f"truncating all to {min_len} for the matrix.\n")

		matrix = pd.DataFrame({k: v[:min_len] for k, v in columns.items()})
		matrix.index.name = "repeat"

		Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
		matrix.to_csv(args.output_csv)
		print(f"Saved heterogeneous ensemble AUC matrix to {args.output_csv} "
		      f"({matrix.shape[0]} rows x {matrix.shape[1]} columns: "
		      f"{len(HETEROGENEOUS_ENSEMBLE_METHODS)} methods + {len(ensemble_variants)} "
		      f"ensemble variants)\n")


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


def compare_reductions(X_train, train_labels, args):
	# Fixes --method/--n-components, varies --reduce across --reduce-sweep's
	# comma-separated list - same repeated_random_subsampling_metrics
	# machinery and output shape as compare_methods, just a different axis.
	reduces = [r.strip() for r in args.reduce_sweep.split(",") if r.strip()]

	json_override = None
	if getattr(args, "detector_params_json", None):
		import json as _json
		json_override = _json.loads(args.detector_params_json)
	use_tuned = getattr(args, "tuned", False) and json_override is None
	if use_tuned and args.method not in TUNED_PARAMS:
		raise ValueError(f"--tuned has no TUNED_PARAMS entry for: {args.method}")

	print("=" * 60)
	print(f"COMPARING REDUCTIONS: {', '.join(reduces)}")
	print(f"(method={args.method}, n_components={args.n_components}, scale={args.scale}, "
	      f"n_repeats={args.n_repeats}"
	      + (" , tuned params)" if (use_tuned or json_override) else ")"))
	print("=" * 60 + "\n")

	if json_override is not None:
		detector_factory = lambda: build_hpo_detector(args.method, json_override, args.random_state)
	elif use_tuned:
		detector_factory = lambda: build_hpo_detector(args.method, TUNED_PARAMS[args.method], args.random_state)
	else:
		detector_factory = None

	results = {}
	for r in reduces:
		print("#" * 60)
		print(f"# REDUCE: {r}")
		print("#" * 60)

		method_args = argparse.Namespace(**{**vars(args), "reduce": r,
		                                     "feature_select": None, "smote": False})
		metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, method_args, n_repeats=args.n_repeats, verbose=False,
			detector_factory=detector_factory,
		)
		results[r] = metrics["repeat_aucs"]
		print(f"AUC: {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}  "
		      f"({metrics['n_repeats']} valid repeats)\n")

	min_len = min(len(v) for v in results.values())
	if any(len(v) != min_len for v in results.values()):
		print(f"WARNING: reductions had different numbers of valid repeats, "
		      f"truncating all to {min_len} for the matrix.\n")

	matrix = pd.DataFrame({r: v[:min_len] for r, v in results.items()})
	matrix.index.name = "repeat"

	print("=" * 60)
	print(f"AUC MATRIX ({matrix.shape[0]} repeats x {matrix.shape[1]} reductions)")
	print("run --stat-test on the saved CSV for the Friedman/Conover-Iman test")
	print("=" * 60)
	print(matrix.round(4).to_string())
	print()
	print("Mean +/- Std per reduction:")
	print((matrix.mean().round(4).astype(str) + " +/- " + matrix.std(ddof=1).round(4).astype(str)).to_string())
	print("=" * 60 + "\n")

	if getattr(args, "output_csv", None):
		Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
		matrix.to_csv(args.output_csv)
		print(f"Saved AUC matrix to {args.output_csv} "
		      f"({matrix.shape[0]} rows x {matrix.shape[1]} columns)\n")


def compare_methods(X_train, train_labels, args):
	# Runs a method list through repeated_random_subsampling_metrics with
	# identical reduce/n_components/scale settings (fair comparison, same
	# feature space for every method), prints the per-repeat AUC matrix to
	# the terminal. Always uses repeated_subsampling regardless of
	# --eval-strategy - this comparison mode is separate from the single-
	# method CV/repeated_subsampling choice above. If --output-csv is set,
	# also writes the matrix to disk - this is one of the two direct
	# inputs for --stat-test (Friedman/Conover-Iman, Week 4).
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
	elif raw_methods_set == "week6_neural_graph":
		methods = COMPARE_METHODS_WEEK6_NEURAL_GRAPH
	elif raw_methods_set == "week6_final":
		methods = COMPARE_METHODS_WEEK6_FINAL
	else:
		methods = [m.strip() for m in raw_methods_set.split(",") if m.strip()]

	use_tuned = getattr(args, "tuned", False)
	if use_tuned:
		missing = [m for m in methods if m not in TUNED_PARAMS]
		if missing:
			raise ValueError(f"--tuned has no TUNED_PARAMS entry for: {missing}")

	print("="*60)
	print(f"COMPARING METHODS: {', '.join(methods)}" + (" (tuned params)" if use_tuned else ""))
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
		detector_factory = None
		if use_tuned:
			detector_factory = lambda m=method: build_hpo_detector(m, TUNED_PARAMS[m], args.random_state)
		metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, method_args, n_repeats=args.n_repeats, verbose=False,
			detector_factory=detector_factory,
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
	# over the column set.
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

	# Significance table, symmetric 4-symbol scheme (chat fix, replaces the
	# old asymmetric Week 3/4 convention which only had ++/+/- - blank
	# cells and a collapsed "-" meant "row not better" covered both
	# "column significantly better" and "no significant difference",
	# losing information and behaving inconsistently between the two
	# directions of the same pair). Now every pair gets a mirrored
	# annotation:
	#   "++" : p<=0.05, row significantly better than column
	#   "+"  : p<=0.10 (but >0.05), row better than column
	#   "--" : p<=0.05, row significantly WORSE than column (mirror of ++)
	#   "-"  : p<=0.10 (but >0.05), row worse than column (mirror of +)
	#   ""   : p>0.10, not significant either way (regardless of which is
	#          numerically higher)
	means = matrix.mean()
	configs = matrix.columns
	sig_table = pd.DataFrame("", index=configs, columns=configs)
	for row in configs:
		for col in configs:
			if row == col:
				continue
			p_ij = pvals.loc[row, col]
			row_higher = means[row] > means[col]
			if p_ij <= 0.05:
				sig_table.loc[row, col] = "++" if row_higher else "--"
			elif p_ij <= 0.10:
				sig_table.loc[row, col] = "+" if row_higher else "-"
			# else: p > 0.10, not significant - cell stays blank regardless
			# of which mean happens to be numerically higher.

	print("=" * 60)
	print("SIGNIFICANCE TABLE (row vs column)")
	print("++ : p<=0.05, row significantly better  | +  : p<=0.10, row better")
	print("-- : p<=0.05, row significantly worse   | -  : p<=0.10, row worse")
	print("(blank: p>0.10, not significant either way)")
	print("=" * 60)
	print(sig_table.to_string())
	print("=" * 60 + "\n")

	return stat, p, pvals, sig_table


def run_statistical_test(args):
	# Entry point for --stat-test: reads any repeats x configurations AUC
	# matrix CSV and runs friedman_conover_test on it. Doesn't need
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
	                     help="'week5' = COMPARE_METHODS_WEEK5. "
	                          "'week6_neural_graph' = COMPARE_METHODS_WEEK6_NEURAL_GRAPH. "
	                          "'week6_final' = COMPARE_METHODS_WEEK6_FINAL (isolation_forest, "
	                          "knn, deep_svdd, lunar, dif). Otherwise comma-separated --method "
	                          "names. Default: original COMPARE_METHODS.")
	parser.add_argument("--tuned", action="store_true",
	                     help="With --compare-methods or --feature-bagging: use TUNED_PARAMS "
	                          "(best params from --hpo) instead of each method's defaults. "
	                          "Requires every method in the resolved list to have a "
	                          "TUNED_PARAMS entry. Ignored if --detector-params-json is set.")
	parser.add_argument("--detector-params-json", type=str, default=None,
	                     help="With --feature-bagging: JSON dict of hyperparameters to use "
	                          "for the (single) method instead of defaults or TUNED_PARAMS, "
	                          "e.g. '{\"n_estimators\": 84, \"max_samples\": 0.1, "
	                          "\"max_features\": 0.36}'. Takes precedence over --tuned - use "
	                          "this when the tuned params depend on --reduce (e.g. different "
	                          "HPO results for pca vs truncated_svd) rather than just the "
	                          "method.")
	parser.add_argument("--reduce-sweep", type=str, default=None,
	                     help="Comma-separated --reduce techniques to compare for the SAME "
	                          "--method/--n-components, e.g. "
	                          "'none,pca,random_projection,truncated_svd,isomap,umap'. "
	                          "Overrides --compare-methods/--method's --reduce.")
	parser.add_argument("--output-csv", type=str, default=None,
	                     help="Save repeats x configurations AUC matrix to this CSV path. With "
	                          "--stat-test: saves the resulting ++/+/- significance table instead.")
	parser.add_argument("--feature-bagging", action="store_true",
	                     help=f"Step B: for each of {FEATURE_BAGGING_METHODS} separately (or "
	                          f"--bagging-methods, if given), repeated subsampling "
	                          f"(--n-repeats) with --n-subsets random feature subsets fit "
	                          f"INSIDE each repeat, compare no_bagging vs "
	                          f"sum/max/aom/pruned/damped aggregation. Total fits per method = "
	                          f"n_repeats * n_subsets. Overrides --method/--compare-methods.")
	parser.add_argument("--bagging-methods", type=str, default=None,
	                     help="Comma-separated --method names to override "
	                          "FEATURE_BAGGING_METHODS for --feature-bagging, e.g. 'knn' or "
	                          "'isolation_forest,knn'. Default (unset): FEATURE_BAGGING_METHODS "
	                          "(isolation_forest only, Week 5 Step 2 behavior, unchanged).")
	parser.add_argument("--n-subsets", type=int, default=10,
	                     help="Number of random feature subsets per repeat for --feature-bagging")
	parser.add_argument("--subspace-method", type=str, default="random",
	                     choices=["random", "vgan", "vgan_kernel"],
	                     help="Week 5 Step 2: how --feature-bagging picks feature subsets. "
	                          "'random' (default, unchanged from before), 'vgan' "
	                          "(V-GAN_no_kl, no kernel learning) or 'vgan_kernel' "
	                          "(V-GAN with kernel learning, the paper's default variant - "
	                          "trains an encoder/decoder alongside the generator, slower "
	                          "per epoch than 'vgan'). Both trained once per repeat on that "
	                          "repeat's train split - see VGANSubspaceGenerator / "
	                          "VGANKernelSubspaceGenerator.")
	parser.add_argument("--vgan-epochs", type=int, default=50,
	                     help="Training epochs for --subspace-method vgan (paper default is "
	                          "2000, far too slow per-repeat here - 50 is a starting point).")
	parser.add_argument("--vgan-lr", type=float, default=0.001,
	                     help="Learning rate for --subspace-method vgan")
	parser.add_argument("--heterogeneous-ensemble", action="store_true",
	                     help=f"Step 3 (Week 5): {HETEROGENEOUS_ENSEMBLE_METHODS} (Top-3 from "
	                          f"Step 1) on the SAME full feature space (no subspace split, "
	                          f"opposite of --feature-bagging), repeated subsampling "
	                          f"(--n-repeats), scores z-normalized and combined via "
	                          f"sum/max/rank_avg/pruned/damped each repeat (same 5-variant "
	                          f"recipe as --feature-bagging, over methods instead of feature "
	                          f"subsets). Overrides --method/--compare-methods/--feature-bagging.")
	parser.add_argument("--hpo", action="store_true",
	                     help=f"Week 6, Step 2: Optuna hyperparameter search for --hpo-model, "
	                          f"using --hpo-repeats repeated_subsampling for each trial "
	                          f"(cheaper than the 25-repeat default - search AUC is noisier "
	                          f"but comparable across trials), then one --hpo-final-repeats "
	                          f"run with the best found params. Available models: "
	                          f"{list(HPO_PARAM_SPACES)}. Overrides --method/--compare-methods/"
	                          f"--feature-bagging/--heterogeneous-ensemble.")
	parser.add_argument("--hpo-model", type=str, default=None,
	                     choices=list(HPO_PARAM_SPACES) + ["all"],
	                     help="Which model --hpo tunes. 'all' runs every model in "
	                          "HPO_PARAM_SPACES sequentially in one invocation, printing a "
	                          "combined summary at the end.")
	parser.add_argument("--hpo-trials", type=int, default=50,
	                     help="Number of Optuna trials for --hpo")
	parser.add_argument("--hpo-repeats", type=int, default=5,
	                     help="repeated_subsampling repeats PER TRIAL during --hpo search "
	                          "(kept low - cost is trials x this - full 25-repeat precision "
	                          "isn't needed just to rank candidates)")
	parser.add_argument("--hpo-final-repeats", type=int, default=25,
	                     help="repeated_subsampling repeats for the one-off final check of "
	                          "--hpo's best params (0 to skip)")
	parser.add_argument("--hpo-sampler", type=str, default="gp", choices=["gp", "tpe"],
	                     help="Optuna sampler for --hpo. 'gp' = Gaussian Process surrogate + "
	                          "Expected Improvement (precise for our small, mostly-continuous "
	                          "search spaces, chat decision). 'tpe' = Optuna's default "
	                          "Tree-structured Parzen Estimator (scales better to many/"
	                          "categorical dims, not needed here).")
	parser.add_argument("--stat-test", type=str, default=None,
	                     help="Week 4: path to a repeats x configurations AUC matrix CSV. "
	                          "Runs a Friedman test, then (if significant) Conover-Iman "
	                          "post-hoc, and prints a ++/+/- significance table. Save the "
	                          "table with --output-csv. Skips data loading entirely.")
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

	if args.hpo:
		if not args.hpo_model:
			raise ValueError("--hpo requires --hpo-model (choices: "
			                  f"{list(HPO_PARAM_SPACES)})")
		run_hpo(X_train, train_labels, args)
		return

	if args.heterogeneous_ensemble:
		heterogeneous_ensemble_comparison(X_train, train_labels, args)
		return

	if args.feature_bagging:
		feature_bagging_comparison(X_train, train_labels, args)
		return

	if args.compare_methods:
		compare_methods(X_train, train_labels, args)
		return

	if args.reduce_sweep:
		compare_reductions(X_train, train_labels, args)
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