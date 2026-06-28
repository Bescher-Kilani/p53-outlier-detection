"""
p53 Outlier Detection Pipeline
Team 2 - DS Lab 2026

Pipeline Steps:
1. Load train_set/test_set/train_labels (produced by split_data.py)
2. Fit an unsupervised outlier detector on train_set features only (no labels)
   - isolation_forest: optionally preceded by PCA or Random Projection
   - gmd_lof: GMD subspace search + LOF, scores summed across subspaces
3. Score train_set and test_set
4. Internal AUC check: compare train_set scores against train_labels
5. Save test_set scores as team2_prediction.csv

Requires for --method gmd_lof: pip install gmd
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import roc_auc_score


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


def reduce_dimensions(X_train, X_test, method="none", n_components=147, random_state=42):
	# Fit only on X_train, then transform both - same no-leakage rule as the split.
	if method == "none":
		return X_train, X_test

	print("="*60)
	print(f"DIMENSIONALITY REDUCTION ({method}, n_components={n_components})")
	print("="*60)

	t0 = time.time()
	if method == "pca":
		reducer = PCA(n_components=n_components, random_state=random_state)
	elif method == "random_projection":
		reducer = GaussianRandomProjection(n_components=n_components, random_state=random_state)
	elif method == "truncated_svd":
		from sklearn.decomposition import TruncatedSVD
		reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
	elif method == "isomap":
		from sklearn.manifold import Isomap
		reducer = Isomap(n_components=n_components)
	elif method == "umap":
		import umap  # pip install umap-learn
		reducer = umap.UMAP(n_components=n_components, random_state=random_state)
	else:
		raise ValueError(f"Unknown method: {method}")

	X_train_red = reducer.fit_transform(X_train)
	X_test_red = reducer.transform(X_test)

	if method == "pca":
		print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

	print(f"Shape: {X_train.shape} -> {X_train_red.shape}  ({time.time()-t0:.1f}s)")
	print("="*60 + "\n")

	return X_train_red, X_test_red


def fit_isolation_forest(X_train, n_estimators=200, random_state=42):
	print("="*60)
	print("FITTING ISOLATION FOREST")
	print("="*60)

	t0 = time.time()
	model = IsolationForest(
		n_estimators=n_estimators,
		random_state=random_state,
		n_jobs=-1,
	)
	model.fit(X_train)
	print(f"Fitted on {len(X_train):,} samples in {time.time()-t0:.1f}s")
	print("="*60 + "\n")

	return model


def anomaly_scores(model, X):
	# sklearn's score_samples: higher = more normal, so we flip the sign
	# to get higher = more anomalous (matches our AUC convention).
	return -model.score_samples(X)


def fit_score_lof(X_train, X_test, random_state=42):
	from sklearn.neighbors import LocalOutlierFactor

	print("="*60)
	print("FITTING LOF (single model, no subspace ensemble)")
	print("="*60)

	t0 = time.time()
	lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
	lof.fit(X_train)
	print(f"Fitted on {len(X_train):,} samples in {time.time()-t0:.1f}s")
	print("="*60 + "\n")

	train_scores = -lof.score_samples(X_train)
	test_scores = -lof.score_samples(X_test)
	return train_scores, test_scores


def fit_score_gmd_lof(X_train, X_test, random_state=42, max_subspaces=None):
	from gmd import GMD
	from sklearn.neighbors import LocalOutlierFactor

	print("="*60)
	print("FITTING GMD + LOF (subspace ensemble)")
	print("="*60)

	# Works on either a DataFrame (raw) or ndarray (e.g. after PCA) input.
	X_train_arr = np.asarray(X_train)
	X_test_arr = np.asarray(X_test)

	t0 = time.time()
	gmd = GMD()
	gmd.fit(X_train_arr)
	subspace_items = list(gmd.subspaces_.items())
	n_found = len(subspace_items)
	if max_subspaces is not None:
		subspace_items = subspace_items[:max_subspaces]
	print(f"GMD found {n_found} subspaces in {time.time()-t0:.1f}s "
	      f"(using {len(subspace_items)})")

	train_scores = np.zeros(len(X_train_arr))
	test_scores = np.zeros(len(X_test_arr))

	for i, (key, cols) in enumerate(subspace_items):
		# novelty=True lets us score X_test from a model fit only on X_train.
		# Using score_samples on X_train itself (for the internal AUC check
		# only) is a known minor inconsistency in novelty mode - acceptable
		# here since it never touches the official test_set prediction.
		lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
		lof.fit(X_train_arr[:, cols])
		train_scores += -lof.score_samples(X_train_arr[:, cols])
		test_scores += -lof.score_samples(X_test_arr[:, cols])
		if (i + 1) % 50 == 0 or (i + 1) == len(subspace_items):
			print(f"  Subspace {i+1}/{len(subspace_items)}  ({time.time()-t0:.1f}s elapsed)")

	print(f"Done in {time.time()-t0:.1f}s")
	print("="*60 + "\n")

	return train_scores, test_scores


def evaluate_internal_auc(train_scores, train_labels):
	print("="*60)
	print("INTERNAL AUC CHECK (train_set only, never used for fitting)")
	print("="*60)

	y_true = (train_labels == "active").astype(int)
	auc = roc_auc_score(y_true, train_scores)
	print(f"AUC: {auc:.4f}")
	print("="*60 + "\n")

	return auc


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


def main():
	parser = argparse.ArgumentParser(description="p53 Outlier Detection Pipeline")
	parser.add_argument("--method", type=str, default="isolation_forest",
	                     choices=["isolation_forest", "lof", "gmd_lof"],
	                     help="Outlier detection method")
	parser.add_argument("--n-estimators", type=int, default=200,
	                     help="Number of trees in the Isolation Forest")
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--reduce", type=str, default="none",
	                     choices=["none", "pca", "random_projection", "truncated_svd", "isomap", "umap"],
	                     help="Dimensionality reduction before fitting")
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
		      f"(active: {n_active} - AUC below is NOT meaningful, runtime check only)\n")

	if args.sweep:
		k_values = [int(k) for k in args.sweep.split(",")]
		results = []
		for k in k_values:
			X_train_red, X_test_red = reduce_dimensions(
				X_train, X_test,
				method=args.reduce,
				n_components=k,
				random_state=args.random_state,
			)
			if args.method == "isolation_forest":
				model = fit_isolation_forest(X_train_red, n_estimators=args.n_estimators,
				                              random_state=args.random_state)
				train_scores = anomaly_scores(model, X_train_red)
			elif args.method == "lof":
				train_scores, _ = fit_score_lof(X_train_red, X_test_red,
				                                 random_state=args.random_state)
			elif args.method == "gmd_lof":
				train_scores, _ = fit_score_gmd_lof(X_train_red, X_test_red,
				                                     random_state=args.random_state,
				                                     max_subspaces=args.max_subspaces)
			auc = evaluate_internal_auc(train_scores, train_labels)
			results.append((k, auc))

		print("="*60)
		print(f"SWEEP SUMMARY ({args.method} + {args.reduce})")
		print("="*60)
		for k, auc in results:
			print(f"  n_components={k:<5} AUC={auc:.4f}")
		best_k, best_auc = max(results, key=lambda r: r[1])
		print(f"Best: n_components={best_k} (AUC={best_auc:.4f})")
		print("="*60 + "\n")
		return

	X_train_red, X_test_red = reduce_dimensions(
		X_train, X_test,
		method=args.reduce,
		n_components=args.n_components,
		random_state=args.random_state,
	)

	if args.method == "isolation_forest":
		model = fit_isolation_forest(X_train_red, n_estimators=args.n_estimators,
		                              random_state=args.random_state)
		train_scores = anomaly_scores(model, X_train_red)
		test_scores = anomaly_scores(model, X_test_red)

	elif args.method == "lof":
		train_scores, test_scores = fit_score_lof(
			X_train_red, X_test_red,
			random_state=args.random_state,
		)

	elif args.method == "gmd_lof":
		# Raw 5408 dims makes GMD's pairwise search intractable in practice -
		# feed it a reduced space instead, e.g. --reduce pca.
		train_scores, test_scores = fit_score_gmd_lof(
			X_train_red, X_test_red,
			random_state=args.random_state,
			max_subspaces=args.max_subspaces,
		)

	evaluate_internal_auc(train_scores, train_labels)
	save_prediction(test_scores, test_ids)


if __name__ == "__main__":
	main()