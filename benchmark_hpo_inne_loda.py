"""
HPO for INNE and LODA (Week 6 Step 1 add-on).

Same protocol as pipeline --hpo (Optuna GP, 150 trials x 15 repeats/trial,
25-repeat final check, PCA-147 + scale, inlier-only OCC), but reuses the
precomputed scale+PCA splits across trials so search time is dominated by
detector fit rather than re-fitting PCA every trial.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from p53_od_pipeline import (
	HPO_PARAM_SPACES,
	_repeated_split_indices,
	build_hpo_detector,
	fit_scale_reduce,
	load_data,
	transform_scale_reduce,
)


def precompute_splits(X_train, train_labels, n_repeats, n_components, random_state, scale=True):
	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx = np.where(y == 0)[0]
	outlier_idx = np.where(y == 1)[0]
	splits = []
	t0 = time.time()
	for i in range(n_repeats):
		train_idx, val_idx = _repeated_split_indices(
			y, inlier_idx, outlier_idx, random_state + i, 0.8, 0.8,
		)
		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		y_val = y[val_idx]
		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			continue
		X_tr_r, scaler, reducer = fit_scale_reduce(
			X_tr, "pca", n_components, random_state, scale, return_fitted=True,
		)
		X_val_r = transform_scale_reduce(X_val, scaler, reducer)
		splits.append((X_tr_r, X_val_r, y_val))
		print(f"  split {i+1}/{n_repeats}: train={X_tr_r.shape} val={X_val_r.shape}", flush=True)
	print(f"Precomputed {len(splits)} splits in {time.time() - t0:.1f}s\n", flush=True)
	return splits


def eval_on_splits(model_name, params, splits, random_state):
	aucs = []
	for X_tr, X_val, y_val in splits:
		det = build_hpo_detector(model_name, params, random_state)
		det.fit(X_tr)
		# score_samples: higher = more normal; flip for AUC (higher = more anomalous)
		scores = -det.score_samples(X_val)
		aucs.append(roc_auc_score(y_val, scores))
	aucs = np.asarray(aucs, dtype=float)
	return float(aucs.mean()), float(aucs.std(ddof=1)), aucs


def run_hpo_model(model, X_train, train_labels, args):
	suggest_fn, resolve_fn = HPO_PARAM_SPACES[model]
	print("=" * 60)
	sampler_name = getattr(args, "sampler", "tpe")
	print(f"HPO: {model}  ({sampler_name}, trials={args.hpo_trials}, repeats/trial={args.hpo_repeats})")
	print(f"(reduce=pca, n_components={args.n_components}, scale=True, cached splits)")
	print("=" * 60 + "\n")

	print("Precomputing search splits...")
	search_splits = precompute_splits(
		X_train, train_labels, args.hpo_repeats, args.n_components, args.random_state,
	)

	if sampler_name == "gp":
		sampler = optuna.samplers.GPSampler(seed=args.random_state)
	else:
		sampler = optuna.samplers.TPESampler(seed=args.random_state)
	study = optuna.create_study(direction="maximize", sampler=sampler)

	def objective(trial):
		t0 = time.time()
		params = suggest_fn(trial)
		auc_mean, auc_std, _ = eval_on_splits(model, params, search_splits, args.random_state)
		print(
			f"  trial {trial.number + 1}/{args.hpo_trials}: AUC={auc_mean:.4f}+/-{auc_std:.4f}  "
			f"params={trial.params}  ({time.time() - t0:.1f}s)",
			flush=True,
		)
		return auc_mean

	t0 = time.time()
	study.optimize(objective, n_trials=args.hpo_trials)
	print(f"\nBest search AUC: {study.best_value:.4f}")
	print(f"Best params: {study.best_params}")
	print(f"Search time: {(time.time() - t0) / 60:.1f} min\n")

	best_params = resolve_fn(study.best_params)
	print(f"Precomputing final {args.hpo_final_repeats} splits...")
	final_splits = precompute_splits(
		X_train, train_labels, args.hpo_final_repeats, args.n_components, args.random_state,
	)
	auc_mean, auc_std, aucs = eval_on_splits(model, best_params, final_splits, args.random_state)
	print("=" * 60)
	print(f"FINAL CHECK ({args.hpo_final_repeats} repeats)")
	print("=" * 60)
	print(f"AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
	print(f"Best params: {best_params}")
	print("=" * 60 + "\n")

	out_path = Path(args.output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	study.trials_dataframe().to_csv(out_path, index=False)
	print(f"Saved trials to {out_path}")

	summary_path = out_path.with_name(out_path.stem + "_final_summary.csv")
	pd.DataFrame([{
		"method": model,
		"auc_mean": auc_mean,
		"auc_std": auc_std,
		"n_repeats": len(aucs),
		**{f"param_{k}": v for k, v in best_params.items()},
	}]).to_csv(summary_path, index=False)
	print(f"Saved final summary to {summary_path}\n")
	return best_params, auc_mean, auc_std


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hpo-model", choices=["inne", "loda", "isolation_forest", "all"], default="all")
	parser.add_argument("--hpo-trials", type=int, default=150)
	parser.add_argument("--hpo-repeats", type=int, default=15)
	parser.add_argument("--hpo-final-repeats", type=int, default=25)
	parser.add_argument("--n-components", type=int, default=147)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--sampler", choices=["tpe", "gp"], default="tpe")
	parser.add_argument("--output-csv", type=str, default="week6-tables/hpo_inne.csv")
	args = parser.parse_args()

	X_train, _, train_labels = load_data()
	models = ["inne", "loda"] if args.hpo_model == "all" else [args.hpo_model]
	results = {}
	for model in models:
		out = args.output_csv
		if args.hpo_model == "all":
			out = f"week6-tables/hpo_{model}.csv"
		ns = argparse.Namespace(**{**vars(args), "output_csv": out})
		results[model] = run_hpo_model(model, X_train, train_labels, ns)

	print("=" * 60)
	print("HPO SUMMARY")
	print("=" * 60)
	for model, (params, auc, std) in results.items():
		print(f"  {model:<8} {auc:.4f} +/- {std:.4f}  params={params}")
	print("=" * 60)


if __name__ == "__main__":
	main()
