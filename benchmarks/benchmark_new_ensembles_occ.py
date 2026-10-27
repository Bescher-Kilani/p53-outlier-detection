"""
Benchmark untried OCC outlier-ensemble methods from the W5/W6 literature list:
  INNE, LODA  (+ Isolation Forest baseline for the usual table)

Skipped from the list (with reason):
  DIF          already tried (Week 6)
  Feature Bagging  already tried (Weeks 3-6)
  XGBOD        supervised (needs labels in fit) — not OCC
  SUOD         acceleration framework, not a detector
  LSCP         OCC-compatible but does not scale to ~20k rows
               (one fit on PCA-147 did not finish in >5 min)

Protocol matches Step A / Week 6 final compare:
  StandardScaler -> PCA(147) -> detector, inlier-only fit,
  25 repeated 80/20 subsamples, report mean +/- std AUC + timing.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pyod.models.inne import INNE
from pyod.models.loda import LODA


OUT_DIR = Path("outputs")
N_REPEATS = 25
N_COMPONENTS = 147
RANDOM_STATE = 42
INLIER_FRAC = 0.8
OUTLIER_FRAC = 0.8


def load_train():
	X = pd.read_parquet("train_set_p53mutant.parquet")
	labels = pd.read_csv("train_labels_p53mutant.csv", index_col=0)["label"].reindex(X.index)
	if labels.isna().any():
		raise ValueError("labels do not align with train_set index")
	y = (labels == "active").astype(int).to_numpy()
	print(f"Train: {X.shape}  active={int(y.sum())}  inactive={int((y == 0).sum())}")
	return X.to_numpy(dtype=np.float64), y


def split_indices(y, seed):
	rng = np.random.default_rng(seed)
	inlier_idx = np.where(y == 0)[0]
	outlier_idx = np.where(y == 1)[0]
	perm_in = rng.permutation(inlier_idx)
	n_train_in = int(round(INLIER_FRAC * len(inlier_idx)))
	train_idx = perm_in[:n_train_in]
	remaining_in = perm_in[n_train_in:]
	perm_out = rng.permutation(outlier_idx)
	n_val_out = int(round(OUTLIER_FRAC * len(outlier_idx)))
	val_idx = np.concatenate([remaining_in, perm_out[:n_val_out]])
	return train_idx, val_idx


def make_model(name, seed):
	if name == "isolation_forest":
		return IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1), "sklearn"
	if name == "inne":
		return INNE(n_estimators=200, random_state=seed, contamination=0.05), "pyod"
	if name == "loda":
		return LODA(n_bins=10, n_random_cuts=100, contamination=0.05), "pyod"
	raise ValueError(name)


def score_anomaly(model, kind, X):
	# higher = more anomalous (matches pipeline convention for AUC)
	if kind == "sklearn":
		return -model.score_samples(X)
	return model.decision_function(X)


def run_method(name, X, y, n_repeats=N_REPEATS):
	print("\n" + "=" * 60)
	print(f"METHOD: {name}  (n_repeats={n_repeats}, PCA={N_COMPONENTS})")
	print("=" * 60)

	aucs = []
	fit_times = []
	prep_times = []
	score_times = []

	for i in range(n_repeats):
		train_idx, val_idx = split_indices(y, RANDOM_STATE + i)
		X_tr, X_val = X[train_idx], X[val_idx]
		y_val = y[val_idx]
		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			print(f"  repeat {i+1}/{n_repeats}: skipped (single-class val)")
			continue

		t_prep = time.perf_counter()
		scaler = StandardScaler().fit(X_tr)
		X_tr_s = scaler.transform(X_tr)
		X_val_s = scaler.transform(X_val)
		pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
		X_tr_r = pca.fit_transform(X_tr_s)
		X_val_r = pca.transform(X_val_s)
		prep_times.append(time.perf_counter() - t_prep)

		model, kind = make_model(name, RANDOM_STATE + i)
		t_fit = time.perf_counter()
		model.fit(X_tr_r)  # inlier-only, no labels
		fit_dt = time.perf_counter() - t_fit
		fit_times.append(fit_dt)

		t_score = time.perf_counter()
		scores = score_anomaly(model, kind, X_val_r)
		score_times.append(time.perf_counter() - t_score)

		auc = roc_auc_score(y_val, scores)
		aucs.append(auc)
		print(
			f"  repeat {i+1}/{n_repeats}: AUC={auc:.4f}  "
			f"fit={fit_dt:.2f}s  prep={prep_times[-1]:.2f}s  "
			f"score={score_times[-1]:.2f}s",
			flush=True,
		)

	aucs = np.asarray(aucs, dtype=float)
	fit_times = np.asarray(fit_times, dtype=float)
	return {
		"method": name,
		"auc_mean": float(aucs.mean()),
		"auc_std": float(aucs.std(ddof=1)),
		"n_repeats": int(len(aucs)),
		"fit_sec_mean": float(fit_times.mean()),
		"fit_sec_std": float(fit_times.std(ddof=1)),
		"fit_sec_total": float(fit_times.sum()),
		"prep_sec_mean": float(np.mean(prep_times)),
		"score_sec_mean": float(np.mean(score_times)),
		"wall_sec_total": float(np.sum(prep_times) + fit_times.sum() + np.sum(score_times)),
		"repeat_aucs": aucs,
	}


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	X, y = load_train()

	# order: baseline first, then untried ensembles
	methods = ["isolation_forest", "inne", "loda"]
	rows = []
	auc_matrix = {}

	for name in methods:
		res = run_method(name, X, y)
		rows.append({k: v for k, v in res.items() if k != "repeat_aucs"})
		auc_matrix[name] = res["repeat_aucs"]

	summary = pd.DataFrame(rows)
	summary_path = OUT_DIR / "week6_new_ensembles_summary.csv"
	summary.to_csv(summary_path, index=False)

	matrix = pd.DataFrame(auc_matrix)
	matrix.index = [f"repeat_{i+1}" for i in range(len(matrix))]
	matrix_path = OUT_DIR / "week6_new_ensembles_auc_matrix.csv"
	matrix.to_csv(matrix_path)

	print("\n" + "=" * 60)
	print("RESULTS TABLE (PCA-147 + StandardScaler, 25 repeats, OCC)")
	print("=" * 60)
	print(f"{'Method':<20} {'AUC':>18} {'Fit time / repeat':>22} {'Total fit':>12}")
	print("-" * 76)
	for _, r in summary.iterrows():
		auc = f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}"
		fit = f"{r['fit_sec_mean']:.2f} +/- {r['fit_sec_std']:.2f}s"
		print(f"{r['method']:<20} {auc:>18} {fit:>22} {r['fit_sec_total']:>10.1f}s")
	print("-" * 76)
	print(f"Saved {summary_path}")
	print(f"Saved {matrix_path}")


if __name__ == "__main__":
	main()
