"""
Coarse n_components sweep for Isolation Forest + PCA + StandardScaler
(Week 6 IF settings: default IF params, 25 repeated OCC subsamples).

Grid:
  coarse = [10, 20, 30, 50, 80, 110, 140, 147, 170, 200, 250, 300]
  around 147: 147 +/- {10,20,30,40,50}  -> 10 offset values

Reports mean +/- std AUC and delta vs n_components=147.
"""
from __future__ import annotations

import argparse
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
	_repeated_split_indices,
	fit_scale_reduce,
	load_data,
	transform_scale_reduce,
)


COARSE = [10, 20, 30, 50, 80, 110, 140, 147, 170, 200, 250, 300]
OFFSETS_AROUND_147 = [10, 20, 30, 40, 50]  # "10 values +/- 147"


def build_grid():
	vals = set(COARSE)
	for d in OFFSETS_AROUND_147:
		vals.add(147 - d)
		vals.add(147 + d)
	# drop non-positive
	return sorted(v for v in vals if v >= 1)


def eval_n_components(X_train, train_labels, n_components, n_repeats, random_state, n_estimators):
	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx = np.where(y == 0)[0]
	outlier_idx = np.where(y == 1)[0]
	aucs = []
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

		# Cap PCA dims by train samples / features
		n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1])
		X_tr_r, scaler, reducer = fit_scale_reduce(
			X_tr, "pca", n_comp, random_state, True, return_fitted=True,
		)
		X_val_r = transform_scale_reduce(X_val, scaler, reducer)

		det = IsolationForest(
			n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
		)
		det.fit(X_tr_r)  # inlier-only
		scores = -det.score_samples(X_val_r)
		aucs.append(roc_auc_score(y_val, scores))
		print(f"    n={n_components:>4}  repeat {i+1}/{n_repeats}: AUC={aucs[-1]:.4f}", flush=True)

	aucs = np.asarray(aucs, dtype=float)
	return {
		"n_components": n_components,
		"auc_mean": float(aucs.mean()),
		"auc_std": float(aucs.std(ddof=1)),
		"n_repeats": int(len(aucs)),
		"wall_sec": time.time() - t0,
		"repeat_aucs": aucs,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n-repeats", type=int, default=25)
	parser.add_argument("--n-estimators", type=int, default=200)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--output-csv", type=str, default="outputs/if_pca_ncomponents_sweep.csv")
	args = parser.parse_args()

	grid = build_grid()
	print("=" * 60)
	print("IF + PCA + StandardScaler  n_components sweep")
	print(f"grid ({len(grid)}): {grid}")
	print(f"n_repeats={args.n_repeats}, n_estimators={args.n_estimators} (default IF)")
	print("=" * 60)

	X_train, _, train_labels = load_data()
	rows = []
	auc_matrix = {}

	for n in grid:
		print(f"\n--- n_components={n} ---", flush=True)
		res = eval_n_components(
			X_train, train_labels, n, args.n_repeats, args.random_state, args.n_estimators,
		)
		rows.append({k: v for k, v in res.items() if k != "repeat_aucs"})
		auc_matrix[f"pca_{n}"] = res["repeat_aucs"]
		print(
			f"  => {res['auc_mean']:.4f} +/- {res['auc_std']:.4f}  "
			f"({res['wall_sec']:.0f}s)",
			flush=True,
		)

	df = pd.DataFrame(rows)
	base = df.loc[df["n_components"] == 147, "auc_mean"].iloc[0]
	df["delta_vs_147"] = df["auc_mean"] - base
	df["delta_pp"] = df["delta_vs_147"] * 100
	df = df.sort_values("n_components").reset_index(drop=True)

	out = Path(args.output_csv)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out, index=False)
	matrix = pd.DataFrame(auc_matrix)
	matrix.index = [f"repeat_{i+1}" for i in range(len(matrix))]
	matrix.to_csv(out.with_name(out.stem + "_auc_matrix.csv"))

	print("\n" + "=" * 70)
	print("RESULTS (IF + PCA(n) + StandardScaler, 25 repeats, OCC)")
	print("Compared to n_components=147")
	print("=" * 70)
	print(f"{'n':>6}  {'AUC':>18}  {'vs 147':>10}  {'note'}")
	print("-" * 70)
	for _, r in df.iterrows():
		note = ""
		if int(r["n_components"]) == 147:
			note = "<- baseline"
		elif int(r["n_components"]) in COARSE:
			note = "coarse"
		else:
			note = "147 +/- offset"
		sign = "+" if r["delta_vs_147"] >= 0 else ""
		print(
			f"{int(r['n_components']):>6}  "
			f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}  "
			f"{sign}{r['delta_vs_147']:+.4f}  {note}"
		)
	print("-" * 70)
	best = df.loc[df["auc_mean"].idxmax()]
	print(
		f"Best: n={int(best['n_components'])}  "
		f"AUC={best['auc_mean']:.4f} +/- {best['auc_std']:.4f}  "
		f"(delta vs 147: {best['delta_vs_147']:+.4f})"
	)
	print(f"Saved {out}")


if __name__ == "__main__":
	main()
