"""
INNE + PCA(200) + Step 1 tuned params, feature bagging (25 repeats, 10 subsets).

Runs aggregations one at a time (sum → max → aom → pruned → damped).
Subset fits are shared: the first aggregation does the 25×10 detector fits
and caches per-repeat subset score matrices; later aggregations only
re-aggregate from the cache and update README immediately when done.

No-bagging is NOT re-run (uses Step 2 value 0.7364 ± 0.0207 in README).
"""
from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root, so p53_od_pipeline can be imported
from p53_od_pipeline import (
	TUNED_PARAMS,
	_repeated_split_indices,
	build_hpo_detector,
	fit_scale_reduce,
	load_data,
	transform_scale_reduce,
)

README = Path("README.md")
CACHE = Path("week6-tables/inne_pca200_bagging_subset_cache.pkl")
OUT_CSV = Path("week6-tables/week6_feature_bagging_inne_pca200_tuned.csv")

N_REPEATS = 25
N_SUBSETS = 10
SUBSET_MIN_FRAC = 0.5
N_COMPONENTS = 200
RANDOM_STATE = 42
PARAMS = TUNED_PARAMS["inne"]

# (readme_label, result_key)
AGGS = [
	("Sum", "bagging_sum"),
	("Max", "bagging_max"),
	("AOM", "bagging_aom"),
	("Pruned", "bagging_pruned"),
	("Damped", "bagging_damped"),
]


def _aggregate(subset_matrix: np.ndarray) -> dict[str, np.ndarray]:
	agg_sum = subset_matrix.sum(axis=0)
	agg_max = subset_matrix.max(axis=0)
	n_groups = min(5, subset_matrix.shape[0])
	groups = np.array_split(subset_matrix, n_groups, axis=0)
	agg_aom = np.mean([g.max(axis=0) for g in groups], axis=0)

	consensus = subset_matrix.mean(axis=0)
	correlations = np.array([
		np.corrcoef(subset_matrix[s], consensus)[0, 1] for s in range(subset_matrix.shape[0])
	])
	keep = correlations >= np.median(correlations)
	agg_pruned = subset_matrix[keep].mean(axis=0)
	agg_damped = np.clip(subset_matrix, -3, 3).mean(axis=0)
	return {
		"bagging_sum": agg_sum,
		"bagging_max": agg_max,
		"bagging_aom": agg_aom,
		"bagging_pruned": agg_pruned,
		"bagging_damped": agg_damped,
	}


def build_cache(X_train, train_labels):
	"""25 repeats × 10 INNE subset fits; cache z-normalized subset score matrices."""
	y = np.asarray(train_labels == "active", dtype=int)
	inlier_idx = np.where(y == 0)[0]
	outlier_idx = np.where(y == 1)[0]
	cache = []

	print("=" * 60)
	print(f"Building subset-score cache (INNE tuned, PCA-{N_COMPONENTS}, "
	      f"{N_REPEATS} repeats x {N_SUBSETS} subsets)")
	print(f"params={PARAMS}")
	print("=" * 60)

	t0 = time.time()
	for i in range(N_REPEATS):
		train_idx, val_idx = _repeated_split_indices(
			y, inlier_idx, outlier_idx, RANDOM_STATE + i, 0.8, 0.8,
		)
		X_tr = X_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		y_val = y[val_idx]
		if y_val.sum() == 0 or y_val.sum() == len(y_val):
			print(f"  repeat {i+1}/{N_REPEATS}: skipped")
			continue

		X_tr_red, scaler, reducer = fit_scale_reduce(
			X_tr, "pca", N_COMPONENTS, RANDOM_STATE, True, return_fitted=True,
		)
		X_val_red = transform_scale_reduce(X_val, scaler, reducer)
		d = X_tr_red.shape[1]

		rng = np.random.default_rng(RANDOM_STATE)
		subset_scores = []
		for j in range(N_SUBSETS):
			k = int(rng.integers(int(SUBSET_MIN_FRAC * d), d + 1))
			cols = rng.choice(d, size=k, replace=False)
			det = build_hpo_detector("inne", PARAMS, RANDOM_STATE + j + 1)
			det.fit(X_tr_red[:, cols])
			raw = -det.score_samples(X_val_red[:, cols])
			z = (raw - raw.mean()) / (raw.std() + 1e-12)
			subset_scores.append(z)

		mat = np.vstack(subset_scores)
		cache.append({"y_val": y_val, "subset_matrix": mat})
		print(f"  repeat {i+1}/{N_REPEATS}: cached subsets  ({time.time()-t0:.0f}s)", flush=True)

	CACHE.parent.mkdir(parents=True, exist_ok=True)
	with CACHE.open("wb") as f:
		pickle.dump(cache, f)
	print(f"Saved cache ({len(cache)} repeats) to {CACHE}  ({time.time()-t0:.0f}s total)\n")
	return cache


def load_or_build_cache(X_train, train_labels):
	if CACHE.exists():
		print(f"Loading cache from {CACHE}")
		with CACHE.open("rb") as f:
			return pickle.load(f)
	return build_cache(X_train, train_labels)


def eval_aggregation(cache, key: str) -> tuple[float, float, list[float]]:
	aucs = []
	for item in cache:
		scores = _aggregate(item["subset_matrix"])[key]
		aucs.append(roc_auc_score(item["y_val"], scores))
	aucs = np.asarray(aucs, dtype=float)
	return float(aucs.mean()), float(aucs.std(ddof=1)), aucs.tolist()


def update_readme_agg(label: str, mean: float, std: float):
	"""Update only the INNE + PCA(200) bagging table row (not other Week tables)."""
	text = README.read_text(encoding="utf-8")
	marker = "**INNE + PCA(200) + Step 1 tuned**"
	idx = text.find(marker)
	if idx < 0:
		raise RuntimeError(f"INNE bagging section marker not found: {marker}")
	# Limit edits to this section (until Step 4 or next ###)
	end = text.find("\n### Step 4", idx)
	if end < 0:
		end = idx + 1200
	head, block, tail = text[:idx], text[idx:end], text[end:]

	value = f"{mean:.4f} ± {std:.4f}"
	pattern = rf"(\| {re.escape(label)} \|)[^\n]*"
	new_block, n = re.subn(pattern, rf"\g<1> {value} |", block, count=1)
	if n != 1:
		raise RuntimeError(f"Could not update INNE bagging row for {label!r} (matches={n})")
	README.write_text(head + new_block + tail, encoding="utf-8")
	print(f"README updated: {label} = {mean:.4f} ± {std:.4f}")


def bold_best_inne_bagging_rows():
	text = README.read_text(encoding="utf-8")
	marker = "**INNE + PCA(200) + Step 1 tuned**"
	idx = text.find(marker)
	if idx < 0:
		return
	end = text.find("\n### Step 4", idx)
	if end < 0:
		end = idx + 1200
	head, block, tail = text[:idx], text[idx:end], text[end:]
	row_re = re.compile(
		r"\| (Sum|Max|AOM|Pruned|Damped) \| ?(?:\*\*)?([0-9.]+) ± ([0-9.]+)(?:\*\*)? \|"
	)
	rows = list(row_re.finditer(block))
	if not rows:
		return
	best_mean = max(float(m.group(2)) for m in rows)
	best_labels = {m.group(1) for m in rows if float(m.group(2)) == best_mean}

	def sub(m):
		label, mean, std = m.group(1), m.group(2), m.group(3)
		val = f"{mean} ± {std}"
		if label in best_labels:
			return f"| {label} | **{val}** |"
		return f"| {label} | {val} |"

	README.write_text(head + row_re.sub(sub, block) + tail, encoding="utf-8")
	print(f"README: bolded best INNE bagging row(s) = {sorted(best_labels)}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--rebuild-cache", action="store_true")
	args = parser.parse_args()

	X_train, _, train_labels = load_data()
	if args.rebuild_cache and CACHE.exists():
		CACHE.unlink()
	cache = load_or_build_cache(X_train, train_labels)

	matrix = {"repeat": list(range(len(cache)))}
	for label, key in AGGS:
		print("=" * 60)
		print(f"Aggregation: {label}")
		print("=" * 60)
		mean, std, aucs = eval_aggregation(cache, key)
		matrix[key] = aucs
		print(f"  AUC={mean:.4f} +/- {std:.4f}  (n={len(aucs)})")
		update_readme_agg(label, mean, std)
		print()

	bold_best_inne_bagging_rows()
	OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(matrix).to_csv(OUT_CSV, index=False)
	print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
	main()
