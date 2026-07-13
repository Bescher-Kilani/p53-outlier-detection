"""Run Deep SVDD (OCC, PCA 147 + scale) and merge into Week 4 AUC matrices + stat tests."""
import argparse
from pathlib import Path

import pandas as pd

from p53_od_pipeline import (
	feature_bagging_repeated_metrics,
	friedman_conover_test,
	load_data,
	repeated_random_subsampling_metrics,
)


def merge_compare(existing_path, new_col_name, new_values, out_path):
	df = pd.read_csv(existing_path, index_col=0)
	if len(new_values) != len(df):
		raise ValueError(f"repeat count mismatch: existing {len(df)} vs new {len(new_values)}")
	df[new_col_name] = new_values
	df.to_csv(out_path)
	print(f"Wrote {out_path} ({df.shape[0]} x {df.shape[1]})")
	return df


def merge_bagging(existing_path, prefix, summary, out_path):
	df = pd.read_csv(existing_path, index_col=0)
	for key, label in [("no_bagging", "no"), ("bagging_sum", "sum"),
	                   ("bagging_max", "max"), ("bagging_aom", "aom")]:
		col = f"{prefix}_{label}"
		vals = summary[key]["values"]
		if len(vals) != len(df):
			raise ValueError(f"{col}: repeat count mismatch")
		df[col] = vals
	df.to_csv(out_path)
	print(f"Wrote {out_path} ({df.shape[0]} x {df.shape[1]})")
	return df


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n-repeats", type=int, default=25)
	parser.add_argument("--n-subsets", type=int, default=10)
	parser.add_argument("--skip-compare", action="store_true")
	parser.add_argument("--skip-bagging", action="store_true")
	args = parser.parse_args()

	ns = argparse.Namespace(
		method="deep_svdd",
		reduce="pca",
		n_components=147,
		n_estimators=200,
		random_state=42,
		max_subspaces=None,
		scale=True,
		feature_select=None,
		smote=False,
		n_repeats=args.n_repeats,
		n_subsets=args.n_subsets,
	)

	out_dir = Path("outputs")
	compare_in = out_dir / "auc_matrix_compare_methods.csv"
	compare_out = out_dir / "auc_matrix_compare_methods.csv"
	bagging_in = out_dir / "auc_feature_bagging_matrix.csv"
	bagging_out = out_dir / "auc_feature_bagging_matrix.csv"

	if not args.skip_compare:
		X_train, _, train_labels = load_data()
		print("=" * 60)
		print("STEP A: Deep SVDD repeated subsampling (inlier-only fit)")
		print("=" * 60)
		metrics = repeated_random_subsampling_metrics(
			X_train, train_labels, ns, n_repeats=args.n_repeats,
		)
		print(f"Deep SVDD AUC: {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}")
		merge_compare(compare_in, "deep_svdd", metrics["repeat_aucs"], compare_out)

	if not args.skip_bagging:
		if args.skip_compare:
			X_train, _, train_labels = load_data()
		print("\n" + "=" * 60)
		print("STEP B: Deep SVDD feature bagging (repeated subsampling)")
		print("=" * 60)
		summary = feature_bagging_repeated_metrics(
			X_train, train_labels, ns, method="deep_svdd",
			n_repeats=args.n_repeats, n_subsets=args.n_subsets,
		)
		for k in ["no_bagging", "bagging_sum", "bagging_max", "bagging_aom"]:
			s = summary[k]
			print(f"  {k}: {s['mean']:.4f} +/- {s['std']:.4f}")
		merge_bagging(bagging_in, "deep_svdd", summary, bagging_out)

	print("\n" + "=" * 60)
	print("STAT TEST: Step A (6 methods)")
	print("=" * 60)
	matrix_a = pd.read_csv(compare_out, index_col=0)
	print(matrix_a.mean().round(4).to_string())
	_, p_a, _, sig_a = friedman_conover_test(matrix_a)
	sig_a.to_csv(out_dir / "significance_table_compare_methods.csv")

	print("\n" + "=" * 60)
	print("STAT TEST: Step B (16 configs)")
	print("=" * 60)
	matrix_b = pd.read_csv(bagging_out, index_col=0)
	print(matrix_b.mean().round(4).to_string())
	_, p_b, _, sig_b = friedman_conover_test(matrix_b)
	sig_b.to_csv(out_dir / "significance_table_feature_bagging.csv")

	print("\nSUMMARY")
	print(f"  Step A Friedman p = {p_a:.6f}")
	print(f"  Step B Friedman p = {p_b:.6f}")
	print(f"  Deep SVDD mean AUC (Step A) = {matrix_a['deep_svdd'].mean():.4f}")


if __name__ == "__main__":
	main()
