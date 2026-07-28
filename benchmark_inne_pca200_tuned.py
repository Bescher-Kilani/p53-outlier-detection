"""Tuned INNE (README Step 1) with PCA-200 instead of PCA-147."""
import argparse

from p53_od_pipeline import (
	TUNED_PARAMS,
	build_hpo_detector,
	load_data,
	repeated_random_subsampling_metrics,
)


def main():
	X_train, _, train_labels = load_data()
	args = argparse.Namespace(
		method="inne",
		reduce="pca",
		n_components=200,
		n_estimators=200,
		random_state=42,
		max_subspaces=None,
		scale=True,
		feature_select=None,
		smote=False,
	)
	params = TUNED_PARAMS["inne"]
	print("INNE tuned params:", params)
	print("Setting: scale + PCA(200), 25-repeat OCC, inlier-only fit")
	factory = lambda: build_hpo_detector("inne", params, 42)
	metrics = repeated_random_subsampling_metrics(
		X_train, train_labels, args, n_repeats=25, verbose=True,
		detector_factory=factory,
	)
	print("=" * 60)
	print(f"INNE + PCA(200) + tuned: AUC={metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}")
	print("(README PCA-147 tuned was 0.7289 +/- 0.0214)")
	print("=" * 60)


if __name__ == "__main__":
	main()
