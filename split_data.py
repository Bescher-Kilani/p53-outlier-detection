# split_data.py - clean the raw data file and create a stratified train/test
# split for the p53 OD pipeline. Replicates the cleaning steps from
# notbooks/data_expolartion.ipynb (cells 1, 9, 11) so the notebook itself
# never needs to be run to produce the pipeline's input files.
#
# Input format is detected from the file extension, not hardcoded to .data -
# supports .data/.csv/.tsv/.txt (delimited text) and .parquet.
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CHUNK_SIZE = 2000  # rows per chunk, same as the notebook


def read_chunks(path, chunk_size):
	# Yields DataFrame chunks regardless of input format.
	suffix = Path(path).suffix.lower()

	if suffix in (".csv", ".data", ".txt", ".tsv"):
		sep = "\t" if suffix == ".tsv" else ","
		yield from pd.read_csv(path, sep=sep, na_values="?", chunksize=chunk_size)
	elif suffix == ".parquet":
		# Already columnar/efficient - read once, then re-chunk manually so
		# the cleaning loop below stays identical for every format.
		df_full = pd.read_parquet(path)
		for start in range(0, len(df_full), chunk_size):
			yield df_full.iloc[start:start + chunk_size]
	else:
		raise ValueError(f"Unsupported file format for chunked reading: {suffix}")


def read_table(path):
	# Full (non-chunked) read for small files like the labels - used for
	# whatever format the file happens to be in.
	suffix = Path(path).suffix.lower()

	if suffix in (".csv", ".data", ".txt", ".tsv"):
		sep = "\t" if suffix == ".tsv" else ","
		return pd.read_csv(path, sep=sep, na_values="?")
	elif suffix == ".parquet":
		return pd.read_parquet(path)
	elif suffix in (".xlsx", ".xls"):
		return pd.read_excel(path)
	elif suffix == ".json":
		return pd.read_json(path)
	else:
		raise ValueError(f"Unsupported file format: {suffix}")


def load_and_clean_data(path, chunk_size=CHUNK_SIZE):
	# Same logic as notebook cell 9: drop fully-empty rows, drop rows
	# missing the 2D feature block, drop duplicate rows, downcast to
	# float32, drop the empty trailing artifact column ("5409").
	seen_hashes = {}
	duplicate_count = 0
	cleaned_chunks = []

	stats = {
		"rows_read": 0,
		"fully_empty_dropped": 0,
		"missing_2d_dropped": 0,
		"duplicates_dropped": 0,
		"rows_remaining": 0,
	}

	for chunk in read_chunks(path, chunk_size):
		if "Unnamed: 0" in chunk.columns:
			chunk = chunk.rename(columns={"Unnamed: 0": "id"})
		if chunk.index.name != "id" and "id" in chunk.columns:
			chunk = chunk.set_index("id")
		if "5409" in chunk.columns:
			chunk = chunk.drop(columns=["5409"])

		stats["rows_read"] += len(chunk)

		n_features = chunk.shape[1]
		row_missing = chunk.isna().sum(axis=1)

		fully_empty_mask = row_missing == n_features
		stats["fully_empty_dropped"] += fully_empty_mask.sum()
		chunk = chunk[~fully_empty_mask]
		row_missing = row_missing[~fully_empty_mask]

		missing_2d_mask = row_missing == 4826
		stats["missing_2d_dropped"] += missing_2d_mask.sum()
		chunk = chunk[~missing_2d_mask]

		chunk = chunk.astype("float32")

		row_hashes = pd.util.hash_pandas_object(chunk, index=False)
		keep_mask = []
		for row_id, h in zip(chunk.index, row_hashes):
			if h in seen_hashes:
				duplicate_count += 1
				keep_mask.append(False)
			else:
				seen_hashes[h] = row_id
				keep_mask.append(True)

		chunk = chunk[keep_mask]
		stats["duplicates_dropped"] = duplicate_count

		cleaned_chunks.append(chunk)

	df_clean = pd.concat(cleaned_chunks)
	stats["rows_remaining"] = len(df_clean)

	for k, v in stats.items():
		print(f"{k}: {v}")

	return df_clean


def main():
	parser = argparse.ArgumentParser(description="Clean raw data and create a stratified split")
	parser.add_argument("--data-file", type=str, default="train_data.data",
	                     help="Raw feature file (.data/.csv/.tsv/.txt/.parquet)")
	parser.add_argument("--labels-file", type=str, default="train_gt.csv",
	                     help="Ground truth labels file (.csv/.tsv/.parquet/.xlsx/.json)")
	parser.add_argument("--test-size", type=float, default=0.20)
	parser.add_argument("--random-state", type=int, default=42)
	args = parser.parse_args()

	print("Cleaning raw data...")
	df = load_and_clean_data(args.data_file)
	print(f"Cleaned features shape: {df.shape}")

	print("Loading ground truth labels...")
	gt = read_table(args.labels_file)
	if "Unnamed: 0" in gt.columns:
		gt = gt.rename(columns={"Unnamed: 0": "id"})
	if "5408" in gt.columns:
		gt = gt.rename(columns={"5408": "label"})
	if gt.index.name != "id" and "id" in gt.columns:
		gt = gt.set_index("id")
	labels = gt.loc[df.index, "label"]
	print(f"Label distribution:\n{labels.value_counts()}")

	print(f"Stratified {1-args.test_size:.0%}/{args.test_size:.0%} split "
	      f"(random_state={args.random_state})...")
	train_idx, test_idx = train_test_split(
		df.index,
		test_size=args.test_size,
		stratify=labels,
		random_state=args.random_state,
	)

	train_set = df.loc[train_idx]
	test_set = df.loc[test_idx]
	train_labels = labels.loc[train_idx]
	test_labels = labels.loc[test_idx]

	print(f"train_set: {train_set.shape}  (active: {(train_labels == 'active').sum()})")
	print(f"test_set:  {test_set.shape}  (active: {(test_labels == 'active').sum()})")

	train_set.to_parquet("train_set_p53mutant.parquet")
	size_mb = Path("train_set_p53mutant.parquet").stat().st_size / 1e6
	print(f"Saved train_set_p53mutant.parquet  {train_set.shape}  ({size_mb:.1f} MB)")

	test_set.to_parquet("test_set_p53mutant.parquet")
	size_mb = Path("test_set_p53mutant.parquet").stat().st_size / 1e6
	print(f"Saved test_set_p53mutant.parquet   {test_set.shape}  ({size_mb:.1f} MB)")

	train_labels.to_csv("train_labels_p53mutant.csv")
	size_mb = Path("train_labels_p53mutant.csv").stat().st_size / 1e6
	print(f"Saved train_labels_p53mutant.csv   {train_labels.shape}  ({size_mb:.1f} MB)")

	test_labels.to_csv("_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv")
	size_mb = Path("_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv").stat().st_size / 1e6
	print(f"Saved _internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv  {test_labels.shape}  ({size_mb:.1f} MB)")

	print("Done.")


if __name__ == "__main__":
	main()