# compute_correlations.py - fast Pearson, Spearman, and Mutual Information
# via matrix multiplication tricks, safe because we already verified
# df_clean has zero NaNs and zero constant columns
import pandas as pd
import numpy as np
import time

DATA_FILE = "train_data_clean.parquet"

print("Loading data...")
df = pd.read_parquet(DATA_FILE)
print(f"Loaded shape: {df.shape}")
n = len(df)
p = df.shape[1]

def fast_corr(values, columns):
	X = values - values.mean(axis=0, keepdims=True)
	std = values.std(axis=0, ddof=1, keepdims=True)
	Xn = X / std
	corr = (Xn.T @ Xn) / (n - 1)
	return pd.DataFrame(corr, index=columns, columns=columns)

print("Computing Pearson correlation (matrix-multiplication based)...")
start = time.time()
pearson_corr = fast_corr(df.values.astype(np.float64), df.columns)
print(f"Pearson done in {time.time() - start:.1f}s")
pearson_corr.to_parquet("pearson_corr.parquet")

print("Ranking columns for Spearman...")
start = time.time()
ranked = df.rank(method="average")
print(f"Ranking done in {time.time() - start:.1f}s")

print("Computing Spearman correlation (Pearson correlation of the ranks)...")
start = time.time()
spearman_corr = fast_corr(ranked.values.astype(np.float64), df.columns)
print(f"Spearman done in {time.time() - start:.1f}s")
spearman_corr.to_parquet("spearman_corr.parquet")

# --- Mutual Information ---
# mutual_info_regression (sklearn) works fine for ~20 candidate pairs but is
# far too slow for all p*(p-1)/2 ~ 14.6M pairs. Same trick as fast_corr:
# discretize each feature into quantile bins, one-hot encode, then get ALL
# pairwise joint bin counts in a single matrix multiplication.
MI_BINS = 8      # bins per feature; matmul cost scales with bins^2
MI_BLOCK = 500   # features processed per block when extracting MI, to bound memory

print(f"Discretizing into {MI_BINS} quantile bins per feature for MI...")
start = time.time()
bin_edges = np.linspace(0, n, MI_BINS + 1)
bins = np.digitize(ranked.values, bin_edges[1:-1])  # shape (n, p), values in [0, MI_BINS-1]
print(f"Binning done in {time.time() - start:.1f}s")

print("Building one-hot indicator matrix...")
start = time.time()
feature_offset = (np.arange(p) * MI_BINS).astype(np.int64)
col_idx = (bins + feature_offset[None, :]).ravel()
row_idx = np.repeat(np.arange(n), p)
H = np.zeros((n, p * MI_BINS), dtype=np.float32)
H[row_idx, col_idx] = 1.0
print(f"One-hot done in {time.time() - start:.1f}s")

print("Computing joint bin counts (H^T @ H)...")
start = time.time()
joint_counts = (H.T @ H).reshape(p, MI_BINS, p, MI_BINS)
marginal_counts = H.sum(axis=0).reshape(p, MI_BINS)
print(f"Matrix multiplication done in {time.time() - start:.1f}s")

print("Extracting pairwise mutual information from joint counts...")
start = time.time()
mi = np.zeros((p, p), dtype=np.float64)
Pj_full = marginal_counts / n  # (p, MI_BINS), reused for every block
for block_start in range(0, p, MI_BLOCK):
	block_end = min(block_start + MI_BLOCK, p)
	Pij = joint_counts[block_start:block_end] / n               # (block, k, p, k)
	Pi = Pj_full[block_start:block_end]                          # (block, k)
	outer = Pi[:, :, None, None] * Pj_full[None, None, :, :]    # (block, k, p, k)
	with np.errstate(divide="ignore", invalid="ignore"):
		term = np.where(Pij > 0, Pij * np.log(Pij / outer), 0.0)
	mi[block_start:block_end] = term.sum(axis=(1, 3))
	print(f"  MI block {block_start}-{block_end} done ({time.time() - start:.1f}s elapsed)")
print(f"MI extraction done in {time.time() - start:.1f}s")

mi_corr = pd.DataFrame(mi, index=df.columns, columns=df.columns)
mi_corr.to_parquet("mi_corr.parquet")

print("Done.")