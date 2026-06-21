# compute_correlations.py - fast Pearson and Spearman via matrix multiplication,
# safe because we already verified df_clean has zero NaNs and zero constant columns
import pandas as pd
import numpy as np
import time

DATA_FILE = "train_data_clean.parquet"

print("Loading data...")
df = pd.read_parquet(DATA_FILE)
print(f"Loaded shape: {df.shape}")
n = len(df)

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

print("Done.")