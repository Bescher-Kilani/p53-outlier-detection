"""
Process figures for README.md and docs/FULL_DOCUMENTATION.md.

Reads the per-repeat AUC matrices in outputs/ and week6-tables/ and writes
PNGs to docs/figures/. Run from the repository root:

	python plots/plot_process_figures.py

Figures built from published table values (no per-repeat file exists) are
marked in the function docstring.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("docs/figures")
TEAL = "#009682"
TEAL_LIGHT = "#8FD0C6"
GREY = "#B8BEC4"
DARK = "#2B2F33"
RED = "#C8553D"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
	"figure.facecolor": "white",
	"axes.facecolor": "white",
	"savefig.facecolor": "white",
	"axes.edgecolor": DARK,
	"axes.labelcolor": DARK,
	"text.color": DARK,
	"xtick.color": DARK,
	"ytick.color": DARK,
})


def save(fig, name: str) -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"wrote {OUT_DIR / name}")


def box(ax, data: pd.DataFrame, order, highlight, horizontal=True):
	"""Boxplot of per-repeat AUCs with the mean as a diamond."""
	colors = [TEAL if c in highlight else GREY for c in order]
	long = data[order].melt(var_name="model", value_name="auc")
	if horizontal:
		sns.boxplot(data=long, y="model", x="auc", order=order, palette=colors,
			hue="model", legend=False, ax=ax, width=0.6, fliersize=2, linewidth=1)
		means = data[order].mean().values
		ax.scatter(means, range(len(order)), marker="D", color="white",
			edgecolor=DARK, zorder=5, s=28)
	else:
		sns.boxplot(data=long, x="model", y="auc", order=order, palette=colors,
			hue="model", legend=False, ax=ax, width=0.6, fliersize=2, linewidth=1)
		means = data[order].mean().values
		ax.scatter(range(len(order)), means, marker="D", color="white",
			edgecolor=DARK, zorder=5, s=28)


def fig_pipeline() -> None:
	"""Pipeline overview (schematic)."""
	fig, ax = plt.subplots(figsize=(15, 3.6))
	ax.set_xlim(0, 14.2)
	ax.set_ylim(0, 3.4)
	ax.axis("off")
	steps = [
		("Raw data", "31,020 × 5,408"),
		("Cleaning", "30,754 rows"),
		("Stratified\nsplit", "train / test"),
		("Scale +\nPCA(200)", "fit on train only"),
		("Isolation Forest\n× 10 subsets", "inlier-only fit"),
		("Pruned\naggregation", "one outlier score"),
		("AUC", "labels used\nhere only"),
	]
	n = len(steps)
	w, gap = 1.7, 0.3
	x0 = 0.1
	for i, (title, sub) in enumerate(steps):
		x = x0 + i * (w + gap)
		last = i == n - 1
		patch = FancyBboxPatch((x, 1.1), w, 1.5, boxstyle="round,pad=0.03,rounding_size=0.12",
			fc=TEAL if i in (3, 4, 5) else "white", ec=TEAL if not last else RED, lw=2)
		ax.add_patch(patch)
		txt = "white" if i in (3, 4, 5) else DARK
		ax.text(x + w / 2, 2.0, title, ha="center", va="center", fontsize=10,
			fontweight="bold", color=txt)
		ax.text(x + w / 2, 1.4, sub, ha="center", va="center", fontsize=8.5, color=txt)
		if i < n - 1:
			ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.85), (x + w + gap - 0.02, 1.85),
				arrowstyle="-|>", mutation_scale=14, color=DARK, lw=1.5))
	ax.text(7.1, 0.55, "OCC rule: labels never enter .fit() or tuning",
		ha="center", fontsize=11, color=RED, fontweight="bold")
	save(fig, "01_pipeline.png")


def fig_data() -> None:
	"""Week 1: cleaning funnel and class imbalance (values from the Week 1 notes)."""
	fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
	labels = ["Raw", "− empty rows", "− no 2D block", "− duplicates"]
	vals = [31020, 31020 - 137, 31020 - 137 - 124, 31020 - 137 - 124 - 5]
	cols = [GREY, GREY, GREY, TEAL]
	a.bar(labels, vals, color=cols)
	for i, v in enumerate(vals):
		a.text(i, v + 60, f"{v:,}", ha="center", fontsize=9)
	a.set_ylim(29500, 31400)
	a.set_ylabel("rows (axis truncated)")
	a.set_title("Cleaning: 31,020 → 30,754 rows", fontsize=11)
	b.bar(["inactive", "active"], [30693, 61], color=[GREY, RED])
	b.set_yscale("log")
	b.text(0, 30693 * 1.15, "30,693", ha="center", fontsize=9)
	b.text(1, 61 * 1.15, "61", ha="center", fontsize=9)
	b.set_title("Class imbalance ≈ 503 : 1", fontsize=11)
	b.set_ylabel("mutants (log)")
	b.set_ylim(10, 1e5)
	fig.tight_layout()
	save(fig, "02_data_cleaning_imbalance.png")


def fig_reduction() -> None:
	"""Week 2 (published CV + norm table) and Week 6 sweep (per-repeat CSV)."""
	fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.1]})
	w2 = pd.DataFrame({
		"model": ["PCA", "Truncated SVD", "Isomap", "Random proj.", "none (raw)"],
		"auc": [0.7291, 0.7285, 0.6851, 0.6689, 0.5926],
		"std": [0.0543, 0.0504, 0.0807, 0.0249, 0.0624],
	})
	a.barh(w2["model"], w2["auc"], xerr=w2["std"], color=[TEAL, TEAL, GREY, GREY, GREY],
		error_kw={"ecolor": DARK, "lw": 1})
	a.invert_yaxis()
	a.set_xlim(0.5, 0.85)
	a.set_xlabel("CV AUC (Isolation Forest, 147 dims, scaled)")
	a.set_title("Week 2: reduction method", fontsize=11)
	sweep = pd.read_csv("week6-tables/week6_reduce_sweep_auc_matrix.csv").drop(columns="repeat")
	sweep = sweep.rename(columns={"pca": "PCA", "truncated_svd": "Truncated SVD",
		"random_projection": "Random proj.", "umap": "UMAP"})
	order = ["PCA", "Truncated SVD", "Random proj.", "UMAP"]
	box(b, sweep, order, {"PCA", "Truncated SVD"})
	b.set_xlabel("AUC over 25 repeats (200 dims)")
	b.set_ylabel("")
	b.set_title("Week 6: sweep at 200 dims", fontsize=11)
	fig.tight_layout()
	save(fig, "03_dimensionality_reduction.png")


def fig_methods() -> None:
	"""Week 3-4: six detectors, per-repeat AUCs, paired by repeat."""
	df = pd.read_csv("outputs/auc_matrix_compare_methods.csv").drop(columns="repeat")
	df = df.rename(columns={"isolation_forest": "Isolation Forest", "knn": "KNN",
		"deep_svdd": "Deep SVDD", "ocsvm": "OCSVM", "abod": "ABOD", "lof": "LOF"})
	order = ["Isolation Forest", "KNN", "Deep SVDD", "OCSVM", "ABOD", "LOF"]
	fig, ax = plt.subplots(figsize=(8, 4.2))
	box(ax, df, order, {"Isolation Forest"})
	ax.axvline(0.5, color=RED, ls="--", lw=1)
	ax.text(0.505, 5.45, "random", color=RED, fontsize=8)
	ax.set_xlabel("AUC over 25 repeats (PCA 147, inlier-only fit)")
	ax.set_ylabel("")
	ax.set_title("Detector comparison: Isolation Forest wins (Friedman p < 0.001)", fontsize=11)
	save(fig, "04_detector_comparison.png")


def fig_bagging() -> None:
	"""Feature bagging gain, paired per repeat, at 147 and 200 dims."""
	fb147 = pd.read_csv("outputs/auc_feature_bagging_matrix.csv")
	fb200 = pd.read_csv("week6-tables/week6_feature_bagging_if_pca200.csv")
	rows = []
	for d, (df, cols) in {
		"PCA 147": (fb147, {"no bagging": "isolation_forest_no_bagging", "sum": "isolation_forest_sum",
			"max": "isolation_forest_max", "AOM": "isolation_forest_aom"}),
		"PCA 200": (fb200, {"no bagging": "isolation_forest_no_bagging", "sum": "isolation_forest_sum",
			"max": "isolation_forest_max", "AOM": "isolation_forest_aom",
			"pruned": "isolation_forest_pruned", "damped": "isolation_forest_damped"}),
	}.items():
		base = df["isolation_forest_no_bagging"]
		for name, col in cols.items():
			if name == "no bagging":
				continue
			for g in (df[col] - base):
				rows.append({"dims": d, "aggregation": name, "gain": g * 100})
	gain = pd.DataFrame(rows)
	fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True,
		gridspec_kw={"width_ratios": [4, 6]})
	for ax, d in zip(axes, ["PCA 147", "PCA 200"]):
		sub = gain[gain["dims"] == d]
		order = list(dict.fromkeys(sub["aggregation"]))
		sns.boxplot(data=sub, x="aggregation", y="gain", order=order, color=TEAL_LIGHT,
			ax=ax, width=0.6, fliersize=2, linewidth=1)
		ax.axhline(0, color=DARK, lw=1)
		ax.set_title(f"Isolation Forest, {d}", fontsize=11)
		ax.set_xlabel("")
	axes[0].set_ylabel("AUC gain vs. no bagging (pp, per repeat)")
	axes[1].set_ylabel("")
	fig.suptitle("Feature bagging: small but consistent gain", fontsize=12, y=1.02)
	fig.tight_layout()
	save(fig, "05_feature_bagging_gain.png")


def fig_hpo() -> None:
	"""Week 6: default vs tuned (per-repeat) and Optuna history for IF at 200 dims."""
	dflt = pd.read_csv("week6-tables/week6_final_default_auc_matrix.csv").drop(columns="repeat")
	tuned = pd.read_csv("week6-tables/week6_final_tuned_auc_matrix.csv").drop(columns="repeat")
	names = {"isolation_forest": "Isolation Forest", "knn": "KNN", "deep_svdd": "Deep SVDD",
		"lunar": "LUNAR", "dif": "DIF"}
	rows = []
	for label, df in (("default", dflt), ("tuned", tuned)):
		for c, n in names.items():
			for v in df[c]:
				rows.append({"model": n, "setting": label, "auc": v})
	long = pd.DataFrame(rows)
	hist = pd.read_csv("week6-tables/hpo_isolation_forest_pca200.csv")
	hist = hist[hist["state"] == "COMPLETE"]
	fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
	sns.boxplot(data=long, x="model", y="auc", hue="setting", order=list(names.values()),
		palette={"default": GREY, "tuned": TEAL}, ax=a, width=0.7, fliersize=2, linewidth=1)
	a.set_xlabel("")
	a.set_ylabel("AUC over 25 repeats (PCA 147)")
	a.set_title("HPO: default vs. tuned", fontsize=11)
	a.legend(title="", loc="lower left")
	b.scatter(hist["number"], hist["value"], s=14, color=GREY)
	b.plot(hist["number"], hist["value"].cummax(), color=TEAL, lw=2, label="best so far")
	b.axhline(0.7392, color=RED, ls="--", lw=1.2, label="default params, 25 repeats (0.7392)")
	b.set_xlabel("Optuna trial")
	b.set_ylabel("AUC (15 repeats / trial)")
	b.set_title("Isolation Forest at 200 dims: no gain from HPO", fontsize=11)
	b.legend(loc="lower right", fontsize=8)
	fig.tight_layout()
	save(fig, "06_hpo.png")


def fig_progress() -> None:
	"""Best OCC AUC per week (values from the weekly slides / Phase 2 report; protocols differ per week)."""
	weeks = ["W2", "W3", "W4", "W5", "W6"]
	auc = [0.7291, 0.7248, 0.7354, 0.7314, 0.7438]
	labels = ["IF + PCA(147)", "IF, 5 detectors", "+ feature bagging\n(AOM)", "+ TabPFN, V-GAN\n(no gain)",
		"PCA(200) + bagging\n(pruned)"]
	fig, ax = plt.subplots(figsize=(9, 4))
	x = np.arange(len(weeks))
	ax.plot(x, auc, color=TEAL, marker="o", lw=2.2, ms=7)
	for xi, a, l in zip(x, auc, labels):
		ax.text(xi, a + 0.006, f"{a:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
		ax.text(xi, 0.705, l, ha="center", va="center", fontsize=8, color=DARK)
	ax.set_xticks(x, weeks)
	ax.set_ylim(0.69, 0.76)
	ax.set_xlim(-0.5, 4.5)
	ax.set_ylabel("best OCC AUC")
	ax.set_title("Progress: best OCC model per week", fontsize=10)
	fig.text(0.5, -0.02, "Protocols differ: W2 3-fold CV, W4 full-train post-hoc, W5/W6 repeated subsampling (as reported on the slides)",
		ha="center", fontsize=7.5, color=DARK)
	save(fig, "07_progress.png")


def fig_final() -> None:
	"""Final train / held-out test AUC (published table; test set has only 12 actives)."""
	rows = [
		("IF + PCA(147)\n+ bagging", "OCC", 0.7298, 0.8153),
		("IF + PCA(200)\n+ bagging", "OCC", 0.7532, 0.8431),
	]
	fig, ax = plt.subplots(figsize=(6.5, 4))
	x = np.arange(len(rows))
	w = 0.36
	ax.bar(x - w / 2, [r[2] for r in rows], w, color=GREY, label="train (post-hoc)")
	ax.bar(x + w / 2, [r[3] for r in rows], w, color=TEAL, label="held-out test")
	for xi, r in zip(x, rows):
		ax.text(xi - w / 2, r[2] + 0.008, f"{r[2]:.3f}", ha="center", fontsize=9)
		ax.text(xi + w / 2, r[3] + 0.008, f"{r[3]:.3f}", ha="center", fontsize=9)
	ax.set_xticks(x, [r[0] for r in rows])
	ax.set_ylim(0.5, 0.95)
	ax.set_ylabel("AUC")
	ax.set_title("Final OCC models (test: 12 actives, noisy)", fontsize=11)
	ax.legend(loc="upper left")
	save(fig, "08_final_train_test.png")


def main() -> None:
	fig_pipeline()
	fig_data()
	fig_reduction()
	fig_methods()
	fig_bagging()
	fig_hpo()
	fig_progress()
	fig_final()


if __name__ == "__main__":
	main()
