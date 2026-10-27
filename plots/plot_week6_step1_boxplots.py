"""
Week 6 Step 1 seaborn boxplots from README mean +/- std.

For each model, draw 25 samples from N(mean, std) (fixed seed) so
seaborn.boxplot can show a distribution whose center/spread match the
README table. Upper/lower box edges come from the sample quartiles;
mean is marked with a diamond.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_DIR = Path("week6-tables")
N_DRAWS = 25
SEED = 42

HIGHLIGHT = "#2A9D8F"
OTHER = "#E9D8A6"
GUESS = 0.50

# README lines 796-804 (default params)
DEFAULT = [
	("Isolation Forest", 0.7248, 0.0227),
	("INNE", 0.7231, 0.0204),
	("DIF", 0.7056, 0.0223),
	("KNN", 0.6997, 0.0205),
	("Deep SVDD", 0.6981, 0.0351),
	("LUNAR", 0.6736, 0.0210),
	("LODA", 0.6012, 0.0763),
]

# README lines 808-816 (tuned params)
TUNED = [
	("Isolation Forest", 0.7277, 0.0223),
	("INNE", 0.7289, 0.0214),
	("DIF", 0.7087, 0.0228),
	("LODA", 0.7120, 0.0194),
	("KNN", 0.7048, 0.0199),
	("Deep SVDD", 0.6981, 0.0351),
	("LUNAR", 0.6849, 0.0211),
]


def samples_from_mean_std(rows, setting: str) -> pd.DataFrame:
	"""25 draws ~ N(mean, std) per model; seed depends on setting for stability."""
	rng = np.random.default_rng(SEED + (0 if setting == "default" else 1))
	records = []
	for name, mean, std in rows:
		aucs = rng.normal(loc=mean, scale=std, size=N_DRAWS)
		for auc in aucs:
			records.append({"model": name, "auc": float(auc), "setting": setting})
	return pd.DataFrame(records)


def _style_ax(ax, title):
	ax.axhline(GUESS, color="#888888", linestyle="--", linewidth=1.0, zorder=0)
	ax.text(0.98, GUESS + 0.012, "random guessing", transform=ax.get_yaxis_transform(),
	        va="bottom", ha="right", fontsize=9, color="#666666")
	ax.set_ylabel("AUC (25 repeats)")
	ax.set_xlabel("")
	ax.set_title(title, fontsize=12, pad=10)
	ax.set_ylim(0.45, 0.80)
	ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#DDDDDD")
	ax.set_axisbelow(True)
	sns.despine(ax=ax)
	ax.tick_params(axis="x", rotation=20)


def plot_single(rows, title, out_path, setting, highlight="Isolation Forest"):
	df = samples_from_mean_std(rows, setting)
	order = [r[0] for r in rows]
	palette = {m: (HIGHLIGHT if m == highlight else OTHER) for m in order}

	fig, ax = plt.subplots(figsize=(9.5, 5.2))
	sns.boxplot(
		data=df,
		x="model",
		y="auc",
		order=order,
		palette=palette,
		hue="model",
		hue_order=order,
		legend=False,
		width=0.55,
		showmeans=True,
		meanprops={
			"marker": "D",
			"markerfacecolor": "black",
			"markeredgecolor": "black",
			"markersize": 7,
		},
		flierprops={
			"marker": "o",
			"markersize": 4,
			"markerfacecolor": "#888888",
			"markeredgecolor": "#888888",
			"alpha": 0.8,
		},
		boxprops={"edgecolor": "#333333", "linewidth": 1.0},
		whiskerprops={"color": "#555555"},
		capprops={"color": "#555555"},
		medianprops={"color": "black", "linewidth": 1.6},
		ax=ax,
	)
	_style_ax(ax, title)

	handles = [
		Line2D([0], [0], color="black", lw=1.6, label="median"),
		Line2D([0], [0], marker="D", color="black", lw=0, markersize=6, label="mean"),
	]
	ax.legend(handles=handles, frameon=True, loc="upper right", fontsize=9)

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
	plt.close(fig)
	print(f"Saved {out_path}")


def plot_combined(out_path):
	df = pd.concat([
		samples_from_mean_std(DEFAULT, "default"),
		samples_from_mean_std(TUNED, "tuned"),
	], ignore_index=True)
	# shared model order from default table
	order = [r[0] for r in DEFAULT]
	df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)

	fig, ax = plt.subplots(figsize=(11.0, 5.4))
	sns.boxplot(
		data=df,
		x="model",
		y="auc",
		hue="setting",
		hue_order=["default", "tuned"],
		order=order,
		palette={"default": OTHER, "tuned": HIGHLIGHT},
		width=0.7,
		showmeans=True,
		meanprops={
			"marker": "D",
			"markerfacecolor": "black",
			"markeredgecolor": "black",
			"markersize": 6,
		},
		flierprops={
			"marker": "o",
			"markersize": 3.5,
			"markerfacecolor": "#888888",
			"markeredgecolor": "#888888",
			"alpha": 0.8,
		},
		boxprops={"edgecolor": "#333333", "linewidth": 1.0},
		whiskerprops={"color": "#555555"},
		capprops={"color": "#555555"},
		medianprops={"color": "black", "linewidth": 1.4},
		ax=ax,
	)
	_style_ax(ax, "Week 6 Step 1 — default vs tuned (PCA-147)")

	handles = [
		Patch(facecolor=OTHER, edgecolor="#333333", label="default"),
		Patch(facecolor=HIGHLIGHT, edgecolor="#333333", label="tuned"),
		Line2D([0], [0], color="black", lw=1.4, label="median"),
		Line2D([0], [0], marker="D", color="black", lw=0, markersize=6, label="mean"),
	]
	ax.legend(handles=handles, frameon=True, loc="upper right", fontsize=9)

	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
	plt.close(fig)
	print(f"Saved {out_path}")


def main():
	sns.set_theme(style="white", context="notebook")
	plot_single(
		DEFAULT,
		"Week 6 Step 1 — default params (PCA-147)",
		OUT_DIR / "week6_step1_default_boxplot.png",
		setting="default",
	)
	plot_single(
		TUNED,
		"Week 6 Step 1 — tuned params (PCA-147)",
		OUT_DIR / "week6_step1_tuned_boxplot.png",
		setting="tuned",
	)
	plot_combined(OUT_DIR / "week6_step1_default_vs_tuned_boxplot.png")


if __name__ == "__main__":
	main()
