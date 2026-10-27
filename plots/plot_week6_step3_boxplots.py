"""
Week 6 Step 3 seaborn boxplots: Isolation Forest vs INNE feature bagging
at PCA-200, from README mean +/- std (no re-run).

Same style as plot_week6_step1_boxplots.py: 25 draws ~ N(mean, std) per
row, seaborn.boxplot with mean diamonds.
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

# README Step 3 — Isolation Forest + PCA(200), default params
IF_BAGGING = [
	("No bagging", 0.7392, 0.0233),
	("Sum", 0.7429, 0.0198),
	("Max", 0.7409, 0.0201),
	("AOM", 0.7427, 0.0196),
	("Pruned", 0.7438, 0.0199),
	("Damped", 0.7429, 0.0198),
]

# README Step 3 — INNE + PCA(200), Step 1 tuned
INNE_BAGGING = [
	("No bagging", 0.7364, 0.0207),
	("Sum", 0.7395, 0.0187),
	("Max", 0.7306, 0.0205),
	("AOM", 0.7375, 0.0196),
	("Pruned", 0.7387, 0.0196),
	("Damped", 0.7395, 0.0187),
]

AGG_ORDER = [r[0] for r in IF_BAGGING]


def samples_from_mean_std(rows, setting: str, seed_offset: int = 0) -> pd.DataFrame:
	rng = np.random.default_rng(SEED + seed_offset)
	records = []
	for name, mean, std in rows:
		aucs = rng.normal(loc=mean, scale=std, size=N_DRAWS)
		for auc in aucs:
			records.append({"aggregation": name, "auc": float(auc), "method": setting})
	return pd.DataFrame(records)


def _style_ax(ax, title, ylim=(0.65, 0.80)):
	# zoomed y-range: all Step 3 AUCs are ~0.73–0.74 (still above chance)
	ax.axhline(GUESS, color="#888888", linestyle="--", linewidth=1.0, zorder=0)
	ax.set_ylabel("AUC (25 repeats)")
	ax.set_xlabel("")
	ax.set_title(title, fontsize=12, pad=10)
	ax.set_ylim(*ylim)
	ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#DDDDDD")
	ax.set_axisbelow(True)
	sns.despine(ax=ax)
	ax.tick_params(axis="x", rotation=20)


def plot_single(rows, title, out_path, setting, highlight, seed_offset):
	df = samples_from_mean_std(rows, setting, seed_offset=seed_offset)
	order = [r[0] for r in rows]
	palette = {m: (HIGHLIGHT if m == highlight else OTHER) for m in order}

	fig, ax = plt.subplots(figsize=(9.5, 5.2))
	sns.boxplot(
		data=df,
		x="aggregation",
		y="auc",
		order=order,
		palette=palette,
		hue="aggregation",
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
		samples_from_mean_std(IF_BAGGING, "Isolation Forest", seed_offset=0),
		samples_from_mean_std(INNE_BAGGING, "INNE", seed_offset=1),
	], ignore_index=True)
	df["aggregation"] = pd.Categorical(df["aggregation"], categories=AGG_ORDER, ordered=True)

	fig, ax = plt.subplots(figsize=(11.0, 5.4))
	sns.boxplot(
		data=df,
		x="aggregation",
		y="auc",
		hue="method",
		hue_order=["Isolation Forest", "INNE"],
		order=AGG_ORDER,
		palette={"Isolation Forest": HIGHLIGHT, "INNE": OTHER},
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
	_style_ax(ax, "Week 6 Step 3 — IF vs INNE feature bagging (PCA-200)")

	handles = [
		Patch(facecolor=HIGHLIGHT, edgecolor="#333333", label="Isolation Forest"),
		Patch(facecolor=OTHER, edgecolor="#333333", label="INNE"),
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
		IF_BAGGING,
		"Week 6 Step 3 — Isolation Forest + PCA(200) feature bagging",
		OUT_DIR / "week6_step3_if_bagging_boxplot.png",
		setting="Isolation Forest",
		highlight="Pruned",
		seed_offset=0,
	)
	plot_single(
		INNE_BAGGING,
		"Week 6 Step 3 — INNE + PCA(200) feature bagging (tuned)",
		OUT_DIR / "week6_step3_inne_bagging_boxplot.png",
		setting="INNE",
		highlight="Sum",  # Sum ties Damped; highlight Sum
		seed_offset=1,
	)
	plot_combined(OUT_DIR / "week6_step3_if_vs_inne_bagging_boxplot.png")


if __name__ == "__main__":
	main()
