# eval_internal_test_auc.py - compare a saved prediction file against the
# internal test labels split_data.py held back. Run this yourself only,
# completely separate from the pipeline - never feed these labels into
# p53_od_pipeline.py itself.
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root, so p53_od_pipeline can be imported
from p53_od_pipeline import evaluate_labeled_scores

PREDICTION_FILE = "outputs/team2_prediction.csv"
LABELS_FILE = "_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv"

prediction = pd.read_csv(PREDICTION_FILE)
labels = pd.read_csv(LABELS_FILE, index_col=0)["label"]

merged = prediction.set_index("id").join(labels)
evaluate_labeled_scores(
	merged["score"],
	merged["label"],
	title="TEST-SET METRICS (internal, never seen by the pipeline)",
)
