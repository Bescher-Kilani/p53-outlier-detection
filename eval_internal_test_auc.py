# eval_internal_test_auc.py - compare a saved prediction file against the
# internal test labels split_data.py held back. Run this yourself only,
# completely separate from the pipeline - never feed these labels into
# p53_od_pipeline.py itself.
import pandas as pd
from sklearn.metrics import roc_auc_score

PREDICTION_FILE = "outputs/team2_prediction.csv"
LABELS_FILE = "_internal_test_labels_DO_NOT_USE_IN_PIPELINE.csv"

prediction = pd.read_csv(PREDICTION_FILE)
labels = pd.read_csv(LABELS_FILE, index_col=0)["label"]

merged = prediction.set_index("id").join(labels)
y_true = (merged["label"] == "active").astype(int)
auc = roc_auc_score(y_true, merged["score"])

print(f"Test-set AUC (internal, never seen by the pipeline): {auc:.4f}")
print(f"Active in test set: {y_true.sum()} / {len(y_true)}")
