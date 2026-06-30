"""mRMR feature ranking + IFS (pick subset size by train AUC with Isolation Forest)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


class MRMRIFSSelector(BaseEstimator, TransformerMixin):
	def __init__(
		self,
		max_features=147,
		n_prefilter=300,
		ifs_candidates=None,
		n_estimators=100,
		random_state=42,
	):
		self.max_features = max_features
		self.n_prefilter = n_prefilter
		self.ifs_candidates = ifs_candidates
		self.n_estimators = n_estimators
		self.random_state = random_state

	def _mrmr_rank(self, X, y):
		n_features = X.shape[1]
		mi_xy = mutual_info_classif(X, y, random_state=self.random_state)
		corr = np.abs(np.corrcoef(X.T))
		np.fill_diagonal(corr, 0.0)

		first = int(np.argmax(mi_xy))
		selected = [first]
		remaining = set(range(n_features)) - {first}
		ranking = [first]

		while remaining and len(ranking) < self.max_features:
			best_score, best_j = -np.inf, None
			for j in remaining:
				redundancy = corr[j, selected].mean()
				score = mi_xy[j] - redundancy
				if score > best_score:
					best_score, best_j = score, j
			selected.append(best_j)
			ranking.append(best_j)
			remaining.remove(best_j)

		return np.asarray(ranking, dtype=int)

	def fit(self, X, y):
		X = np.asarray(X, dtype=np.float32)
		y = np.asarray(y, dtype=int)

		n_prefilter = min(self.n_prefilter, X.shape[1])
		mi = mutual_info_classif(X, y, random_state=self.random_state)
		prefilter_idx = np.argsort(-mi)[:n_prefilter]
		X_sub = X[:, prefilter_idx]

		local_rank = self._mrmr_rank(X_sub, y)
		ranked_global = prefilter_idx[local_rank]

		if self.ifs_candidates is None:
			step = max(25, self.max_features // 6)
			candidates = list(range(step, self.max_features + 1, step))
			if self.max_features not in candidates:
				candidates.append(self.max_features)
		else:
			candidates = list(self.ifs_candidates)

		candidates = [k for k in candidates if k <= len(ranked_global)]
		if not candidates:
			candidates = [min(25, len(ranked_global))]

		best_auc, best_k = -1.0, candidates[0]
		for k in candidates:
			cols = ranked_global[:k]
			clf = IsolationForest(
				n_estimators=self.n_estimators,
				random_state=self.random_state,
				n_jobs=-1,
			)
			clf.fit(X[:, cols])
			scores = -clf.score_samples(X[:, cols])
			auc = roc_auc_score(y, scores)
			if auc > best_auc:
				best_auc, best_k = auc, k

		self.selected_features_ = ranked_global[:best_k]
		self.n_features_selected_ = best_k
		self.ifs_best_auc_ = best_auc
		self.mrmr_ranking_ = ranked_global
		return self

	def transform(self, X):
		X = np.asarray(X)
		return X[:, self.selected_features_]
