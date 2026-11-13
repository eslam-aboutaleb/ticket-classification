"""Downstream supervised classification suite, pseudo-label QA, and Open-Set novelty gate."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


class ClassifierSuite:
    """Trains and compares multiple classifiers against discovered cluster pseudo-labels."""

    def __init__(
        self,
        min_data_per_class: int = 3,
        confidence_quantile_cutoff: float = 0.10,
        open_set_distance_quantile: float = 0.96,
        random_state: int = 42,
    ) -> None:
        self.min_data_per_class = min_data_per_class
        self.confidence_quantile_cutoff = confidence_quantile_cutoff
        self.open_set_distance_quantile = open_set_distance_quantile
        self.random_state = random_state

        self.models_: dict[str, object] = {}
        self.best_model_name_: str = "logistic_regression"
        self.centroids_: np.ndarray | None = None
        self.distance_threshold_: float = 0.65
        self.valid_classes_: set[int] = set()

    def filter_pseudo_labels(
        self,
        X: np.ndarray,
        cluster_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter low-confidence boundary points and gate rare classes (karolzak + DeepAligned pattern)."""
        if not len(confidences):
            self.valid_classes_ = set()
            return X[:0], cluster_ids[:0]
        conf_floor = (
            float(
                np.quantile(
                    confidences[cluster_ids >= 0], self.confidence_quantile_cutoff
                )
            )
            if np.any(cluster_ids >= 0)
            else 1.0
        )
        mask = confidences >= conf_floor

        # Enforce min_data_per_class support gating
        unique, counts = np.unique(cluster_ids[mask], return_counts=True)
        valid_classes = {
            int(c)
            for c, cnt in zip(unique, counts)
            if cnt >= self.min_data_per_class and c >= 0
        }
        mask = mask & np.isin(cluster_ids, list(valid_classes))

        self.valid_classes_ = valid_classes
        self.training_mask_ = mask
        return X[mask], cluster_ids[mask]

    def fit_all(
        self,
        X_train: np.ndarray,
        embeddings_train: np.ndarray,
        cluster_ids_train: np.ndarray,
        confidences_train: np.ndarray,
    ) -> dict[str, object]:
        """Fit baseline, linear, SVM, and gradient boosted classifiers on filtered pseudo-labels."""
        self.best_model_name_ = "logistic_regression"
        X_clean, y_clean = self.filter_pseudo_labels(
            X_train, cluster_ids_train, confidences_train
        )
        self.models_ = {}
        if not len(y_clean):
            self.centroids_ = np.empty((0, embeddings_train.shape[1]))
            self.best_model_name_ = "unassigned"
            return self.models_

        # Compute cluster centroids in normalized embedding space for Open-Set distance gate
        norm_emb = normalize(embeddings_train, norm="l2")
        unique_classes = sorted(self.valid_classes_)
        centroids = []
        for c in unique_classes:
            c_vecs = norm_emb[(cluster_ids_train == c) & self.training_mask_]
            centroids.append(
                normalize(c_vecs.mean(axis=0, keepdims=True), norm="l2")[0]
            )
        self.centroids_ = np.vstack(centroids)
        self.class_index_to_id_ = {idx: c for idx, c in enumerate(unique_classes)}

        # Calibrate Open-Set cosine distance threshold (1 - max_cosine_similarity) on training data
        cos_sims = norm_emb @ self.centroids_.T
        min_dists = 1.0 - np.max(cos_sims, axis=1)
        self.distance_threshold_ = float(
            np.quantile(min_dists, self.open_set_distance_quantile)
        )

        # Majority class baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_clean, y_clean)
        if len(unique_classes) == 1:
            self.models_ = {"single_cluster": dummy}
            self.best_model_name_ = "single_cluster"
            return self.models_

        # Multinomial logistic regression with balanced class weights
        logreg = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        logreg.fit(X_clean, y_clean)

        # Calibrated linear support vector classifier
        min_class_count = int(np.min(np.unique(y_clean, return_counts=True)[1]))
        cv_folds = max(2, min(3, min_class_count))
        base_svc = LinearSVC(
            C=1.0, class_weight="balanced", random_state=self.random_state
        )
        if min_class_count >= 2:
            svc = CalibratedClassifierCV(base_svc, cv=cv_folds)
            svc.fit(X_clean, y_clean)
        else:
            svc = logreg

        # Gradient boosted tree ensemble
        gbdt = GradientBoostingClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
        )
        gbdt.fit(X_clean, y_clean)

        self.models_ = {
            "naive_majority": dummy,
            "logistic_regression": logreg,
            "calibrated_linear_svc": svc,
            "gradient_boosting": gbdt,
            "frozen_prototype": None,
        }
        return self.models_

    def predict_with_open_set(
        self,
        X: np.ndarray,
        embeddings: np.ndarray,
        model_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (pred_cluster_ids, confidences, margins, is_open_set_novel, centroid_distances)."""
        target_name = model_name or self.best_model_name_
        if not len(X):
            return (
                np.empty(0, dtype=int),
                np.empty(0),
                np.empty(0),
                np.empty(0, dtype=bool),
                np.empty(0),
            )
        if not self.models_:
            n = len(X)
            return (
                np.full(n, -1),
                np.zeros(n),
                np.zeros(n),
                np.ones(n, dtype=bool),
                np.ones(n),
            )
        if target_name == "frozen_prototype":
            similarities = normalize(embeddings) @ self.centroids_.T
            weights = np.exp(
                (similarities - similarities.max(axis=1, keepdims=True)) / 0.1
            )
            probs = weights / weights.sum(axis=1, keepdims=True)
            classes = np.array(
                [self.class_index_to_id_[i] for i in range(len(self.centroids_))]
            )
        else:
            clf = self.models_[target_name]
            probs = clf.predict_proba(X)
            classes = clf.classes_
        sorted_probs = np.sort(probs, axis=1)
        top_idx = np.argmax(probs, axis=1)
        preds = np.array([int(classes[i]) for i in top_idx], dtype=int)
        confidences = sorted_probs[:, -1]
        margins = (
            sorted_probs[:, -1] - sorted_probs[:, -2]
            if probs.shape[1] > 1
            else confidences
        )

        assert self.centroids_ is not None
        centroid_dists, is_open_set = self.novelty_scores(embeddings)

        return preds, confidences, margins, is_open_set, centroid_dists

    def novelty_scores(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shared evaluation/serving novelty gate, with no projected-space threshold mixing."""
        if self.centroids_ is None or not len(self.centroids_):
            return np.ones(len(embeddings)), np.ones(len(embeddings), dtype=bool)
        if not len(embeddings):
            return np.empty(0), np.empty(0, dtype=bool)
        distances = 1.0 - np.max(normalize(embeddings) @ self.centroids_.T, axis=1)
        return distances, distances > self.distance_threshold_

    def calibrate_novelty(self, embeddings: np.ndarray) -> None:
        """Fit the serving gate on held-out known examples in original embedding space."""
        if self.centroids_ is not None and len(self.centroids_) and len(embeddings):
            distances = 1.0 - np.max(normalize(embeddings) @ self.centroids_.T, axis=1)
            self.distance_threshold_ = float(
                np.quantile(distances, self.open_set_distance_quantile)
            )
