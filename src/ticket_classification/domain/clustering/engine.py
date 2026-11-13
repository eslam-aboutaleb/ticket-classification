"""Clustering workflows, cluster count estimation, centroid alignment, and keyword extraction."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from ticket_classification.domain.base_stage import LeakageSafePipelineStage
from ticket_classification.models.entities import AlignmentResult


class ClusteringEngine(
    LeakageSafePipelineStage[np.ndarray, tuple[np.ndarray, np.ndarray]]
):
    """Implements DeepAligned K-Means, Density Clustering, Silhouette optimization, and Hungarian alignment."""

    def __init__(
        self,
        strategy: str = "aligned_kmeans",
        k_min: int = 4,
        k_max: int = 15,
        n_components: int = 10,
        outlier_recovery_threshold: float = 0.55,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.k_min = k_min
        self.k_max = k_max
        self.n_components = n_components
        self.outlier_recovery_threshold = outlier_recovery_threshold
        self.random_state = random_state

        self._projector: PCA | None = None
        self.centroids_: np.ndarray | None = None
        self.raw_space_centroids_: np.ndarray | None = None
        self.train_labels_: np.ndarray | None = None
        self.optimal_k_: int = k_min
        self.dropout_viable_k_: int = k_min
        self.hit_k_max_boundary_: bool = False
        self.best_silhouette_: float = 0.0
        self.silhouette_history_: dict[int, float] = {}
        self.last_alignment_: AlignmentResult | None = None

    def _project_embeddings(self, embeddings: np.ndarray, is_fit: bool) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must be a finite two-dimensional matrix")
        if not len(embeddings):
            width = (
                self._projector.n_components_
                if self._projector is not None
                else embeddings.shape[1]
            )
            return np.empty((0, width))
        normed = normalize(embeddings, norm="l2")
        if not is_fit:
            return (
                normalize(self._projector.transform(normed), norm="l2")
                if self._projector is not None
                else normed
            )
        self._projector = None
        n_comp = min(self.n_components, normed.shape[0] - 1, normed.shape[1])
        if n_comp < 2:
            return normed
        self._projector = PCA(n_components=n_comp, random_state=self.random_state)
        return normalize(self._projector.fit_transform(normed), norm="l2")

    def fit_transform(self, train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return discovery assignments, preserving fitted noise and recovery decisions."""
        self.fit(train_data)
        assert self.train_labels_ is not None
        _, strengths = self.transform(train_data)
        return self.train_labels_.copy(), np.where(
            self.train_labels_ >= 0, strengths, 0.0
        )

    def predict_k_with_dropout_and_silhouette(
        self, projected: np.ndarray
    ) -> tuple[int, float]:
        """Estimates cluster count by combining density-filtered upper-bound clustering with silhouette validation."""
        n_samples = projected.shape[0]
        if n_samples < 4 or len(np.unique(projected, axis=0)) < 2:
            return 1, 0.0
        max_k = min(
            self.k_max,
            max(self.k_min, n_samples // 4),
            n_samples - 1,
            len(np.unique(projected, axis=0)),
        )
        min_k = min(self.k_min, max_k)

        km_upper = KMeans(
            n_clusters=max_k, n_init=10, random_state=self.random_state
        ).fit(projected)
        dropout_threshold = n_samples / max_k
        viable_count = sum(
            1
            for c in np.unique(km_upper.labels_)
            if np.sum(km_upper.labels_ == c) >= dropout_threshold * 0.5
        )
        self.dropout_viable_k_ = max(min_k, viable_count)

        best_k = self.dropout_viable_k_
        best_score = -1.0
        best_adjusted = -1.0
        self.silhouette_history_ = {}

        for k in range(min_k, max_k + 1):
            if k >= n_samples:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(
                projected
            )
            if len(np.unique(km.labels_)) < 2:
                continue
            score = float(silhouette_score(projected, km.labels_, metric="cosine"))
            self.silhouette_history_[k] = score
            adjusted = score - 0.0015 * abs(k - self.dropout_viable_k_)
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_score = score
                best_k = k

        self.hit_k_max_boundary_ = best_k >= max_k
        return best_k, best_score

    def align_centroids_hungarian(
        self,
        new_centroids: np.ndarray,
        labels: np.ndarray,
        reference_centroids: np.ndarray | None = None,
        candidate_raw_centroids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
        """Align new cluster centroids to previous centroids via linear_sum_assignment."""
        ref = (
            reference_centroids if reference_centroids is not None else self.centroids_
        )
        cand = (
            candidate_raw_centroids
            if candidate_raw_centroids is not None
            else new_centroids
        )
        if ref is None or ref.shape != cand.shape:
            identity_map = {i: i for i in range(len(new_centroids))}
            return new_centroids, labels, identity_map

        dist_matrix = np.linalg.norm(
            ref[:, np.newaxis, :] - cand[np.newaxis, :, :],
            axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        aligned_centroids = np.zeros_like(new_centroids)
        pseudo_to_aligned: dict[int, int] = {}

        for canonical_idx, raw_idx in zip(row_ind, col_ind):
            aligned_centroids[canonical_idx] = new_centroids[raw_idx]
            pseudo_to_aligned[int(raw_idx)] = int(canonical_idx)

        aligned_labels = np.array(
            [pseudo_to_aligned.get(int(lbl), int(lbl)) for lbl in labels],
            dtype=int,
        )
        return aligned_centroids, aligned_labels, pseudo_to_aligned

    def recover_outliers(
        self,
        projected: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Reassigns unassigned points to the nearest centroid if cosine similarity meets the threshold."""
        recovered = labels.copy()
        outlier_mask = recovered == -1
        if not np.any(outlier_mask) or len(centroids) == 0:
            return recovered, 0, int(np.sum(outlier_mask))

        norm_pts = normalize(projected[outlier_mask], norm="l2")
        norm_cents = normalize(centroids, norm="l2")
        sims = norm_pts @ norm_cents.T
        best_cluster = np.argmax(sims, axis=1)
        best_sim = np.max(sims, axis=1)

        reassigned_count = 0
        outlier_indices = np.where(outlier_mask)[0]
        for idx, clu_id, sim in zip(outlier_indices, best_cluster, best_sim):
            if sim >= self.outlier_recovery_threshold:
                recovered[idx] = int(clu_id)
                reassigned_count += 1

        remaining_outliers = int(np.sum(recovered == -1))
        return recovered, reassigned_count, remaining_outliers

    def _fit_impl(self, train_data: np.ndarray) -> None:
        if len(train_data) == 0:
            raise ValueError("Discovery requires at least one embedding")
        projected = self._project_embeddings(train_data, is_fit=True)

        if self.strategy == "density":
            db = DBSCAN(eps=0.35, min_samples=3, metric="cosine").fit(projected)
            raw_labels = db.labels_
            unique_valid = [c for c in np.unique(raw_labels) if c != -1]
            if len(unique_valid) < 2:
                # Fallback to aligned K-Means if density parameters collapse
                self.optimal_k_, self.best_silhouette_ = (
                    self.predict_k_with_dropout_and_silhouette(projected)
                )
                km = KMeans(
                    n_clusters=self.optimal_k_,
                    n_init=10,
                    random_state=self.random_state,
                ).fit(projected)
                raw_labels = km.labels_
                raw_centroids = normalize(km.cluster_centers_, norm="l2")
            else:
                self.optimal_k_ = len(unique_valid)
                raw_centroids = normalize(
                    np.vstack(
                        [projected[raw_labels == c].mean(axis=0) for c in unique_valid]
                    ),
                    norm="l2",
                )
                remap = {old_c: new_c for new_c, old_c in enumerate(unique_valid)}
                raw_labels = np.array(
                    [remap.get(int(c), -1) for c in raw_labels], dtype=int
                )
                valid_mask = raw_labels != -1
                self.best_silhouette_ = (
                    float(
                        silhouette_score(
                            projected[valid_mask],
                            raw_labels[valid_mask],
                            metric="cosine",
                        )
                    )
                    if np.sum(valid_mask) > self.optimal_k_
                    else 0.0
                )
        else:
            self.optimal_k_, self.best_silhouette_ = (
                self.predict_k_with_dropout_and_silhouette(projected)
            )
            km = KMeans(
                n_clusters=self.optimal_k_, n_init=10, random_state=self.random_state
            ).fit(projected)
            raw_labels = km.labels_
            raw_centroids = normalize(km.cluster_centers_, norm="l2")
            # Flag extreme low-similarity peripheral points as noise (-1) candidates before outlier recovery
            pt_sims = np.sum(projected * raw_centroids[raw_labels], axis=1)
            raw_labels = np.where(
                pt_sims < (self.outlier_recovery_threshold * 0.65), -1, raw_labels
            )

        normed_raw = normalize(train_data, norm="l2")
        candidate_raw_cents = normalize(
            np.vstack(
                [
                    normed_raw[raw_labels == c].mean(axis=0)
                    if np.any(raw_labels == c)
                    else normed_raw.mean(axis=0)
                    for c in range(len(raw_centroids))
                ]
            ),
            norm="l2",
        )

        aligned_cents, aligned_labels, align_map = self.align_centroids_hungarian(
            raw_centroids,
            raw_labels,
            reference_centroids=self.raw_space_centroids_,
            candidate_raw_centroids=candidate_raw_cents,
        )
        recovered_labels, rec_count, rem_outliers = self.recover_outliers(
            projected, aligned_labels, aligned_cents
        )
        self.train_labels_ = recovered_labels
        self.centroids_ = aligned_cents
        # Store aligned raw-space centroids so subsequent refits align in invariant embedding space
        aligned_raw_cents = np.zeros_like(candidate_raw_cents)
        for raw_idx, canonical_idx in align_map.items():
            aligned_raw_cents[canonical_idx] = candidate_raw_cents[raw_idx]
        self.raw_space_centroids_ = aligned_raw_cents

        self.last_alignment_ = AlignmentResult(
            num_clusters=self.optimal_k_,
            silhouette_score=self.best_silhouette_,
            recovered_outliers=rec_count,
            remaining_outliers=rem_outliers,
            alignment_map=align_map,
        )

    def _transform_impl(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (cluster_ids, cosine_confidences) for any dataset split."""
        assert self.centroids_ is not None
        projected = self._project_embeddings(data, is_fit=False)
        if len(projected) == 0:
            return np.empty(0, dtype=int), np.empty(0)
        norm_cents = normalize(self.centroids_, norm="l2")
        sims = projected @ norm_cents.T
        cluster_ids = np.argmax(sims, axis=1).astype(int)
        confidences = np.clip(np.max(sims, axis=1), 0.0, 1.0)
        return cluster_ids, confidences

    def project_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 2D via PCA for visual scatterplots."""
        normed = normalize(embeddings, norm="l2")
        if normed.shape[0] < 3 or normed.shape[1] < 2:
            return np.zeros((normed.shape[0], 2))
        pca_2d = PCA(n_components=2, random_state=self.random_state)
        return pca_2d.fit_transform(normed)


def extract_c_tfidf_keywords_and_medoids(
    lexical_texts: list[str],
    contextual_texts: list[str] | None = None,
    embeddings: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    top_n_words: int = 8,
    top_n_medoids: int = 3,
    raw_texts: list[str] | None = None,
) -> dict[int, tuple[list[str], list[str]]]:
    """Extracts distinctive class-based TF-IDF keywords and representative medoid texts per cluster."""
    medoid_source = (
        contextual_texts if contextual_texts is not None else (raw_texts or [])
    )
    assert embeddings is not None and cluster_ids is not None
    unique_clusters = sorted(int(c) for c in np.unique(cluster_ids) if c >= 0)
    if not unique_clusters:
        return {}

    cluster_docs: list[str] = []
    for c in unique_clusters:
        docs = [
            lexical_texts[i] for i in range(len(lexical_texts)) if cluster_ids[i] == c
        ]
        joined = " ".join(docs).strip()
        cluster_docs.append(joined if joined else "ticket")

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf_mat = vec.fit_transform(cluster_docs).toarray()
    vocab = np.array(vec.get_feature_names_out())

    norm_emb = normalize(embeddings, norm="l2")
    results: dict[int, tuple[list[str], list[str]]] = {}

    for idx, c in enumerate(unique_clusters):
        row = tfidf_mat[idx]
        top_idx = np.argsort(row)[::-1][:top_n_words]
        keywords = [str(vocab[i]) for i in top_idx if row[i] > 0]

        member_indices = np.where(cluster_ids == c)[0]
        member_vecs = norm_emb[member_indices]
        centroid = normalize(member_vecs.mean(axis=0, keepdims=True), norm="l2")
        sims = (member_vecs @ centroid.T).ravel()
        best_local = np.argsort(sims)[::-1][:top_n_medoids]
        medoids = [medoid_source[int(member_indices[m])] for m in best_local]
        results[c] = (keywords, medoids)

    return results
