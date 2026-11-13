"""Dataset-driven discovery with explicit fitted assignments and frozen prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from ticket_classification.domain.base_stage import LeakageSafePipelineStage
from ticket_classification.models.entities import AlignmentResult


class ClusteringEngine(
    LeakageSafePipelineStage[np.ndarray, tuple[np.ndarray, np.ndarray]]
):
    """Discover groups offline; predict by frozen prototypes online.

    HDBSCAN is sklearn's implementation. Prototype prediction is a separate
    inductive baseline, not HDBSCAN approximate_predict or discovery of new groups.
    """

    def __init__(
        self,
        strategy: str = "aligned_kmeans",
        k_min: int | None = None,
        k_max: int | None = None,
        n_components: int = 10,
        outlier_recovery_threshold: float = 0.55,
        random_state: int = 42,
        reducer: str = "pca",
        min_cluster_support: int = 3,
        candidate_budget: int = 24,
        silhouette_sample_size: int = 1000,
        recover_noise: bool = False,
        single_cluster_min_similarity: float = 0.75,
    ) -> None:
        super().__init__()
        if strategy not in {"aligned_kmeans", "hdbscan_density", "density"}:
            raise ValueError(f"Unsupported discovery strategy: {strategy}")
        if reducer not in {"none", "pca", "umap"}:
            raise ValueError(f"Unsupported reducer: {reducer}")
        if k_min is not None and (k_min < 2 or k_max is not None and k_min > k_max):
            raise ValueError("Invalid K bounds")
        self.strategy, self.k_min, self.k_max = strategy, k_min, k_max
        self.n_components, self.reducer = n_components, reducer
        self.outlier_recovery_threshold = outlier_recovery_threshold
        self.random_state, self.min_cluster_support = random_state, min_cluster_support
        self.candidate_budget, self.silhouette_sample_size = (
            candidate_budget,
            silhouette_sample_size,
        )
        self.recover_noise, self.single_cluster_min_similarity = (
            recover_noise,
            single_cluster_min_similarity,
        )
        self._projector: Any = None
        self.centroids_: np.ndarray | None = None
        self.raw_space_centroids_: np.ndarray | None = None
        self.train_labels_: np.ndarray | None = None
        self.train_strengths_: np.ndarray | None = None
        self.optimal_k_ = self.dropout_viable_k_ = 0
        self.hit_k_max_boundary_ = False
        self.best_silhouette_ = 0.0
        self.silhouette_history_: dict[int, float] = {}
        self.last_alignment_: AlignmentResult | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _project_embeddings(self, embeddings: np.ndarray, is_fit: bool) -> np.ndarray:
        data = np.asarray(embeddings, dtype=float)
        if data.ndim != 2 or not np.isfinite(data).all() or data.shape[1] == 0:
            raise ValueError(
                "Embeddings must be a finite two-dimensional matrix with features"
            )
        if not len(data):
            width = getattr(self, "projected_width_", data.shape[1])
            return np.empty((0, width))
        normed = normalize(data)
        if not is_fit:
            if data.shape[1] != self.input_width_:
                raise ValueError(
                    "Embedding dimensions differ from fitted discovery model"
                )
            return (
                self._projector.transform(normed)
                if self._projector is not None
                else normed
            )
        self.input_width_ = data.shape[1]
        self._projector = None
        n_comp = min(self.n_components, len(data) - 1, data.shape[1])
        if self.reducer == "none" or n_comp < 2 or len(np.unique(normed, axis=0)) < 2:
            projected = normed
        elif self.reducer == "umap":
            try:
                from umap import UMAP
            except ImportError as exc:
                raise ValueError("UMAP requires uv sync --extra discovery") from exc
            if len(data) < 4:
                projected = normed
            else:
                self._projector = UMAP(
                    n_components=n_comp,
                    n_neighbors=min(15, len(data) - 1),
                    metric="cosine",
                    init="random",
                    random_state=self.random_state,
                    transform_seed=self.random_state,
                    n_jobs=1,
                )
                projected = self._projector.fit_transform(normed)
        else:
            self._projector = PCA(n_components=n_comp, random_state=self.random_state)
            projected = self._projector.fit_transform(normed)
        self.projected_width_ = projected.shape[1]
        return projected

    def fit_transform(self, train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.fit(train_data)
        return self.train_labels_.copy(), self.train_strengths_.copy()

    def _quality(self, original: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        valid = labels >= 0
        groups, counts = np.unique(labels[valid], return_counts=True)
        coverage = float(np.mean(valid))
        if len(groups) < 2 or np.sum(valid) <= len(groups):
            return -1.0, 0.0
        # A fixed sample makes candidate comparisons reproducible and bounds pairwise work.
        indices = np.random.default_rng(self.random_state).permutation(len(labels))[
            : self.silhouette_sample_size
        ]
        indices = indices[valid[indices]]
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(
            indices
        ):
            return -1.0, 0.0
        separation = float(
            silhouette_score(original[indices], sample_labels, metric="cosine")
        )
        unsupported = float(
            np.sum(counts[counts < self.min_cluster_support]) / len(labels)
        )
        return separation * coverage - 0.25 * (1 - coverage) - unsupported, separation

    def predict_k_with_dropout_and_silhouette(
        self, projected: np.ndarray
    ) -> tuple[int, float]:
        """Adaptive K-Means baseline; method name retained for source compatibility."""
        original = self._fit_original
        unique_count = len(np.unique(projected, axis=0))
        feasible = min(
            unique_count,
            max(1, len(projected) // self.min_cluster_support),
            len(projected) - 1,
        )
        if feasible < 2:
            return 1, 0.0
        lower = min(self.k_min or 2, feasible)
        # This is an explicit compute ceiling, not a claimed upper bound on true intents.
        upper = min(self.k_max or feasible, feasible)
        candidates = set(
            np.geomspace(
                lower, upper, min(max(2, self.candidate_budget // 2), upper - lower + 1)
            )
            .round()
            .astype(int)
        )
        evaluated: dict[int, tuple[float, float, float]] = {}

        def evaluate(k: int) -> None:
            if (
                k in evaluated
                or len(evaluated) >= self.candidate_budget
                or not lower <= k <= upper
            ):
                return
            km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state).fit(
                projected
            )
            other = KMeans(
                n_clusters=k, n_init=3, random_state=self.random_state + 1
            ).fit(projected)
            quality, separation = self._quality(original, km.labels_)
            stability = float(adjusted_rand_score(km.labels_, other.labels_))
            evaluated[k] = (quality + 0.1 * stability, separation, stability)

        for k in sorted(candidates):
            evaluate(int(k))
        while len(evaluated) < self.candidate_budget:
            best = max(evaluated, key=lambda k: evaluated[k][0])
            ordered = sorted(evaluated)
            neighbors = [lower - 1, *ordered, upper + 1]
            pos = neighbors.index(best)
            additions = [
                (neighbors[pos - 1] + best) // 2,
                (best + neighbors[pos + 1] + 1) // 2,
            ]
            before = len(evaluated)
            for k in additions:
                evaluate(k)
            if len(evaluated) == before:
                break
        best = max(evaluated, key=lambda k: evaluated[k][0])
        self.silhouette_history_ = {
            k: result[1] for k, result in sorted(evaluated.items())
        }
        self.hit_k_max_boundary_ = best == upper
        self.dropout_viable_k_ = best
        self.diagnostics_.update(
            candidate_bounds=[lower, upper],
            candidate_budget=self.candidate_budget,
            budget_exhausted=len(evaluated) >= self.candidate_budget,
            candidates=[
                {"k": k, "score": v[0], "silhouette": v[1], "seed_stability_ari": v[2]}
                for k, v in sorted(evaluated.items())
            ],
        )
        return best, evaluated[best][1]

    def align_centroids_hungarian(
        self,
        new_centroids: np.ndarray,
        labels: np.ndarray,
        reference_centroids: np.ndarray | None = None,
        candidate_raw_centroids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
        ref = (
            reference_centroids if reference_centroids is not None else self.centroids_
        )
        cand = (
            candidate_raw_centroids
            if candidate_raw_centroids is not None
            else new_centroids
        )
        if ref is None or ref.shape != cand.shape:
            return new_centroids, labels, {i: i for i in range(len(new_centroids))}
        rows, columns = linear_sum_assignment(
            np.linalg.norm(ref[:, None, :] - cand[None, :, :], axis=2)
        )
        mapping = {int(c): int(r) for r, c in zip(rows, columns)}
        aligned = np.zeros_like(new_centroids)
        for source, target in mapping.items():
            aligned[target] = new_centroids[source]
        return aligned, np.array([mapping.get(int(x), int(x)) for x in labels]), mapping

    def recover_outliers(
        self, projected: np.ndarray, labels: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        recovered = labels.copy()
        indices = np.where(labels == -1)[0]
        if not len(indices) or not len(centroids):
            return recovered, 0, len(indices)
        sims = normalize(projected[indices]) @ normalize(centroids).T
        accepted = sims.max(axis=1) >= self.outlier_recovery_threshold
        recovered[indices[accepted]] = sims.argmax(axis=1)[accepted]
        return recovered, int(accepted.sum()), int(np.sum(recovered == -1))

    def _fit_impl(self, train_data: np.ndarray) -> None:
        if not len(train_data):
            raise ValueError("Discovery requires at least one embedding")
        projected = self._project_embeddings(train_data, is_fit=True)
        original = normalize(train_data)
        self._fit_original = original
        self.diagnostics_ = {
            "strategy": self.strategy,
            "reducer": self.reducer,
            "prediction": "frozen_centroid_cosine",
            "input_rows": len(train_data),
            "unique_embeddings": len(np.unique(original, axis=0)),
        }
        self.silhouette_history_ = {}
        self.hit_k_max_boundary_ = False
        labels = np.full(len(projected), -1, dtype=int)
        strengths = np.zeros(len(projected))
        if not np.any(original):
            self.diagnostics_["outcome"] = "no_usable_vectors"
        elif (
            len(projected) < self.min_cluster_support
            or len(np.unique(original, axis=0)) == 1
        ):
            labels[:] = 0
            strengths[:] = 1.0
            self.diagnostics_["outcome"] = "single_cluster"
        elif self.strategy == "aligned_kmeans":
            k, self.best_silhouette_ = self.predict_k_with_dropout_and_silhouette(
                projected
            )
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(
                projected
            )
            labels = km.labels_
            centers = normalize(
                np.vstack([original[labels == c].mean(axis=0) for c in range(k)])
            )
            strengths = np.sum(original * centers[labels], axis=1).clip(0, 1)
            labels[strengths < self.outlier_recovery_threshold * 0.65] = -1
        elif self.strategy == "density":
            labels = DBSCAN(
                eps=0.35, min_samples=self.min_cluster_support, metric="cosine"
            ).fit_predict(normalize(projected))
            strengths = (labels >= 0).astype(float)
        else:
            best_score = -np.inf
            sizes = sorted(
                {
                    self.min_cluster_support,
                    max(self.min_cluster_support, int(np.sqrt(len(projected)))),
                    max(self.min_cluster_support, len(projected) // 20),
                }
            )
            trials = []
            for size in sizes:
                if size > len(projected):
                    continue
                for samples in sorted({2, min(size, 5)}):
                    if len(trials) >= self.candidate_budget:
                        break
                    model = HDBSCAN(
                        min_cluster_size=size,
                        min_samples=samples,
                        allow_single_cluster=False,
                        copy=True,
                    ).fit(projected)
                    score, separation = self._quality(original, model.labels_)
                    subset = np.sort(
                        np.random.default_rng(self.random_state).choice(
                            len(projected),
                            max(size, int(len(projected) * 0.8)),
                            replace=False,
                        )
                    )
                    repeated = HDBSCAN(
                        min_cluster_size=size,
                        min_samples=samples,
                        allow_single_cluster=False,
                        copy=True,
                    ).fit_predict(projected[subset])
                    stability = float(
                        adjusted_rand_score(model.labels_[subset], repeated)
                    )
                    score += 0.1 * stability * float(np.mean(model.labels_ >= 0))
                    trials.append(
                        {
                            "min_cluster_size": size,
                            "min_samples": samples,
                            "k": len(set(model.labels_) - {-1}),
                            "score": score,
                            "coverage": float(np.mean(model.labels_ >= 0)),
                            "silhouette": separation,
                            "resampling_stability_ari": stability,
                        }
                    )
                    if score > best_score:
                        best_score, labels, strengths = (
                            score,
                            model.labels_.copy(),
                            model.probabilities_.copy(),
                        )
                        self.best_silhouette_ = separation
            self.diagnostics_["candidates"] = trials
            self.diagnostics_["budget_exhausted"] = len(trials) >= self.candidate_budget
            if not np.any(labels >= 0):
                center = normalize(original.mean(axis=0, keepdims=True))[0]
                if (
                    float(np.min(original @ center))
                    >= self.single_cluster_min_similarity
                ):
                    labels[:] = 0
                    strengths = (original @ center).clip(0, 1)
        # Zero vectors cannot support semantic discovery.
        labels[np.linalg.norm(train_data, axis=1) == 0] = -1
        groups = sorted(set(labels) - {-1})
        remap = {int(c): i for i, c in enumerate(groups)}
        labels = np.array([remap.get(int(c), -1) for c in labels])
        self.optimal_k_ = len(groups)
        cents = (
            normalize(
                np.vstack(
                    [projected[labels == c].mean(axis=0) for c in range(len(groups))]
                )
            )
            if groups
            else np.empty((0, projected.shape[1]))
        )
        raw = (
            normalize(
                np.vstack(
                    [original[labels == c].mean(axis=0) for c in range(len(groups))]
                )
            )
            if groups
            else np.empty((0, original.shape[1]))
        )
        cents, labels, mapping = self.align_centroids_hungarian(
            cents, labels, self.raw_space_centroids_, raw
        )
        raw_aligned = np.zeros_like(raw)
        for source, target in mapping.items():
            raw_aligned[target] = raw[source]
        recovered = 0
        if self.recover_noise:
            labels, recovered, _ = self.recover_outliers(original, labels, raw_aligned)
        self.centroids_, self.raw_space_centroids_ = cents, raw_aligned
        self.train_labels_ = labels
        self.train_strengths_ = np.where(labels >= 0, strengths, 0.0)
        self.last_alignment_ = AlignmentResult(
            self.optimal_k_,
            self.best_silhouette_,
            recovered,
            int(np.sum(labels == -1)),
            mapping,
        )
        self.diagnostics_.update(
            selected_k=self.optimal_k_,
            coverage=float(np.mean(labels >= 0)),
            outcome="clusters" if groups else "all_noise",
            hit_boundary=self.hit_k_max_boundary_,
        )
        del self._fit_original

    def _transform_impl(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._project_embeddings(data, is_fit=False)  # validate fitted feature contract
        if not len(data):
            return np.empty(0, dtype=int), np.empty(0)
        if not len(self.raw_space_centroids_):
            return np.full(len(data), -1), np.zeros(len(data))
        sims = normalize(data) @ self.raw_space_centroids_.T
        strengths = np.max(sims, axis=1).clip(0, 1)
        return np.where(
            strengths >= self.outlier_recovery_threshold * 0.65, sims.argmax(axis=1), -1
        ), strengths

    def project_2d(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 3 or embeddings.shape[1] < 2:
            return np.zeros((len(embeddings), 2))
        return PCA(n_components=2, random_state=self.random_state).fit_transform(
            normalize(embeddings)
        )


def extract_c_tfidf_keywords_and_medoids(
    lexical_texts: list[str],
    contextual_texts: list[str] | None = None,
    embeddings: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    top_n_words: int = 8,
    top_n_medoids: int = 3,
    raw_texts: list[str] | None = None,
) -> dict[int, tuple[list[str], list[str]]]:
    """Extracts distinctive class-based TF-IDF keywords and centroid-nearest representative texts (legacy medoid field name) per cluster."""
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

    vec = CountVectorizer(
        max_features=3000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    counts = vec.fit_transform(cluster_docs).toarray().astype(float)
    term_frequency = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    inverse_frequency = np.log1p(
        counts.sum(axis=1).mean() / np.maximum(counts.sum(axis=0), 1)
    )
    tfidf_mat = term_frequency * inverse_frequency
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
