"""Application Service Layer orchestrating Ingestion, Cleaning, Cloud Embeddings, Clustering, Classification, and Evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from llm_gateway_sdk.exceptions import GatewaySDKError
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from ticket_classification.config.settings import AppSettings, LLMProvider, ModelTier
from ticket_classification.domain.classification.engine import ClassifierSuite
from ticket_classification.domain.cleaning.handlers import build_default_cleaning_chain
from ticket_classification.domain.clustering.engine import (
    ClusteringEngine,
    extract_c_tfidf_keywords_and_medoids,
)
from ticket_classification.domain.features.metadata_encoder import (
    MetadataFeaturePipeline,
)
from ticket_classification.gateways.llm_gateway_adapter import BaseCloudGateway
from ticket_classification.models.entities import (
    ClusterProfile,
    PredictionResult,
    TicketRecord,
)
from ticket_classification.models.schemas import (
    ClusterTaxonomyLLMOutput,
    RerankDecisionLLMOutput,
)
from ticket_classification.repositories.dataset_repo import TicketDatasetRepository
from ticket_classification.repositories.embedding_cache_repo import (
    EmbeddingCacheRepository,
)
from ticket_classification.repositories.model_registry_repo import (
    ModelRegistryRepository,
)

OOS_PROBE_TICKETS: tuple[str, ...] = (
    "What is the best recipe for homemade sourdough pizza dough with mozzarella?",
    "Can you check the weather forecast and UV index in Tokyo this weekend?",
    "Who won the Formula 1 Monaco Grand Prix championship in 1998?",
    "My office espresso machine grinder is jammed with dark roast coffee beans.",
    "Play classical Beethoven symphony music on the living room speaker.",
    "How many calories are in a medium avocado and scrambled egg breakfast?",
)


class TicketPipelineService:
    """Orchestrates the end-to-end clustering-to-classification lifecycle."""

    def __init__(
        self,
        settings: AppSettings,
        gateway: BaseCloudGateway,
        dataset_repo: TicketDatasetRepository,
        cache_repo: EmbeddingCacheRepository,
        model_registry_repo: ModelRegistryRepository | None = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.dataset_repo = dataset_repo
        self.cache_repo = cache_repo
        self.model_registry_repo = model_registry_repo or ModelRegistryRepository(
            f"{self.settings.data_dir}/artifacts/model_registry.joblib"
        )
        self.cleaning_chain = build_default_cleaning_chain()

        self.metadata_pipeline: MetadataFeaturePipeline | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.clustering_engine: ClusteringEngine | None = None
        self.classifier_suite: ClassifierSuite | None = None
        self.cluster_profiles: dict[int, ClusterProfile] = {}
        self.cluster_to_gt_label: dict[int, str] = {}
        self.fitted_embedding_provider: LLMProvider | None = None
        self.latest_metrics: dict[str, Any] = {}

        self.load_persisted_state()

    def save_persisted_state(self) -> str:
        """Persist fitted models, encoders, centroids, and taxonomy to ModelRegistryRepository."""
        bundle = {
            "metadata_pipeline": self.metadata_pipeline,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "clustering_engine": self.clustering_engine,
            "classifier_suite": self.classifier_suite,
            "cluster_profiles": self.cluster_profiles,
            "cluster_to_gt_label": self.cluster_to_gt_label,
            "fitted_embedding_provider": self.fitted_embedding_provider,
            "latest_metrics": self.latest_metrics,
        }
        return self.model_registry_repo.save_bundle(bundle)

    def load_persisted_state(self) -> bool:
        """Restore fitted pipeline state from ModelRegistryRepository if available on disk."""
        bundle = self.model_registry_repo.load_bundle()
        if not bundle:
            return False
        self.metadata_pipeline = bundle.get("metadata_pipeline")
        self.tfidf_vectorizer = bundle.get("tfidf_vectorizer")
        self.clustering_engine = bundle.get("clustering_engine")
        self.classifier_suite = bundle.get("classifier_suite")
        self.cluster_profiles = bundle.get("cluster_profiles", {})
        self.cluster_to_gt_label = bundle.get("cluster_to_gt_label", {})
        self.fitted_embedding_provider = bundle.get("fitted_embedding_provider")
        self.latest_metrics = bundle.get("latest_metrics", {})
        return self.clustering_engine is not None and self.classifier_suite is not None

    def switch_provider(self, provider: LLMProvider | str) -> dict[str, Any]:
        """Switch active cloud provider ('litellm', 'openai', or 'gemini')."""
        from llm_gateway_sdk.integrations.pydantic_settings import connection_options

        self.gateway.switch_provider(provider)
        active = self.gateway.get_active_provider()
        opts = connection_options(self.settings)
        return {
            "active_provider": active.value,
            "gateway_base_url": str(opts.get("api_base", "")),
            "models_by_tier": {
                tier.value: self.settings.resolve_model(tier, provider_override=active)
                for tier in ModelTier
            },
            "embedding_identity": self.settings.embedding_identity(
                provider_override=active
            ),
        }

    def audit_and_split_dataset(
        self,
        df: pd.DataFrame,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Audits dataset schema, tracks missing values, and produces stratified splits without leakage."""
        total_rows = len(df)
        missing_text = int(
            df["raw_text"].isna().sum()
            + (df["raw_text"].fillna("").str.strip() == "").sum()
        )
        missing_label = int(df["label"].isna().sum()) if "label" in df.columns else 0

        df = df.copy()
        df["is_text_missing"] = (
            df["raw_text"].isna() | (df["raw_text"].fillna("").str.strip() == "")
        ).astype(int)
        df["raw_text"] = df["raw_text"].fillna("empty_ticket_narrative").astype(str)

        # Strict split BEFORE any featurization
        if "split" not in df.columns or set(df["split"].unique()) == {"unassigned"}:
            strat_col = (
                df["label"]
                if "label" in df.columns and df["label"].nunique() > 1
                else None
            )
            try:
                train_df, temp_df = train_test_split(
                    df, test_size=0.30, stratify=strat_col, random_state=random_state
                )
                strat_temp = temp_df["label"] if strat_col is not None else None
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=0.50,
                    stratify=strat_temp,
                    random_state=random_state,
                )
            except ValueError:
                train_df, temp_df = train_test_split(
                    df, test_size=0.30, random_state=random_state
                )
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.50, random_state=random_state
                )

            train_df = train_df.copy()
            val_df = val_df.copy()
            test_df = test_df.copy()
            train_df["split"] = "train"
            val_df = val_df.assign(split="val")
            test_df = test_df.assign(split="test")
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        audit_summary = {
            "total_rows": total_rows,
            "missing_text_rows_retained_with_indicator": missing_text,
            "missing_label_rows": missing_label,
            "split_counts": df["split"].value_counts().to_dict(),
            "unique_ground_truth_labels": int(df["label"].nunique())
            if "label" in df.columns
            else 0,
        }
        return df, audit_summary

    def clean_records(self, df: pd.DataFrame) -> list[TicketRecord]:
        """Applies text cleaning and normalization across input records."""
        records: list[TicketRecord] = []
        for row in df.to_dict(orient="records"):
            rec = TicketRecord(
                ticket_id=str(row.get("ticket_id", "")),
                raw_text=str(row.get("raw_text", "")),
                ground_truth_label=str(row["label"])
                if row.get("label") is not None
                else None,
                split=str(row.get("split", "train")),
                metadata={
                    k: v
                    for k, v in row.items()
                    if k not in {"ticket_id", "raw_text", "label", "split"}
                },
            )
            self.cleaning_chain.handle(rec)
            records.append(rec)
        return records

    def embed_with_dedup_and_cache(
        self,
        texts: list[str],
        batch_size: int = 64,
        provider_override: LLMProvider | str | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Deduplicates input texts, looks up cached vectors, and generates missing embeddings via gateway."""
        emb_id = self.settings.embedding_identity(provider_override=provider_override)
        unique_texts = list(dict.fromkeys(texts))
        cached_map = self.cache_repo.get_many(emb_id, unique_texts)

        missing_texts = [
            t for t in unique_texts if self.cache_repo.hash_text(t) not in cached_map
        ]

        api_calls = 0
        new_text_to_vec: dict[str, list[float]] = {}
        for i in range(0, len(missing_texts), batch_size):
            batch = missing_texts[i : i + batch_size]
            vectors = self.gateway.embed_texts(
                batch, provider_override=provider_override
            )
            api_calls += 1
            for text_item, vec in zip(batch, vectors):
                new_text_to_vec[text_item] = vec
                cached_map[self.cache_repo.hash_text(text_item)] = vec

        if new_text_to_vec:
            self.cache_repo.put_many(emb_id, new_text_to_vec)

        matrix = np.array(
            [cached_map[self.cache_repo.hash_text(t)] for t in texts],
            dtype=float,
        )
        stats = {
            "total_texts": len(texts),
            "unique_texts_after_stage1_dedup": len(unique_texts),
            "cache_hits": len(unique_texts) - len(missing_texts),
            "gateway_embedded_texts": len(missing_texts),
            "gateway_batch_requests": api_calls,
            "embedding_dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        }
        return matrix, stats

    def _name_cluster_via_gateway(
        self,
        cluster_id: int,
        keywords: list[str],
        medoids: list[str],
        use_llm: bool = True,
        provider_override: LLMProvider | str | None = None,
    ) -> tuple[str, str, str]:
        """Generate human-readable cluster taxonomy via Cloud LLM (with deterministic c-TF-IDF fallback)."""
        fallback_label = (
            " / ".join(w.title() for w in keywords[:3]) or f"Cluster {cluster_id}"
        )
        fallback_sig = f"Tickets regarding {', '.join(keywords[:5])}"
        fallback_queue = (
            f"{keywords[0].title()} Support Queue" if keywords else "General Triage"
        )

        if not use_llm:
            return fallback_label, fallback_sig, fallback_queue

        role = self.settings.system_role_description
        domain = self.settings.domain_context
        system_prompt = (
            f"You are an expert {role} in the domain of {domain}. "
            "Given top distinctive keywords and representative text items of a discovered cluster, "
            "produce a concise category title, a 50-150 character signature summary, and the target routing queue or responsible department."
        )
        user_prompt = (
            f"Cluster ID: {cluster_id}\n"
            f"Top Keywords: {', '.join(keywords)}\n"
            f"Representative Items:\n" + "\n".join(f"- {m[:200]}" for m in medoids[:3])
        )
        try:
            out = self.gateway.complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_cls=ClusterTaxonomyLLMOutput,
                tier=ModelTier.CHEAP,
                provider_override=provider_override,
            )
            return out.short_label, out.signature, out.routing_queue
        except (GatewaySDKError, ValueError, TypeError, KeyError):
            # GatewaySDKError covers timeouts/429/5xx; ValueError covers pydantic ValidationError.
            return fallback_label, fallback_sig, fallback_queue

    @staticmethod
    def _compute_bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Any,
        n_bootstraps: int = 200,
        random_state: int = 42,
    ) -> tuple[float, float, float]:
        """Compute point estimate and 95% confidence interval via bootstrapping."""
        rng = np.random.default_rng(random_state)
        point = float(metric_fn(y_true, y_pred))
        scores: list[float] = []
        n = len(y_true)
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        low, high = np.percentile(scores, [2.5, 97.5])
        return point, float(low), float(high)

    @staticmethod
    def _hungarian_cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute unsupervised clustering accuracy using Hungarian optimal assignment."""
        true_labels, true_inv = np.unique(y_true, return_inverse=True)
        pred_labels, pred_inv = np.unique(y_pred, return_inverse=True)
        dim = max(len(true_labels), len(pred_labels))
        w = np.zeros((dim, dim), dtype=np.int64)
        for t, p in zip(true_inv, pred_inv):
            w[p, t] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return float(sum(w[r, c] for r, c in zip(row_ind, col_ind)) / len(y_true))

    def run_benchmark_pipeline(
        self,
        dataset_name: str = "textoir_banking77",
        max_samples: int = 400,
        top_n_classes: int = 10,
        k_min: int = 5,
        k_max: int = 14,
        use_llm_naming: bool = True,
        provider_override: LLMProvider | str | None = None,
    ) -> dict[str, Any]:
        """Runs dataset ingestion, cleaning, clustering, classifier training, and benchmarking."""
        t0 = time.perf_counter()

        # Ingestion, schema audit, and split
        raw_df = self.dataset_repo.load_benchmark(
            dataset_name=dataset_name,
            max_samples=max_samples,
            top_n_classes=top_n_classes,
        )
        split_df, audit_info = self.audit_and_split_dataset(raw_df)

        # Text cleaning and metadata encoding
        records = self.clean_records(split_df)
        split_df["text_contextual"] = [r.text_contextual for r in records]
        split_df["text_lexical"] = [r.text_lexical for r in records]
        split_df["template_signature"] = [r.template_signature for r in records]

        train_mask = (split_df["split"] == "train").to_numpy()
        val_mask = (split_df["split"] == "val").to_numpy()
        test_mask = (split_df["split"] == "test").to_numpy()

        self.metadata_pipeline = MetadataFeaturePipeline()
        meta_train = self.metadata_pipeline.fit_transform(split_df[train_mask])
        meta_val = self.metadata_pipeline.transform(split_df[val_mask])
        meta_test = self.metadata_pipeline.transform(split_df[test_mask])

        # Lexical term frequency fitting on training partition
        self.tfidf_vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
        tfidf_train = self.tfidf_vectorizer.fit_transform(
            split_df.loc[train_mask, "text_lexical"]
        ).toarray()
        tfidf_val = self.tfidf_vectorizer.transform(
            split_df.loc[val_mask, "text_lexical"]
        ).toarray()
        tfidf_test = self.tfidf_vectorizer.transform(
            split_df.loc[test_mask, "text_lexical"]
        ).toarray()

        # Embedding generation with caching
        active_provider = (
            LLMProvider(provider_override.lower())
            if isinstance(provider_override, str)
            else (provider_override or self.gateway.get_active_provider())
        )
        self.fitted_embedding_provider = active_provider

        all_embeddings, embed_stats = self.embed_with_dedup_and_cache(
            split_df["text_contextual"].tolist(),
            provider_override=self.fitted_embedding_provider,
        )
        emb_train = all_embeddings[train_mask]
        emb_val = all_embeddings[val_mask]
        emb_test = all_embeddings[test_mask]

        # Unsupervised clustering and Hungarian centroid alignment
        self.clustering_engine = ClusteringEngine(
            strategy="aligned_kmeans",
            k_min=k_min,
            k_max=k_max,
            n_components=min(15, emb_train.shape[0] // 3),
        )
        train_clusters, train_confs = self.clustering_engine.fit_transform(emb_train)
        val_clusters, _val_confs = self.clustering_engine.transform(emb_val)
        test_clusters, _test_confs = self.clustering_engine.transform(emb_test)

        # Template signature deduplication summary
        unique_templates = int(split_df["template_signature"].nunique())

        # Discovered cluster to ground truth label mapping on training split
        self.cluster_to_gt_label = {}
        if "label" in split_df.columns and split_df["label"].nunique() > 1:
            train_gt = split_df.loc[train_mask, "label"].astype(str).to_numpy()
            for cid in np.unique(train_clusters):
                if cid < 0:
                    continue
                mask_c = train_clusters == cid
                if np.any(mask_c):
                    labels_in_c, counts = np.unique(
                        train_gt[mask_c], return_counts=True
                    )
                    self.cluster_to_gt_label[int(cid)] = str(
                        labels_in_c[np.argmax(counts)]
                    )

        # Cluster taxonomy generation using c-TF-IDF and model naming
        # Contextual masked texts are used to avoid sending raw sensitive tokens
        kw_medoids = extract_c_tfidf_keywords_and_medoids(
            lexical_texts=split_df.loc[train_mask, "text_lexical"].tolist(),
            contextual_texts=split_df.loc[train_mask, "text_contextual"].tolist(),
            embeddings=emb_train,
            cluster_ids=train_clusters,
        )
        self.cluster_profiles = {}
        for cid, (keywords, medoids) in kw_medoids.items():
            short_lbl, sig, queue = self._name_cluster_via_gateway(
                cluster_id=cid,
                keywords=keywords,
                medoids=medoids,
                use_llm=use_llm_naming,
                provider_override=provider_override,
            )
            size = int(np.sum(train_clusters == cid))
            self.cluster_profiles[cid] = ClusterProfile(
                cluster_id=cid,
                size=size,
                top_keywords=keywords,
                medoid_texts=medoids,
                short_label=short_lbl,
                signature=sig,
                routing_queue=queue,
                centroid=self.clustering_engine.centroids_[cid].tolist()
                if self.clustering_engine.centroids_ is not None
                else [],
            )

        # Downstream supervised classification suite
        X_train = np.hstack(
            [normalize(emb_train, norm="l2"), tfidf_train * 0.25, meta_train * 0.05]
        )
        X_val = np.hstack(
            [normalize(emb_val, norm="l2"), tfidf_val * 0.25, meta_val * 0.05]
        )
        X_test = np.hstack(
            [normalize(emb_test, norm="l2"), tfidf_test * 0.25, meta_test * 0.05]
        )

        self.classifier_suite = ClassifierSuite()
        self.classifier_suite.fit_all(
            X_train=X_train,
            embeddings_train=emb_train,
            cluster_ids_train=train_clusters,
            confidences_train=train_confs,
        )

        # Model evaluation with bootstrap confidence intervals and ground truth metrics
        has_gt = "label" in split_df.columns and split_df["label"].nunique() > 1
        y_test_gt = (
            split_df.loc[test_mask, "label"].astype(str).to_numpy()
            if has_gt
            else np.array([])
        )

        model_benchmarks: dict[str, dict[str, Any]] = {}
        best_val_f1 = -1.0
        best_model_name = "logistic_regression"

        for m_name in self.classifier_suite.models_:
            t_infer0 = time.perf_counter()
            val_preds, _, _, _, _ = self.classifier_suite.predict_with_open_set(
                X_val, emb_val, model_name=m_name
            )
            test_preds, _test_p_conf, _test_margins, _test_open_set, _ = (
                self.classifier_suite.predict_with_open_set(
                    X_test, emb_test, model_name=m_name
                )
            )
            infer_ms_per_sample = ((time.perf_counter() - t_infer0) * 1000.0) / max(
                1, len(X_val) + len(X_test)
            )

            val_f1 = float(
                f1_score(val_clusters, val_preds, average="macro", zero_division=0)
            )
            if val_f1 > best_val_f1 and m_name != "naive_majority":
                best_val_f1 = val_f1
                best_model_name = m_name

            # Agreement with clustering
            acc_pt, acc_lo, acc_hi = self._compute_bootstrap_ci(
                test_clusters, test_preds, accuracy_score
            )
            f1_pt, f1_lo, f1_hi = self._compute_bootstrap_ci(
                test_clusters,
                test_preds,
                lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            )
            wf1_pt = float(
                f1_score(test_clusters, test_preds, average="weighted", zero_division=0)
            )
            prec_pt = float(
                precision_score(
                    test_clusters, test_preds, average="macro", zero_division=0
                )
            )
            rec_pt = float(
                recall_score(
                    test_clusters, test_preds, average="macro", zero_division=0
                )
            )

            m_metrics: dict[str, Any] = {
                "val_macro_f1": round(val_f1, 4),
                "test_cluster_agreement_acc": round(acc_pt, 4),
                "test_accuracy": round(acc_pt, 4),
                "test_accuracy_95ci": [round(acc_lo, 4), round(acc_hi, 4)],
                "test_macro_f1": round(f1_pt, 4),
                "test_macro_f1_95ci": [round(f1_lo, 4), round(f1_hi, 4)],
                "test_weighted_f1": round(wf1_pt, 4),
                "test_macro_precision": round(prec_pt, 4),
                "test_macro_recall": round(rec_pt, 4),
                "latency_ms_per_ticket": round(infer_ms_per_sample, 4),
            }

            # Non-circular True Ground-Truth Classification Metrics
            if has_gt and len(self.cluster_to_gt_label) > 0:
                mapped_test_preds = np.array(
                    [
                        self.cluster_to_gt_label.get(int(p), "unassigned")
                        for p in test_preds
                    ]
                )
                gt_acc_pt, gt_acc_lo, gt_acc_hi = self._compute_bootstrap_ci(
                    y_test_gt, mapped_test_preds, accuracy_score
                )
                gt_f1_pt, gt_f1_lo, gt_f1_hi = self._compute_bootstrap_ci(
                    y_test_gt,
                    mapped_test_preds,
                    lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
                )
                m_metrics["test_ground_truth_accuracy"] = round(gt_acc_pt, 4)
                m_metrics["test_ground_truth_accuracy_95ci"] = [
                    round(gt_acc_lo, 4),
                    round(gt_acc_hi, 4),
                ]
                m_metrics["test_ground_truth_macro_f1"] = round(gt_f1_pt, 4)
                m_metrics["test_ground_truth_macro_f1_95ci"] = [
                    round(gt_f1_lo, 4),
                    round(gt_f1_hi, 4),
                ]

            model_benchmarks[m_name] = m_metrics

        self.classifier_suite.best_model_name_ = best_model_name

        # Slice-based error analysis on test set using the winning model
        best_test_preds, _, _, best_open_flags, _ = (
            self.classifier_suite.predict_with_open_set(
                X_test, emb_test, model_name=best_model_name
            )
        )
        test_lengths = split_df.loc[test_mask, "raw_text"].str.len().to_numpy()
        median_len = float(np.median(test_lengths))
        short_mask = test_lengths <= median_len
        long_mask = test_lengths > median_len

        # Quantitative Open-Set Evaluation using probe tickets
        oos_embeddings, _ = self.embed_with_dedup_and_cache(
            list(OOS_PROBE_TICKETS), provider_override=self.fitted_embedding_provider
        )
        norm_cents = normalize(self.clustering_engine.centroids_, norm="l2")
        oos_proj = self.clustering_engine._project_embeddings(
            oos_embeddings, is_fit=False
        )
        oos_sims = oos_proj @ norm_cents.T
        oos_dists = 1.0 - np.max(oos_sims, axis=1)
        oos_flagged = int(
            np.sum(oos_dists >= self.classifier_suite.distance_threshold_)
        )
        oos_recall = round(float(oos_flagged / max(1, len(OOS_PROBE_TICKETS))), 4)
        in_dist_fpr = round(float(np.mean(best_open_flags)), 4)

        slice_analysis = {
            "short_tickets_lte_median_chars": {
                "count": int(np.sum(short_mask)),
                "macro_f1": round(
                    float(
                        f1_score(
                            test_clusters[short_mask],
                            best_test_preds[short_mask],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "accuracy": round(
                    float(
                        accuracy_score(
                            test_clusters[short_mask], best_test_preds[short_mask]
                        )
                    ),
                    4,
                ),
            },
            "long_tickets_gt_median_chars": {
                "count": int(np.sum(long_mask)),
                "macro_f1": round(
                    float(
                        f1_score(
                            test_clusters[long_mask],
                            best_test_preds[long_mask],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "accuracy": round(
                    float(
                        accuracy_score(
                            test_clusters[long_mask], best_test_preds[long_mask]
                        )
                    ),
                    4,
                ),
            },
            "open_set_flagged_count_on_test": int(np.sum(best_open_flags)),
            "open_set_in_distribution_fpr": in_dist_fpr,
            "open_set_oos_recall": oos_recall,
            "open_set_probes_tested": len(OOS_PROBE_TICKETS),
        }

        # External clustering validity vs ground-truth labels (Test-only AND Full Split)
        external_clustering_metrics: dict[str, float] = {}
        if has_gt:
            gt_all = split_df["label"].astype(str).to_numpy()
            all_cluster_ids = np.zeros(len(split_df), dtype=int)
            all_cluster_ids[train_mask] = train_clusters
            all_cluster_ids[val_mask] = val_clusters
            all_cluster_ids[test_mask] = test_clusters

            # Test-only metrics
            gt_test = split_df.loc[test_mask, "label"].astype(str).to_numpy()
            external_clustering_metrics = {
                "test_NMI_vs_ground_truth": round(
                    float(normalized_mutual_info_score(gt_test, test_clusters)), 4
                ),
                "test_ARI_vs_ground_truth": round(
                    float(adjusted_rand_score(gt_test, test_clusters)), 4
                ),
                "test_Hungarian_ACC_vs_ground_truth": round(
                    self._hungarian_cluster_accuracy(gt_test, test_clusters), 4
                ),
                "full_NMI_vs_ground_truth": round(
                    float(normalized_mutual_info_score(gt_all, all_cluster_ids)), 4
                ),
                "full_ARI_vs_ground_truth": round(
                    float(adjusted_rand_score(gt_all, all_cluster_ids)), 4
                ),
                "full_Hungarian_ACC_vs_ground_truth": round(
                    self._hungarian_cluster_accuracy(gt_all, all_cluster_ids), 4
                ),
            }

        coords_2d = self.clustering_engine.project_2d(emb_train)
        cm = confusion_matrix(test_clusters, best_test_preds).tolist()

        active_provider = (
            LLMProvider(provider_override.lower())
            if isinstance(provider_override, str)
            else (provider_override or self.gateway.get_active_provider())
        )

        self.latest_metrics = {
            "dataset": dataset_name,
            "provider_used": active_provider.value,
            "embedding_model": self.settings.resolve_model(
                ModelTier.EMBEDDING, active_provider
            ),
            "chat_model": self.settings.resolve_model(ModelTier.CHEAP, active_provider),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "audit_and_split": audit_info,
            "embedding_stats": {
                **embed_stats,
                "unique_templates_after_stage2_funnel": unique_templates,
            },
            "clustering": {
                "optimal_k": self.clustering_engine.optimal_k_,
                "dropout_viable_k": self.clustering_engine.dropout_viable_k_,
                "hit_k_max_boundary": self.clustering_engine.hit_k_max_boundary_,
                "best_silhouette_cosine": round(
                    self.clustering_engine.best_silhouette_, 4
                ),
                "silhouette_sweep": {
                    str(k): round(v, 4)
                    for k, v in self.clustering_engine.silhouette_history_.items()
                },
                "alignment": asdict(self.clustering_engine.last_alignment_)
                if self.clustering_engine.last_alignment_
                else {},
                "external_validity": external_clustering_metrics,
                "clusters": [asdict(p) for p in self.cluster_profiles.values()],
            },
            "classification": {
                "selected_production_model": best_model_name,
                "open_set_cosine_distance_threshold": round(
                    self.classifier_suite.distance_threshold_, 4
                ),
                "model_comparison": model_benchmarks,
                "slice_error_analysis": slice_analysis,
                "test_confusion_matrix": cm,
            },
            "_plot_data": {
                "coords_2d": coords_2d.tolist(),
                "train_clusters": train_clusters.tolist(),
                "train_ground_truth": split_df.loc[train_mask, "label"]
                .astype(str)
                .tolist()
                if "label" in split_df.columns
                else [],
                "test_true_clusters": test_clusters.tolist(),
                "test_pred_clusters": best_test_preds.tolist(),
            },
        }

        # Auto-persist fitted state to ModelRegistryRepository
        self.save_persisted_state()
        return self.latest_metrics

    def classify_single_ticket(
        self,
        ticket_id: str,
        raw_text: str,
        metadata: dict[str, Any] | None = None,
        provider_override: LLMProvider | str | None = None,
        enable_llm_reranker: bool = True,
    ) -> PredictionResult:
        """Classify a live incoming ticket against discovered clusters with Open-Set & LLM Reranker."""
        if (
            self.clustering_engine is None
            or self.classifier_suite is None
            or self.metadata_pipeline is None
            or self.tfidf_vectorizer is None
        ):
            raise RuntimeError(
                "Pipeline has not been fitted yet. Run run_benchmark_pipeline() first."
            )

        rec = TicketRecord(
            ticket_id=ticket_id,
            raw_text=raw_text,
            metadata=metadata or {},
        )
        self.cleaning_chain.handle(rec)

        df_single = pd.DataFrame(
            [{"ticket_id": ticket_id, "raw_text": raw_text, **(metadata or {})}]
        )
        meta_vec = self.metadata_pipeline.transform(df_single)
        tfidf_vec = self.tfidf_vectorizer.transform([rec.text_lexical]).toarray()

        # Always pin embedding generation to the fitted embedding provider space
        embed_provider = (
            self.fitted_embedding_provider or self.gateway.get_active_provider()
        )
        emb_vec, _ = self.embed_with_dedup_and_cache(
            [rec.text_contextual], provider_override=embed_provider
        )

        X_single = np.hstack(
            [normalize(emb_vec, norm="l2"), tfidf_vec * 0.25, meta_vec * 0.05]
        )
        preds, confs, margins, is_open, dists = (
            self.classifier_suite.predict_with_open_set(X_single, emb_vec)
        )

        pred_cid = int(preds[0])
        conf = float(confs[0])
        margin = float(margins[0])
        novel = bool(is_open[0])
        dist = float(dists[0])
        used_reranker = False
        reasoning = "Classified via fast vector + metadata linear classifier."

        active_provider = (
            LLMProvider(provider_override.lower())
            if isinstance(provider_override, str)
            else (provider_override or self.gateway.get_active_provider())
        )

        # Ambiguous boundary ticket -> Escalate to Cloud LLM Few-Shot Reranker
        if enable_llm_reranker and (margin < 0.15 or novel) and self.cluster_profiles:
            try:
                candidates_desc = "\n".join(
                    f"- cluster_id={cid}: {prof.short_label} | {prof.signature} (keywords: {', '.join(prof.top_keywords[:5])})"
                    for cid, prof in self.cluster_profiles.items()
                )
                role = self.settings.system_role_description
                domain = self.settings.domain_context
                rerank_out = self.gateway.complete_structured(
                    system_prompt=(
                        f"You are a precision {role} for {domain}. "
                        "Select the best matching cluster_id from the discovered cluster taxonomy, "
                        "or set is_novel_issue=true and chosen_cluster_id=-1 if it represents an unseen emerging issue."
                    ),
                    user_prompt=(
                        f"Incoming Item: {rec.text_contextual}\n\n"
                        f"Discovered Clusters:\n{candidates_desc}"
                    ),
                    schema_cls=RerankDecisionLLMOutput,
                    tier=ModelTier.CHEAP,
                    provider_override=active_provider,
                )
                used_reranker = True
                reasoning = rerank_out.reasoning
                novel = rerank_out.is_novel_issue
                if not novel and rerank_out.chosen_cluster_id in self.cluster_profiles:
                    pred_cid = rerank_out.chosen_cluster_id
                    conf = rerank_out.confidence
            except (GatewaySDKError, ValueError, TypeError, KeyError):
                reasoning = "Fallback to distance-based cluster assignment"

        profile = self.cluster_profiles.get(pred_cid)
        label_str = (
            "UNASSIGNED / EMERGING_CLUSTER"
            if novel and pred_cid not in self.cluster_profiles
            else (profile.short_label if profile else f"Cluster {pred_cid}")
        )
        queue_str = (
            "Emerging Issues Triage Queue"
            if novel
            else (profile.routing_queue if profile else "General Support Queue")
        )
        top_terms = (
            profile.top_keywords[:5] if profile else rec.text_lexical.split()[:5]
        )

        return PredictionResult(
            ticket_id=ticket_id,
            predicted_cluster_id=pred_cid,
            cluster_label=label_str,
            routing_queue=queue_str,
            confidence=round(conf, 4),
            margin=round(margin, 4),
            is_open_set_novel=novel,
            centroid_distance=round(dist, 4),
            used_llm_reranker=used_reranker,
            provider_used=active_provider.value,
            top_contributing_terms=top_terms,
            reasoning=reasoning,
        )
