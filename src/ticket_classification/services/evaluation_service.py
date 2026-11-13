"""Held-out evaluation, language slices and release evidence."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ticket_classification.config.pipeline import DomainProfile, PipelineConfig
from ticket_classification.domain.classification.engine import ClassifierSuite
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.models.entities import ClusterProfile, TicketRecord
from ticket_classification.repositories.model_registry_repo import (
    runtime_environment_revisions,
)


def compute_bootstrap_ci(
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


def hungarian_cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute unsupervised clustering accuracy using Hungarian optimal assignment."""
    true_labels, true_inv = np.unique(y_true, return_inverse=True)
    pred_labels, pred_inv = np.unique(y_pred, return_inverse=True)
    dim = max(len(true_labels), len(pred_labels))
    w = np.zeros((dim, dim), dtype=np.int64)
    for t, p in zip(true_inv, pred_inv):
        w[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return float(sum(w[r, c] for r, c in zip(row_ind, col_ind)) / len(y_true))


def evaluate_open_set(
    classifier_suite: ClassifierSuite,
    known_embeddings: np.ndarray,
    oos_embeddings: np.ndarray,
    primary_quantile: float,
) -> dict[str, Any]:
    """Score every supplied known row; select thresholds only from known validation distances."""
    oos_scores, oos_flags = classifier_suite.novelty_scores(oos_embeddings)
    oos_recall = round(float(np.mean(oos_flags)), 4)
    known_scores, known_flags = classifier_suite.novelty_scores(known_embeddings)
    if len(known_scores):
        scores = np.concatenate([known_scores, oos_scores])
        targets = np.concatenate(
            [np.zeros(len(known_scores)), np.ones(len(oos_scores))]
        )
        operating_points = []
        calibration_distances = classifier_suite.calibration_distances_
        if len(calibration_distances):
            for quantile in sorted({0.96, primary_quantile}):
                threshold = float(np.quantile(calibration_distances, quantile))
                known_positive = int(np.sum(known_scores > threshold))
                unknown_positive = int(np.sum(oos_scores > threshold))
                operating_points.append(
                    {
                        "calibration_quantile": quantile,
                        "threshold": threshold,
                        "known_test_count": len(known_scores),
                        "known_false_positive_count": known_positive,
                        "known_false_positive_rate": known_positive / len(known_scores),
                        "oos_test_count": len(oos_scores),
                        "oos_true_positive_count": unknown_positive,
                        "oos_recall": unknown_positive / len(oos_scores),
                    }
                )
        open_set_metrics = {
            "cohort": "all supplied in-domain test records; no prediction-based filtering",
            "known_test_count": len(known_scores),
            "known_false_positive_count": int(np.sum(known_flags)),
            "oos_test_count": len(oos_scores),
            "oos_true_positive_count": int(np.sum(oos_flags)),
            "operating_points": operating_points,
            "calibration_uses_oos_test": False,
            "auroc": float(roc_auc_score(targets, scores)),
            "average_precision": float(average_precision_score(targets, scores)),
            "known_false_positive_rate": float(np.mean(known_flags)),
            "probe_recall": oos_recall,
            "probe_precision": float(
                np.sum(oos_flags) / max(1, np.sum(oos_flags) + np.sum(known_flags))
            ),
        }
    return open_set_metrics


def evaluate_fitted_pipeline(
    *,
    dataset_name: str,
    split_df: pd.DataFrame,
    records: list[TicketRecord],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    train_clusters: np.ndarray,
    val_clusters: np.ndarray,
    test_clusters: np.ndarray,
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    emb_test: np.ndarray,
    X_val: np.ndarray | sparse.spmatrix,
    X_test: np.ndarray | sparse.spmatrix,
    clustering_engine: ClusteringEngine,
    classifier_suite: ClassifierSuite,
    cluster_profiles: dict[int, ClusterProfile],
    cluster_to_gt_label: dict[int, str],
    profile: DomainProfile,
    config: PipelineConfig,
    base_metrics: dict[str, Any],
    audit_info: dict[str, Any],
    embed_stats: dict[str, Any],
    unique_templates: int,
    active_provider_name: str,
    embedding_model_name: str,
    chat_model_name: str,
    elapsed_seconds: float,
    oos_texts: list[str] | None = None,
    oos_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run held-out classifier comparison, slice error analysis, multilingual checks, and release gates."""
    has_gt = (
        "label" in split_df.columns
        and split_df["label"].notna().all()
        and split_df["label"].nunique() > 1
    )
    y_test_gt = (
        split_df.loc[test_mask, "label"].astype(str).to_numpy()
        if has_gt
        else np.array([])
    )

    model_benchmarks: dict[str, dict[str, Any]] = {}
    best_val_f1 = -1.0
    best_model_name = classifier_suite.best_model_name_

    for m_name in classifier_suite.models_:
        t_infer0 = time.perf_counter()
        val_preds, _, _, _, _ = classifier_suite.predict_with_open_set(
            X_val, emb_val, model_name=m_name
        )
        test_preds, _test_p_conf, _test_margins, _test_open_set, _ = (
            classifier_suite.predict_with_open_set(X_test, emb_test, model_name=m_name)
        )
        infer_ms_per_sample = ((time.perf_counter() - t_infer0) * 1000.0) / max(
            1, X_val.shape[0] + X_test.shape[0]
        )

        val_f1 = float(
            f1_score(val_clusters, val_preds, average="macro", zero_division=0)
        )
        if val_f1 > best_val_f1 and m_name not in {
            "naive_majority",
            "frozen_prototype",
        }:
            best_val_f1 = val_f1
            best_model_name = m_name

        acc_pt, acc_lo, acc_hi = compute_bootstrap_ci(
            test_clusters, test_preds, accuracy_score
        )
        f1_pt, f1_lo, f1_hi = compute_bootstrap_ci(
            test_clusters,
            test_preds,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        )
        wf1_pt = float(
            f1_score(test_clusters, test_preds, average="weighted", zero_division=0)
        )
        prec_pt = float(
            precision_score(test_clusters, test_preds, average="macro", zero_division=0)
        )
        rec_pt = float(
            recall_score(test_clusters, test_preds, average="macro", zero_division=0)
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

        if has_gt and len(cluster_to_gt_label) > 0:
            mapped_test_preds = np.array(
                [cluster_to_gt_label.get(int(p), "unassigned") for p in test_preds]
            )
            gt_acc_pt, gt_acc_lo, gt_acc_hi = compute_bootstrap_ci(
                y_test_gt, mapped_test_preds, accuracy_score
            )
            gt_f1_pt, gt_f1_lo, gt_f1_hi = compute_bootstrap_ci(
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

    classifier_suite.best_model_name_ = best_model_name

    best_test_preds, _, _, best_open_flags, _ = classifier_suite.predict_with_open_set(
        X_test, emb_test, model_name=best_model_name
    )
    test_lengths = split_df.loc[test_mask, "raw_text"].str.len().to_numpy()
    median_len = float(np.median(test_lengths))
    short_mask = test_lengths <= median_len
    long_mask = test_lengths > median_len

    oos_recall = None
    open_set_metrics = {}
    if oos_texts and oos_embeddings is not None and len(emb_test):
        open_set_metrics = evaluate_open_set(
            classifier_suite,
            emb_test,
            oos_embeddings,
            config.classification.open_set_distance_quantile,
        )
        oos_recall = open_set_metrics["probe_recall"]
    in_dist_fpr = round(float(np.mean(best_open_flags)), 4)

    def slice_metrics(mask):
        count = int(np.sum(mask))
        return {
            "count": count,
            "macro_f1": round(
                float(
                    f1_score(
                        test_clusters[mask],
                        best_test_preds[mask],
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            )
            if count
            else None,
            "accuracy": round(
                float(accuracy_score(test_clusters[mask], best_test_preds[mask])), 4
            )
            if count
            else None,
        }

    slice_analysis: dict[str, Any] = {
        "open_set_evaluation": open_set_metrics,
        "short_tickets_lte_median_chars": slice_metrics(short_mask),
        "long_tickets_gt_median_chars": slice_metrics(long_mask),
        "open_set_flagged_count_on_test": int(np.sum(best_open_flags)),
        "open_set_in_distribution_fpr": in_dist_fpr,
        "open_set_oos_recall": oos_recall,
        "open_set_probes_tested": len(oos_texts or []),
    }

    language_evaluation = {}
    test_records = [record for record, selected in zip(records, test_mask) if selected]
    for language in profile.language.supported:
        selected = np.array([language in record.languages for record in test_records])
        if np.any(selected):
            language_evaluation[language] = {
                "count": int(selected.sum()),
                "cluster_agreement_macro_f1": float(
                    f1_score(
                        test_clusters[selected],
                        best_test_preds[selected],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "accepted_coverage": float(np.mean(~best_open_flags[selected])),
                "novelty_flag_rate": float(np.mean(best_open_flags[selected])),
            }
    slice_analysis["by_language"] = language_evaluation
    slice_analysis["language_status_counts"] = {
        status: sum(record.language_status == status for record in test_records)
        for status in ("supported", "unsupported", "undetermined")
    }

    external_clustering_metrics: dict[str, float] = {}
    if has_gt:
        gt_all = split_df["label"].astype(str).to_numpy()
        all_cluster_ids = np.zeros(len(split_df), dtype=int)
        all_cluster_ids[train_mask] = train_clusters
        all_cluster_ids[val_mask] = val_clusters
        all_cluster_ids[test_mask] = test_clusters

        gt_test = split_df.loc[test_mask, "label"].astype(str).to_numpy()
        external_clustering_metrics = {
            "test_NMI_vs_ground_truth": round(
                float(normalized_mutual_info_score(gt_test, test_clusters)), 4
            ),
            "test_ARI_vs_ground_truth": round(
                float(adjusted_rand_score(gt_test, test_clusters)), 4
            ),
            "test_Hungarian_ACC_vs_ground_truth": round(
                hungarian_cluster_accuracy(gt_test, test_clusters), 4
            ),
            "full_NMI_vs_ground_truth": round(
                float(normalized_mutual_info_score(gt_all, all_cluster_ids)), 4
            ),
            "full_ARI_vs_ground_truth": round(
                float(adjusted_rand_score(gt_all, all_cluster_ids)), 4
            ),
            "full_Hungarian_ACC_vs_ground_truth": round(
                hungarian_cluster_accuracy(gt_all, all_cluster_ids), 4
            ),
        }

    coords_2d = clustering_engine.project_2d(emb_train)
    cm = confusion_matrix(test_clusters, best_test_preds).tolist()

    lineage = base_metrics.get("lineage", {})
    requires_human_approval = bool(
        lineage.get("births") or lineage.get("splits") or lineage.get("merges")
    )

    return {
        **base_metrics,
        "evaluated": True,
        "dataset": dataset_name,
        "provider_used": active_provider_name,
        "embedding_model": embedding_model_name,
        "chat_model": chat_model_name,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "reproducibility": {
            "random_seed": int(config.project.get("random_seed", 42)),
            "known_classes": sorted(int(c) for c in classifier_suite.valid_classes_),
            "environment": runtime_environment_revisions(),
        },
        "release_gate": {
            "calibration_status": classifier_suite.calibration_status_,
            "calibration_sample_count": classifier_suite.calibration_sample_count_,
            "requires_human_taxonomy_review": requires_human_approval,
            "supported_language_coverage": language_evaluation,
        },
        "audit_and_split": audit_info,
        "embedding_stats": {
            **embed_stats,
            "unique_templates_after_stage2_funnel": unique_templates,
        },
        "clustering": {
            "diagnostics": clustering_engine.diagnostics_,
            "optimal_k": clustering_engine.optimal_k_,
            "dropout_viable_k": clustering_engine.dropout_viable_k_,
            "hit_k_max_boundary": clustering_engine.hit_k_max_boundary_,
            "best_silhouette_cosine": round(clustering_engine.best_silhouette_, 4),
            "silhouette_sweep": {
                str(k): round(v, 4)
                for k, v in clustering_engine.silhouette_history_.items()
            },
            "alignment": asdict(clustering_engine.last_alignment_)
            if clustering_engine.last_alignment_
            else {},
            "external_validity": external_clustering_metrics,
            "clusters": [asdict(p) for p in cluster_profiles.values()],
        },
        "classification": {
            "calibration": classifier_suite.calibration_status_,
            "calibration_sample_count": classifier_suite.calibration_sample_count_,
            "selected_production_model": best_model_name,
            "open_set_cosine_distance_threshold": round(
                classifier_suite.distance_threshold_, 4
            ),
            "model_comparison": model_benchmarks,
            "slice_error_analysis": slice_analysis,
            "test_confusion_matrix": cm,
        },
        "_plot_data": {
            "coords_2d": coords_2d.tolist(),
            "train_clusters": train_clusters.tolist(),
            "train_ground_truth": split_df.loc[train_mask, "label"].astype(str).tolist()
            if "label" in split_df.columns
            else [],
            "test_true_clusters": test_clusters.tolist(),
            "test_pred_clusters": best_test_preds.tolist(),
        },
    }
