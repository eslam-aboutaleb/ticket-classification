"""MVC View Layer: Generates visual plots (2D cluster projections, confusion matrices) and Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class ReportAndPlotView:
    """Renders benchmark evaluation metrics, 2D cluster scatterplots, and confusion matrices."""

    def __init__(self, artifacts_dir: str | Path = "data/artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_plots(self, metrics: dict[str, Any]) -> dict[str, str]:
        """Save 2D PCA/UMAP scatterplots (by cluster & ground-truth) and confusion matrix."""
        plot_data = metrics.get("_plot_data", {})
        coords = np.array(plot_data.get("coords_2d", []))
        clusters = np.array(plot_data.get("train_clusters", []))
        gt_labels = plot_data.get("train_ground_truth", [])
        cm = np.array(
            metrics.get("classification", {}).get("test_confusion_matrix", [])
        )

        saved_paths: dict[str, str] = {}

        if coords.ndim == 2 and len(coords) > 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=[f"C{c}" for c in clusters],
                palette="tab10",
                s=45,
                alpha=0.85,
                ax=axes[0],
            )
            axes[0].set_title(
                f"2D Projection Colored by Discovered Cluster (K={metrics['clustering']['optimal_k']})"
            )
            axes[0].set_xlabel("Component 1")
            axes[0].set_ylabel("Component 2")

            if gt_labels:
                sns.scatterplot(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    hue=gt_labels,
                    palette="husl",
                    s=45,
                    alpha=0.85,
                    ax=axes[1],
                )
                axes[1].set_title("2D Projection Colored by Ground-Truth Feature")
                axes[1].set_xlabel("Component 1")
                axes[1].set_ylabel("Component 2")
            plt.tight_layout()
            cluster_plot_path = self.artifacts_dir / "clusters_2d_projection.png"
            plt.savefig(cluster_plot_path, dpi=150)
            plt.close(fig)
            saved_paths["clusters_2d_plot"] = str(cluster_plot_path)

        if cm.ndim == 2 and cm.size > 0:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(
                f"Test Confusion Matrix ({metrics['classification']['selected_production_model']})"
            )
            ax.set_xlabel("Predicted Cluster ID")
            ax.set_ylabel("True Cluster ID")
            plt.tight_layout()
            cm_path = self.artifacts_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close(fig)
            saved_paths["confusion_matrix_plot"] = str(cm_path)

        return saved_paths

    def render_markdown_report(
        self,
        metrics: dict[str, Any],
        plot_paths: dict[str, str] | None = None,
        report_path: str | Path = "BENCHMARK_REPORT.md",
    ) -> str:
        """Generate a comprehensive Markdown benchmark report following ML Best Practices."""
        clu = metrics["clustering"]
        clf = metrics["classification"]
        emb = metrics["embedding_stats"]
        audit = metrics["audit_and_split"]

        if not metrics.get("evaluated", True):
            text = f"# Fit Report: `{metrics['dataset']}`\n\nEvaluation requires supported classes and nonempty validation/test splits.\n\n```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n"
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return text

        lines = [
            f"# Benchmark Execution Report: `{metrics['dataset']}`",
            "",
            f"- **Active Cloud Provider:** `{metrics['provider_used']}` (via `llm-gateway-sdk`)",
            f"- **Embedding Model:** `{metrics['embedding_model']}` ({emb['embedding_dimensions']} dims)",
            f"- **Chat / Taxonomy Model:** `{metrics['chat_model']}`",
            f"- **Total Runtime:** `{metrics['elapsed_seconds']}s`",
            "",
            "---",
            "",
            "## 1. Schema Audit, Missing Value Handling & Leakage-Free Split",
            f"- **Total Rows Processed:** `{audit['total_rows']}`",
            f"- **Missing Text Rows Retained (with `is_text_missing` indicator):** `{audit['missing_text_rows_retained_with_indicator']}`",
            f"- **Split Distribution (Strict Fit-on-Train-Only):** `{audit['split_counts']}`",
            f"- **Ground-Truth Categories Present:** `{audit['unique_ground_truth_labels']}`",
            "",
            "## 2. Exact Deduplication & SQLite SHA-256 Embedding Cache",
            f"- **Input Tickets:** `{emb['total_texts']}` $\\rightarrow$ **Unique Contextual Texts (Stage-1 Exact):** `{emb['unique_texts_after_stage1_dedup']}`",
            f"- **Unique Structural Templates (diagnostic only):** `{emb.get('unique_templates_after_stage2_funnel', 'N/A')}`",
            f"- **SQLite Cache Hits:** `{emb['cache_hits']}` | **New Cloud Gateway Embeddings:** `{emb['gateway_embedded_texts']}` (`{emb['gateway_batch_requests']}` batch calls)",
            "",
            "## 3. Discovery and Dataset-Driven Cluster Selection",
            f"- **Selected Cluster Count ($K$):** `{clu['optimal_k']}` (boundary limit reached = `{clu.get('hit_k_max_boundary', False)}`)",
            f"- **Selection Diagnostics:** `{clu.get('diagnostics', {})}`",
            f"- **Best Cosine Silhouette Score:** `{clu['best_silhouette_cosine']}`",
            f"- **Silhouette Score by Candidate $K$:** `{clu['silhouette_sweep']}`",
            f"- **External Validity vs Ground-Truth Labels:** `{clu['external_validity']}`",
            "",
            "### Discovered Cluster Taxonomy (c-TF-IDF Keywords + Cloud LLM Naming)",
            "| Cluster ID | Train Size | Short Label | Routing Queue | Top c-TF-IDF Keywords | Signature |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for c in clu["clusters"]:
            kws = ", ".join(c["top_keywords"][:5])
            lines.append(
                f"| `C{c['cluster_id']}` | {c['size']} | **{c['short_label']}** | `{c['routing_queue']}` | `{kws}` | {c['signature']} |"
            )

        # Check if true ground truth metrics were computed
        has_gt_eval = any(
            "test_ground_truth_accuracy" in m for m in clf["model_comparison"].values()
        )

        lines.extend(
            [
                "",
                "## 4. Downstream Classifier Comparison (with 95% Bootstrap CIs)",
                f"- **Selected Production Model:** `{clf['selected_production_model']}`",
                f"- **Calibrated Open-Set Cosine Distance Threshold:** `{clf['open_set_cosine_distance_threshold']}`",
                "",
            ]
        )

        if has_gt_eval:
            lines.extend(
                [
                    "### True End-to-End Ground-Truth Test Performance",
                    "| Model | Val Macro-F1 | Test Ground-Truth Accuracy (95% CI) | Test Ground-Truth Macro-F1 (95% CI) | Latency (ms/ticket) |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for m_name, m_stats in clf["model_comparison"].items():
                if "test_ground_truth_accuracy" in m_stats:
                    gt_acc_ci = m_stats["test_ground_truth_accuracy_95ci"]
                    gt_f1_ci = m_stats["test_ground_truth_macro_f1_95ci"]
                    lines.append(
                        f"| `{m_name}` | {m_stats['val_macro_f1']} | "
                        f"**{m_stats['test_ground_truth_accuracy']}** (`[{gt_acc_ci[0]}, {gt_acc_ci[1]}]`) | "
                        f"**{m_stats['test_ground_truth_macro_f1']}** (`[{gt_f1_ci[0]}, {gt_f1_ci[1]}]`) | "
                        f"{m_stats['latency_ms_per_ticket']} ms |"
                    )
            lines.append("")

        lines.extend(
            [
                "### Cluster Agreement Metrics (Prediction vs Discovered Cluster)",
                "| Model | Val Macro-F1 | Cluster Agreement Acc (95% CI) | Cluster Macro-F1 (95% CI) | Weighted-F1 | Latency (ms/ticket) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        for m_name, m_stats in clf["model_comparison"].items():
            acc_ci = m_stats["test_accuracy_95ci"]
            f1_ci = m_stats["test_macro_f1_95ci"]
            lines.append(
                f"| `{m_name}` | {m_stats['val_macro_f1']} | "
                f"**{m_stats['test_accuracy']}** (`[{acc_ci[0]}, {acc_ci[1]}]`) | "
                f"**{m_stats['test_macro_f1']}** (`[{f1_ci[0]}, {f1_ci[1]}]`) | "
                f"{m_stats['test_weighted_f1']} | {m_stats['latency_ms_per_ticket']} ms |"
            )

        slices = clf["slice_error_analysis"]
        lines.extend(
            [
                "",
                "## 5. Slice-Based Error Analysis & Quantitative Open-Set Evaluation",
                (
                    f"- **Short Tickets ($\\le$ median length):** Count = `{slices['short_tickets_lte_median_chars']['count']}`, "
                    f"Macro-F1 = `{slices['short_tickets_lte_median_chars']['macro_f1']}`, "
                    f"Accuracy = `{slices['short_tickets_lte_median_chars']['accuracy']}`"
                ),
                (
                    f"- **Long Tickets ($>$ median length):** Count = `{slices['long_tickets_gt_median_chars']['count']}`, "
                    f"Macro-F1 = `{slices['long_tickets_gt_median_chars']['macro_f1']}`, "
                    f"Accuracy = `{slices['long_tickets_gt_median_chars']['accuracy']}`"
                ),
                f"- **Open-Set Novel/Out-of-Distribution Flagged on Test:** `{slices['open_set_flagged_count_on_test']}` (In-Distribution FPR: `{slices.get('open_set_in_distribution_fpr', 'N/A')}`)",
                f"- **Open-Set Out-of-Scope (OOS) Probe Recall:** `{slices.get('open_set_oos_recall', 'N/A')}` (tested on `{slices.get('open_set_probes_tested', 0)}` out-of-scope probe tickets)",
            ]
        )

        if plot_paths:
            lines.extend(
                [
                    "",
                    "## 6. Generated Visual Artifacts",
                    f"- 2D Cluster & Ground-Truth Scatterplot: `{plot_paths.get('clusters_2d_plot', '')}`",
                    f"- Test Confusion Matrix Heatmap: `{plot_paths.get('confusion_matrix_plot', '')}`",
                ]
            )

        content = "\n".join(lines) + "\n"
        Path(report_path).write_text(content, encoding="utf-8")
        return content
