"""CLI Controller for running benchmark datasets, switching providers, and classifying tickets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ticket_classification.controllers.api import ServiceContainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ticket-clf",
        description="Cloud-Native Ticket & Claim Clustering and Classification CLI (powered by llm-gateway-sdk)",
    )
    parser.add_argument(
        "--dataset",
        default="textoir_banking77",
        choices=[
            "textoir_banking77",
            "airweave_incidents",
            "it_service_tickets",
            "textoir_clinc_oos",
            "cfpb_claims",
        ],
        help="Benchmark dataset to download and evaluate",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["litellm", "openai", "gemini"],
        help="Cloud model provider routed through llm-gateway-sdk",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Maximum stratified samples to run for the benchmark",
    )
    parser.add_argument(
        "--top-n-classes",
        type=int,
        default=8,
        help="Number of ground-truth categories to sample",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=5,
        help="Minimum candidate clusters K for Silhouette sweep",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=12,
        help="Maximum candidate clusters K for Silhouette sweep",
    )
    parser.add_argument(
        "--report-path",
        default="BENCHMARK_REPORT.md",
        help="Output path for the Markdown benchmark report",
    )

    args = parser.parse_args(argv)
    container = ServiceContainer()

    if args.provider:
        container.pipeline_service.switch_provider(args.provider)

    print(
        f"==> Running benchmark on dataset='{args.dataset}' "
        f"(provider='{container.gateway.get_active_provider().value}', max_samples={args.max_samples})..."
    )
    metrics = container.pipeline_service.run_benchmark_pipeline(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        top_n_classes=args.top_n_classes,
        k_min=args.k_min,
        k_max=args.k_max,
        use_llm_naming=True,
        provider_override=args.provider,
    )
    plot_paths = container.report_view.save_plots(metrics)
    container.report_view.render_markdown_report(
        metrics, plot_paths=plot_paths, report_path=args.report_path
    )

    # Also test live inference on a sample in-distribution ticket and an open-set novel claim
    sample_in = container.pipeline_service.classify_single_ticket(
        ticket_id="live-test-001",
        raw_text="From: user@example.com\nSubject: Pending card top-up\nHi, I made a cash deposit and top-up to my card account $250.00 yesterday (INC99812) but my balance is still not updated!",
        provider_override=args.provider,
    )
    sample_novel = container.pipeline_service.classify_single_ticket(
        ticket_id="live-test-002",
        raw_text="The office espresso machine on the 4th floor is leaking steam and needs a replacement gasket.",
        provider_override=args.provider,
    )

    summary = {
        "dataset": metrics["dataset"],
        "provider_used": metrics["provider_used"],
        "embedding_model": metrics["embedding_model"],
        "chat_model": metrics["chat_model"],
        "optimal_k": metrics["clustering"]["optimal_k"],
        "best_silhouette": metrics["clustering"]["best_silhouette_cosine"],
        "external_validity": metrics["clustering"]["external_validity"],
        "selected_classifier": metrics["classification"]["selected_production_model"],
        "classifier_metrics": metrics["classification"]["model_comparison"][
            metrics["classification"]["selected_production_model"]
        ],
        "sample_predictions": [asdict(sample_in), asdict(sample_novel)],
        "report_saved_to": args.report_path,
        "plots": plot_paths,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
