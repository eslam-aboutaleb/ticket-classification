"""Batch fitting for benchmark datasets or user-supplied multilingual data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ticket_classification.config.settings import AppSettings
from ticket_classification.controllers.api import ServiceContainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ticket-clf", description=__doc__)
    parser.add_argument("--dataset", default="textoir_banking77")
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV, JSON records or JSONL input; columns come from the domain profile",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--domain-id",
        default=None,
        help="Built-in or configured domain profile ID",
    )
    parser.add_argument("--domain", help="Optional context override for naming")
    parser.add_argument("--provider", choices=["litellm", "openai", "gemini"])
    parser.add_argument(
        "--strategy", choices=["aligned_kmeans", "hdbscan_density", "density"]
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--top-n-classes",
        type=int,
        default=None,
        help="Optional labeled benchmark subset; omitted by default",
    )
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--no-llm-naming", action="store_true")
    parser.add_argument(
        "--oos-file",
        type=Path,
        help="JSON array of domain-appropriate unknown-intent probe texts",
    )
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args(argv)
    options = {
        key: value
        for key, value in {
            "config_path": args.config,
            "domain_id": args.domain_id,
        }.items()
        if value is not None
    }
    if args.domain:
        options["domain_context"] = args.domain
    settings = AppSettings(**options)
    container = ServiceContainer(settings=settings)
    service = container.pipeline_service
    if args.provider:
        service.switch_provider(args.provider)
    kwargs = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "strategy": args.strategy,
        "use_llm_naming": not args.no_llm_naming,
        "provider_override": args.provider,
    }
    if args.oos_file:
        kwargs["oos_texts"] = json.loads(args.oos_file.read_text(encoding="utf-8"))
    if args.input:
        if args.input.suffix.lower() == ".csv":
            frame = pd.read_csv(args.input)
        else:
            frame = pd.read_json(
                args.input, lines=args.input.suffix.lower() == ".jsonl"
            )
        metrics = service.fit_dataframe(frame, dataset_name=args.input.stem, **kwargs)
    else:
        metrics = service.run_benchmark_pipeline(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            top_n_classes=args.top_n_classes,
            **kwargs,
        )
    plots = (
        container.report_view.save_plots(metrics) if metrics.get("evaluated") else {}
    )
    container.report_view.render_markdown_report(
        metrics,
        plot_paths=plots,
        report_path=args.report_path
        or Path(settings.data_dir) / "reports" / "fit-report.md",
    )
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if not k.startswith("_")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
