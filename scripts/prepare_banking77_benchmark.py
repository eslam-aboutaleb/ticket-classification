"""Prepare a full Banking77 run and independently selected CLINC150 OOS test probes."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

from ticket_classification.repositories.dataset_repo import TicketDatasetRepository

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
FINANCIAL_INTENTS = {
    "apr",
    "credit_limit",
    "credit_limit_change",
    "credit_score",
    "improve_credit_score",
    "interest_rate",
    "min_payment",
    "rollover_401k",
    "insurance",
    "insurance_change",
}
NONFINANCIAL_INTENTS = {
    "weather",
    "time",
    "date",
    "timer",
    "alarm",
    "calculator",
    "spelling",
    "definition",
    "recipe",
    "cook_time",
    "food_last",
    "ingredient_substitution",
    "nutrition_info",
    "tell_joke",
    "fun_fact",
    "play_music",
    "next_song",
    "what_is_your_name",
    "who_made_you",
    "meaning_of_life",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/benchmarks/banking77-full")
    )
    parser.add_argument("--source-data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pipeline_config.yaml")
    )
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "clinc-data-full.json"
    base = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}"
    if not source.exists():
        source.write_bytes(
            urllib.request.urlopen(base + "/data/data_full.json", timeout=30).read()
        )
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("CLINC source does not match the pinned dataset digest")
    license_path = root / "clinc-license.txt"
    if not license_path.exists():
        license_path.write_bytes(
            urllib.request.urlopen(base + "/LICENSE", timeout=30).read()
        )
    data = json.loads(source.read_text())
    rows = [
        {
            "text": text,
            "source_intent": label,
            "slice": "financial_unseen"
            if label in FINANCIAL_INTENTS
            else "nonfinancial",
        }
        for text, label in data["test"]
        if label in FINANCIAL_INTENTS | NONFINANCIAL_INTENTS
    ]
    if len(rows) != 900 or len({row["source_intent"] for row in rows}) != 30:
        raise ValueError("Unexpected CLINC probe coverage")
    (root / "oos-test.json").write_text(
        json.dumps([row["text"] for row in rows], ensure_ascii=False, indent=2)
    )
    (root / "oos-labels.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    config = yaml.safe_load(args.config.read_text())
    config["profiles"] = {
        "banking77": {
            "id": "banking77",
            "context": "Banking77 consumer banking support: the 77 documented benchmark intents",
            "language": {"supported": ["en"]},
            "cross_split_duplicates": "prefer_holdout",
        }
    }
    config.setdefault("classification", {}).update(
        {
            "open_set_distance_quantile": 0.99,
            "enable_llm_reranker": False,
            "candidate_models": [
                "naive_majority",
                "logistic_regression",
                "frozen_prototype",
            ],
        }
    )
    config.setdefault("clustering", {}).update(
        {"max_discovery_samples": 20000, "max_work_budget_seconds": 120}
    )
    (root / "benchmark.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run_data = root / "run-data"
    (run_data / "raw").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        name = f"textoir_banking77_{split}.tsv"
        cached = args.source_data_dir / "raw" / name
        target = run_data / "raw" / name
        if cached.exists() and not target.exists():
            shutil.copyfile(cached, target)
    frame = TicketDatasetRepository(run_data, config.get("datasets")).load_benchmark(
        "textoir_banking77"
    )
    if len(frame) != 13083 or frame.label.nunique() != 77:
        raise ValueError(
            "Expected all 13,083 Banking77 records and 77 labels; cached data may be truncated"
        )
    protocol = {
        "dataset": "textoir_banking77",
        "input_row_count": len(frame),
        "label_count": int(frame.label.nunique()),
        "input_splits": frame["split"].value_counts().to_dict(),
        "dataset_files": {
            split: {
                "sha256": hashlib.sha256(
                    (run_data / "raw" / f"textoir_banking77_{split}.tsv").read_bytes()
                ).hexdigest()
            }
            for split in ("train", "val", "test")
        },
        "duplicate_policy": "Prefer the official test partition, then validation; remove overlapping earlier copies. Record removals in the fit audit.",
        "clinc_source": "https://github.com/clinc/oos-eval",
        "clinc_revision": REVISION,
        "clinc_sha256": SOURCE_SHA256,
        "oos_source_split": "test",
        "oos_count": len(rows),
        "oos_slices": dict(Counter(row["slice"] for row in rows)),
        "oos_sha256": hashlib.sha256((root / "oos-test.json").read_bytes()).hexdigest(),
        "oos_selection": "30 intents chosen before scoring: financial intents outside the 77 Banking77 labels and clearly nonfinancial intents. A curated external challenge set, not an official Banking77 OOS split.",
        "calibration": "Known validation only; compare quantiles .96 and .99 without OOS test threshold tuning.",
        "sample_cap": None,
        "class_cap": None,
        "discovery_sample_cap": 20000,
        "candidate_models": config["classification"]["candidate_models"],
    }
    (root / "prepared-protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2)
    )
    print(
        f"Prepared {len(frame)} known records, 77 labels and {len(rows)} OOS test probes in {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
