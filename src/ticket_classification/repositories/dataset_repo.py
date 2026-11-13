"""Repository for downloading and loading benchmark ticket and claim datasets."""

from __future__ import annotations

import ast
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd


class TicketDatasetRepository:
    """Handles downloading, caching, and parsing benchmark datasets from the studied repos."""

    BENCHMARK_URLS: ClassVar[dict[str, dict[str, str]]] = {
        "textoir_banking77": {
            "train": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/banking/train.tsv",
            "val": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/banking/dev.tsv",
            "test": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/banking/test.tsv",
            "format": "tsv",
        },
        "textoir_clinc_oos": {
            "train": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/clinc/train.tsv",
            "val": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/clinc/dev.tsv",
            "test": "https://raw.githubusercontent.com/thuiar/TEXTOIR/main/data/clinc/test.tsv",
            "format": "tsv",
        },
        "it_service_tickets": {
            "train": "https://raw.githubusercontent.com/cibhibaskar/IT-service-ticket-classification/main/all_tickets_processed_improved_v3.csv",
            "format": "csv",
        },
        "airweave_incidents": {
            "train": "https://raw.githubusercontent.com/airweave-ai/error-monitoring-agent/main/backend/samples/errors.py",
            "format": "airweave_py",
        },
        "cfpb_claims": {
            "train": "https://raw.githubusercontent.com/RitikKumar3/Automatic-Ticket-Classification/main/complaints-2021-05-14_08_16.json",
            "format": "cfpb_json",
        },
    }

    def __init__(
        self, data_dir: str | Path = "data", specs: dict | None = None
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_urls = {k: dict(v) for k, v in self.BENCHMARK_URLS.items()}
        for name, spec in (specs or {}).items():
            self.benchmark_urls[name] = {
                **self.benchmark_urls.get(name, {}),
                **{k.removesuffix("_url"): v for k, v in spec.items()},
            }

    @staticmethod
    def adapt_frame(data: pd.DataFrame, profile) -> pd.DataFrame:
        """Map arbitrary tabular records without requiring ground-truth labels."""
        frame = data.copy()
        if profile.text_col not in frame:
            raise ValueError(f"Missing configured text column: {profile.text_col}")
        rename = {profile.text_col: "raw_text", profile.id_col: "ticket_id"}
        if profile.label_col and profile.label_col in frame:
            rename[profile.label_col] = "label"
        if profile.language_hint_col and profile.language_hint_col in frame:
            rename[profile.language_hint_col] = "language_hint"
        frame = frame.rename(columns=rename)
        if "ticket_id" not in frame:
            frame["ticket_id"] = [f"item-{i}" for i in range(len(frame))]
        if frame["ticket_id"].isna().any() or frame["ticket_id"].duplicated().any():
            raise ValueError("Input item IDs must be non-null and unique")
        if "metadata" in frame:
            nested = pd.json_normalize(
                [x if isinstance(x, dict) else {} for x in frame.pop("metadata")]
            )
            for col in nested:
                if col not in frame:
                    frame[col] = nested[col].to_numpy()
        return frame.reset_index(drop=True)

    def _download_text(self, url: str, dest: Path) -> str:
        if dest.exists():
            return dest.read_text(encoding="utf-8", errors="ignore")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        dest.write_text(content, encoding="utf-8")
        return content

    def load_benchmark(
        self,
        dataset_name: str = "textoir_banking77",
        max_samples: int | None = None,
        top_n_classes: int | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Download (if needed) and return a unified DataFrame with [ticket_id, raw_text, label, split, metadata]."""
        if dataset_name not in self.benchmark_urls:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(self.BENCHMARK_URLS)}"
            )

        spec = self.benchmark_urls[dataset_name]
        fmt = spec["format"]

        if fmt == "tsv":
            frames: list[pd.DataFrame] = []
            for split_name in ("train", "val", "test"):
                url = spec[split_name]
                local_file = self.raw_dir / f"{dataset_name}_{split_name}.tsv"
                raw_tsv = self._download_text(url, local_file)
                df_split = pd.read_csv(io.StringIO(raw_tsv), sep="\t")
                df_split = df_split.rename(
                    columns={
                        spec.get("text_col", "text"): "raw_text",
                        spec.get("label_col", "label"): "label",
                    }
                )
                df_split["split"] = split_name
                frames.append(df_split)
            df = pd.concat(frames, ignore_index=True)

        elif fmt == "csv":
            local_file = self.raw_dir / f"{dataset_name}.csv"
            raw_csv = self._download_text(spec["train"], local_file)
            df = pd.read_csv(io.StringIO(raw_csv))
            df = df.rename(
                columns={
                    spec.get("text_col", "Document"): "raw_text",
                    spec.get("label_col", "Topic_group"): "label",
                }
            )
            df["split"] = "unassigned"

        elif fmt == "airweave_py":
            local_file = self.raw_dir / "airweave_errors.py"
            py_source = self._download_text(spec["train"], local_file)
            records = self._parse_airweave_samples(py_source)
            df = pd.DataFrame(records)
            df["split"] = "unassigned"

        elif fmt == "cfpb_json":
            local_file = self.raw_dir / "cfpb_claims.json"
            raw_json = self._download_text(spec["train"], local_file)
            data = json.loads(raw_json)
            rows = []
            for idx, item in enumerate(data):
                src = item.get("_source", item)
                rows.append(
                    {
                        "ticket_id": f"cfpb-{idx}",
                        "raw_text": src.get("complaint_what_happened", ""),
                        "label": src.get("product", "unknown"),
                        "sub_product": src.get("sub_product"),
                        "issue": src.get("issue"),
                        "split": "unassigned",
                    }
                )
            df = pd.DataFrame(rows)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        if top_n_classes is not None and "label" in df.columns:
            top_labels = (
                df["label"].dropna().value_counts().head(top_n_classes).index.tolist()
            )
            df = df[df["label"].isin(top_labels)].copy()

        if max_samples is not None and len(df) > max_samples:
            # Stratified subsampling by (split, label) preserving all columns in pandas 2.x/3.x
            group_cols = [c for c in ("split", "label") if c in df.columns]
            if group_cols:
                frac = max_samples / len(df)
                sampled_parts = [
                    g.sample(n=max(1, round(len(g) * frac)), random_state=random_state)
                    for _, g in df.groupby(group_cols)
                ]
                df = pd.concat(sampled_parts, ignore_index=True)
                if len(df) > max_samples:
                    df = df.sample(
                        n=max_samples, random_state=random_state
                    ).reset_index(drop=True)
            else:
                df = df.sample(n=max_samples, random_state=random_state).reset_index(
                    drop=True
                )

        if "ticket_id" not in df.columns:
            df["ticket_id"] = [f"{dataset_name}-{i:05d}" for i in range(len(df))]

        return df.reset_index(drop=True)

    @staticmethod
    def _parse_airweave_samples(py_source: str) -> list[dict[str, Any]]:
        """Extract SAMPLE_ERRORS list from airweave-ai/error-monitoring-agent backend/samples/errors.py."""
        tree = ast.parse(py_source)
        records: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "ERROR" in target.id.upper():
                        try:
                            val = ast.literal_eval(node.value)
                            if isinstance(val, list):
                                for idx, item in enumerate(val):
                                    if isinstance(item, dict):
                                        msg = (
                                            item.get("message")
                                            or item.get("error_message")
                                            or str(item)
                                        )
                                        err_type = (
                                            item.get("error_type")
                                            or item.get("type")
                                            or "Incident"
                                        )
                                        service = (
                                            item.get("service")
                                            or item.get("module")
                                            or "core"
                                        )
                                        records.append(
                                            {
                                                "ticket_id": item.get(
                                                    "id", f"aw-{idx:03d}"
                                                ),
                                                "raw_text": f"[{err_type}] ({service}) {msg}",
                                                "label": err_type,
                                                "service": service,
                                                "http_status": item.get("http_status")
                                                or item.get("status_code"),
                                            }
                                        )
                        except (ValueError, SyntaxError, TypeError, KeyError):
                            continue
        if not records:
            # Fallback regex extraction if AST uses function calls like datetime.now()
            import re

            blocks = re.findall(
                r'\{[^{}]*"message"\s*:\s*"([^"]+)"[^{}]*\}', py_source, re.DOTALL
            )
            for idx, msg in enumerate(blocks):
                records.append(
                    {
                        "ticket_id": f"aw-{idx:03d}",
                        "raw_text": msg,
                        "label": "production_error",
                    }
                )
        return records
