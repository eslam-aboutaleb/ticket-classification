"""Integration boundaries that cannot be verified by classifier accuracy alone."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ticket_classification.config.pipeline import (
    CleaningPolicy,
    LanguagePolicy,
    PipelineConfig,
)
from ticket_classification.domain.cleaning.handlers import build_default_cleaning_chain
from ticket_classification.domain.cleaning.languages import (
    LanguageResult,
    register_language_detector,
    register_tokenizer,
)
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.domain.clustering.strategies import (
    FACTORIES,
    create_discovery_strategy,
    register_discovery_strategy,
)
from ticket_classification.models.entities import TicketRecord
from ticket_classification.views.report_view import ReportAndPlotView


def test_dynamic_k_exceeds_old_cap_and_accepts_all_noise():
    rng = np.random.default_rng(88)
    data = np.vstack([row + rng.normal(0, 0.003, (8, 24)) for row in np.eye(24)])
    engine = ClusteringEngine(strategy="aligned_kmeans", reducer="none")
    engine.fit(data)
    assert engine.optimal_k_ == 24
    assert engine.diagnostics_["candidate_bounds"][1] > 20
    noise = ClusteringEngine(strategy="hdbscan_density")
    labels, strengths = noise.fit_transform(np.zeros((8, 5)))
    assert noise.optimal_k_ == 0 and np.all(labels == -1) and np.all(strengths == 0)
    assert noise.transform(np.zeros((1, 5)))[0][0] == -1


def test_optional_umap_freezes_reducer():
    pytest.importorskip("umap")
    rng = np.random.default_rng(7)
    data = rng.normal(size=(24, 6))
    engine = ClusteringEngine(reducer="umap", n_components=2, k_min=2, k_max=3)
    engine.fit(data)
    reducer = engine._projector
    assert engine._project_embeddings(data[:1], False).shape == (1, 2)
    engine.transform(data[:2])
    assert engine._projector is reducer


def test_registered_language_and_tokenization_are_used():
    register_language_detector(
        "test_detector",
        lambda text, policy, hint: LanguageResult(["fr"], 1.0, "supported", hint),
    )
    register_tokenizer("test_tokenizer", lambda text: ["service", "paiement"])
    rec = TicketRecord("x", "Bonjour les problèmes de paiement", language_hint="fr")
    build_default_cleaning_chain(
        language=LanguagePolicy(
            supported=["fr"], detector="test_detector", tokenizer="test_tokenizer"
        )
    ).handle(rec)
    assert rec.languages == ["fr"] and rec.language_hint == "fr"
    assert rec.text_lexical == "service paiement"
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_default_cleaning_chain(
            language=LanguagePolicy(tokenizer="missing")
        ).handle(TicketRecord("x", "This is a payment issue"))
    with pytest.raises(ValueError, match="Invalid cleaning pattern"):
        CleaningPolicy(pii_patterns={"ACCOUNT": "["})


def test_discovery_registration_and_invalid_profiles():
    name = "test_custom"
    try:
        register_discovery_strategy(
            name, lambda **options: ClusteringEngine(reducer="none")
        )
        assert (
            PipelineConfig.model_validate(
                {"clustering": {"default_strategy": name}}
            ).clustering.default_strategy
            == name
        )
        assert create_discovery_strategy(name).reducer == "none"
        with pytest.raises(ValueError, match="already registered"):
            register_discovery_strategy(name, ClusteringEngine)
    finally:
        FACTORIES.pop(name, None)
    with pytest.raises(ValueError, match="default_route"):
        PipelineConfig.model_validate(
            {"profiles": {"x": {"id": "x", "routes": {"claims": "Claims"}}}}
        )


def test_fit_only_report_serializes_tiny_results(tmp_path: Path):
    metrics = {
        "dataset": "tiny",
        "evaluated": False,
        "clustering": {"optimal_k": 0},
        "classification": {},
        "embedding_stats": {},
        "audit_and_split": {},
    }
    path = tmp_path / "report.md"
    text = ReportAndPlotView(tmp_path).render_markdown_report(metrics, report_path=path)
    assert path.exists() and "Evaluation requires" in text
    assert (
        json.loads(text.split("```json\n")[1].split("\n```")[0])["evaluated"] is False
    )


def test_cli_custom_file_and_benchmark_have_no_fixed_k_defaults(
    tmp_path, monkeypatch, capsys
):
    from ticket_classification.controllers import cli

    metrics = {
        "dataset": "custom",
        "evaluated": False,
        "clustering": {},
        "classification": {},
        "embedding_stats": {},
        "audit_and_split": {},
    }
    service = MagicMock()
    service.fit_dataframe.return_value = metrics
    service.run_benchmark_pipeline.return_value = metrics
    container = SimpleNamespace(
        pipeline_service=service, report_view=ReportAndPlotView(tmp_path)
    )
    monkeypatch.setattr(cli, "AppSettings", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, "ServiceContainer", lambda **kwargs: container)
    path = tmp_path / "claims.jsonl"
    path.write_text('{"raw_text":"مشكلة في الدفع"}\n', encoding="utf-8")
    report = tmp_path / "report.md"
    assert (
        cli.main(
            ["--input", str(path), "--no-llm-naming", "--report-path", str(report)]
        )
        == 0
    )
    assert (
        service.fit_dataframe.call_args.args[0].iloc[0]["raw_text"] == "مشكلة في الدفع"
    )
    assert report.exists()
    assert json.loads(capsys.readouterr().out)["evaluated"] is False
    assert (
        cli.main(["--dataset", "textoir_banking77", "--report-path", str(report)]) == 0
    )
    options = service.run_benchmark_pipeline.call_args.kwargs
    assert (
        options["k_min"] is None
        and options["k_max"] is None
        and options["top_n_classes"] is None
    )
    assert json.loads(capsys.readouterr().out)["evaluated"] is False
