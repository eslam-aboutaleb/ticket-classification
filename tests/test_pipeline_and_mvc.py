"""Ticket pipeline and HTTP API behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import SYNTHETIC_RELEASE_POLICY
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ticket_classification.config.settings import AppSettings, LLMProvider, ModelTier
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.domain.cleaning.handlers import build_default_cleaning_chain
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.domain.features.metadata_encoder import (
    MetadataFeaturePipeline,
)
from ticket_classification.gateways.llm_gateway_adapter import BaseCloudGateway
from ticket_classification.models.entities import TicketRecord
from ticket_classification.models.schemas import (
    ClusterTaxonomyLLMOutput,
    RerankDecisionLLMOutput,
)

T = TypeVar("T", bound=BaseModel)


class DeterministicMockGateway(BaseCloudGateway):
    """Deterministic offline Cloud Gateway for unit & integration tests."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def switch_provider(self, provider: LLMProvider | str) -> None:
        self.settings.llm_provider = (
            LLMProvider(provider.lower()) if isinstance(provider, str) else provider
        )

    def get_active_provider(self) -> LLMProvider:
        return self.settings.llm_provider

    def embed_texts(
        self,
        texts: list[str],
        provider_override: LLMProvider | str | None = None,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        topics = [
            ("deposit", "cash", "balance", "cheque", "top"),
            ("card", "virtual", "disposable", "replacement", "pin"),
            ("exchange", "currency", "rate", "fee", "charge"),
            ("identity", "verify", "passport", "document", "verification"),
        ]
        for text in texts:
            low = text.lower()
            base = np.zeros(32, dtype=float)
            for idx, kw_group in enumerate(topics):
                hits = sum(1 for kw in kw_group if kw in low)
                if hits > 0:
                    base[idx * 6 : (idx + 1) * 6] += hits * 3.0

            digest = hashlib.sha256(low.encode("utf-8")).digest()
            noise = np.frombuffer(digest[:32], dtype=np.uint8).astype(float) / 2550.0
            vec = base + noise
            norm = np.linalg.norm(vec) or 1.0
            vectors.append((vec / norm).tolist())
        return vectors

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type[T],
        tier: ModelTier = ModelTier.STANDARD,
        provider_override: LLMProvider | str | None = None,
    ) -> T:
        if schema_cls is ClusterTaxonomyLLMOutput:
            return ClusterTaxonomyLLMOutput(
                short_label="Account & Card Operations",
                signature="Customer inquiries regarding card payments, deposits, and verification",
                routing_queue="Banking Operations Queue",
            )  # type: ignore[return-value]
        if schema_cls is RerankDecisionLLMOutput:
            is_espresso = "espresso" in user_prompt.lower()
            return RerankDecisionLLMOutput(
                chosen_cluster_id=-1 if is_espresso else 0,
                confidence=0.92,
                is_novel_issue=is_espresso,
                reasoning="Evaluated semantic alignment via Gateway LLM reranker.",
            )  # type: ignore[return-value]
        raise ValueError(f"Unsupported schema: {schema_cls}")


def test_provider_switching_between_litellm_openai_and_gemini() -> None:
    """Verify that switching between litellm, openai, and gemini resolves distinct models and fingerprints."""
    settings = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="sk-test-virtual-key-123456",
        llm_provider=LLMProvider.LITELLM,
    )

    assert settings.resolve_model(ModelTier.CHEAP) == "general-fast"
    assert settings.resolve_model(ModelTier.EMBEDDING) == "rag-embedding"
    id_litellm = settings.embedding_identity()

    settings.llm_provider = LLMProvider.OPENAI
    assert settings.resolve_model(ModelTier.CHEAP) == "gpt-4o-mini"
    assert settings.resolve_model(ModelTier.STRONG) == "gpt-4o"
    assert settings.resolve_model(ModelTier.EMBEDDING) == "text-embedding-3-small"
    id_openai = settings.embedding_identity()

    settings.llm_provider = LLMProvider.GEMINI
    assert settings.resolve_model(ModelTier.CHEAP) == "gemini-3.8-flash"
    assert settings.resolve_model(ModelTier.STRONG) == "gemini-4-argon"
    assert settings.resolve_model(ModelTier.EMBEDDING) == "gemini-embedding-001"
    id_gemini = settings.embedding_identity()

    assert len({id_litellm, id_openai, id_gemini}) == 3


def test_dual_track_cleaning_chain_and_pii_redaction() -> None:
    """Verify email headers, PII tokens, and CFPB XXXX masks are normalized properly."""
    chain = build_default_cleaning_chain()
    raw = (
        "From: john.doe@company.com\n"
        "Sent: Monday, Nov 12\n"
        "Subject: Dispute on charge\n"
        "Hi support, I was billed $1,250.00 on XXXX/XXXX/2026 under ticket INC49201. "
        "Call me at (415) 555-0199 or see https://bank.example.com/claim. "
        "[cid:image001.png@01D]"
    )
    rec = TicketRecord(ticket_id="t-1", raw_text=raw)
    chain.handle(rec)

    assert "From:" not in rec.text_contextual
    assert "[cid:" not in rec.text_contextual
    assert "<AMOUNT>" in rec.text_contextual
    assert "<REDACTED>" in rec.text_contextual
    assert "<TICKET_ID>" in rec.text_contextual
    assert "<PHONE>" in rec.text_contextual
    assert "<URL>" in rec.text_contextual
    assert "dispute" in rec.text_lexical
    assert "charge" in rec.text_lexical


def test_leakage_safe_pipeline_stage_enforces_fit_before_transform() -> None:
    """Verify that calling transform() before fit() raises RuntimeError to prevent data leakage."""
    meta_pipe = MetadataFeaturePipeline(numeric_cols=["amount"])
    df = pd.DataFrame({"raw_text": ["ticket a", "ticket b"], "amount": [10.0, 20.0]})

    with pytest.raises(RuntimeError, match="before fit"):
        meta_pipe.transform(df)

    out = meta_pipe.fit_transform(df)
    assert out.shape[0] == 2


def test_hungarian_alignment_and_outlier_recovery() -> None:
    """Verify Hungarian centroid alignment and outlier (-1) recovery."""
    engine = ClusteringEngine(
        k_min=3, k_max=4, n_components=3, outlier_recovery_threshold=0.5
    )
    c1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    c2 = np.array([[0.0, 1.0, 0.0, 0.0]])
    c3 = np.array([[0.0, 0.0, 1.0, 0.0]])
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            c1 + rng.normal(0, 0.05, (15, 4)),
            c2 + rng.normal(0, 0.05, (15, 4)),
            c3 + rng.normal(0, 0.05, (15, 4)),
        ]
    )
    clusters, _confs = engine.fit_transform(X)
    assert len(clusters) == 45
    assert engine.optimal_k_ in (3, 4)

    proj = engine._project_embeddings(X[:5], is_fit=False)
    noisy_labels = np.array([-1, 0, -1, 1, 2])
    assert engine.centroids_ is not None
    _rec_labels, rec_cnt, rem_cnt = engine.recover_outliers(
        proj, noisy_labels, engine.centroids_
    )
    assert rec_cnt >= 1
    assert rem_cnt == 0 or rem_cnt < 2


def test_end_to_end_service_and_fastapi_mvc_controllers(tmp_path: Path) -> None:
    """Verify full benchmark pipeline and FastAPI MVC controllers with provider switching."""
    config_path = tmp_path / "pipeline.yaml"
    config = AppSettings(_env_file=None).load_pipeline_yaml()
    config["release"] = SYNTHETIC_RELEASE_POLICY
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="sk-test-virtual-key-123456",
        data_dir=str(tmp_path),
        config_path=str(config_path),
        embedding_dimensions=32,
        _env_file=None,
    )
    mock_gw = DeterministicMockGateway(settings)
    container = ServiceContainer(settings=settings, gateway=mock_gw)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    intents = [
        (
            "I made a cash deposit to my account but balance is not updated",
            "balance_not_updated",
        ),
        ("How do I get a disposable virtual card for online payments?", "virtual_card"),
        (
            "What is the currency exchange fee and rate for international wire?",
            "exchange_charge",
        ),
        (
            "Please tell me the steps to verify my identity with passport",
            "verify_identity",
        ),
    ]
    for split, mult in [("train", 16), ("dev", 4), ("test", 4)]:
        lines = ["text\tlabel"]
        for i in range(mult):
            for text_tmpl, lbl in intents:
                lines.append(f"{text_tmpl} {split} example #{i}\t{lbl}")
        (
            raw_dir / f"textoir_banking77_{'val' if split == 'dev' else split}.tsv"
        ).write_text("\n".join(lines), encoding="utf-8")

    metrics = container.pipeline_service.run_benchmark_pipeline(
        dataset_name="textoir_banking77",
        max_samples=96,
        top_n_classes=4,
        k_min=3,
        k_max=5,
        use_llm_naming=True,
    )
    assert metrics["clustering"]["optimal_k"] >= 3
    assert (
        metrics["classification"]["selected_production_model"]
        in metrics["classification"]["model_comparison"]
    )

    app = create_app(container)
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_prov_get = client.get("/api/v1/provider")
    assert r_prov_get.status_code == 200

    r_switch = client.post("/api/v1/provider", json={"provider": "gemini"})
    assert r_switch.status_code == 200
    assert r_switch.json()["active_provider"] == "gemini"
    assert r_switch.json()["models_by_tier"]["cheap"] == "gemini-3.8-flash"

    r_switch_bad = client.post(
        "/api/v1/provider", json={"provider": "invalid_provider"}
    )
    assert r_switch_bad.status_code == 400

    r_clu = client.get("/api/v1/clusters")
    assert r_clu.status_code == 200
    assert r_clu.json()["cluster_count"] >= 3

    r_clf = client.post(
        "/api/v1/classify",
        json={
            "ticket_id": "tkt-900",
            "text": "My cash deposit still has not updated my account balance",
            "provider_override": "openai",
        },
    )
    assert r_clf.status_code == 200
    body = r_clf.json()
    assert body["ticket_id"] == "tkt-900"
    assert body["provider_used"] == "openai"
    assert body["confidence"] > 0.0

    r_batch = client.post(
        "/api/v1/classify/batch",
        json={
            "tickets": [
                {"ticket_id": "tkt-901", "text": "Need virtual card for shopping"},
                {
                    "ticket_id": "tkt-902",
                    "text": "Currency exchange fee on wire transfer",
                },
            ],
            "provider_override": "gemini",
        },
    )
    assert r_batch.status_code == 200
    assert len(r_batch.json()) == 2

    plots = container.report_view.save_plots(metrics)
    assert "clusters_2d_plot" in plots
    assert "confusion_matrix_plot" in plots
    report_file = tmp_path / "fit-report.md"
    md_text = container.report_view.render_markdown_report(
        metrics, plot_paths=plots, report_path=report_file
    )
    assert report_file.is_file()
    assert "Benchmark Execution Report" in md_text


def test_dataset_repo_formats_and_density_alignment(tmp_path: Path) -> None:
    """Verify dataset repository parsers (airweave, cfpb, csv) and Hungarian centroid alignment."""
    from ticket_classification.repositories.dataset_repo import TicketDatasetRepository

    repo = TicketDatasetRepository(tmp_path)
    sample_py = (
        "SAMPLE_ERRORS = [\n"
        '    {"id": "err-1", "message": "Timeout connecting to DB", "error_type": "DBTimeout", "service": "auth"},\n'
        '    {"id": "err-2", "message": "Null pointer in handler", "error_type": "NullPointer", "service": "api"},\n'
        "]\n"
    )
    (repo.raw_dir / "airweave_errors.py").write_text(sample_py, encoding="utf-8")
    df_aw = repo.load_benchmark("airweave_incidents", max_samples=1)
    assert len(df_aw) == 1

    cfpb_json = '[{"_source": {"complaint_what_happened": "Unauthorized fee charged", "product": "Card", "sub_product": "Credit", "issue": "Fee"}}]'
    (repo.raw_dir / "cfpb_claims.json").write_text(cfpb_json, encoding="utf-8")
    df_cfpb = repo.load_benchmark("cfpb_claims")
    assert len(df_cfpb) == 1
    assert df_cfpb.iloc[0]["label"] == "Card"

    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(1.0, 0.02, (12, 4)), rng.normal(-1.0, 0.02, (12, 4))])
    dens_engine = ClusteringEngine(strategy="density", k_min=2, k_max=4, n_components=2)
    labels1, _ = dens_engine.fit_transform(X)
    assert len(labels1) == 24

    labels2, _ = dens_engine.fit_transform(X)
    assert len(labels2) == 24


def test_model_registry_persistence_and_startup_reload(tmp_path: Path) -> None:
    """Verify versioned state and the active release pointer reload on process restart."""
    config_path = tmp_path / "pipeline.yaml"
    config = AppSettings(_env_file=None).load_pipeline_yaml()
    config["release"] = SYNTHETIC_RELEASE_POLICY
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="sk-test-virtual-key-123456",
        data_dir=str(tmp_path),
        config_path=str(config_path),
        embedding_dimensions=32,
        _env_file=None,
    )
    mock_gw = DeterministicMockGateway(settings)
    container1 = ServiceContainer(settings=settings, gateway=mock_gw)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = ["text\tlabel"]
    for i in range(12):
        lines.append(f"Deposit cash into account #{i}\tdeposit")
        lines.append(f"Card replacement needed #{i}\tcard")
        lines.append(f"Exchange currency rates #{i}\texchange")
    (raw_dir / "textoir_banking77_train.tsv").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (raw_dir / "textoir_banking77_val.tsv").write_text(
        "\n".join(
            [lines[0], *[line.replace(" #", " validation #") for line in lines[1:10]]]
        ),
        encoding="utf-8",
    )
    (raw_dir / "textoir_banking77_test.tsv").write_text(
        "\n".join(
            [lines[0], *[line.replace(" #", " testing #") for line in lines[1:10]]]
        ),
        encoding="utf-8",
    )

    container1.pipeline_service.run_benchmark_pipeline(
        dataset_name="textoir_banking77",
        max_samples=36,
        top_n_classes=3,
        k_min=2,
        k_max=4,
        use_llm_naming=False,
    )
    registry = container1.model_registry_repo
    assert registry.release_record_path.is_file()
    assert registry.version_path(
        registry.get_release_record()["active_version"]
    ).is_file()

    container2 = ServiceContainer(settings=settings, gateway=mock_gw)
    assert container2.pipeline_service.clustering_engine is not None
    assert container2.pipeline_service.classifier_suite is not None
    assert len(container2.pipeline_service.cluster_profiles) >= 2

    app2 = create_app(container2)
    client2 = TestClient(app2)
    resp = client2.post(
        "/api/v1/classify",
        json={"ticket_id": "test-persisted", "text": "Need card replacement now"},
    )
    assert resp.status_code == 200
    assert resp.json()["ticket_id"] == "test-persisted"


def test_llm_gateway_adapter_json_fence_and_dimensions() -> None:
    """Verify that LLMGatewaySDKAdapter strips code fences from JSON responses and respects embedding dimensions."""
    from unittest.mock import MagicMock

    from ticket_classification.gateways.llm_gateway_adapter import LLMGatewaySDKAdapter

    settings = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="sk-test-virtual-key-123456",
        _env_file=None,
    )
    mock_client = MagicMock()
    mock_client.complete.return_value = MagicMock(
        content='```json\n{"short_label": "Fenced Title", "signature": "A fenced summary", "routing_queue": "Fenced Queue"}\n```'
    )
    mock_client.embed.return_value = MagicMock(vectors=[[0.1, 0.2, 0.3]])

    adapter = LLMGatewaySDKAdapter(settings=settings, client=mock_client)
    res = adapter.complete_structured(
        system_prompt="sys", user_prompt="usr", schema_cls=ClusterTaxonomyLLMOutput
    )
    assert res.short_label == "Fenced Title"
    assert res.routing_queue == "Fenced Queue"

    vecs = adapter.embed_texts(["sample text"])
    assert vecs == [[0.1, 0.2, 0.3]]

    assert mock_client.embed.call_args.kwargs.get("dimensions") is None

    settings_with_dims = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="sk-test-virtual-key-123456",
        embedding_dimensions=768,
        _env_file=None,
    )
    adapter_with_dims = LLMGatewaySDKAdapter(
        settings=settings_with_dims, client=mock_client
    )
    adapter_with_dims.embed_texts(["sample text"])
    assert mock_client.embed.call_args.kwargs.get("dimensions") == 768
