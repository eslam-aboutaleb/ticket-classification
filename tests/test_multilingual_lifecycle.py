"""Offline contract tests; synthetic vectors do not benchmark a real language model."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from ticket_classification.config.settings import AppSettings, LLMProvider
from ticket_classification.controllers.api import ServiceContainer, create_app
from ticket_classification.gateways.llm_gateway_adapter import BaseCloudGateway


class MultilingualGateway(BaseCloudGateway):
    def __init__(self):
        self.provider = LLMProvider.LITELLM
        self.calls = 0
        self.malformed = False

    def get_active_provider(self):
        return self.provider

    def switch_provider(self, provider):
        self.provider = LLMProvider(provider)

    def embed_texts(self, texts, provider_override=None):
        self.calls += 1
        if self.malformed:
            return [[float("nan")]]
        vectors = []
        for text in texts:
            vector = np.zeros(8)
            lower = text.casefold()
            index = (
                0
                if "payment" in lower or "دفع" in text
                else 1
                if "delivery" in lower or "توصيل" in text
                else 2
            )
            vector[index] = 1.0
            vector[3:] = (
                np.frombuffer(
                    hashlib.sha256(text.encode()).digest()[:5], dtype=np.uint8
                )
                / 25500
            )
            vectors.append(vector.tolist())
        return vectors

    def complete_structured(self, **kwargs):
        raise ValueError(
            "Naming/reranking is disabled in these offline lifecycle tests"
        )


@pytest.fixture
def setup(tmp_path):
    config = {
        "classification": {
            "enable_llm_reranker": False,
            "lexical_weight": 0.0,
            "metadata_weight": 0.0,
        },
        "profiles": {
            "claims": {
                "id": "claims",
                "text_col": "narrative",
                "id_col": "id",
                "group_col": "conversation",
                "routes": {"triage": "Claims triage"},
                "default_route": "triage",
                "metadata": {"numeric_cols": ["amount"]},
            }
        },
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = AppSettings(
        gateway_base="https://gateway.example/v1",
        gateway_key="test-key",
        embedding_dimensions=8,
        config_path=str(path),
        data_dir=str(tmp_path),
        _env_file=None,
    )
    gateway = MultilingualGateway()
    return ServiceContainer(settings, gateway), gateway, path


def records():
    templates = [
        "My payment failed and I need help with this transaction",
        "حصلت مشكلة في دفع الفلوس ومحتاج مساعدة في المعاملة",
        "My delivery has not arrived and I need help finding it",
        "عندي مشكلة في توصيل الطلب ومحتاج اعرف هو فين دلوقتي",
    ]
    return [
        {
            "id": f"{i}-{j}",
            "narrative": f"{text} example {i}",
            "conversation": f"thread-{i}",
            "amount": i,
        }
        for i in range(18)
        for j, text in enumerate(templates)
    ]


def test_multilingual_fit_classify_reload_and_domain_isolation(setup):
    container, gateway, _ = setup
    client = TestClient(create_app(container))
    fitted = client.post(
        "/api/v1/fit",
        json={"domain_id": "claims", "records": records(), "use_llm_naming": False},
    )
    assert fitted.status_code == 200, fitted.text
    body = fitted.json()
    assert body["clustering"]["optimal_k"] == 2
    version = body["taxonomy_version"]
    service = container.get_service("claims")
    english = service.classify_single_ticket(
        "en",
        "My payment failed and I need help with this transaction",
        enable_llm_reranker=False,
    )
    arabic = service.classify_single_ticket(
        "ar",
        "حصلت مشكلة في دفع الفلوس ومحتاج مساعدة في المعاملة",
        enable_llm_reranker=False,
    )
    assert english.predicted_cluster_id == arabic.predicted_cluster_id >= 0
    assert english.stable_cluster_id == arabic.stable_cluster_id
    assert arabic.input_languages == ["ar"]
    assert arabic.taxonomy_version == version
    assert container.get_service("generic").clustering_engine is None
    reloaded = ServiceContainer(container.settings, gateway).get_service("claims")
    result = reloaded.classify_single_ticket(
        "reload",
        "حصلت مشكلة في دفع الفلوس ومحتاج مساعدة في المعاملة",
        enable_llm_reranker=False,
        display_language="ar",
    )
    assert result.stable_cluster_id == arabic.stable_cluster_id
    assert result.confidence == arabic.confidence
    assert (
        Path(container.settings.data_dir)
        / "artifacts/claims/versions"
        / f"{version}.joblib"
    ).is_file()


def test_unknown_and_language_coverage_are_distinct_and_queued(setup):
    container, gateway, _ = setup
    service = container.get_service("claims")
    metrics = service.fit_dataframe(
        pd.DataFrame(records()),
        use_llm_naming=False,
        oos_texts=[
            "The office espresso machine needs maintenance and a replacement gasket"
        ],
    )
    assert (
        metrics["classification"]["slice_error_analysis"]["open_set_evaluation"][
            "auroc"
        ]
        == 1.0
    )
    unknown = service.classify_single_ticket(
        "novel",
        "The office espresso machine needs maintenance and a replacement gasket",
    )
    assert unknown.is_open_set_novel and unknown.predicted_cluster_id == -1
    assert unknown.stable_cluster_id is None and unknown.route_id is None
    assert unknown.status == "unknown"
    calls = gateway.calls
    for ticket_id, text in [
        ("empty", ""),
        (
            "fr",
            "Bonjour je souhaite contester ce paiement et recevoir un remboursement",
        ),
    ]:
        response = service.classify_single_ticket(ticket_id, text)
        assert response.status == "needs_review" and not response.is_open_set_novel
        assert response.predicted_cluster_id == -1
    assert gateway.calls == calls
    items = service.review_queue.list_items("claims")
    assert {item["ticket_id"] for item in items} == {"novel", "empty", "fr"}


def test_failed_refit_preserves_previous_model_and_embedding_identity(setup):
    container, gateway, _ = setup
    service = container.get_service("claims")
    frame = pd.DataFrame(records())
    service.fit_dataframe(frame, use_llm_naming=False)
    version = service.taxonomy_version
    gateway.malformed = True
    frame.loc[0, "narrative"] = "A new payment question that has not been cached"
    with pytest.raises(ValueError, match="finite vector"):
        service.fit_dataframe(frame, use_llm_naming=False)
    assert service.taxonomy_version == version
    gateway.malformed = False
    service.settings.embedding_revision = "different-revision"
    with pytest.raises(ValueError, match="refit"):
        service.classify_single_ticket("x", "My payment failed and I need help")


def test_group_split_validation_and_missing_metadata(setup):
    container, _, _ = setup
    service = container.get_service("claims")
    frame = pd.DataFrame(records())
    canonical = service.dataset_repo.adapt_frame(frame, service.profile)
    split, audit = service.audit_and_split_dataset(canonical)
    assert split.groupby("conversation")["split"].nunique().max() == 1
    assert audit["split_groups"] == 18
    bad = canonical.iloc[:2].copy()
    bad["split"] = ["train", "test"]
    with pytest.raises(ValueError, match="spans"):
        service.audit_and_split_dataset(bad)
    service.fit_dataframe(frame, use_llm_naming=False)
    result = service.classify_single_ticket(
        "missing-metadata", "My payment failed and I need help with this transaction"
    )
    assert result.predicted_cluster_id >= 0


def test_tiny_unlabeled_fit_and_api_errors(setup):
    container, _, _ = setup
    client = TestClient(create_app(container))
    result = client.post(
        "/api/v1/fit",
        json={
            "records": [{"raw_text": "Just one payment issue"}],
            "use_llm_naming": False,
        },
    )
    assert result.status_code == 200, result.text
    assert not result.json()["evaluated"]
    assert (
        client.post("/api/v1/classify", json={"text": "", "ticket_id": "empty"}).json()[
            "status"
        ]
        == "needs_review"
    )
    assert (
        client.post(
            "/api/v1/fit", json={"records": [{"raw_text": "x"}], "strategy": "invalid"}
        ).status_code
        == 400
    )
    assert client.get("/api/v1/clusters?domain_id=unknown").status_code == 400
    assert client.get("/api/v1/review-items?limit=0").status_code == 400
    assert client.get("/api/v1/review-items").status_code == 200


def test_temporal_splits_validate_supplied_chronology(setup):
    container, _, config_path = setup
    config = yaml.safe_load(config_path.read_text())
    config["profiles"]["claims"].update(split_mode="temporal", timestamp_col="created")
    config_path.write_text(yaml.safe_dump(config))
    service = container.get_service("claims")
    frame = pd.DataFrame(records())
    frame["created"] = pd.date_range("2025-01-01", periods=len(frame))
    frame = service.dataset_repo.adapt_frame(frame, service.profile)
    split, _ = service.audit_and_split_dataset(frame)
    assert (
        split.loc[split["split"].eq("train"), "created"].max()
        <= split.loc[split["split"].eq("val"), "created"].min()
    )
    split["created"] = split["created"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="temporal boundaries"):
        service.audit_and_split_dataset(split)
    frame["created"] = None
    with pytest.raises(ValueError, match="valid timestamps"):
        service.audit_and_split_dataset(frame)
