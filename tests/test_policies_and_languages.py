"""Domain policies, multilingual preservation and geometry-driven discovery."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ticket_classification.config.pipeline import (
    CleaningPolicy,
    LanguagePolicy,
    PipelineConfig,
)
from ticket_classification.domain.cleaning.handlers import build_default_cleaning_chain
from ticket_classification.domain.clustering.engine import ClusteringEngine
from ticket_classification.domain.clustering.lineage import build_lineage
from ticket_classification.domain.features.metadata_encoder import (
    MetadataFeaturePipeline,
)
from ticket_classification.models.entities import TicketRecord


def test_unicode_mixed_language_and_preserved_codes():
    rec = TicketRecord(
        "1",
        "مرحبا عندي مشكلة في الحساب وعايز استرجع الفلوس payment failed HTTP 403 not 404 <ERROR>",
    )
    build_default_cleaning_chain().handle(rec)
    assert {"ar", "en"} <= set(rec.languages)
    assert "مشكلة" in rec.text_lexical
    assert "not" in rec.text_lexical
    assert "403" in rec.text_lexical and "404" in rec.text_lexical
    assert "<ERROR>" in rec.text_contextual
    assert rec.language_status == "supported"
    assert "403" in rec.template_signature


def test_arabic_technical_code_is_not_an_unsupported_third_language():
    record = TicketRecord(
        "ar-http", "السيرفر بيرجع خطأ HTTP 500 كل ما افتح لوحة التحكم في التطبيق"
    )
    build_default_cleaning_chain().handle(record)
    assert record.languages == ["ar"]
    assert record.language_status == "supported"
    assert "http" in record.text_lexical and "500" in record.text_lexical


def test_configurable_cleaning_and_arabic_normalization():
    rec = TicketRecord("1", "From: user@example.com\nأَنَا لم أدفع ORDER-123")
    build_default_cleaning_chain(
        CleaningPolicy(strip_email_headers=False, mask_pii=False),
        LanguagePolicy(arabic_remove_diacritics=True, arabic_fold_alef=True),
    ).handle(rec)
    assert "user@example.com" in rec.text_contextual
    assert "انا" in rec.text_lexical and "لم" in rec.text_lexical


def test_short_unsupported_and_cjk_inputs_do_not_disappear():
    for text, status in [
        ("hi", "undetermined"),
        (
            "Bonjour je souhaite contester ce paiement et recevoir un remboursement",
            "unsupported",
        ),
    ]:
        rec = TicketRecord("1", text)
        build_default_cleaning_chain().handle(rec)
        assert rec.language_status == status
    rec = TicketRecord("2", "银行卡支付失败需要退款")
    build_default_cleaning_chain(language=LanguagePolicy(supported=["zh-cn"])).handle(
        rec
    )
    assert "银" in rec.text_lexical and "退" in rec.text_lexical


def test_strict_config_and_metadata_columns_are_frozen():
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"clustering": {"default_strategy": "nmf_topic"}})
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"cleaning": {"unknown_toggle": True}})
    pipeline = MetadataFeaturePipeline(
        numeric_cols=["amount"], nominal_cols=["product"], date_cols=["created"]
    )
    data = pd.DataFrame(
        {"raw_text": ["first", "second"], "amount": [1, 3], "product": ["a", "b"]}
    )
    fitted = pipeline.fit_transform(data)
    assert (
        pipeline.transform(pd.DataFrame({"raw_text": ["third"]})).shape[1]
        == fitted.shape[1]
    )
    assert pipeline.transform(data[:0]).shape == (0, fitted.shape[1])


def test_hdbscan_and_kmeans_discover_embedding_groups():
    rng = np.random.default_rng(5)
    data = np.vstack([center + rng.normal(0, 0.01, (16, 6)) for center in np.eye(6)])
    for strategy in ("hdbscan_density", "aligned_kmeans"):
        engine = ClusteringEngine(strategy=strategy, reducer="none")
        labels, _ = engine.fit_transform(data)
        assert engine.optimal_k_ == 6
        assert np.mean(labels >= 0) > 0.9
        assert engine.diagnostics_["candidates"]


def test_lineage_handles_births_retirement_and_changed_k():
    old = np.eye(3)
    current = np.vstack(
        [
            old[1],
            [
                0,
                0,
                0,
            ],
        ]
    )
    current[1] = [-1, 0, 0]
    ids, lineage = build_lineage(old, current, {0: "a", 1: "b", 2: "c"}, True)
    assert ids[0] == "b"
    assert lineage["retired"] == ["a", "c"]
    assert len(lineage["births"]) == 1
