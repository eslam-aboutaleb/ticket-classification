"""Independent known/OOS cohorts, leakage handling and validation-only thresholds."""

import numpy as np
import pandas as pd
import pytest
from test_production_readiness import ControlledGateway, _make_settings

from ticket_classification.controllers.api import ServiceContainer
from ticket_classification.domain.classification.engine import ClassifierSuite
from ticket_classification.services.evaluation_service import evaluate_open_set


def _vectors(distances):
    similarities = 1 - np.asarray(distances)
    return np.column_stack([similarities, np.sqrt(1 - similarities**2)])


def test_lower_fpr_threshold_uses_known_validation_and_all_known_test_rows():
    suite = ClassifierSuite(open_set_distance_quantile=0.99)
    suite.centroids_ = np.array([[1.0, 0.0]])
    suite.calibrate_novelty(_vectors(np.linspace(0.01, 1.0, 100)))
    known = _vectors([0.5, 0.97, 0.985])
    oos = _vectors([1.2, 1.3])
    threshold = suite.distance_threshold_
    result = evaluate_open_set(suite, known, oos, 0.99)
    baseline, lower_fpr = result["operating_points"]
    assert result["known_test_count"] == 3
    assert baseline["known_false_positive_count"] == 2
    assert lower_fpr["known_false_positive_count"] == 0
    assert lower_fpr["oos_true_positive_count"] == 2
    assert result["known_false_positive_rate"] == 0
    assert result["probe_recall"] == 1
    evaluate_open_set(suite, known, _vectors([0.01, 0.02]), 0.99)
    assert suite.distance_threshold_ == threshold
    assert result["calibration_uses_oos_test"] is False


def test_explicit_overlap_cleanup_preserves_test_and_default_remains_strict(tmp_path):
    container = ServiceContainer(_make_settings(tmp_path), ControlledGateway())
    frame = pd.DataFrame(
        {
            "ticket_id": [
                "train-duplicate",
                "val-duplicate",
                "test-original",
                "train-safe",
                "val-safe",
            ],
            "raw_text": ["A repeated payment question"] * 3
            + ["My delivery is late", "I need a card replacement"],
            "split": ["train", "val", "test", "train", "val"],
            "label": ["payment"] * 3 + ["delivery", "card"],
        }
    )
    try:
        service = container.pipeline_service
        with pytest.raises(ValueError, match="spans explicit dataset splits"):
            service.audit_and_split_dataset(frame)
        profile = service.profile.model_copy(
            update={"cross_split_duplicates": "prefer_holdout"}, deep=True
        )
        retained, audit = service.audit_and_split_dataset(frame, profile=profile)
        assert retained.loc[retained.split == "test", "ticket_id"].tolist() == [
            "test-original"
        ]
        assert set(retained.ticket_id) == {"test-original", "train-safe", "val-safe"}
        assert audit["input_rows"] == 5
        assert audit["total_rows"] == 3
        assert audit["cross_split_duplicates_removed"] == 2
    finally:
        container.close()
