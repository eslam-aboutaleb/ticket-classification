"""Regression coverage for discovery assignments and inductive prediction."""

import numpy as np
import pytest

from ticket_classification.domain.classification.engine import ClassifierSuite
from ticket_classification.domain.clustering.engine import ClusteringEngine


def test_discovery_preserves_noise_and_projection_is_batch_independent():
    data = np.random.default_rng(42).normal(size=(100, 32))
    engine = ClusteringEngine(k_min=2, k_max=4, n_components=10)
    labels, strengths = engine.fit_transform(data)
    np.testing.assert_array_equal(labels, engine.train_labels_)
    assert np.any(labels == -1)
    assert np.all(strengths[labels == -1] == 0)
    batch_labels, batch_scores = engine.transform(data[:3])
    for i in range(3):
        label, score = engine.transform(data[i : i + 1])
        assert label[0] == batch_labels[i]
        assert score[0] == pytest.approx(batch_scores[i])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_discovery(n):
    engine = ClusteringEngine()
    labels, _ = engine.fit_transform(np.ones((n, 8)))
    assert labels.tolist() == [0] * n
    assert engine.optimal_k_ == 1


def test_no_rare_class_relaxation_and_single_class_model():
    data = np.eye(6)
    suite = ClassifierSuite(min_data_per_class=3)
    assert suite.fit_all(data, data, np.arange(6), np.ones(6)) == {}
    preds, _, _, novel, _ = suite.predict_with_open_set(data, data)
    assert np.all(preds == -1) and np.all(novel)
    suite.fit_all(data, data, np.zeros(6, dtype=int), np.ones(6))
    assert suite.best_model_name_ == "single_cluster"
    assert suite.predict_with_open_set(data, data)[0].tolist() == [0] * 6


def test_held_out_novelty_calibration_matches_serving():
    suite = ClassifierSuite(min_data_per_class=2)
    data = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
    suite.fit_all(data, data, np.repeat([0, 1], 4), np.ones(8))
    held_out = np.array([[1.0, 0.1], [0.1, 1.0]])
    suite.calibrate_novelty(held_out)
    _, _, _, novel, distances = suite.predict_with_open_set(held_out, held_out)
    np.testing.assert_array_equal(novel, distances > suite.distance_threshold_)
