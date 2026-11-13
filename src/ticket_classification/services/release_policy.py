"""Evaluate immutable candidate evidence before permitting production activation."""

import math
from typing import Any

from ticket_classification.config.pipeline import DomainProfile, ReleasePolicy


def assess_release(
    metrics: dict[str, Any], profile: DomainProfile, policy: ReleasePolicy
) -> dict[str, Any]:
    blockers = []
    if policy.require_evaluation and not metrics.get("evaluated"):
        blockers.append("held_out_evaluation_required")
    calibration = metrics.get("classification", {}).get("calibration")
    if policy.require_calibration and calibration != "held_out_known":
        blockers.append("held_out_calibration_required")
    evidence = metrics.get("release_gate", {})
    languages = evidence.get("supported_language_coverage", {})
    for language in profile.language.supported:
        item = languages.get(language, {})
        if item.get("count", 0) < policy.min_language_samples:
            blockers.append(f"insufficient_language_samples:{language}")
        if item.get("count", 0):
            for metric, threshold in [
                ("cluster_agreement_macro_f1", policy.min_language_macro_f1),
                ("accepted_coverage", policy.min_language_accepted_coverage),
            ]:
                value = item.get(metric)
                if value is None or not math.isfinite(value) or value < threshold:
                    blockers.append(f"language_quality:{language}:{metric}")
    if policy.min_oos_recall is not None:
        value = (
            metrics.get("classification", {})
            .get("slice_error_analysis", {})
            .get("open_set_oos_recall")
        )
        if value is None or not math.isfinite(value) or value < policy.min_oos_recall:
            blockers.append("unknown_intent_recall_below_threshold")
    classification = metrics.get("classification", {})
    if policy.min_ground_truth_macro_f1 is not None:
        selected = classification.get("selected_production_model")
        value = (
            classification.get("model_comparison", {})
            .get(selected, {})
            .get("test_ground_truth_macro_f1")
        )
        if (
            value is None
            or not math.isfinite(value)
            or value < policy.min_ground_truth_macro_f1
        ):
            blockers.append("business_macro_f1_below_threshold")
    if policy.max_known_false_positive_rate is not None:
        value = (
            classification.get("slice_error_analysis", {})
            .get("open_set_evaluation", {})
            .get("known_false_positive_rate")
        )
        if (
            value is None
            or not math.isfinite(value)
            or value > policy.max_known_false_positive_rate
        ):
            blockers.append("known_false_positive_rate_above_threshold")
    lineage = metrics.get("lineage", {})
    review = bool(
        lineage.get("births")
        or lineage.get("splits")
        or lineage.get("merges")
        or lineage.get("route_changes")
    )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "requires_human_taxonomy_review": policy.require_taxonomy_approval and review,
        "policy": policy.model_dump(),
    }
