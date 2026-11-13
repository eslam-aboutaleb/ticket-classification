"""Explicit release policy for small synthetic fixtures that test mechanics, not ML quality."""

SYNTHETIC_RELEASE_POLICY = {
    "require_evaluation": False,
    "require_calibration": False,
    "min_language_samples": 0,
    "min_language_macro_f1": 0.0,
    "min_language_accepted_coverage": 0.0,
    "require_taxonomy_approval": False,
}
