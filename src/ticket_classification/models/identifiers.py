"""Opaque model versions are identifiers, never filesystem paths."""

VERSION_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$"

EXPECTED_VERSION_PATTERN = r"^(?:[a-zA-Z0-9][a-zA-Z0-9_-]{0,127})?$"
