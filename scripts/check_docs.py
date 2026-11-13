"""Validate shipped user documentation links and example configurations."""

import re
from pathlib import Path

import yaml

from ticket_classification.config.pipeline import PipelineConfig

root = Path(__file__).resolve().parents[1]
for path in [root / "README.md", *sorted((root / "docs").rglob("*.md"))]:
    for target in re.findall(
        r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", path.read_text(encoding="utf-8")
    ):
        if target.startswith(("http:", "https:", "#", "mailto:")):
            continue
        relative = target.split("#", 1)[0]
        if relative:
            assert (path.parent / relative).exists(), (
                f"Broken link in {path.relative_to(root)}: {target}"
            )
for path in (root / "configs").glob("*.yaml"):
    PipelineConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
print("User documentation links and configuration examples verified.")
