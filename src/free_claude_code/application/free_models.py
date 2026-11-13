"""Curated list of free-tier models shown in the admin UI.

Data lives in ``core/data/free_models.json`` (seeded from freeaiapi.org and the
Free API Hunter list, ``docs/FREE_AI_APIS.md``). ``rank`` orders models from
smartest to weakest; the loader keeps that order stable.
"""

import json
from pathlib import Path
from typing import Any

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "core" / "data" / "free_models.json"
)


def load_free_models_payload() -> dict[str, Any]:
    """Return the free-models payload with models sorted smartest-first."""
    with DATA_PATH.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    models = sorted(
        payload.get("free_models", []), key=lambda model: model.get("rank", 10**9)
    )
    return {**payload, "free_models": models}
