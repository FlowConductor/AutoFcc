"""Automatic model ranking from public AI leaderboards.

Two sources are supported:

* ``lmarena`` (default): daily structured snapshots of the Arena AI
  (LMArena) text leaderboard published as JSON on GitHub. No API key
  is required.
* ``artificial_analysis``: the Artificial Analysis models API sorted by
  intelligence index. Requires ``ARTIFICIAL_ANALYSIS_API_KEY``.

The resulting order is cached in memory and persisted to
``~/.fcc/auto_rank_cache.json`` so restarts can reuse the last successful
ranking without network access.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Protocol

import httpx
from loguru import logger

from free_claude_code.config.paths import config_dir_path
from free_claude_code.config.settings import Settings

LMARENA_SNAPSHOT_INDEX_URL = (
    "https://raw.githubusercontent.com/oolong-tea-2026/"
    "arena-ai-leaderboards/main/data/latest.json"
)
LMARENA_SNAPSHOT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/oolong-tea-2026/"
    "arena-ai-leaderboards/main/data/{date}/text.json"
)
ARTIFICIAL_ANALYSIS_MODELS_URL = (
    "https://artificialanalysis.ai/api/v2/data/llms/models"
    "?is_metric_independent=true"
)
AUTO_RANK_CACHE_FILENAME = "auto_rank_cache.json"
_FETCH_TIMEOUT_SECONDS = 20.0
_MIN_MATCH_SCORE = 0.5


@dataclass(frozen=True, slots=True)
class RankedModel:
    """One leaderboard model and its ordering score (higher is better)."""

    name: str
    rank: int
    score: float


class LeaderboardFetchError(RuntimeError):
    """One leaderboard source could not be fetched or parsed."""


class LeaderboardSource(Protocol):
    """One public AI ranking source."""

    name: str

    async def fetch(self, client: httpx.AsyncClient) -> tuple[RankedModel, ...]: ...


class LMArenaSnapshotSource:
    """Arena AI (LMArena) text leaderboard via daily JSON snapshots."""

    name = "lmarena"

    async def fetch(self, client: httpx.AsyncClient) -> tuple[RankedModel, ...]:
        index_response = await client.get(LMARENA_SNAPSHOT_INDEX_URL)
        index_response.raise_for_status()
        snapshot_date = (index_response.json()).get("date")
        if not isinstance(snapshot_date, str) or not snapshot_date:
            raise LeaderboardFetchError("LMArena snapshot index has no date")
        response = await client.get(
            LMARENA_SNAPSHOT_URL_TEMPLATE.format(date=snapshot_date)
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise LeaderboardFetchError("LMArena snapshot has no models list")
        entries: list[RankedModel] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("model")
            rank = item.get("rank")
            score = item.get("score")
            if not isinstance(name, str) or not name:
                continue
            entries.append(
                RankedModel(
                    name=name,
                    rank=rank if isinstance(rank, int) else len(entries) + 1,
                    score=float(score) if isinstance(score, int | float) else 0.0,
                )
            )
        if not entries:
            raise LeaderboardFetchError("LMArena snapshot contained no models")
        return tuple(entries)


class ArtificialAnalysisSource:
    """Artificial Analysis models API ordered by intelligence index."""

    name = "artificial_analysis"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    async def fetch(self, client: httpx.AsyncClient) -> tuple[RankedModel, ...]:
        if not self._api_key:
            raise LeaderboardFetchError(
                "ARTIFICIAL_ANALYSIS_API_KEY is required for "
                "AUTO_RANK_SOURCE=artificial_analysis"
            )
        response = await client.get(
            ARTIFICIAL_ANALYSIS_MODELS_URL,
            headers={"x-api-key": self._api_key},
        )
        response.raise_for_status()
        payload = response.json()
        models = payload if isinstance(payload, list) else payload.get("models")
        if not isinstance(models, list):
            raise LeaderboardFetchError(
                "Artificial Analysis response has no models list"
            )
        entries: list[RankedModel] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model_name")
            if not isinstance(name, str) or not name:
                continue
            score = next(
                (
                    float(item[key])
                    for key in ("intelligence_index", "quality_index", "index")
                    if isinstance(item.get(key), int | float)
                ),
                0.0,
            )
            entries.append(RankedModel(name=name, rank=0, score=score))
        entries.sort(key=lambda entry: entry.score, reverse=True)
        for position, entry in enumerate(entries, start=1):
            entries[position - 1] = RankedModel(
                name=entry.name, rank=position, score=entry.score
            )
        if not entries:
            raise LeaderboardFetchError("Artificial Analysis contained no models")
        return tuple(entries)


def _tokenize(name: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[^a-z0-9]+", name.lower()) if part)


def match_score(leaderboard_name: str, provider_model: str) -> float:
    """Return overlap strength between a leaderboard and provider model name."""

    leaderboard_tokens = set(_tokenize(leaderboard_name))
    provider_tokens = set(_tokenize(provider_model))
    if not leaderboard_tokens or not provider_tokens:
        return 0.0
    overlap = leaderboard_tokens & provider_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(leaderboard_tokens)
    precision = len(overlap) / len(provider_tokens)
    score = coverage * precision
    if coverage == 1.0 and precision == 1.0:
        score = 1.0
    return score


def best_match(
    ranked: tuple[RankedModel, ...], provider_model: str
) -> tuple[RankedModel, float] | None:
    """Return the strongest leaderboard match for one provider model."""

    best: tuple[RankedModel, float] | None = None
    for entry in ranked:
        score = match_score(entry.name, provider_model)
        if score < _MIN_MATCH_SCORE:
            continue
        if best is None or score > best[1]:
            best = (entry, score)
    return best


def order_refs_by_rank(
    provider_model_refs: tuple[str, ...], ranked: tuple[RankedModel, ...]
) -> tuple[str, ...]:
    """Order provider model refs smartest-first; unmatched refs trail."""

    matched: list[tuple[int, float, str]] = []
    unmatched: list[str] = []
    for ref in provider_model_refs:
        provider_model = ref.partition("/")[2] or ref
        match = best_match(ranked, provider_model)
        if match is None:
            unmatched.append(ref)
            continue
        matched.append((match[0].rank, -match[1], ref))
    ordered = (ref for *_head, ref in sorted(matched))
    return (*ordered, *sorted(unmatched))


class AutoRankChain:
    """In-memory holder for the last successfully built ranking order."""

    def __init__(self) -> None:
        self._chain: tuple[str, ...] | None = None
        self._source: str | None = None
        self._entry_count: int = 0
        self._fetched_at: float | None = None

    def update(
        self,
        chain: tuple[str, ...],
        *,
        source: str,
        entry_count: int,
        fetched_at: float | None = None,
    ) -> None:
        self._chain = tuple(chain)
        self._source = source
        self._entry_count = entry_count
        self._fetched_at = fetched_at if fetched_at is not None else time()

    def get(self) -> tuple[str, ...] | None:
        return self._chain

    def age_seconds(self) -> float | None:
        """Return seconds since the last successful refresh, or None."""

        if self._fetched_at is None:
            return None
        return time() - self._fetched_at

    def status(self) -> dict[str, object]:
        return {
            "source": self._source,
            "entry_count": self._entry_count,
            "chain": list(self._chain or ()),
            "fetched_at": (
                self._fetched_at if self._fetched_at is not None else None
            ),
        }

    def clear(self) -> None:
        self._chain = None
        self._source = None
        self._entry_count = 0
        self._fetched_at = None


auto_rank_chain = AutoRankChain()
"""Process-wide default chain consumed by the model router."""


def auto_rank_cache_path() -> Path:
    """Return the persisted ranking cache path under ``~/.fcc``."""

    return config_dir_path() / AUTO_RANK_CACHE_FILENAME


def save_chain_cache(
    path: Path, chain: tuple[str, ...], *, source: str, entry_count: int
) -> None:
    payload = {
        "source": source,
        "entry_count": entry_count,
        "fetched_at": time(),
        "chain": list(chain),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Auto-rank cache write failed: {}", exc)


def load_cached_chain(path: Path) -> tuple[tuple[str, ...], float | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (), None
    chain = payload.get("chain") if isinstance(payload, dict) else None
    fetched_at = payload.get("fetched_at") if isinstance(payload, dict) else None
    if not isinstance(chain, list) or not all(isinstance(item, str) for item in chain):
        return (), None
    if not isinstance(fetched_at, int | float):
        fetched_at = None
    return tuple(chain), fetched_at


def resolve_source(settings: Settings) -> LeaderboardSource:
    if settings.auto_rank_source == "artificial_analysis":
        return ArtificialAnalysisSource(settings.artificial_analysis_api_key)
    return LMArenaSnapshotSource()


async def fetch_ranked_models(
    settings: Settings, *, client: httpx.AsyncClient | None = None
) -> tuple[tuple[RankedModel, ...], str]:
    """Fetch ranked models from the configured source."""

    source = resolve_source(settings)
    if client is not None:
        return await source.fetch(client), source.name
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS) as owned:
        return await source.fetch(owned), source.name


async def refresh_auto_rank_chain(
    settings: Settings,
    provider_model_refs: tuple[str, ...],
    *,
    client: httpx.AsyncClient | None = None,
    cache_path: Path | None = None,
) -> tuple[str, ...] | None:
    """Fetch rankings and rebuild the process-wide auto-ranked chain."""

    if settings.model_routing != "auto":
        return None
    try:
        ranked, source_name = await fetch_ranked_models(settings, client=client)
    except (httpx.HTTPError, LeaderboardFetchError, ValueError) as exc:
        logger.warning(
            "Auto-rank refresh skipped: source={} reason={}",
            settings.auto_rank_source,
            exc,
        )
        return None
    chain = order_refs_by_rank(provider_model_refs, ranked)
    if not chain:
        logger.warning("Auto-rank chain is empty after matching; keeping previous")
        return None
    auto_rank_chain.update(chain, source=source_name, entry_count=len(ranked))
    path = cache_path if cache_path is not None else auto_rank_cache_path()
    from functools import partial

    await asyncio.get_running_loop().run_in_executor(
        None,
        partial(
            save_chain_cache,
            path,
            chain,
            source=source_name,
            entry_count=len(ranked),
        ),
    )
    logger.info(
        "Auto-rank chain updated: source={} models={} primary={}",
        source_name,
        len(ranked),
        chain[0],
    )
    return chain


def restore_cached_chain(
    settings: Settings, cache_path: Path | None = None
) -> tuple[str, ...] | None:
    """Load the persisted chain into memory when auto routing is enabled."""

    if settings.model_routing != "auto":
        return None
    existing = auto_rank_chain.get()
    if existing is not None:
        return existing
    path = cache_path if cache_path is not None else auto_rank_cache_path()
    chain, fetched_at = load_cached_chain(path)
    if chain:
        auto_rank_chain.update(
            chain, source="cache", entry_count=0, fetched_at=fetched_at
        )
    return chain
