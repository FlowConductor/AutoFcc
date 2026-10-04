"""Process-wide failover state for quota-exhausted provider models.

When a provider/model exhausts its quota (billing failures such as HTTP 402
or sustained rate limits), the registry marks it unavailable for a cooldown
window. The executor reorders candidates so healthy models are tried first
and exhausted models are deferred until their cooldown expires.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from loguru import logger

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.trace import trace_event


def is_quota_exhaustion_failure(failure: ExecutionFailure) -> bool:
    """Return whether one failure means the provider/model ran out of quota."""

    if failure.kind is FailureKind.PERMISSION and failure.status_code == 402:
        return True
    return failure.kind is FailureKind.RATE_LIMIT and failure.status_code == 429


def cooldown_seconds_for_failure(
    failure: ExecutionFailure,
    *,
    quota_cooldown_seconds: float,
    rate_limit_cooldown_seconds: float,
) -> float | None:
    """Return the cooldown for one failure, or None when it is not quota-like."""

    if failure.kind is FailureKind.PERMISSION and failure.status_code == 402:
        return quota_cooldown_seconds
    if failure.kind is FailureKind.RATE_LIMIT and failure.status_code == 429:
        return rate_limit_cooldown_seconds
    return None


@dataclass(frozen=True, slots=True)
class FailoverEntry:
    """One recorded exhaustion and when its cooldown expires."""

    provider_model_ref: str
    expires_at: float
    reason: str


class FailoverRegistry:
    """Track quota-exhausted provider models with monotonic cooldowns."""

    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, FailoverEntry] = {}

    def mark_exhausted(
        self, provider_model_ref: str, seconds: float, *, reason: str
    ) -> None:
        """Record one exhaustion with a cooldown window."""

        if seconds <= 0:
            return
        now = self._clock()
        with self._lock:
            self._entries[provider_model_ref] = FailoverEntry(
                provider_model_ref=provider_model_ref,
                expires_at=now + seconds,
                reason=reason,
            )
        logger.info(
            "Failover cooldown: provider_model='{}' seconds={} reason={}",
            provider_model_ref,
            seconds,
            reason,
        )

    def cooldown_remaining(self, provider_model_ref: str) -> float:
        """Return remaining cooldown seconds for one provider model."""

        now = self._clock()
        with self._lock:
            entry = self._entries.get(provider_model_ref)
            if entry is None:
                return 0.0
            if entry.expires_at <= now:
                del self._entries[provider_model_ref]
                return 0.0
            return entry.expires_at - now

    def is_exhausted(self, provider_model_ref: str) -> bool:
        """Return whether one provider model is still in cooldown."""

        return self.cooldown_remaining(provider_model_ref) > 0.0

    def active_entries(self) -> tuple[FailoverEntry, ...]:
        """Return every cooldown still active, ordered by expiry."""

        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            return tuple(sorted(self._entries.values(), key=lambda e: e.expires_at))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_expired(self, now: float) -> None:
        expired = [
            ref for ref, entry in self._entries.items() if entry.expires_at <= now
        ]
        for ref in expired:
            del self._entries[ref]


def order_candidates_by_cooldown(
    registry: FailoverRegistry, candidates: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split candidates into healthy and cooling lists, both order-preserving."""

    healthy: list[str] = []
    cooling: list[str] = []
    for candidate in candidates:
        (cooling if registry.is_exhausted(candidate) else healthy).append(candidate)
    return tuple(healthy), tuple(cooling)


failover_registry = FailoverRegistry()
"""Process-wide default registry shared by every provider executor."""


def trace_failover_reorder(
    *, request_id: str, primary_ref: str, healthy: tuple[str, ...], cooling: tuple[str, ...]
) -> None:
    trace_event(
        stage="execution",
        event="free_claude_code.failover.reordered",
        source="application",
        request_id=request_id,
        primary_provider_model_ref=primary_ref,
        healthy_count=len(healthy),
        cooling_provider_model_refs=cooling,
    )
