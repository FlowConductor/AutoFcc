"""Provider-level quota tracking with documented free-tier limits.

Fark: mevcut ``FailoverRegistry`` MODEL bazli cooldown tutar; bu modul
SAGLAYICI bazli limit olcer. Bir saglayicinin gunluk limiti doldugunda
ayni saglayicinin diger modelleri denenmez — zincirde direkt atlanir.

Kurallar:
- Belgelenmis limitler (PROVIDER_LIMITS) FREE_AI_APIS.md / saglayici
  dokumanlarindan gelir: attempts >= rpd → saglayici gun batimina kadar
  tukendi sayilir.
- Gozlemlenen hatalar: 402 (kota/odeme) → saglayici gun batimina kadar
  tukendi; 429 (rate limit) → saglayici kisa cooldown (org seviyesinde
  limitler modeller arasinda paylasilir).
- Kullanim kalici olarak ``~/.fcc/provider_usage.json`` dosyasina yazilir;
  gun (UTC) degisince sayaclar sifirlanir.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Yerlesik yedek katalog (siteden veri gelmezse). Siteden cekilen
# provider_limits.json bu degerlerle birlestirilir - SITEDEN GELEN KAZANIR.
FALLBACK_LIMITS: dict[str, dict[str, int]] = {
    "groq": {"rpm": 30, "rpd": 1000, "tpd": 200_000},
    "cerebras": {"rpm": 5, "tpm": 30_000, "tpd": 1_000_000},
    "llm7": {"rpd": 200},
    "cloudflare": {"neurons_per_day": 10_000},
    "huggingface": {},
    "cohere": {},
    "sambanova": {},
    "nvidia_nim": {},
    "gmicloud": {},
    "zai_api": {},
    "mistral": {},
    "gemini": {},
    "kilo": {},
    "open_router": {"rpd": 50},
}

_LIMIT_KEYS = ("rpm", "rpd", "tpm", "tpd")


def _load_limits() -> (
    tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]
):
    """Kaynak sitelerden cekilen limitler + yerlesik yedek (siteden gelen kazanir).

    core/data/provider_limits.json, freeApi scraper'lari tarafindan
    (scrapers/extract_limits.py) saglayici sitelerinden cekilir. Site limiti
    gercek davranisla celisirse o saglayici yerel olcume gecer (learned_rpm).
    """
    scraped: dict[str, dict[str, int]] = {}
    scraped_models: dict[str, dict[str, dict[str, int]]] = {}
    try:
        raw = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "core"
                / "data"
                / "provider_limits.json"
            ).read_text(encoding="utf-8")
        )
        for pid, lim in (raw.get("providers") or {}).items():
            numeric = {
                key: value
                for key, value in lim.items()
                if key in _LIMIT_KEYS and isinstance(value, int)
            }
            if numeric:
                scraped[pid] = numeric
            models_raw = lim.get("models")
            if isinstance(models_raw, dict):
                model_map: dict[str, dict[str, int]] = {}
                for model_name, model_lim in models_raw.items():
                    if isinstance(model_lim, dict):
                        clean = {
                            key: value
                            for key, value in model_lim.items()
                            if key in _LIMIT_KEYS and isinstance(value, int)
                        }
                        if clean:
                            model_map[model_name] = clean
                if model_map:
                    scraped_models[pid] = model_map
    except Exception as exc:
        logger.warning("provider_limits.json okunamiyor, yerlesik katalog: {}", exc)
    merged: dict[str, dict[str, int]] = {}
    for pid, fallback in FALLBACK_LIMITS.items():
        merged[pid] = {**fallback, **scraped.get(pid, {})}
    for pid, limits in scraped.items():
        merged.setdefault(pid, limits)
    return merged, scraped_models


# Birincil limit katalogu: siteden cekilen + yerlesik yedek birlesimi
PROVIDER_LIMITS, PROVIDER_MODEL_LIMITS = _load_limits()


def parse_limit_overrides(spec: str | None) -> dict[str, int]:
    """'provider=rpd,provider=rpd' metnini {provider: rpd} sozlugune cevir."""
    overrides: dict[str, int] = {}
    for part in (spec or "").split(","):
        provider, separator, limit = part.strip().partition("=")
        if not separator or not provider.strip() or not limit.strip().isdigit():
            continue
        overrides[provider.strip()] = int(limit.strip())
    return overrides


def apply_limit_overrides(spec: str | None) -> dict[str, int]:
    """Belgelenmis gunluk limitleri saglayici bazinda gecersiz kil.

    OpenRouter'in ucretsiz model tavani 50 istek/gun; hesaba BIR KEZ 10$
    kredi yuklendiginde tavan omur boyu 1.000 istek/gun olur. Kazinan
    katalog bunu bilemez, bu yuzden kullanici
    ``PROVIDER_DAILY_LIMIT_OVERRIDES=open_router=1000`` ile bildirir.
    """
    overrides = parse_limit_overrides(spec)
    for provider, rpd in overrides.items():
        PROVIDER_LIMITS.setdefault(provider, {})["rpd"] = rpd
    if overrides:
        logger.info("Provider gunluk limit gecersiz kilindi: {}", overrides)
    return overrides


def _utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _seconds_until_utc_midnight() -> float:
    now = datetime.now(UTC)
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if tomorrow <= now:
        import calendar

        tomorrow = tomorrow.replace(day=now.day)
        tomorrow = datetime.fromtimestamp(
            calendar.timegm(tomorrow.timetuple()) + 86400, tz=UTC
        )
    return max(1.0, (tomorrow - now).total_seconds())


@dataclass
class ProviderState:
    """One provider's counters and exhaustion windows for a UTC day."""

    day: str = field(default_factory=_utc_day)
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    exhausted_reason: str | None = None
    exhausted_until: float | None = None
    rate_limited_until: float | None = None
    # Site limiti gercek davranisla celisince ogrenilen etkili RPM (yerel olcum)
    learned_rpm: int | None = None
    # MODEL bazli havuzlar (org ile ayni olmayan limitler - or. Gemini Pro 50
    # RPD vs Flash 1.500 RPD): model adi -> gunluk istek sayisi
    model_attempts: dict[str, int] = field(default_factory=dict)
    # gunu dolan modeller: model adi -> tekrar acilma zamani (epoch)
    model_exhausted: dict[str, float] = field(default_factory=dict)


class ProviderQuotaTracker:
    """Thread-safe, persistent provider-level quota measurement."""

    def __init__(self, path: Path | None = None) -> None:
        from free_claude_code.config.paths import config_dir_path

        self._path = (
            path if path is not None else config_dir_path() / "provider_usage.json"
        )
        # RLock: status() icinde sorgu metodlari tekrar kilit alabilir
        self._lock = threading.RLock()
        self._states: dict[str, ProviderState] = {}
        # RPM penceresi: provider -> son 60sn'deki istek zamanlari
        self._recent_attempts: dict[str, list[float]] = {}
        self._overrides_applied = False
        self._load()

    def _ensure_overrides(self) -> None:
        """PROVIDER_DAILY_LIMIT_OVERRIDES ayarini bir kez uygula.

        Ayarlar import aninda hazir olmadigi icin tembel calisir; ilk kota
        sorgusunda okunur. Ayar degisirse surec yeniden baslatilmalidir.
        """
        if self._overrides_applied:
            return
        self._overrides_applied = True
        try:
            from free_claude_code.config.loader import get_settings

            apply_limit_overrides(get_settings().provider_daily_limit_overrides)
        except Exception as exc:
            logger.warning("Provider limit gecersiz kilma okunamadi: {}", exc)

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for provider, payload in raw.items():
                self._states[provider] = ProviderState(**payload)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Provider usage dosyasi okunamiyor, sifirdan: {}", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: asdict(state) for name, state in self._states.items()}
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Provider usage kaydi yazilamadi: {}", exc)

    def _state(self, provider: str) -> ProviderState:
        self._ensure_overrides()
        state = self._states.setdefault(provider, ProviderState())
        if state.day != _utc_day():
            # gun degisti: sayaclar sifir, cooldown pencereleri kapanmis sayilir
            state.day = _utc_day()
            state.attempts = state.successes = state.failures = 0
            state.exhausted_reason = None
            state.exhausted_until = None
            state.rate_limited_until = None
        return state

    # ------------------------------------------------------------ recording
    def record_attempt(self, provider: str, model: str | None = None) -> None:
        """Bir saglayiciya istek denendi (basari veya hata fark etmez).

        RPM limiti YEREDE olculur: bilinen limit dolmak uzereyse saglayici,
        ilk slot acilana kadar rate-limited isaretlenir - 429 hatasi
        karsimiza gelmeden onlenir.
        """
        now = time.time()
        with self._lock:
            state = self._state(provider)
            state.attempts += 1
            # 60 sn'lik kayan pencere (RPM)
            recent = self._recent_attempts.setdefault(provider, [])
            recent.append(now)
            self._recent_attempts[provider] = [
                stamp for stamp in recent if now - stamp <= 60.0
            ]
            rpm = state.learned_rpm or PROVIDER_LIMITS.get(provider, {}).get("rpm")
            if rpm is not None and len(self._recent_attempts[provider]) >= rpm:
                window_until = self._recent_attempts[provider][0] + 60.0
                state.rate_limited_until = max(
                    state.rate_limited_until or 0.0, window_until
                )
                logger.info(
                    "Provider RPM penceresi doldu: provider={} rpm={} -> "
                    "ilk slot acilana kadar atlanacak (kalan {:.0f}s)",
                    provider,
                    rpm,
                    max(0.0, window_until - now),
                )
            # MODEL bazli RPD: ayri havuz (org limitinden bagimsiz)
            if model is not None:
                state.model_attempts[model] = state.model_attempts.get(model, 0) + 1
                model_rpd = (PROVIDER_MODEL_LIMITS.get(provider, {}).get(model, {}) or {}).get("rpd")
                if (
                    model_rpd is not None
                    and state.model_attempts[model] >= model_rpd
                    and model not in state.model_exhausted
                ):
                    state.model_exhausted[model] = (
                        time.time() + _seconds_until_utc_midnight()
                    )
                    logger.info(
                        "Model gunluk limiti doldu: provider={} model={} rpd={} "
                        "-> gun batimina kadar atlanacak (diger modeller devam)",
                        provider,
                        model,
                        model_rpd,
                    )
            limit = PROVIDER_LIMITS.get(provider, {}).get("rpd")
            if (
                limit is not None
                and state.attempts >= limit
                and state.exhausted_until is None
            ):
                state.exhausted_reason = f"documented_rpd:{limit}"
                state.exhausted_until = time.time() + _seconds_until_utc_midnight()
                logger.info(
                    "Provider kota (belgelenmis limit): provider={} attempts={} rpd={} "
                    "-> gun batimina kadar atlanacak",
                    provider,
                    state.attempts,
                    limit,
                )
            self._save()

    def record_success(self, provider: str, *, tokens: int = 0) -> None:
        with self._lock:
            state = self._state(provider)
            state.successes += 1
            if tokens > 0:
                limit = PROVIDER_LIMITS.get(provider, {}).get("tpd")
                if limit is not None and state.attempts >= limit:
                    state.exhausted_reason = f"documented_tpd:{limit}"
                    state.exhausted_until = time.time() + _seconds_until_utc_midnight()
            self._save()

    def record_failure(self, provider: str, *, status_code: int | None) -> None:
        """Hata kaydet; erisim/kota/rate hatalari saglayici seviyesinde olcer.

        401/403 (erisim): hesap bu saglayiciyi kullanamiyor - gun batimina
        kadar denenmez. 402 (kota/odeme): gun bitti. 429 (rate limit):
        org seviyesinde paylasilan kisa pencere.
        """
        now = time.time()
        with self._lock:
            state = self._state(provider)
            state.failures += 1
            if status_code in (401, 403):
                # erisim hatasi: hesap bazli - her istekte tekrar denemek zaman kaybi
                state.exhausted_reason = f"access_denied:{status_code}"
                state.exhausted_until = max(
                    state.exhausted_until or 0.0,
                    now + max(_seconds_until_utc_midnight(), 1800.0),
                )
                logger.info(
                    "Provider erisim hatasi ({}): provider={} -> gun batimina "
                    "kadar atlanacak (hesap erisimi yok)",
                    status_code,
                    provider,
                )
            elif status_code == 402:
                # kota/odeme: saglayicinin gunu bitti — diger modelleri de deneme
                state.exhausted_reason = "quota_402"
                state.exhausted_until = max(
                    state.exhausted_until or 0.0,
                    now + max(_seconds_until_utc_midnight(), 1800.0),
                )
                logger.info(
                    "Provider kota tukendi (402): provider={} -> gun batimina kadar "
                    "atlanacak, ayni saglayicinin diger modelleri denenmeyecek",
                    provider,
                )
            elif status_code == 429:
                # ADAPTIF GERI DUSUS: site limitine guvendik ama limit ALTINDAYKEN
                # 429 geldi -> site verisi bu hesap icin yanlis. O saglayici icin
                # YEREL olcume gec: ogrenilen etkili RPM = son pencerede
                # basarisiz oldugu noktanin bir alti.
                recent_count = len(self._recent_attempts.get(provider, []))
                documented = PROVIDER_LIMITS.get(provider, {}).get("rpm")
                if documented is not None and 0 < recent_count < documented:
                    state.learned_rpm = max(1, recent_count - 1)
                    logger.info(
                        "Site RPM ({}) altindayken 429 geldi (son 60sn: {} istek) "
                        "-> SAGLAYICI YEREL OLCUME GECTI: etkili RPM={}",
                        documented,
                        recent_count,
                        state.learned_rpm,
                    )
                # rate limit: org seviyesinde paylasilir — kisa pencere
                state.rate_limited_until = max(
                    state.rate_limited_until or 0.0, now + 60.0
                )
                logger.info(
                    "Provider rate limit (429): provider={} -> 60sn provider cooldown",
                    provider,
                )
            self._save()

    # ------------------------------------------------------------ queries
    def record_model_failure(
        self, provider: str, model: str, *, status_code: int | None
    ) -> None:
        """Model bazli kalici hatalar: 404 = model o saglayicida yok.

        Olu model ID'leri her istekte tekrar denenmez - gun batimina kadar
        isaretlenir (saglayicinin DIGER modelleri etkilenmez).
        """
        if status_code != 404:
            return
        now = time.time()
        with self._lock:
            state = self._state(provider)
            if model not in state.model_exhausted:
                state.model_exhausted[model] = now + max(
                    _seconds_until_utc_midnight(), 1800.0
                )
                logger.info(
                    "Model bulunamadi (404): provider={} model={} -> gun "
                    "batimina kadar zincirden cikarildi",
                    provider,
                    model,
                )
                self._save()

    def candidate_available(
        self, provider: str, model: str | None = None
    ) -> tuple[bool, str | None]:
        """Bir aday (saglayici+model) uygun mu? Zincir elemanlari icin.

        Org seviyesi (saglayici havuzu) VE model seviyesi (ayri havuzu olan
        modeller - or. Gemini Pro) ayni anda kontrol edilir.
        """
        ok, reason = self.provider_available(provider)
        if not ok:
            return ok, reason
        if model is None:
            return True, None
        now = time.time()
        with self._lock:
            state = self._state(provider)
            until = state.model_exhausted.get(model)
            if until is not None and now < until:
                return False, "model_daily_limit"
            if until is not None and now >= until:
                del state.model_exhausted[model]
        return True, None

    def provider_available(self, provider: str) -> tuple[bool, str | None]:
        """(uygun mu, gerekce). Tukenmis saglayicilar zincirden atlanir."""
        now = time.time()
        with self._lock:
            state = self._state(provider)
            if state.exhausted_until is not None and now < state.exhausted_until:
                return False, f"exhausted:{state.exhausted_reason}"
            if state.rate_limited_until is not None and now < state.rate_limited_until:
                return False, "rate_limited"
            return True, None

    def seconds_until_available(self, provider: str) -> float:
        now = time.time()
        with self._lock:
            state = self._state(provider)
            windows = [
                w - now
                for w in (state.exhausted_until, state.rate_limited_until)
                if w is not None and w > now
            ]
            return max(windows) if windows else 0.0

    def status(self) -> dict[str, Any]:
        """Admin/ileti icin ozet."""
        with self._lock:
            return {
                name: {
                    "day": state.day,
                    "attempts": state.attempts,
                    "successes": state.successes,
                    "failures": state.failures,
                    "exhausted_reason": state.exhausted_reason,
                    "learned_rpm": state.learned_rpm,
                    "seconds_until_available": self.seconds_until_available(name),
                    "limits": PROVIDER_LIMITS.get(name, {}),
                }
                for name, state in sorted(self._states.items())
            }


provider_quota_tracker = ProviderQuotaTracker()
"""Process-wide default tracker shared by every provider executor."""
