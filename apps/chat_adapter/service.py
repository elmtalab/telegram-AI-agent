"""Normalisation utilities for the chat adapter.

The real project performs a large number of pre‑processing steps before a
message is handed over to the router.  For the purposes of the kata we
implement a small but representative subset driven entirely by configuration
values loaded from ``chat_adapter.yaml``.

The module exposes the :func:`normalize_message` function which accepts a raw
message mapping and returns a new mapping that follows the
``MessageEnvelope@1`` structure defined in :mod:`lib.contracts.envelope`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import time
import re
import unicodedata

from lib.config.chat_adapter_loader import AdapterConfig


# ---------------------------------------------------------------------------
# Chat session memory
# ---------------------------------------------------------------------------

@dataclass
class ChatMemory:
    """In‑memory state for a chat session.

    The adapter keeps track of the most recent attachments and user text as
    well as a few intent related preferences.  The original stickies used by
    ``normalize_message`` are retained for backwards compatibility.
    """

    last_voice_file_id: Optional[str] = None
    last_image_file_id: Optional[str] = None
    last_doc_file_id: Optional[str] = None
    last_user_text: Optional[str] = None
    last_turn_ts: int = field(default_factory=lambda: int(time.time()))
    pending_key: Optional[str] = None
    pending_question: Optional[str] = None
    user_lang_pref: Optional[str] = None  # 'fa' or 'en'
    intended_pipeline: Optional[str] = None  # 'audio.stt', 'audio.transcribe_translate'
    intended_target_lang: Optional[str] = None  # 'fa'/'en'
    # Tutor preferences
    tutor_target_lang: Optional[str] = None
    tutor_native_lang: Optional[str] = None
    tutor_daily_minutes: Optional[int] = None
    # Legacy sticky fields
    last_file: Optional[str] = None
    last_url: Optional[str] = None
    last_text: Optional[str] = None
    turns_since_update: int = 0

    def decay(self, ttl: int) -> None:
        """Expire stickies if they have not been referenced for ``ttl`` turns."""

        if ttl and self.turns_since_update >= ttl:
            self.last_file = None
            self.last_url = None
            self.last_text = None
            self.turns_since_update = 0
        else:
            self.turns_since_update += 1


# ---------------------------------------------------------------------------
# Basic sanitisation helpers
# ---------------------------------------------------------------------------

def _clean_text(raw: str, cfg: Dict[str, Any]) -> str:
    """Apply the ``preprocess.clean`` rules from the configuration."""

    s = raw or ""
    if cfg.get("unicode_nfkc", True):
        s = unicodedata.normalize("NFKC", s)
    if cfg.get("strip_control_chars", True):
        s = re.sub(r"[\u0000-\u001F\u007F]", "", s)
    if cfg.get("normalize_quotes", True):
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if cfg.get("normalize_dashes", True):
        s = s.replace("—", "-").replace("–", "-")
    # whitespace handling
    if cfg.get("trim_whitespace", True):
        s = s.strip()
    if cfg.get("collapse_whitespace", True):
        s = re.sub(r"\s+", " ", s)
    # digit normalisation
    digits_cfg = cfg.get("normalize_digits", {})
    if digits_cfg.get("fa_to_ascii", True):
        s = s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    if digits_cfg.get("ar_to_ascii", True):
        s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    max_len = int(cfg.get("max_text_len", 8000))
    if len(s) > max_len:
        if cfg.get("truncate_strategy") == "hard_tail":
            s = s[-max_len:]
        else:
            s = s[:max_len]
    return s


def _detect_urls(text: str, cfg: Dict[str, Any], defaults: Dict[str, Any]) -> List[str]:
    """Return URLs found in ``text`` according to configuration."""

    if not cfg.get("enabled", True) or not text:
        return []
    pattern = cfg.get("regex_ref")
    if pattern and pattern.startswith("defaults."):
        key = pattern.split(".", 1)[1]
        pattern = defaults.get(key, "")
    regex = re.compile(pattern or r"https?://\S+")
    allow = {s.lower() for s in cfg.get("allow_schemes", ["http", "https"])}
    max_urls = int(cfg.get("max_urls", 5))
    found = [u for u in regex.findall(text) if u.split(":", 1)[0].lower() in allow]
    return found[:max_urls]


def _label_attachments(attachments: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assign ``media_type`` to attachments based on simple rules."""

    if not attachments or not cfg.get("enabled", True):
        return attachments or []
    rules = cfg.get("rules", [])
    labelled: List[Dict[str, Any]] = []
    for att in attachments:
        mime = (att.get("mime") or "").lower()
        media_type = "unknown"
        for rule in rules:
            cond = rule.get("when", "")
            if cond == "attachment.mime.startswith('image/')" and mime.startswith("image/"):
                media_type = rule.get("as_media_type", "image")
                break
            if cond == "attachment.mime.startswith('audio/')" and mime.startswith("audio/"):
                media_type = rule.get("as_media_type", "audio")
                break
            if cond == "attachment.mime.startswith('video/')" and mime.startswith("video/"):
                media_type = rule.get("as_media_type", "video")
                break
            if cond == "attachment.mime == 'application/pdf'" and mime == "application/pdf":
                media_type = rule.get("as_media_type", "document")
                break
            if cond == "attachment.mime.startswith('application/')" and mime.startswith("application/"):
                media_type = rule.get("as_media_type", "document")
                break
        a = dict(att)
        a.setdefault("media_type", media_type)
        labelled.append(a)
    return labelled


def _attachments_guard(attachments: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Perform early sanity checks on the attachments list."""

    if not cfg:
        return
    if cfg.get("reject_if_missing_id", True):
        for att in attachments:
            if not att.get("file_id"):
                raise ValueError("attachment missing file_id")
    max_files = int(cfg.get("max_files", 5))
    if len(attachments) > max_files:
        raise ValueError("too many attachments")
    total_bytes = 0
    for att in attachments:
        total_bytes += int(att.get("size", 0))
    if total_bytes > int(cfg.get("max_total_bytes", 0)) > 0:
        raise ValueError("attachments exceed size limit")


def _infer_locale_tz(explicit_locale: str | None, explicit_tz: str | None, cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Infer locale and timezone with simple fallbacks."""

    locale = explicit_locale or cfg.get("default_locale", "en-US")
    tz_cfg = cfg.get("timezone", {})
    tz = explicit_tz or tz_cfg.get("tz_by_locale", {}).get(locale)
    if not tz:
        tz = tz_cfg.get("default_tz", "UTC")
    return locale, tz


def _apply_stickies(text: str, urls: List[str], attachments: List[Dict[str, Any]], memory: ChatMemory, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Update memory based on the current turn and return the snapshot."""

    snapshot = {
        "last_file": memory.last_file,
        "last_url": memory.last_url,
        "last_text": memory.last_text,
        "last_intent_hint": None,
    }

    # clear on negation
    clear_cfg = cfg.get("clear_on_negation", {})
    if clear_cfg.get("enabled", True):
        markers = [m.lower() for m in clear_cfg.get("negation_markers", {}).get("en", [])]
        markers += [m.lower() for m in clear_cfg.get("negation_markers", {}).get("fa", [])]
        low = text.lower()
        if any(m in low for m in markers):
            scope = clear_cfg.get("scope", [])
            if "file" in scope:
                snapshot["last_file"] = None
                memory.last_file = None
            if "url" in scope:
                snapshot["last_url"] = None
                memory.last_url = None
            if "text" in scope:
                snapshot["last_text"] = None
                memory.last_text = None

    # precedence – current turn artefacts win
    prec = cfg.get("precedence", {})
    if attachments and prec.get("prefer_current_attachments_over_stickies", True):
        snapshot["last_file"] = None
    if urls and prec.get("prefer_current_urls_over_stickies", True):
        snapshot["last_url"] = None

    # update memory with current artefacts
    if attachments:
        memory.last_file = attachments[-1].get("file_id")
        memory.turns_since_update = 0
    if urls:
        memory.last_url = urls[-1]
        memory.turns_since_update = 0
    if text:
        memory.last_text = text
        memory.turns_since_update = 0

    return snapshot


def _build_adapter_hints(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the non‑binding adapter hints."""

    hints = {
        "target_lang": cfg.get("target_lang_default"),
        "provider": cfg.get("provider_default", {}),
        "safety_mode": cfg.get("safety_mode", "standard"),
    }
    propagate = cfg.get("propagate_to_router", [])
    return {k: v for k, v in hints.items() if k in propagate and v is not None}


# ---------------------------------------------------------------------------
# Public normalisation entry point
# ---------------------------------------------------------------------------

def normalize_message(message: Dict[str, Any], cfg: AdapterConfig, memory: ChatMemory) -> Dict[str, Any]:
    """Normalise ``message`` and return an envelope mapping.

    Parameters
    ----------
    message:
        Mapping with optional ``text``, ``attachments``, ``locale`` and
        ``timezone`` keys.
    cfg:
        :class:`AdapterConfig` instance.
    memory:
        Session memory for the chat the message belongs to.  The memory is
        updated in place.
    """

    raw_text = message.get("text", "")
    sanitized = _clean_text(raw_text, cfg.clean)
    urls = _detect_urls(sanitized, cfg.detect_urls, cfg.raw.get("defaults", {}))
    attachments = _label_attachments(message.get("attachments", []), cfg.detect_media_types)
    _attachments_guard(attachments, cfg.attachments_guard)
    locale, tz = _infer_locale_tz(message.get("locale"), message.get("timezone"), cfg.locale)
    snapshot = _apply_stickies(sanitized, urls, attachments, memory, cfg.sticky)
    hints = _build_adapter_hints(cfg.hints)

    envelope = {
        "text": sanitized,
        "urls": urls,
        "attachments": attachments,
        "locale": locale,
        "timezone": tz,
        "session_snapshot": snapshot,
        "adapter_hints": hints,
    }
    return envelope


__all__ = ["normalize_message", "ChatMemory"]
