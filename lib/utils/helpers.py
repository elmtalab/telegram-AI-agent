"""General helper utilities."""

import hmac
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ANSI_ESC_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
LLM_TOKEN_MARKER_RE = re.compile(
    r"<\|/?(system|user|assistant|end|begin)[^>]*\|>",
    re.I,
)
URL_RE = re.compile(r"https?://\S+", re.I)


def extract_urls(text: str) -> List[str]:
    return re.findall(URL_RE, text or "")


def detect_provider_from_url(url: str) -> Optional[str]:
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "instagram.com" in u or "instagr.am" in u:
        return "instagram"
    if "spotify.com" in u:
        return "spotify"
    if "soundcloud.com" in u:
        return "soundcloud"
    if "tiktok.com" in u:
        return "tiktok"
    if "x.com" in u or "twitter.com" in u:
        return "twitter"
    if "podcasts.apple.com" in u or "apple.com/podcasts" in u:
        return "apple_podcasts"
    return None


def sanitize_user_text(raw: str) -> str:
    if not raw:
        return ""
    s = ANSI_ESC_RE.sub("", raw)
    s = LLM_TOKEN_MARKER_RE.sub("", s)
    s = re.sub(r"[\u202A-\u202E]", "", s)
    s = re.sub(
        r"\s*\[adapter_hint[^\]]*\].*?$",
        "",
        s,
        flags=re.I | re.M,
    )
    if len(s) > 6000:
        s = s[:6000]
    return s


def _utcnow_iso() -> str:
    try:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tokenize_simple(text: str) -> List[str]:
    t = URL_RE.sub(" ", text or "")
    t = re.sub(r"[^\w\u0600-\u06FF]+", " ", t, flags=re.UNICODE)
    toks = [x for x in t.strip().split() if x]
    return toks


def _length_bucket(text: str, locale_hint: str = "fa") -> Tuple[str, int]:
    toks = _tokenize_simple(text)
    n = len(toks)
    if n <= 4:
        return ("ultra_short", n)
    if n <= 8:
        return ("short", n)
    if n <= 35:
        return ("medium", n)
    if n <= 200:
        return ("long", n)
    return ("very_long", n)


def _has_negation(text: str) -> bool:
    t = (text or "").lower()
    return bool(
        re.search(r"\b(don'?t|do not|no|never)\b", t)
        or re.search(r"(نکن|نمی‌?خوام|نه\s+ترجمه)", t)
    )


def _strip_quoted(text: str) -> str:
    s = re.sub(r'"[^"]+"', " ", text or "")
    s = re.sub(r"'[^']+'", " ", s)
    s = re.sub(r"«[^»]+»", " ", s)
    return s


def _imperative_hits(text: str) -> Dict[str, bool]:
    t = (_strip_quoted(text) or "").lower()
    return {
        "translate": bool(
            re.search(r"\btranslate\b|ترجمه|tarjome(h)?", t)
        ),
        "transcribe": bool(
            re.search(r"\btranscribe\b|تبدیل\s*به\s*متن|متن(?: کن|سازی)", t)
        ),
        "summarize": bool(
            re.search(
                r"\b(tl;?dr|tldr|summari[sz]e|summary|condense)\b",
                t,
            )
            or re.search(r"خلاصه|جمع\s*بندی", t)
        ),
        "rewrite": bool(
            re.search(r"\brewrite\b|بازنویسی|دوباره بنویس", t)
        ),
        "fix_grammar": bool(
            re.search(r"fix grammar|اشتباه(?:ات)?(?:\s*|‌)گرامر", t)
        ),
        "outline": bool(
            re.search(r"\boutline\b|فهرست|طرح کلی", t)
        ),
        "remove_bg": bool(
            re.search(
                r"\b(remove|erase)\s+bg\b"
                r"|پس ?زمینه(?: را)?(?: حذف|بردار)"
                r"|background",
                t,
            )
        ),
        "gif": bool(re.search(r"\bgif\b|گیف", t)),
        "thumbnail": bool(
            re.search(r"\bthumbnail\b|کاور|بند[ِ ]?انگشتی", t)
        ),
        "tutor": bool(
            re.search(r"\btutor\b|معلم(?:\s*زبان)?|آموزش زبان|start tutor", t)
        ),
        "slots": bool(
            re.search(r"\bslots?\b|زمان(?:‌های)? خالی|available times", t)
        ),
        "book": bool(
            re.search(r"\bbook\b|رزرو|نوبت بگیر|ثبت کن", t)
        ),
        "reschedule": bool(
            re.search(r"\breschedul(e|ing)\b|جابجا کن|تغییر زمان", t)
        ),
        "cancel": bool(re.search(r"\bcancel\b|لغو", t)),
        "face_blur": bool(
            re.search(
                r"\bblur\b.*\bfaces?\b|تاری(?:\s*|‌)چهره"
                r"|تار(?:\s*|‌)کن(?:\s*|‌)چهره",
                t,
            )
        ),
    }


_TIME_RE = re.compile(
    r"(?:^|\b)(?:at\s*)?("
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})"
    r"|(\d{1,2})m(\d{1,2})s"
    r"|(\d{1,2})s"
    r"|(\d{1,2}):(\d{2})"
    r")(?:\b|$)",
    re.I,
)


def _normalize_hhmmss(h: int, m: int, s: int) -> str:
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_timecode(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TIME_RE.search(text.strip())
    if not m:
        return None
    if m.group(1) and m.group(2) is not None:
        h = int(m.group(2) or 0)
        mm = int(m.group(3) or 0)
        ss = int(m.group(4) or 0)
        if mm > 59 or ss > 59:
            return None
        return _normalize_hhmmss(h, mm, ss)
    if m.group(5) and m.group(6):
        mm = int(m.group(5))
        ss = int(m.group(6))
        if mm > 59 or ss > 59:
            return None
        return _normalize_hhmmss(0, mm, ss)
    if m.group(7):
        ss = int(m.group(7))
        if ss > 59:
            return None
        return _normalize_hhmmss(0, 0, ss)
    if m.group(8) and m.group(9):
        mm = int(m.group(8))
        ss = int(m.group(9))
        if mm > 59 or ss > 59:
            return None
        return _normalize_hhmmss(0, mm, ss)
    return None


_DURATION_RE = re.compile(
    r"(?:(\d{1,2})\s*h(?:ours?)?)?\s*(?:(\d{1,3})\s*m(?:in(?:utes)?)?)",
    re.I,
)


def _parse_duration_minutes(text: str) -> Optional[int]:
    """Parse patterns like '2h 15m', '90 min', '30m' from text."""
    if not text:
        return None
    m = _DURATION_RE.search(text)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    return h * 60 + mins if (h or mins) else None


def _detect_lang_to(text: str) -> Optional[str]:
    t = (text or "").lower().strip()
    if (
        "to english" in t
        or "به انگلیسی" in t
        or (t in ("en", "english"))
    ):
        return "en"
    if (
        "to persian" in t
        or "to farsi" in t
        or "به فارسی" in t
        or (t in ("fa", "farsi", "persian"))
    ):
        return "fa"
    return None


def _is_noise_gibberish(
    text: str, has_attachment: bool, has_url: bool
) -> bool:
    toks = _tokenize_simple(text)
    n = len(toks)
    if has_attachment or has_url:
        return False
    if n < 12:
        return False
    unique = len(set(toks))
    div = unique / max(1, n)
    symbols = re.findall(
        r"[^A-Za-z0-9\u0600-\u06FF\s]",
        URL_RE.sub("", text or ""),
    )
    sym_ratio = len(symbols) / max(1, len(text))
    rep = bool(re.search(r"(.)\1{4,}", text or ""))
    if (
        (n > 250 and div < 0.35)
        or (len(text) > 2000 and div < 0.45)
        or sym_ratio > 0.25
        or rep
    ):
        return True
    return False


def _hmac_tag(payload: Dict[str, Any], secret: str) -> str:
    msg = json.dumps(
        payload, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"), msg, digestmod="sha256"
    ).hexdigest()[:10]


def _is_missing_val(v: Any) -> bool:
    """Treat empty strings/lists/dicts/None as missing."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False
