"""Configurable LLM based router.

This module contains a compact yet featureful implementation of the
``ConfigLLMRouter`` described in the long design document provided by the
user.  The goal of this implementation is not to be production perfect – the
real project spans thousands of lines and depends on numerous external
packages – but to mirror the structure and behaviour of the original router so
that the rest of this kata can exercise a realistic code path.

The router reads its configuration from ``router.yaml`` and information about
available intents from ``route_registry.yaml``.  Five processing stages are
modelled: ``decide_domain``, ``plan_pipeline``, ``merge_prepare``,
``apply_policy`` and ``finalize``.  Each stage consumes a portion of the
configuration file that matches the layout described in the design notes.  The
logic inside the stages is intentionally lightweight; it performs simple
keyword based heuristics rather than expensive semantic ranking.  Nevertheless
the public API matches the production counterpart which makes the component a
useful stand‑in for tests and demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Optional

from lib.contracts.envelope import Attachment, MessageEnvelope
from lib.contracts.router_output import RouterOutput
from .models import PlanOut, DecideOut, RouterState

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------



def intent_modality(intent: str) -> str:
    if intent.startswith(("image.", "ocr.", "qr.", "barcode.", "receipt.", "table.")):
        return "image" if intent.startswith("image.") else "file"
    if intent.startswith(("stt.", "audio.")):
        return "voice"
    if intent.startswith(("files.",)):
        return "file"
    if intent.startswith(("music.",)):
        return "file"   # ← important: music ops act on files
    if intent.startswith(("link.","video.","media.","podcast.")):
        return "url"
    return "text"


def new_turn_modality(env: MessageEnvelope) -> str:
    if env.media_type == "image": return "image"
    if env.media_type == "voice": return "voice"
    if env.file_id: return "file"
    if env.urls: return "url"
    if env.text: return "text"
    return "none"

CHANGE_PATTERNS = [
    r"\binstead\b", r"\bforget (that|it|previous)\b", r"\bcancel (that|previous)\b", r"\bno, just\b",
    r"به جای", r"بی‌?خیال", r"لغو", r"کنسل", r"دیگه نه", r"نه فقط"
  ]
DIRCHANGE_RE = re.compile("|".join(CHANGE_PATTERNS), re.I)
def is_direction_change(text: str) -> bool:
    return bool(DIRCHANGE_RE.search(text or ""))

def _dp_unique_preserve(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _dp_snip(text: str, n: int = 240) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else (t[:n] + "…")

# ---------------------------------------------------------------------------
# Data structures used by the simplified router
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Tiny session object used to keep track of the previous turn."""

    last_file: Optional[str] = None
    last_url: Optional[str] = None
    last_intent: Optional[str] = None


# ---------------------------------------------------------------------------
# ConfigLLMRouter implementation
# ---------------------------------------------------------------------------


class ConfigLLMRouter:
    RISKY_URL_INTENT_PREFIXES = ()

    # FA/AR normalizer map
    _FA_MAP = str.maketrans({"ي":"ی","ك":"ک"})

    def __init__(self, settings: Dict[str, Any], routes: Dict[str, Any]):
        self.settings_raw = settings or {}
        self.routes_raw = routes or {}

        defaults = {
            "timezone": "Europe/Amsterdam",
            "image_size": "768",
            "media_quality": "720p",
            "tts_voice": "neutral",
            "router_model": os.environ.get("ROUTER_MODEL", "gpt-5-nano"),
            "incomplete_max_age_turns": 2,
            "conf_domain_dispatch": 1.00,
            "low_conf_threshold": 0.70,
            "crypto_exchange": "bitstamp",
            "clarify_mode": "when_missing",
        }
        defaults = {**defaults, **(settings or {}).get("settings", {}).get("defaults", {})}
        self.defaults = defaults

        s = (settings or {}).get("settings", {})
        self.policy = {
            "download": "owned_or_authorized_only",
            "third_party_scrape": "deny",
            "fallback_on_violation": "inline.tldr",
            "provider_allowlist": ["youtube"],  # allowlisted media providers
        }
        self.policy.update(s.get("policy", {}))
        self.thresholds = {
            "conf_domain_dispatch": float(s.get("conf_domain_dispatch", 1.0)),
            "low_conf_threshold": float(s.get("low_conf_threshold", defaults["low_conf_threshold"])),
            "incomplete_max_age_turns": int(s.get("incomplete_max_age_turns", defaults["incomplete_max_age_turns"])),

        }
        self._slot_model_client = None
        self._slot_calls_this_turn = 0


        self.clarify_mode = s.get("clarify_mode", defaults.get("clarify_mode", "when_missing"))

        self.caps = {
            "media_transcript_max_minutes": int((s.get("caps") or {}).get("media_transcript_max_minutes", 15)),
            "audio_stt_max_minutes": int((s.get("caps") or {}).get("audio_stt_max_minutes", 30)),
            "image_max_px": int((s.get("caps") or {}).get("image_max_px", 2048)),
        }

        self.friendly = {
            "text": "متن را بفرستید. / Please send the text.",
            "file_id": "فایل/عکس/صوت را بفرستید. / Send the file/photo/voice.",
            "url": "لینک را بفرستید. / Send the link.",
            "target_lang": "به فارسی یا انگلیسی؟ (fa/en)",
            "language": "زبان خروجی؟ (fa/en)",
            "tone": "لحن؟ (formal/casual/neutral)",
            "bullet_count": "چند بولت؟ (۳ تا ۱۰)",
            "size": "اندازه تصویر؟ (512/768/1024)",
            "provider": "سرویس؟ (youtube/instagram/drive/…)",
            "auth_token": "ابتدا باید اتصال مجاز شود. وصل کنم؟ (yes/no)",
            "media_id": "شناسهٔ رسانه؟ / Media ID?",
            "quality": "کیفیت؟ (1080p/720p/480p/audio_only)",
            "start": "شروع؟ (HH:MM:SS یا ثانیه)",
            "end": "پایان؟ (HH:MM:SS یا ثانیه)",
            "duration_minutes": "مدت جلسه چند دقیقه؟ / Duration (minutes)?",
            "date": "تاریخ؟ (YYYY-MM-DD)",
            "time": "ساعت؟ (HH:MM)",
            "datetime": "زمان دقیق؟ (YYYY-MM-DD HH:MM)",
            "attendee": "نام یا ایمیل شرکت‌کننده؟",
            "booking_ref": "کد رزرو را وارد کنید.",
            "deck_id": "شناسهٔ دِک چیست؟",
            "exam": "نوع آزمون؟ (IELTS/TOEFL/…)",
            "role": "نقش/سمت چیست؟ (مثلاً مصاحبه‌گر/سخنران)",
            # NEW labels
            "date_range": "بازهٔ تاریخ؟ (مثلاً week یا 2025-08-24..2025-08-30)",
            "exchange": "صرافی؟ (bitstamp/…)",
            "interval": "بازهٔ زمانی (مثلاً 1m/5m/1h/4h/1d)",
            "amount": "مقدار چقدر؟",
            "from_currency": "از چه ارزی؟ (USD/EUR/…)",
            "to_currency": "به چه ارزی؟ (USD/EUR/…)",
            "purpose": "کاربرد چیست؟ (streaming/gaming/…)",
            "region": "منطقه؟ (US/EU/…)",
            "budget": "بودجه؟",
            "devices": "چند دستگاه؟",
        }
        self.friendly.update(s.get("friendly_labels", {}))

        self.router_model = s.get("router_model", defaults["router_model"])
        self.lang_hint = s.get("language", "fa")

        # Domain map by namespace
        self.domain_map = {
            "text": "core", "audio": "core", "doc": "core", "image": "images",
            "media": "media", "music": "media",
            "tutor": "tutor", "sched": "scheduler",
            "finance": "utilities", "network": "utilities",
            "writing": "core", "translate": "core", "tools": "utilities",
            "qr": "utilities", "barcode": "utilities", "receipt": "core", "table": "core",
            "files": "core", "video": "media", "link": "media", "podcast": "media", "thread": "core", "tts": "core",
        }
        self.domain_map.update({
            "video": "media",
            "link":  "media",
        })
        self._sr_trace: Dict[str, Any] = {"domain": None, "intent": None}

        self._debug = bool(s.get("debug", False) or ((s.get("logging") or {}).get("debug", False) if isinstance(s.get("logging"), dict) else False))

        # Domain & intent semantic configs
        sem_block = s.get("semantic", {}) or {}
        self.semantic_enabled: bool = bool(s.get("semantic_enabled", sem_block.get("enabled", True)))
        self.semantic_domain_threshold: float = float(sem_block.get("domain_threshold", 0.55))
        self.semantic_intent_threshold: float = float(sem_block.get("intent_threshold", 0.60))
        self.semantic_intent_limit: int = int(sem_block.get("intent_limit", 10))
        self.semantic_config = sem_block

        costs_cfg = (self.settings_raw.get("settings", {}).get("costs", {}) or {}).get("models", {}) or {}
        self._meter = TokenCostMeter(model=self.router_model, prices_by_model=costs_cfg)

        # Build intent specs from routes config
        intents_list = []
        routes = routes or {}
        if "routes" in routes and isinstance(routes["routes"], list):
            intents_list = routes["routes"]
        elif "intents" in routes and isinstance(routes["intents"], dict):
            for name, io in routes["intents"].items():
                intents_list.append({"name": name, "required": list(io.get("required", [])), "optional": list(io.get("optional", [])), "expected_output": io.get("expected_output", {})})

        self.intent_specs: Dict[str, Dict[str, Any]] = {}
        self.intent_outputs: Dict[str, Dict[str, Any]] = {}
        for it in intents_list:
            name = it.get("name") or it.get("intent")
            if not name:
                continue
            self.intent_specs[name] = {"required": list(it.get("required", []) or []), "optional": list(it.get("optional", []) or [])}
            self.intent_outputs[name] = it.get("expected_output", {}) or {}
        self.intent_set: List[str] = list(self.intent_specs.keys())
        self.url_required_intents = {i for i in self.intent_set if i.startswith(("link.","video.","media.","podcast."))}

        # ─── Patch D: Canonicalize duplicates to reduce SR collisions ─────────
        ALIASES = {
            "translate.text": "text.translate",
            "stt.transcribe": "audio.stt",
            "inline.tldr":    "text.summarize"
        }

        for alias, canon in list(ALIASES.items()):
            if alias in self.intent_specs and canon in self.intent_specs:
                self.intent_specs.pop(alias, None)
                self.intent_outputs.pop(alias, None)
        self.intent_set = list(self.intent_specs.keys())
        self.url_required_intents = {i for i in self.intent_set if i.startswith(("link.","video.","media.","podcast."))}
        # ──────────────────────────────────────────────────────────────────────

        self._sessions: Dict[str, SessionState] = {}

        self.llm = self._init_llm()

        # Domain hint & SR authority flag
        self._current_domain_hint: Optional[str] = None
        self._domain_from_sr: bool = False  # NEW: True only when SR confidently picked the domain

        # ── Semantic Router setup (intent + domain) ───────────────────────────
        self._sr_intent = None
        self._sr_domain = None
        self._sr_error = None
        # Negatives store (intent + domain)
        self._sr_neg_utt: Dict[str, Set[str]] = {name: set() for name in self.intent_set}
        self._sr_domain_neg_utt: Dict[str, Set[str]] = {
            "core": set(), "media": set(), "images": set(), "tutor": set(), "scheduler": set(), "utilities": set(), "other": set()
        }

        if self.semantic_enabled and _HAS_SEMANTIC_ROUTER:
            try:
                # Curated utterances from config
                curated = (self.semantic_config.get("utterances") or {})
                curated_domain_pos = (self.semantic_config.get("domain_utterances") or {})
                curated_domain_neg = (self.semantic_config.get("domain_negatives") or {})

                # Auto exemplars v1 + v2
                auto_ex_v1 = generate_intent_exemplars(self.intent_set)  # pos only
                auto_ex_v2 = generate_intent_exemplars_v2(self.intent_set)  # pos+neg

                intent_routes = []
                for name in self.intent_set:
                    pos_cur = list((curated.get(name, {}) or {}).get("positives") or [])
                    neg_cur = list((curated.get(name, {}) or {}).get("negatives") or [])
                    pos_v1 = list(auto_ex_v1.get(name, []) or [])
                    pos_v2 = list((auto_ex_v2.get(name, {}) or {}).get("positives") or [])
                    neg_v2 = list((auto_ex_v2.get(name, {}) or {}).get("negatives") or [])

                    # Merge positives (v2 first, then curated, then v1), cap
                    merged_pos = _dp_unique_preserve([*pos_v2, *pos_cur, *pos_v1])[:12]
                    if not merged_pos:
                        merged_pos = [name]
                    intent_routes.append(Route(name=name, utterances=merged_pos))

                    # Merge negatives (curated + v2)
                    neg_all = _dp_unique_preserve([*neg_cur, *neg_v2])[:12]
                    self._sr_neg_utt[name] = set(neg_all)

                # Domain v2 + v1 + curated merge
                domain_v2 = generate_domain_exemplars_v2()  # pos+neg
                domain_v1 = generate_domain_exemplars()     # pos only

                domain_routes = []
                for dom in ["core","media","images","tutor","scheduler","utilities","other"]:
                    pos_v2 = list((domain_v2.get(dom, {}) or {}).get("positives") or [])
                    neg_v2 = list((domain_v2.get(dom, {}) or {}).get("negatives") or [])
                    pos_v1 = list(domain_v1.get(dom, []) or [])
                    pos_cur = list((curated_domain_pos.get(dom, []) if isinstance(curated_domain_pos, dict) else []) or [])
                    neg_cur = list((curated_domain_neg.get(dom, []) if isinstance(curated_domain_neg, dict) else []) or [])

                    merged_pos = _dp_unique_preserve([*pos_cur, *pos_v2, *pos_v1])[:12]
                    domain_routes.append(Route(name=dom, utterances=merged_pos))

                    neg_all = _dp_unique_preserve([*neg_cur, *neg_v2])[:12]
                    self._sr_domain_neg_utt[dom] = set(neg_all)

                # Prefer local encoder; fallback to OpenAI encoder if necessary
                # 🔒 Force OpenAI embeddings only
                sem_block = s.get("semantic", {}) or {}
                embedding_model = (
                    sem_block.get("embedding_model")
                    or os.getenv("OPENAI_EMBEDDING_MODEL")
                    or "text-embedding-3-small"   # or "text-embedding-3-large"
                )

                # Optional: support custom base URL / org (Azure/proxy)
                encoder = OpenAIEncoder(
                  name=embedding_model,
                  openai_api_key=os.getenv("OPENAI_API_KEY"),
              )

                self._sr_intent = SemanticRouter(
                    encoder=encoder,
                    index=LocalIndex())




                self._sr_domain = SemanticRouter(
                    encoder=encoder,
                    index=LocalIndex())

                def _sr_add_and_ready(router, routes, label):
                    try:
                        if hasattr(router, "add"):
                            router.add(routes)  # populates the index
                        elif hasattr(router, "index") and hasattr(router.index, "add"):
                            # older variants expose .add() on index instead of router
                            router.index.add(routes, encoder=encoder)
                        else:
                            raise RuntimeError("Neither router.add() nor router.index.add() available")

                        # Optional: spin until index says it's ready (if API exists)
                        idx = getattr(router, "index", None)
                        if idx and hasattr(idx, "is_ready"):
                            for _ in range(40):  # ~2s max
                                if idx.is_ready():
                                    break
                                time.sleep(0.05)
                            if hasattr(idx, "is_ready") and not idx.is_ready():
                                raise RuntimeError(f"{label} index did not become ready after add()")
                    except Exception as e:
                        self._sr_error = f"semantic-router {label} init failed: {e}"
                        if self._debug:
                            print("[DEBUG]", self._sr_error)




                _sr_add_and_ready(self._sr_intent, intent_routes, "intent")
                _sr_add_and_ready(self._sr_domain,  domain_routes,  "domain")
                # Precompute embedding banks for positives and negatives (for fast runtime scoring)
                self._emb_bank_intent = self._precompute_route_embeddings(intent_routes)
                self._emb_bank_neg    = self._precompute_negative_embeddings()

                if self._debug:
                    print("[SR] error:", self._sr_error)
                    print("[SR] intent router:", bool(self._sr_intent), "domain router:", bool(self._sr_domain))
                    print("[SR] embedding model:", embedding_model)
            except Exception as e:
                self._sr_error = f"semantic-router init failed: {e}"
        elif self.semantic_enabled and not _HAS_SEMANTIC_ROUTER:
            self._sr_error = "semantic-router package not available; skipping semantic routing."

        if self._sr_error and self._debug:
            print("[DEBUG] ", self._sr_error)

        if not _HAS_LANGGRAPH:
            raise RuntimeError("LangGraph is required. Please `pip install -U langgraph`.")
        graph = StateGraph(RouterState)
        graph.add_node("decide_domain", self._node_decide_domain)
        graph.add_node("plan_pipeline", self._node_plan_pipeline)
        graph.add_node("merge_and_prepare", self._node_merge_and_prepare)
        graph.add_node("apply_policy", self._node_apply_policy)
        graph.add_node("finalize", self._node_finalize)
        graph.set_entry_point("decide_domain")
        graph.add_edge("decide_domain", "plan_pipeline")
        graph.add_edge("plan_pipeline", "merge_and_prepare")
        graph.add_edge("merge_and_prepare", "apply_policy")
        graph.add_edge("apply_policy", "finalize")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    # Small helper for debug logs
    def _log_debug(self, where: str, data: Dict[str, Any]):
        if self._debug:
            try:
                print(f"[DEBUG] {where}: {json.dumps(data, ensure_ascii=False)}")
            except Exception:
                print(f"[DEBUG] {where}: {data}")

    # ---- SR scoring helpers -----------------------------------------------------
    # ─── Slot-filler config + normalization helpers ─────────────────────────────
    def _slotfill_cfg(self):
        return (self.settings_raw.get("settings", {}) or {}).get("slotfill", {}) or {}

    def _slotfill_specs_for_intent(self, intent: str) -> Optional[Dict[str,Any]]:
        sf = self._slotfill_cfg()
        if not sf.get("enabled"):
            return None
        return (sf.get("routes") or {}).get(intent)

    def _digits_en(self, s: str) -> str:
        # Persian/Arabic digits → ASCII
        trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
        return (s or "").translate(trans)

    def _norm_text_for_offsets(self, s: str) -> str:
        # unify FA/AR variants + digits, strip bidi marks
        return self._digits_en(self._normalize_fa_ar(s or ""))

    def _regex_harvest_generic(self, text: str, ent: Dict[str,Any], spec: Dict[str,Any]) -> Dict[str,Any]:
        """Try declared patterns first; fall back to defaults library when absent."""
        import re as _re
        norm = self._norm_text_for_offsets(text)
        fields = (spec.get("fields") or {})
        defaults = (self._slotfill_cfg().get("defaults") or {})

        for key, fcfg in fields.items():
            if key in ent and not _is_missing_val(ent.get(key)):
                continue
            pats = list(fcfg.get("patterns") or [])
            for t in (fcfg.get("type","") or "").split("|"):
                t = t.strip()
                if t in defaults: pats.append(defaults[t])

            val = None
            for rx in pats:
                try:
                    m = _re.search(rx, norm, _re.I)
                    if m:
                        val = m.group(1) if m.lastindex else m.group(0)
                        break
                except Exception:
                    continue
            if val:
                ent[key] = val.strip()
        return ent

    def _validate_span(self, norm: str, s: SlotSpan, allowed: Set[str]) -> bool:
        if s.key not in allowed: return False
        if s.start < 0 or s.end > len(norm) or s.start >= s.end: return False
        return norm[s.start:s.end] == s.text

    # cache for a dedicated slot-fill model if different from router model
    def _slot_llm(self):
        sf = self._slotfill_cfg()
        mdl = sf.get("model") or self.router_model
        # use router client if same model
        if mdl == self.router_model:
            return self.llm
        # otherwise make a small, cached client (or fallback to heuristic if not available)
        if not hasattr(self, "_slot_model_client"):
            self._slot_model_client = None
        if self._slot_model_client is None:
            try:
                if ChatOpenAI is None:
                    return _HeuristicLLM()
                self._slot_model_client = ChatOpenAI(model=mdl, temperature=0, max_retries=1, timeout=12)  # type: ignore
            except Exception:
                self._slot_model_client = _HeuristicLLM()
        return self._slot_model_client

    _SLOTFILL_SYS = """Extract ONLY the fields you are asked for.
    Return strict JSON with a list of spans (key, text, start, end, normalized?).
    Rules:
    - key MUST be one of the allowed keys.
    - text MUST be an exact substring of the provided TEXT (validate case-sensitively).
    - start/end are byte offsets on the given TEXT AFTER normalizing Persian/Arabic digits to ASCII.
    - If uncertain, omit the field. Never invent.
    """

    def _llm_slotfill_generic(self, text: str, allowed_keys: List[str]) -> Optional[SlotPack]:
        # budget guard
        sf = self._slotfill_cfg(); budget = int(sf.get("max_calls_per_turn", 1))
        if getattr(self, "_slot_calls_this_turn", 0) >= budget:
            return None
        if not text.strip():
            return None
        llm = self._slot_llm()
        if isinstance(llm, _HeuristicLLM):
            return None
        try:
            messages = [
                SystemMessage(content=self._SLOTFILL_SYS.strip()),
                HumanMessage(content=f"ALLOWED_KEYS = {allowed_keys}\n\nTEXT:\n{text}")
            ]
            structured = llm.with_structured_output(SlotPack, method="json_schema", strict=True)
            out: SlotPack = structured.invoke(messages, config={"callbacks":[UsageCallback(self._meter, "slotfill_llm")]})
            norm = self._norm_text_for_offsets(text)
            ok_items = [s for s in out.items if self._validate_span(norm, s, set(allowed_keys))]
            # count budget
            self._slot_calls_this_turn = getattr(self, "_slot_calls_this_turn", 0) + 1
            return SlotPack(items=ok_items)
        except Exception:
            return None

    def _slotfill_for_intent(self, intent: str, ent: Dict[str,Any], user_text: str) -> Dict[str,Any]:
        spec = self._slotfill_specs_for_intent(intent)
        if not spec:
            return ent

        fields = (spec.get("fields") or {})
        allowed = list(fields.keys())
        if not allowed:
            return ent

        # 1) Regex harvest first
        ent = self._regex_harvest_generic(user_text, ent, spec)

        # Decide what's still missing
        missing = [k for k in allowed if (k not in ent or _is_missing_val(ent.get(k)))]

        # 2) If still missing → LLM JSON span extractor
        if missing:
            pack = self._llm_slotfill_generic(user_text, allowed)
            if pack:
                for item in pack.items:
                    if (item.key in missing) and (item.key not in ent or _is_missing_val(ent.get(item.key))):
                        ent[item.key] = (item.normalized or item.text).strip()

        # 3) No write-through for allow_names_for_ids here; we handle acceptance in missing_for_node (below).
        return ent










    def _expand_via_map(self, top_intent: str) -> List[Dict[str,Any]]:
        ex = (self.settings_raw.get("settings", {}) or {}).get("expansions", {})
        plan = ex.get(top_intent) or [{"intent": top_intent}]
        nodes: List[Dict[str,Any]] = []
        for i, step in enumerate(plan, 1):
            nodes.append({
                "task_id": f"t{i}",
                "intent": step["intent"],
                "entities": {},
                "bind": {},
                "confidence": 0.92 if i == 1 else 0.90,
            })
        # simple text-binding for common chains
        for j in range(1, len(nodes)):
            if nodes[j]["intent"] in ("text.translate","ocr.tldr","media.quotes","media.chapters"):
                nodes[j]["bind"]["text"] = {"from_task": nodes[j-1]["task_id"], "key":"text"}
        return nodes



    def _get_encoder(self):
        try:
            return getattr(self._sr_intent, "encoder", None)
        except Exception:
            return None

    def _embed_strings(self, encoder, texts: List[str]) -> List[List[float]]:
        """
        Try SR encoder first; if not available, use OpenAI Embeddings directly.
        """
        if not texts:
            return []

        # 1) SR encoder variants
        try:
            if hasattr(encoder, "embed") and callable(getattr(encoder, "embed")):
                return encoder.embed(texts)  # some SR versions
            if hasattr(encoder, "encode") and callable(getattr(encoder, "encode")):
                return encoder.encode(texts)  # other SR versions
        except Exception:
            pass

        # 2) Direct OpenAI fallback (works regardless of SR encoder internals)
        model = (self.semantic_config.get("embedding_model")
                or os.getenv("OPENAI_EMBEDDING_MODEL")
                or "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            # No key → skip refinement silently
            return []

        # Try new SDK first, then legacy as a fallback
        try:
            from openai import OpenAI  # v1 SDK
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            try:
                import openai  # legacy SDK
                openai.api_key = api_key
                resp = openai.Embedding.create(model=model, input=texts)
                return [d["embedding"] for d in resp["data"]]
            except Exception:
                # As a last resort, skip refinement
                return []



    @staticmethod
    def _cos(u: List[float], v: List[float]) -> float:
        # cosine similarity
        if not u or not v:
            return 0.0
        s = sum(a*b for a, b in zip(u, v))
        nu = (sum(a*a for a in u) ** 0.5) or 1e-9
        nv = (sum(b*b for b in v) ** 0.5) or 1e-9
        return s / (nu * nv)

    @staticmethod
    def _centroid_topk(q_emb: List[float], vecs: List[List[float]], k: int = 3) -> float:
        if not vecs:
            return 0.0
        sims = sorted((ConfigLLMRouter._cos(q_emb, v) for v in vecs), reverse=True)
        top = sims[:max(1, min(k, len(sims)))]
        return sum(top) / len(top)

    def _precompute_route_embeddings(self, routes) -> Dict[str, List[List[float]]]:
        """
        Cache embeddings for each route's positive exemplars one time at init.
        routes: list of semantic_router.Route (has .name and .utterances)
        """
        enc = self._get_encoder()
        bank: Dict[str, List[List[float]]] = {}
        if not enc:
            return bank
        for r in routes or []:
            utts = getattr(r, "utterances", None) or []
            try:
                bank[r.name] = self._embed_strings(enc, utts)
            except Exception:
                bank[r.name] = []
        return bank

    def _precompute_negative_embeddings(self) -> Dict[str, List[List[float]]]:
        """
        Cache embeddings for curated hard negatives per intent (self._sr_neg_utt).
        """
        enc = self._get_encoder()
        bank: Dict[str, List[List[float]]] = {}
        if not enc:
            return bank
        for name, negs in (self._sr_neg_utt or {}).items():
            try:
                bank[name] = self._embed_strings(enc, list(negs or []))
            except Exception:
                bank[name] = []
        return bank






    def _to_list(self, obj):
        if obj is None: return []
        return obj if isinstance(obj, list) else [obj]

    def _choice_name(self, c):
        if isinstance(c, str): return c
        # objects or dicts
        for k in ("name", "route", "route_name", "class_name"):
            v = getattr(c, k, None) if not isinstance(c, dict) else c.get(k)
            if v: return v
        return None

    def _choice_score(self, c):
        if isinstance(c, str):
            return 1.0  # older versions return just the route name; treat as strong hit
        # objects or dicts
        v = (getattr(c, "score", None) or getattr(c, "similarity", None)
            or getattr(c, "similarity_score", None)
            or (c.get("score") if isinstance(c, dict) else None)
            or (c.get("similarity") if isinstance(c, dict) else None)
            or (c.get("similarity_score") if isinstance(c, dict) else None))
        try:
            return float(v)
        except Exception:
            return 0.0




    def _is_market_query(self, t: str) -> bool:
        """Detect crypto/FX market queries and avoid common pricing false-positives."""
        t = self._normalize_fa_ar(t or "").lower()
        has_pair = bool(re.search(r"\b([a-z]{3})/([a-z]{3})\b", t))
        has_crypto = bool(re.search(r"\b(btc|eth|crypto|coin|token|usdt|usd|eur|gbp|jpy|cad|chf)\b", t))
        has_mkt = bool(re.search(r"\b(price|ticker|ohlc|candle|fx|rate|convert)\b", t))
        neg = bool(re.search(r"\b(price\s+of\s+(admission|entry|tickets?)|pricing\s+page|price\s+list|price\s+elasticity)\b", t))
        neg = neg or bool(re.search(r"(قیمت\s+(بلیت|ورودیه|اشتراک)|صفحه(?:‌| )?قیمت(?:‌گذاری)?|پلن(?:‌| )قیمت|لیست(?:‌| )قیمت|کشش(?:‌| )قیمتی)", t))
        return (has_mkt and (has_pair or has_crypto)) and not neg

    # NEW: Explicit command guard for core menu/help/settings
    def _explicit_core_cmd(self, text: str) -> bool:
        low = (text or "").lower()
        # allow surrounding whitespace and trailing punctuation (! ? Persian '؟' fullwidth '！')
        return bool(re.fullmatch(r"\s*(help|راهنما|menu|منو|settings|تنظیمات)\s*[!！؟?]*\s*", low))


    # NEW helper: restrict to domain
    def _restrict_to_domain(self, intents: List[str], domain: Optional[str]) -> List[str]:
        """Keep only intents whose namespace maps to the selected domain."""
        if not domain:
            return intents
        kept = [i for i in intents if self.domain_map.get(_ns(i), "core") == domain]
        return kept or intents

    def _init_llm(self):
        use_fallback = False
        if ChatOpenAI is None or HumanMessage is None or SystemMessage is None:
            use_fallback = True
        elif (self.router_model or "").lower() in ("none","disabled","heuristic"):
            use_fallback = True
        elif not os.getenv("OPENAI_API_KEY"):
            use_fallback = True
        if use_fallback:
            return _HeuristicLLM()
        try:
            return ChatOpenAI(model=self.router_model, temperature=0, max_retries=1, timeout=20)  # type: ignore
        except Exception:
            return _HeuristicLLM()

    VPN_WORDS = re.compile(r"\b(vpn|proxy|mtproto|socks5|http proxy|فیلترشکن)\b", re.I)
    def _is_vpn_related(self, text: str) -> bool:
        return bool(self.VPN_WORDS.search(text or ""))

    def _guess_size_from_text(self, text: str) -> Optional[str]:
        if not text: return None
        m = re.search(r"\b(256|512|768|1024)\b", text)
        return m.group(1) if m else None

    def _guess_lang_to(self, text: str) -> Optional[str]:
        return _detect_lang_to(text)

    def _parse_timecode(self, text: str) -> Optional[str]:
        return _parse_timecode(text)

    def _guess_symbol_from_text(self, text: str) -> Optional[str]:
        exch = self.defaults.get("crypto_exchange","bitstamp")
        t = (text or "").lower()
        if "btc" in t or "bitcoin" in t:
            return "btcusd" if exch=="bitstamp" else "btcusdt"
        if "eth" in t or "ether" in t or "ethereum" in t:
            return "ethusd" if exch=="bitstamp" else "ethusdt"
        m = re.search(r"\b([a-z]{2,5})\s*/?\s*(usd|usdt|eur)\b", t)
        if m:
            base = m.group(1); quote = m.group(2)
            if exch=="bitstamp" and quote=="usdt": quote="usd"
            return base+quote
        return None

    def _expand_keys_expr(self, expr: str) -> List[str]:
        out=[]
        if not expr: return out
        for alt in expr.split("|"):
            for k in alt.split("+"):
                k = k.strip()
                if k: out.append(k)
        return out

    def _allowed_keys_for_intent(self, intent: str) -> set:
        spec = self.intent_specs.get(intent, {})
        keys = set()
        for expr in (spec.get("required", []) + spec.get("optional", [])):
            keys.update(self._expand_keys_expr(expr))
        return keys

    def _filter_entities_for_intent(self, intent: str, entities: Dict[str,Any]) -> Dict[str,Any]:
        allowed = self._allowed_keys_for_intent(intent)
        return {k:v for k,v in (entities or {}).items() if (k in allowed)}

    # Normalizer
    def _normalize_fa_ar(self, s: str) -> str:
        s = (s or "").translate(self._FA_MAP)
        s = re.sub(r"[\u200c\u200f\u200e]", "", s)
        return s

    # ───────────── Minimal Dynamic Prompt (shortlist + essentials) ────────────

    def _is_solvable_now(self, intent: str, shared: Dict[str, Any], text_present: bool) -> bool:
        ent = dict(shared or {})
        if text_present and "text" not in ent:
            ent["text"] = "__present__"
        # NEW: infer entities from the actual user text BEFORE checking missing
        user_text = (shared.get("_user_text") or "")
        ent = self._canonicalize_entities_for_intent(intent, ent, user_text)
        miss = self.missing_for_node(intent, ent, {})
        return len(miss) == 0

















    def _gate_by_modality(self, env: MessageEnvelope) -> List[str]:
        """
        Hybrid gate (state/modality first + strict lexical triggers).
        Major changes:
        - Do NOT admit qr./barcode. on plain text (only explicit or image state).
        - Do NOT admit image.generate on plain text unless explicitly requested.
        - Only admit core.* when an explicit core command is typed (help|menu|settings).
        - Keep candidate list small and relevant for SR.
        """
        import re
        intents = self.intent_set
        mod = new_turn_modality(env)

        def pref(prefixes: Tuple[str, ...]) -> List[str]:
            return [i for i in intents if i.startswith(prefixes)]

        def include(*names: str) -> List[str]:
            return [n for n in names if n in intents]

        t = (env.text or "").strip()
        tlow = t.lower()

        hits = _imperative_hits(t)

        # tight triggers / negatives
        RE_REPLACE_BG = re.compile(r"\breplace\b.*\bbackground\b|تعویض\s*پس[‌\u200c ]?زمینه", re.I)
        RE_ANNOTATE   = re.compile(r"\bred\s+circle\b|دایره\s*قرمز|soft\s+shadow|سایه\s*نرم", re.I)
        RE_TIME_CUE   = re.compile(r"\b(?:gif|thumbnail|thumb)\b|(\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d{1,3}s\b)", re.I)
        RE_SCREENSHOT = re.compile(r"\b(screenshot|full\s*page|اسکرین\s*شات)\b", re.I)

        # precision negative to avoid finance false positives
        RE_FIN_NEG = re.compile(r"price of admission|pricing page|price list|price elasticity|"
                                r"(قیمت\s+(بلیت|ورودیه|اشتراک)|صفحه(?:‌| )?قیمت(?:‌گذاری)?|پلن(?:‌| )قیمت|کشش(?:‌| )قیمتی)", re.I)
        RE_NET_NEG = re.compile(r"proxy\s+brush|photoshop|design\s+pattern", re.I)

        # 1) Start from a conservative base by modality
        if mod == "image":
            base = pref(("ocr.","qr.","barcode.","receipt.","table.","image.")) + ["chat.general"]
            base = _dp_unique_preserve(include("image.annotate") + base)

        elif mod == "voice":
            base = pref(("audio.","stt.","music.")) + ["chat.general"]

        elif mod == "file":
            mt = (env.media_type or "").lower()
            if mt == "image":
                base = pref(("ocr.","qr.","barcode.","receipt.","table.","image.")) + ["chat.general"]
            elif mt == "voice":
                base = pref(("audio.","stt.","music.")) + ["chat.general"]
            else:
                # generic document: doc/ocr/table/receipt + image (possible scans) + video (file)
                base = pref(("doc.","ocr.","table.","receipt.","image.","video.","music.")) + ["chat.general"]
                if RE_TIME_CUE.search(t):
                    base = _dp_unique_preserve(include("video.gif","video.thumbnail") + base)

        elif mod == "url":
            # NOTE: drop qr./barcode. on url-only turns; keep media/video/link/music
            base = pref(("media.","video.","link.","podcast.","music.")) + ["text.summarize","inline.tldr","chat.general"]
            if RE_SCREENSHOT.search(t):
                base = _dp_unique_preserve(include("link.screenshot") + base)
            if RE_TIME_CUE.search(t):
                base = _dp_unique_preserve(include("video.thumbnail","video.gif") + base)

        elif mod == "text":
            # conservative: DO NOT add qr./barcode. nor image.generate by default
            base = (
                pref(("finance.","network.","media.","video.","music.","doc.")) +
                pref(("text.","writing.","translate.","tools.","tts.","thread.","tutor.","sched."))
            ) + ["chat.general"]

            # fine-grained boosters (only when user text says so)
            if hits.get("face_blur"):
                base = _dp_unique_preserve(include("image.face_blur") + base)
            if hits.get("remove_bg"):
                base = _dp_unique_preserve(include("image.remove_bg") + base)
            if RE_REPLACE_BG.search(t):
                base = _dp_unique_preserve(include("image.background_replace","image.remove_bg") + base)
            if RE_ANNOTATE.search(t):
                base = _dp_unique_preserve(include("image.annotate") + base)
            if RE_TIME_CUE.search(t):
                base = _dp_unique_preserve(include("video.gif","video.thumbnail") + base)
            if RE_SCREENSHOT.search(t):
                base = _dp_unique_preserve(include("link.screenshot") + base)

            # guard finance/network negatives
            if RE_FIN_NEG.search(t):
                base = [i for i in base if not i.startswith("finance.")]
            if RE_NET_NEG.search(t):
                base = [i for i in base if not i.startswith("network.")]

            # STRICT: only admit image.generate on explicit cues
            GEN_IMG = re.compile(r"\b(generate|create|draw|make)\b.*\b(image|picture|art)\b|تصویر\s*بساز", re.I)
            if GEN_IMG.search(t):
                base = _dp_unique_preserve(include("image.generate") + base)

            # STRICT: core.* only when explicit core command
            if self._explicit_core_cmd(t):
                base = _dp_unique_preserve([x for x in pref(("core.",)) if x != "core.start"] + base)

        else:
            base = ["chat.general"]

        # 2) Final de-dup & return
        return _dp_unique_preserve(base)


















    def _planout_to_plain(self, plan: PlanOut) -> Dict[str, Any]:
        pipeline_nodes: List[Dict[str, Any]] = []
        for item in plan.pipeline or []:
            pipeline_nodes.append({
                "task_id": item.task_id,
                "intent": item.intent,
                "entities": self._kv_list_to_dict(item.entities),
                "bind": self._bind_list_to_dict(item.bind),
                "confidence": item.confidence
            })
        clarify = plan.clarify.model_dump() if plan.clarify else None
        return {"pipeline": pipeline_nodes, "clarify": clarify}

    def _synthesize_plan_from_hints(self, env: MessageEnvelope, hints: Dict[str, Any]) -> Dict[str, Any]:
        plan = (hints or {}).get("plan") or ""
        tlang = (hints or {}).get("target_lang") or self.lang_hint
        and_sum = bool((hints or {}).get("and_summarize"))
        url = (hints or {}).get("url") or (env.urls[0] if env.urls else None)
        provider = (hints or {}).get("provider") or (detect_provider_from_url(url) if url else None)
        nodes: List[Dict[str, Any]] = []

        def renumber(nodes):
            new=[]
            for i, n in enumerate(nodes, 1):
                n2=dict(n); n2["task_id"]=f"t{i}"
                new.append(n2)
            return new

        if plan in ("audio.transcribe_translate","stt->translate"):
            nodes = [
                {"task_id":"t1","intent":"audio.stt","entities":{"file_id":""},"bind":{},"confidence":0.96},
                {"task_id":"t2","intent":"text.translate","entities":{"target_lang":tlang},
                 "bind":{"text":{"from_task":"t1","key":"text"}},"confidence":0.95},
            ]
            if and_sum:
                nodes.append({"task_id":"t3","intent":"text.summarize","entities":{},
                              "bind":{"text":{"from_task":"t2","key":"text"}},"confidence":0.9})
            return {"pipeline": renumber(nodes), "clarify": None}

        if plan in ("audio.stt",):
            nodes = [{"task_id":"t1","intent":"audio.stt","entities":{"file_id":""},"bind":{},"confidence":0.95}]
            return {"pipeline": nodes, "clarify": None}

        if plan in ("text.translate","text.translate_summarize","translate.text"):
            nodes = [{"task_id":"t1","intent":"text.translate","entities":{"text":"","target_lang":tlang},"bind":{},"confidence":0.9}]
            if plan.endswith("_summarize"):
                nodes.append({"task_id":"t2","intent":"text.summarize","entities":{},
                              "bind":{"text":{"from_task":"t1","key":"text"}},"confidence":0.88})
            return {"pipeline": renumber(nodes), "clarify": None}

        # UPDATED: media chains (translate+summarize vs transcript_chain with optional summarize)
        if plan in ("media.translate_summarize","media.transcript_chain"):
            if not url:
                return {"pipeline": [], "clarify": None}
            allow = provider in (self.policy.get("provider_allowlist") or [])
            if allow:
                nodes = [
                    {"task_id":"t1","intent":"media.ingest_url","entities":{"url":url,"provider":provider},"bind":{},"confidence":0.95},
                    {"task_id":"t2","intent":"media.transcript","entities":{"provider":provider,"url":url,"max_minutes": self.caps["media_transcript_max_minutes"]},"bind":{},"confidence":0.93},
                ]
                if plan == "media.translate_summarize":
                    nodes.append({"task_id":"t3","intent":"text.translate","entities":{"target_lang":tlang},
                                  "bind":{"text":{"from_task":"t2","key":"text"}},"confidence":0.92})
                    nodes.append({"task_id":"t4","intent":"text.summarize","entities":{},
                                  "bind":{"text":{"from_task":"t3","key":"text"}},"confidence":0.9})
                else:
                    if and_sum:
                        nodes.append({"task_id":"t3","intent":"text.summarize","entities":{},
                                      "bind":{"text":{"from_task":"t2","key":"text"}},"confidence":0.9})
                return {"pipeline": nodes, "clarify": None}
            else:
                nodes = [
                    {"task_id":"t1","intent":"text.summarize",
                     "entities":{"text": f"User shared a link from non-allowlisted provider '{provider or 'unknown'}'. Summarize using only the URL text (no fetch): {url}"},
                     "bind":{},"confidence":0.9}
                ]
                return {"pipeline": nodes, "clarify": None}

        # NEW: image single-step plans
        if plan in ("image.remove_bg",):
            return {"pipeline":[{"task_id":"t1","intent":"image.remove_bg","entities":{"file_id":""},"bind":{},"confidence":0.95}], "clarify": None}
        if plan in ("image.face_blur",):
            return {"pipeline":[{"task_id":"t1","intent":"image.face_blur","entities":{"file_id":""},"bind":{},"confidence":0.95}], "clarify": None}

        # NEW: music plans
        if plan == "music.convert":
            return {"pipeline":[{"task_id":"t1","intent":"music.convert","entities":{"file_id":"", "format":"mp3", "bitrate":"192kbps"}, "bind":{}, "confidence":0.92}], "clarify": None}
        if plan == "music.preview":
            return {"pipeline":[{"task_id":"t1","intent":"music.preview","entities":{"file_id":"", "length_seconds":"20"}, "bind":{}, "confidence":0.9}], "clarify": None}

        if plan == "tutor.start":
            nodes = [
                {"task_id":"t1","intent":"tutor.level_test","entities":{},"bind":{},"confidence":0.92},
                {"task_id":"t2","intent":"tutor.plan.update","entities":{
                    "target_lang": hints.get("target_lang") or "",
                    "native_lang": hints.get("native_lang") or "",
                    "daily_minutes": hints.get("daily_minutes") or ""
                },"bind":{},"confidence":0.9},
            ]
            return {"pipeline": renumber(nodes), "clarify": None}

        if plan == "sched.show_slots":
            nodes = [{"task_id":"t1","intent":"sched.availability.show",
                      "entities":{"duration_minutes": hints.get("duration_minutes") or "",
                                  "date_range": hints.get("date_range") or hints.get("week") or "week"},
                      "bind":{},"confidence":0.9}]
            return {"pipeline": nodes, "clarify": None}

        if plan == "sched.book":
            ent = {}
            if hints.get("datetime_text"):
                ent["datetime"] = hints["datetime_text"]
            else:
                ent["date"] = hints.get("date") or ""
                ent["time"] = hints.get("time") or ""
            ent["duration_minutes"] = hints.get("duration_minutes") or ""
            ent["attendee"] = hints.get("attendee") or ""
            nodes = [{"task_id":"t1","intent":"sched.book","entities":ent,"bind":{},"confidence":0.92}]
            return {"pipeline": nodes, "clarify": None}

        if plan == "sched.reschedule":
            ent = {"booking_ref": hints.get("booking_ref") or ""}
            if hints.get("datetime_text"):
                ent["datetime"] = hints["datetime_text"]
            nodes = [{"task_id":"t1","intent":"sched.reschedule","entities":ent,"bind":{},"confidence":0.9}]
            return {"pipeline": nodes, "clarify": None}

        if plan == "sched.cancel":
            nodes = [{"task_id":"t1","intent":"sched.cancel",
                      "entities":{"booking_ref": hints.get("booking_ref") or ""},"bind":{},"confidence":0.9}]
            return {"pipeline": nodes, "clarify": None}

        return {"pipeline": [], "clarify": None}


    def _condense_pipeline(self, env: MessageEnvelope, pipeline: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        Condense to ≤3 steps. If the router is running with the no-op heuristic LLM (Option C),
        or if the LLM returns nothing, keep a simple deterministic condense rather than dropping the plan.
        """
        # 0) Fast exit
        if not pipeline or len(pipeline) <= 3:
            return pipeline

        # Helper: dedupe (intent + entities) and cap to 3
        def _safe_condense(pl: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
            seen = set(); out = []
            for n in pl:
                key = (n.get("intent"),
                      json.dumps(n.get("entities", {}), sort_keys=True, ensure_ascii=False))
                if key in seen:
                    continue
                seen.add(key)
                out.append(n)
                if len(out) >= 3:
                    break
            return out or pl[:3]

        # 1) No-op LLM → never call model; keep safe condense
        if isinstance(self.llm, _HeuristicLLM):
            return _safe_condense(pipeline)

        # 2) Real LLM condense
        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=self.condense_prompt.strip()),
                HumanMessage(content=f"Last user message:\n{(env.text or '(no text)')[:1200]}"),
                SystemMessage(content="[current_pipeline]\n" + json.dumps(pipeline, ensure_ascii=False)),
                SystemMessage(content="Constraint: Return <= 3 tasks using the same schema.")
            ]
        else:
            messages = [self.condense_prompt, env.text or "", json.dumps(pipeline, ensure_ascii=False)]

        structured = self.llm.with_structured_output(PlanOut, method="json_schema", strict=True)
        try:
            out: PlanOut = structured.invoke(messages, config={"callbacks": [UsageCallback(self._meter, "condense_pipeline")]})
            plain = self._planout_to_plain(out)
            nodes = plain.get("pipeline", []) or []
            # ← important change: if empty, fall back to safe condense (don’t fabricate chat.general)
            return nodes[:3] if nodes else _safe_condense(pipeline)
        except Exception:
            # Any failure → safe condense
            return _safe_condense(pipeline)

    @staticmethod
    def _bind_list_to_dict(binds: List[BindRef]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for b in binds or []:
            if not b.dest or not b.from_task: continue
            out[b.dest] = {"from_task": b.from_task, "key": (b.key or "text")}
        return out

    # NOTE: There was a duplicate definition of _core_plan_pipeline in v5.
    # Keep a single, corrected version here.
    def _core_plan_pipeline(self, env: MessageEnvelope, prev_pipeline: List[Dict[str,Any]],
                            session_prompts: List[str], direction_changed: bool,
                            adapter_hints: Optional[Dict[str,Any]] = None,
                            sess: Optional[SessionState] = None) -> Dict[str, Any]:
        # ensure we have the current session
        sess = sess or self._load_session(env.user_id)

        # ▶ Patch B: collect shared inputs with sticky session context
        shared = self.collect_shared_inputs(env, sess)

        # ▶▶ NEW: Deterministic music search ("songs by X", "tracks by X", "music from X")
        low_txt = (env.text or "").lower()
        m_artist = re.search(r"(?:songs?|music|tracks?)\s+(?:by|from)\s+([^,.\n]+)", low_txt)
        if m_artist:
            artist = m_artist.group(1).strip()
            return {
                "pipeline": [{
                    "task_id": "t1",
                    "intent": "music.inline.search",
                    "entities": {"query": f"artist:{artist}", "limit": "10"},
                    "bind": {},
                    "confidence": 0.95
                }],
                "clarify": None
            }
        # ◀◀ NEW

        # Prefer adapter hints FIRST if available
        if (adapter_hints or {}).get("plan"):
            synth = self._synthesize_plan_from_hints(env, adapter_hints or {})
            if synth.get("pipeline"):
                # Harmonize anyway to ensure binds/targets complete
                synth["pipeline"] = self._harmonize_with_hints(synth.get("pipeline", []), adapter_hints or {})
                return synth

        # Otherwise, compute shortlist + LLM plan
        shortlist = self._compute_shortlist(env, shared, adapter_hints or {}, sess)
        top = shortlist[0] if shortlist else None
        if top:
            nodes = self._expand_via_map(top)
            return {"pipeline": nodes, "clarify": None}
        # Patch C: FAST PATH for deterministic, one-step intents when already solvable
        deterministic_one_step = {
            "finance.crypto.price","finance.fx.rate","finance.fx.convert",
            "network.vpn.get","network.proxy.get",
            "text.summarize","text.translate","audio.stt",
            "media.transcript",
            "music.convert","music.preview",
            "core.help","core.menu","core.settings","core.language.set",
            "link.screenshot",   

        }
        CORE_FAST_CMDS = {"core.help","core.menu","core.settings"}  # language.set is allowed unconditionally
        for it in shortlist:
            # Guard image ops so they don't hijack non-image attachments
            if it in ("image.remove_bg","image.face_blur","image.describe","image.variation"):
                if env.media_type != "image":
                    hits = _imperative_hits(env.text or "")
                    # Only allow on non-image turns if the user explicitly asked
                    if not (hits.get("remove_bg") or hits.get("face_blur")):
                        continue
            # Skip hijacking fast-path for core help/menu/settings unless explicitly typed
            if it in CORE_FAST_CMDS and not self._explicit_core_cmd(env.text):
                continue
            if it in deterministic_one_step and self._is_solvable_now(it, shared, bool(env.text)):
                return {
                    "pipeline": [{
                        "task_id":"t1","intent": it,"entities":{},"bind":{},"confidence":0.95
                    }],
                    "clarify": None
                }

        prompt = self._build_dynamic_prompt(env, shared, shortlist, adapter_hints or {}, sess, direction_changed)

        if SystemMessage and HumanMessage:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=(env.text or "(no text)")[:1800]),
            ]
        else:
            messages = [prompt, env.text or ""]

        structured = self.llm.with_structured_output(PlanOut, method="json_schema", strict=True)
        try:
            out: PlanOut = structured.invoke(messages, config={"callbacks": [UsageCallback(self._meter, "plan_pipeline")]})
            plain = self._planout_to_plain(out)
        except Exception:
            out = _HeuristicStructured(PlanOut).invoke(messages, config=None)
            plain = self._planout_to_plain(out)

        # If LLM produced nothing, synthesize from hints (if any)
        if not (plain.get("pipeline") or []):
            synth = self._synthesize_plan_from_hints(env, adapter_hints or {})
            if synth.get("pipeline"):
                synth["pipeline"] = self._harmonize_with_hints(synth.get("pipeline", []), adapter_hints or {})
                return synth

        return plain

    def _kv_list_to_dict(self, kvs: List[KV]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for kv in kvs or []:
            try:
                key = kv.key
                val = self._coerce_scalar(kv.value)
                if key:
                    out[key] = val
            except Exception:
                pass
        return out

    def _compute_shortlist(self, env: MessageEnvelope, shared: Dict[str, Any],
                          hints: Dict[str, Any], sess: SessionState) -> List[str]:
        """
        SR-as-SoT (single source of truth) shortlist:
        - Session-aware gating: if current turn is text-only but session carries a sticky file/url, bias gate accordingly.
        - SR owns the candidate set; adapter "prelist" does not expand the pool (only influences ranking via solvability).
        - HARD client-side post-filter of SR picks to the allowed set (gate + optional domain scoping).
        - Score refinement: centroid Top-K over positives + hard-negative penalty + sharpening.
        - Ranking: solvable > unsolvable, then SR tiers/scores; 'chat.general' only as last resort.
        """
        # Clear previous-turn trace
        self._sr_trace["intent"] = {}

        CORE_AMBIG = {"core.cancel","core.help","core.menu"}

        def _penalize_core_ambig(name: str) -> int:
            # push core.* down unless user typed an explicit command
            return 1 if (name in CORE_AMBIG and not self._explicit_core_cmd(env.text)) else 0




        # --- Inputs
        text = self._normalize_fa_ar(env.text or "")
        text_sq = _strip_quoted(text or "")
        intents_all = list(self.intent_set)

        # If SR picked a domain confidently upstream, we scope to it
        domain_hint = self._current_domain_hint if getattr(self, "_domain_from_sr", False) else None

        # ---- Session-aware modality gate ----------------------------------------
        # Base gate from this turn
        gated = [i for i in self._gate_by_modality(env) if i in intents_all]

        # Overlay: if user sent only text *this* turn but session carries a file/url, bias gate
        turn_mod = new_turn_modality(env)
        if turn_mod == "text" and sess and isinstance(sess.entity_bag, dict):
            # Sticky FILE → prefer image/ocr/receipt/table/doc/video families
            if sess.entity_bag.get("file_id"):
                prefixed = tuple(["ocr.", "barcode.", "receipt.", "table.", "image.", "doc.", "video."])
                sticky = [i for i in intents_all if i.startswith(prefixed)]
                gated = _dp_unique_preserve(sticky + [x for x in gated if x not in sticky])
            # Sticky URL → prefer media/video/link/podcast/music families
            if sess.entity_bag.get("url"):
                prefixed = tuple(["media.", "video.", "link.", "podcast.", "music."])
                sticky = [i for i in intents_all if i.startswith(prefixed)]
                gated = _dp_unique_preserve(sticky + [x for x in gated if x not in sticky])

        # Domain scoping (only when SR was confident on domain)
        gated = self._restrict_to_domain(gated, domain_hint)

        # ---- SR call (strictly scoped) + HARD post-filter ------------------------
        sr_scores: Dict[str, float] = {}
        sr_limit = min(getattr(self, "semantic_intent_limit", 10), len(intents_all)) or 10

        def _allowed_routes_for_sr() -> Optional[List[str]]:
            if domain_hint:
                return [i for i in intents_all if self.domain_map.get(_ns(i), "core") == domain_hint] or None
            return gated or None

        def _sr_call(txt: str) -> List[Any]:
            if not (self.semantic_enabled and self._sr_intent and (txt or "").strip()):
                return []
            try:
                route_filter = _allowed_routes_for_sr()
                try:
                    picks = self._sr_intent(text=txt, limit=sr_limit, route_filter=route_filter)
                except TypeError:
                    picks = self._sr_intent(text=txt, limit=sr_limit)
            except Exception as e:
                if self._debug: print("[DEBUG] SR intent call failed:", repr(e))
                return []
            picks = self._to_list(picks)

            # HARD client-side filter: even if SR ignores route_filter, keep only allowed
            allowed = set(_allowed_routes_for_sr() or [])
            if allowed:
                picks = [c for c in picks if (self._choice_name(c) in allowed)]

            if self._debug:
                print("[DEBUG] SR raw:", [(self._choice_name(c), self._choice_score(c)) for c in picks][:5])
            return picks

        for choice in (_sr_call(text) + _sr_call(text_sq)):
            name  = self._choice_name(choice)
            score = float(self._choice_score(choice) or 0.0)
            if name and name in intents_all:
                sr_scores[name] = max(sr_scores.get(name, 0.0), score)

        # ---- Score refinement: centroid Top-K + negatives + sharpening -----------
        enc = self._get_encoder()
        try:
            if enc and (env.text or "").strip() and sr_scores:
                q_vecs = self._embed_strings(enc, [env.text])
                if q_vecs:
                    q = q_vecs[0]
                    W_SR, W_C, K = 0.5, 0.5, 3

                    # Positive centroid blend
                    for name in list(sr_scores.keys()):
                        pos_vecs = (getattr(self, "_emb_bank_intent", {}) or {}).get(name) or []
                        if pos_vecs:
                            cscore = self._centroid_topk(q, pos_vecs, k=K)
                            sr_scores[name] = W_SR * float(sr_scores[name]) + W_C * float(cscore)

                    # Hard-negatives penalty
                    ALPHA = 0.25
                    for name in list(sr_scores.keys()):
                        neg_vecs = (getattr(self, "_emb_bank_neg", {}) or {}).get(name) or []
                        if neg_vecs:
                            nmax = max((self._cos(q, v) for v in neg_vecs), default=0.0)
                            sr_scores[name] = max(0.0, float(sr_scores[name]) - ALPHA * nmax)

                    # Normalize + sharpen margins
                    vals = list(sr_scores.values())
                    if vals:
                        mn, mx = min(vals), max(vals)
                        P = 1.8
                        if mx > mn:
                            for k in list(sr_scores.keys()):
                                norm = (sr_scores[k] - mn) / (mx - mn)
                                sr_scores[k] = norm ** P
        except Exception as e:
            if self._debug: print("[DEBUG] SR refinement failed:", repr(e))

        sr_sorted = sorted(sr_scores.items(), key=lambda kv: kv[1], reverse=True)
        sr_top_name, sr_top_score = (sr_sorted[0] if sr_sorted else (None, 0.0))
        self._sr_trace["intent"] = {
            "scoped_domain": domain_hint,
            "scores": sr_scores,
            "top": sr_top_name,
            "top_score": sr_top_score,
            "list": sr_sorted,
        }
        sr_sorted_names = [k for k, _ in sr_sorted][:sr_limit]

        # SR-lock: if SR was decisive enough and top is solvable, return it first (and stop)
        # SR-lock: if SR was decisive enough and top is solvable, return ONLY the top.
        thr = float(getattr(self, "semantic_intent_threshold", 0.60))
        top_vals = sorted(sr_scores.values(), reverse=True)[:2] + [0.0, 0.0]
        top, second = top_vals[0], top_vals[1]
        rel_margin = (top - second) / max(top, 1e-6) if top > 0 else 0.0

        text_present = bool((env.text or "").strip())
        pool = [i for i in sr_sorted_names if i in gated]  # SR picks that pass gate

        if (
            sr_top_name
            and (sr_top_name in pool)
            and self._is_solvable_now(sr_top_name, shared, text_present)
            and (top >= thr and rel_margin >= 0.12)
        ):
            # LOCK: return only the SR top to prevent planner drift.
            return [sr_top_name]

        # ---- Candidate pool = gate ∪ SR picks (no adapter prelist expansion) ----
        # Safety: if SR returned nothing, fall back to a tiny gated set (prefer solvable)
        if not pool:
            solvables = [i for i in gated if self._is_solvable_now(i, shared, text_present)]
            pool = solvables[:5] or gated[:5]

        pool = self._restrict_to_domain(pool, domain_hint)

        if not pool:
            return ["chat.general"] if "chat.general" in intents_all else []


        # ---- Ranking features -----------------------------------------------------
        thr = float(getattr(self, "semantic_intent_threshold", 0.60))
        top_vals = sorted(sr_scores.values(), reverse=True)[:2] + [0.0, 0.0]
        top, second = top_vals[0], top_vals[1]
        margin = max(0.0, top - second)
        rel_margin = (margin / max(top, 1e-6)) if top > 0 else 0.0
        sr_indecisive = (top < thr) or (rel_margin < 0.12)

        text_present = bool((env.text or "").strip())
        last_intent = sess.last_intent if sess.last_intent in intents_all else None

        def _hit_any(patterns: Set[str], text_low: str) -> bool:
            if not patterns: return False
            for pat in patterns:
                rx = r"\b" + re.escape(pat).replace(r"\ ", r"\s+") + r"\b"
                if re.search(rx, text_low):
                    return True
            return False

        low = (text or "").lower()

        def _tier(score: float) -> int:
            if score <= 0: return 0
            strong_cut = max(thr, top - 0.25 * margin)
            if score >= strong_cut: return 3
            if score >= thr: return 2
            return 1

        feats: Dict[str, Dict[str, Any]] = {}
        for i in pool:
            score = float(sr_scores.get(i, 0.0))
            solvable = self._is_solvable_now(i, shared, text_present)  # <- always prefer solvable
            dom_neg = self._sr_domain_neg_utt.get(self.domain_map.get(_ns(i), "core"), set())
            feats[i] = {
                "tier": _tier(score),
                "sr_score": score,
                "solvable": solvable,
                "gated_ok": i in gated,
                "last_intent_match": (i == last_intent),
                "neg_hit": _hit_any(self._sr_neg_utt.get(i, set()) or set(), low),
                "dom_neg_hit": _hit_any(dom_neg or set(), low),
                "orig_idx": pool.index(i),
            }

        # Primary sort: if SR decisive use SR order, else fall back to solvability-first
        def _key_sr_rank(name: str):
            f = feats[name]
            return (
                0 if f["solvable"] else 1,
                _penalize_core_ambig(name),
                -f["tier"],
                -f["sr_score"],
                0 if f["gated_ok"] else 1,
                1 if (f["neg_hit"] or f["dom_neg_hit"]) else 0,
                0 if f["last_intent_match"] else 1,
                f["orig_idx"],
            )

        def _key_fallback(name: str):
            f = feats[name]
            return (
                0 if f["solvable"] else 1,
                _penalize_core_ambig(name),
                0 if f["gated_ok"] else 1,
                -f["sr_score"],
                f["orig_idx"],
            )

        ranked = sorted(pool, key=_key_fallback if sr_indecisive else _key_sr_rank)

        # Keep concise list; widen when indecisive
        max_n = 10 if sr_indecisive else 6
        shortlist = ranked[:max_n]

        # 'chat.general' only as last resort
        if "chat.general" in shortlist:
            shortlist = [i for i in shortlist if i != "chat.general"] + ["chat.general"]
        elif (domain_hint in (None, "core")) and (new_turn_modality(env) == "text"):
            solvables = [i for i in shortlist if feats[i]["solvable"]]
            if not solvables and "chat.general" in intents_all:
                shortlist = (shortlist + ["chat.general"])[:max_n]

        if self._debug:
            self._log_debug("shortlist", {
                "top": round(top, 3),
                "margin": round(margin, 3),
                "names": shortlist,
                "scores": [round(sr_scores.get(i, 0.0), 3) for i in shortlist]
            })
        return shortlist





    def _harmonize_with_hints(self, nodes: List[Dict[str,Any]], hints: Dict[str,Any]) -> List[Dict[str,Any]]:
        """Ensure adapter hint plans are realized and missing fields are injected."""
        if not hints: return nodes
        plan = hints.get("plan")
        if not plan: return nodes

        # audio.transcribe_translate chain enforcement
        if plan in ("audio.transcribe_translate","stt->translate"):
            has_stt = any(n.get("intent") in ("audio.stt","stt.transcribe") for n in nodes)
            has_tr  = any(n.get("intent") in ("text.translate","translate.text") for n in nodes)
            if not (has_stt and has_tr):
                return self._synthesize_plan_from_hints(MessageEnvelope(user_id="", chat_id=""), hints)["pipeline"]
            # ensure bind translate.text <- stt
            stt_node = next((n for n in nodes if n.get("intent") in ("audio.stt","stt.transcribe")), None)
            tr_node = next((n for n in nodes if n.get("intent") in ("text.translate","translate.text")), None)
            if stt_node and tr_node:
                b = dict(tr_node.get("bind", {}))
                b["text"] = {"from_task": stt_node.get("task_id","t1"), "key":"text"}
                tr_node["bind"] = b
            # backfill target_lang if missing
            if tr_node:
                ents = dict(tr_node.get("entities", {}))
                if not ents.get("target_lang") and hints.get("target_lang"):
                    ents["target_lang"] = hints["target_lang"]
                    tr_node["entities"] = ents
            return nodes

        # media chain (translate+summarize vs transcript_chain)
        if plan in ("media.translate_summarize","media.transcript_chain"):
            # rely on synthesized plan for correctness of chain
            return self._synthesize_plan_from_hints(MessageEnvelope(user_id="", chat_id=""), hints)["pipeline"]

        # text.translate (+ optional summarize)
        if plan in ("text.translate","text.translate_summarize"):
            for n in nodes:
                if n.get("intent") == "text.translate":
                    ents = dict(n.get("entities", {}))
                    if not ents.get("target_lang") and hints.get("target_lang"):
                        ents["target_lang"] = hints["target_lang"]
                        n["entities"] = ents
            return nodes

        # sched flows: inject missing entities from hints
        if plan in ("sched.show_slots","sched.book","sched.reschedule","sched.cancel"):
            mapping = {
                "sched.show_slots": "sched.availability.show",
                "sched.book": "sched.book",
                "sched.reschedule": "sched.reschedule",
                "sched.cancel": "sched.cancel",
            }
            required_intent = mapping[plan]
            found = False
            for n in nodes:
                if n.get("intent") == required_intent:
                    found = True
                    ents = dict(n.get("entities", {}))
                    for key in ("duration_minutes","datetime","date","time","attendee","booking_ref","date_range","week"):
                        if key in hints and (key not in ents or _is_missing_val(ents.get(key))):
                            ents[key] = hints[key]
                    n["entities"] = ents
            if not found:
                return self._synthesize_plan_from_hints(MessageEnvelope(user_id="", chat_id=""), hints)["pipeline"]
            return nodes

        # music flows: prefer synthesized minimal nodes
        if plan in ("music.convert","music.preview"):
            return self._synthesize_plan_from_hints(MessageEnvelope(user_id="", chat_id=""), hints)["pipeline"]

        # Other one-step plans
        if not nodes and plan in ("audio.stt","text.translate","text.translate_summarize","media.translate_summarize"):
            return self._synthesize_plan_from_hints(MessageEnvelope(user_id="", chat_id=""), hints)["pipeline"]

        return nodes

    def _build_dynamic_prompt(self, env: MessageEnvelope, shared: Dict[str, Any], shortlist: List[str],
                              hints: Dict[str, Any], sess: SessionState, direction_changed: bool) -> str:
        req_snip = _dp_snip(env.text, 240)
        available_now = ", ".join([k for k in ["text","url","file_id","provider"] if k in shared]) or "(none)"
        hint_bits = []
        if hints.get("plan"): hint_bits.append(f"plan={hints['plan']}")
        if hints.get("target_lang"): hint_bits.append(f"target_lang={hints['target_lang']}")
        if hints.get("and_summarize"): hint_bits.append("and_summarize=true")
        hint_line = " | ".join(hint_bits) if hint_bits else "(none)"
        last_intent = sess.last_intent if sess.last_intent in self.intent_set else "-"
        pending_key = (hints or {}).get("pending_key") or "-"
        allowed = "\n".join(f"- {i}" for i in shortlist)
        ex = """{
  "pipeline":[
    {"task_id":"t1","intent":"audio.stt","entities":[{"key":"file_id","value":""}],"bind":[],"confidence":0.93},
    {"task_id":"t2","intent":"text.translate","entities":[{"key":"target_lang","value":"fa"}],
     "bind":[{"dest":"text","from_task":"t1","key":"text"}],"confidence":0.91}
  ]
  }"""
        return f"""
  You are CoreRouter. Plan up to 3 tasks from the shortlist. Output ONLY JSON using this schema:

  - pipeline: list of steps
    * task_id: string
    * intent: EXACT name from Allowed intents (below)
    * entities: list of {{"key":"...","value":"..."}}
    * bind: list of {{"dest":"...","from_task":"...","key":"text"}}
    * confidence: 0.60–1.00
  - clarify (optional): {{"question":"...","missing":["<key>"],"node_index":0}}

  User request:
  {req_snip}

  Available inputs: {available_now}
  Adapter hint: {hint_line}
  Session: last_intent={last_intent}, pending_key={pending_key}
  Direction changed: {str(direction_changed).lower()}

  Allowed intents (shortlist):
  {allowed}

  Rules:
  - Prefer tasks solvable with current inputs; avoid off‑modality picks.
  - Use 'bind' to chain outputs (e.g., stt → translate).
  - Enforce provider allowlist for media; otherwise prefer safe summarize with URL text only.
  - At most ONE clarify when a required key is missing (Fa question). After clarify, stop.

  Example (2‑step):
  {ex}
  """.strip()

    @property
    def condense_prompt(self) -> str:
        return ("You are TaskCondenser. Return a NEW 'pipeline' with <=3 tasks using the same schema. "
                "Merge duplicates; keep minimum steps; no scraping. "
                "Remember: 'entities' is a list of {key,value}; 'bind' is a list of {dest,from_task,key}.")

    @property
    def classify_system(self) -> str:
        return "Decide the best domain: core | media | images | tutor | scheduler | utilities | other. Reply as JSON."

    @property
    def explore_prompt(self) -> str:
        return """Ask ONE short clarifying question in Persian (Fa) to disambiguate the user's goal.
  Then provide up to 3 bullet options (very short). No prose before/after."""

    # ---------- REQUIRED/FILL/MISSING helpers ----------

    def _has_all(self, fields: List[str], ent: Dict[str, Any], bind: Dict[str,Any]) -> bool:
        def available(k: str) -> bool:
            if k in ent and not _is_missing_val(ent[k]): return True
            if k in ent and isinstance(ent[k], str) and ent[k].strip() == "__present__":
                return True
            if bind and k in bind and isinstance(bind[k], dict) and bind[k].get("from_task"): return True
            return False
        return all(available(f) for f in fields)

    def _expr_ok(self, expr: str, ent: Dict[str, Any], bind: Dict[str,Any]) -> bool:
        for alt in [a.strip() for a in expr.split("|")]:
            if self._has_all([x.strip() for x in alt.split("+")], ent, bind):
                return True
        return False

    def missing_for_node(self, intent: str, entities: Dict[str, Any], bind: Dict[str,Any]) -> List[str]:
        """
        Return EXACTLY ONE missing key for the node, respecting alternative expressions:
        - Supports "file_id|url" (prefers file_id)
        - Supports "datetime|date+time" (prefer single 'datetime' if both absent; else ask for the missing piece)
        - Considers bind references (dest <- from_task) as "present"
        - Treats the sentinel "__present__" for 'text' as present (auto-satisfied; substituted at execution)
        - NEW: accepts name fallbacks for ID fields via slotfill allow_names_for_ids, e.g. "playlist_id->playlist_name"
        """
        spec = self.intent_specs.get(intent, {}) or {}
        required = list(spec.get("required", []) or [])
        ent = dict(entities or {})
        bn = dict(bind or {})

        # ── NEW: map e.g. {"playlist_id": "playlist_name"} from slotfill config ──
        allow_map: Dict[str, str] = {}
        try:
            sf = self._slotfill_specs_for_intent(intent) or {}
            for rule in (sf.get("allow_names_for_ids") or []):
                try:
                    target, fallback = [p.strip() for p in rule.split("->", 1)]
                    if target and fallback:
                        allow_map[target] = fallback
                except Exception:
                    pass
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────────────

        def _has(k: str) -> bool:
            # accept fallback if configured and present
            if k in allow_map:
                fb = allow_map[k]
                if fb in ent and not _is_missing_val(ent.get(fb)):
                    return True
            if k in ent and not _is_missing_val(ent.get(k)):
                return True
            if k in ent and isinstance(ent.get(k), str) and ent.get(k).strip() == "__present__":
                return True
            if bn and k in bn and isinstance(bn[k], dict) and bn[k].get("from_task"):
                return True
            return False

        def _missing_from_alt_expr(expr: str) -> Optional[str]:
            alts = [a.strip() for a in (expr or "").split("|") if a.strip()]

            # Single-term (possibly composite with '+')
            if len(alts) == 1:
                parts = [p.strip() for p in alts[0].split("+") if p.strip()]
                for p in parts:
                    if not _has(p):
                        return p
                return None  # satisfied

            # Special case: file_id|url  (prefer file_id)
            if "file_id" in alts and "url" in alts:
                if not _has("file_id"):
                    return "file_id"
                if not _has("url"):
                    return "url"
                return None  # one is present

            # Special case: datetime|date+time
            if "datetime" in alts and any(a == "date+time" for a in alts):
                if _has("datetime"):
                    return None
                have_date = _has("date")
                have_time = _has("time")
                if have_date and not have_time:
                    return "time"
                if have_time and not have_date:
                    return "date"
                # neither is present → prefer asking for a single 'datetime'
                return "datetime"

            # Generic: choose the alternative that needs the fewest missing fields.
            def _missing_count(alt: str) -> int:
                parts = [p.strip() for p in alt.split("+") if p.strip()]
                return sum(0 if _has(p) else 1 for p in parts)

            # Tie-breaking preference: file_id > datetime > first
            ordered = sorted(
                alts,
                key=lambda a: (
                    _missing_count(a),
                    0 if a == "file_id" else 1 if a == "datetime" else 2
                )
            )
            chosen = ordered[0]
            for p in [x.strip() for x in chosen.split("+") if x.strip()]:
                if not _has(p):
                    return p
            return None

        for expr in required:
            if self._expr_ok(expr, ent, bn):
                continue
            mk = _missing_from_alt_expr(expr)
            if mk:
                return [mk]

        return []







    def friendly_question(self, intent: str, missing: List[str], ent: Dict[str, Any]) -> str:
        if not missing:
            defaultables = {
                "timezone": self.defaults.get("timezone"),
                "size": self.defaults.get("image_size"),
                "quality": self.defaults.get("media_quality"),
                "voice": self.defaults.get("tts_voice")
            }
            opt = (self.intent_specs.get(intent) or {}).get("optional", [])
            for k,v in defaultables.items():
                if k in opt and k not in ent and v:
                    return f"ست کنم {k} = {v} ؟ (yes/no)"
            return "همه‌چیز آماده‌ست؛ انجام بدهم؟ / All set. Shall I proceed?"
        key = missing[0]
        return self.settings_raw.get("settings", {}).get("friendly_labels", {}).get(
            key, f"لطفا «{key}» را مشخص کنید. / Please provide **{key}**."
        )

    # ---------- Shared inputs & auto-fill/bind ----------

    def collect_shared_inputs(self, env: MessageEnvelope, sess: Optional[SessionState] = None) -> Dict[str, Any]:
        """
        1) Collect inputs from *this* turn.
        2) Soft-merge sticky inputs from the session (file_id/url/provider) if missing this turn.
        3) Update session bag when fresh inputs arrive.
        """
        shared: Dict[str, Any] = {}
        if env.file_id:
            shared["file_id"] = env.file_id
        if env.urls:
            shared["url"] = env.urls[0]
            prov = detect_provider_from_url(env.urls[0])
            if prov:
                shared["provider"] = prov
        if env.text:
            shared["text"] = "__present__"
        shared["_user_text"] = env.text or ""

        # Sticky carry-over (so the next message like "gif 00:00:02–00:00:05" sees the prior file)
        if sess and isinstance(sess.entity_bag, dict):
            if "file_id" not in shared and sess.entity_bag.get("file_id"):
                shared["file_id"] = sess.entity_bag["file_id"]
            if "url" not in shared and sess.entity_bag.get("url"):
                shared["url"] = sess.entity_bag["url"]
                if "provider" not in shared:
                    prov2 = detect_provider_from_url(shared["url"])
                    if prov2:
                        shared["provider"] = prov2

        # Update bag with newly seen attachments/links
        if sess and isinstance(sess.entity_bag, dict):
            if "file_id" in shared:
                sess.entity_bag["file_id"] = shared["file_id"]
            if "url" in shared:
                sess.entity_bag["url"] = shared["url"]
            if "provider" in shared:
                sess.entity_bag["provider"] = shared["provider"]

        return shared

    TEXT_PRODUCERS = {
        # legacy + new
        "writing.summarize","writing.rewrite","writing.expand","writing.outline","writing.fix_grammar",
        "stt.transcribe","ocr.extract","ocr.tldr","image.describe","inline.tldr","chat.general",
        "link.transcript","link.tldr","podcast.tldr","thread.summarize",
        # new families
        "text.summarize","text.rewrite","text.correct","text.translate","text.draft",
        "audio.stt","media.transcript","media.summarize","doc.ocr","doc.qa"
    }

    def auto_bind_text(self, pipeline: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        out=[]
        for i, node in enumerate(pipeline):
            ent = dict(node.get("entities", {}))
            bind = dict(node.get("bind", {}))
            needs_text = False
            spec = self.intent_specs.get(node["intent"],{})
            for req_expr in spec.get("required", []):
                if "text" in req_expr: needs_text = True
                if "text|file_id|url" in req_expr or "text|url" in req_expr: needs_text = True
            if needs_text and "text" not in ent and "text" not in bind:
                src_id = None
                for j in range(i-1, -1, -1):
                    if pipeline[j]["intent"] in self.TEXT_PRODUCERS:
                        src_id = pipeline[j].get("task_id")
                        break
                if src_id: bind["text"] = {"from_task": src_id, "key": "text"}
            node2=dict(node); node2["entities"]=ent; node2["bind"]=bind
            out.append(node2)
        return out

    def fill_required_alternatives_node(self, entities: Dict[str,Any], bind: Dict[str,Any], shared: Dict[str,Any], intent: str):
        spec = self.intent_specs.get(intent) or {}
        for expr in spec.get("required", []):
            expr = expr.strip()
            # Already satisfied? skip
            if ("|" in expr or "+" in expr) and self._expr_ok(expr, entities, bind):
                continue

            if "text|file_id|url" in expr:
                # If the node already binds 'text' from a previous step, respect that and avoid injecting url/file.
                if "text" in bind and isinstance(bind["text"], dict) and bind["text"].get("from_task"):
                    if "text" not in entities:
                        entities["text"] = shared.get("text", "__present__")
                else:
                    # prefer file_id, then url, else text if present in shared
                    if ("file_id" not in entities or _is_missing_val(entities.get("file_id"))) and "file_id" not in bind and "file_id" in shared:
                        entities["file_id"] = shared["file_id"]
                    elif ("url" not in entities or _is_missing_val(entities.get("url"))) and "url" not in bind and "url" in shared:
                        entities["url"] = shared["url"]
                    elif "text" in shared and ("text" not in entities or _is_missing_val(entities.get("text"))):
                        entities["text"] = shared["text"]  # "__present__" sentinel

            elif "text|url" in expr:
                if "url" in shared and ("url" not in entities or _is_missing_val(entities.get("url"))) and "url" not in bind:
                    entities["url"] = shared["url"]
                elif "text" in shared and ("text" not in entities or _is_missing_val(entities.get("text"))):
                    entities["text"] = shared["text"]
            elif "file_id|url" in expr:
                # prefer file_id from shared, then url
                if "file_id" in shared and ("file_id" not in entities or _is_missing_val(entities.get("file_id"))) and "file_id" not in bind:
                    entities["file_id"] = shared["file_id"]
                elif "url" in shared and ("url" not in entities or _is_missing_val(entities.get("url"))) and "url" not in bind:
                    entities["url"] = shared["url"]
            elif "url|file_id" in expr:
                # NEW: prefer url from shared, else file_id
                if "url" in shared and ("url" not in entities or _is_missing_val(entities.get("url"))) and "url" not in bind:
                    entities["url"] = shared["url"]
                elif "file_id" in shared and ("file_id" not in entities or _is_missing_val(entities.get("file_id"))) and "file_id" not in bind:
                    entities["file_id"] = shared["file_id"]
            elif "datetime|date+time" in expr:
                has_datetime = ("datetime" in entities and not _is_missing_val(entities["datetime"])) or ("datetime" in bind)
                has_date_time = (
                    (("date" in entities) and not _is_missing_val(entities.get("date"))) or ("date" in shared)
                ) and (
                    (("time" in entities) and not _is_missing_val(entities.get("time"))) or ("time" in shared)
                )
                if not (has_datetime or has_date_time):
                    if "datetime" in shared and ("datetime" not in entities or _is_missing_val(entities.get("datetime"))):
                        entities["datetime"] = shared["datetime"]
                    else:
                        if "date" in shared and ("date" not in entities or _is_missing_val(entities.get("date"))):
                            entities["date"] = shared["date"]
                        if "time" in shared and ("time" not in entities or _is_missing_val(entities.get("time"))) :
                            entities["time"] = shared["time"]

    def autofill_inputs_pipeline(self, pipeline: List[Dict[str,Any]], shared: Dict[str,Any]) -> List[Dict[str,Any]]:
        # NEW: robust fallback extraction for URLs from raw _user_text
        raw_user = (shared.get("_user_text") or "")
        fallback_urls = extract_urls(raw_user)
        fallback_url = fallback_urls[0] if fallback_urls else None

        out=[]
        for node in pipeline:
            ent = dict(node.get("entities", {}))
            bind = dict(node.get("bind", {}))
            self.fill_required_alternatives_node(ent, bind, shared, node["intent"])

            # URL-required intents
            if node["intent"] in self.url_required_intents:
                if ("url" not in ent or _is_missing_val(ent.get("url"))) and "url" not in bind:
                    if "url" in shared:
                        ent["url"] = shared["url"]
                    elif fallback_url:
                        ent["url"] = fallback_url

            # Provider inference
            if (("provider" not in ent) or _is_missing_val(ent.get("provider"))) and "provider" not in bind and "url" in ent and not _is_missing_val(ent["url"]):
                prov = detect_provider_from_url(ent["url"])
                if prov: ent["provider"] = prov

            # NEW: generic file auto-fill for any intent whose spec includes 'file_id'
            spec = self.intent_specs.get(node["intent"], {})
            req_exprs = (spec.get("required", []) or [])
            req_joined = " ".join(req_exprs)
            if (("file_id" not in ent) or _is_missing_val(ent.get("file_id"))) \
                and ("file_id" not in bind) \
                and ("file_id" in shared) and (not _is_missing_val(shared.get("file_id"))) \
                and ("file_id" in req_joined):

                ent["file_id"] = shared["file_id"]

            # Legacy: modality-based file fill (kept for back-compat)
            if ("file_id" not in ent or _is_missing_val(ent.get("file_id"))) and "file_id" not in bind and "file_id" in shared and intent_modality(node["intent"]) in ("image","voice","file"):
                ent["file_id"] = shared["file_id"]

            # NEW: If text required and shared has "__present__", use it
            if "text" not in ent and "text" not in bind and "text" in shared:
                # only if the spec contains text somewhere (hard or alternative)
                if any("text" in (r or "") for r in (spec.get("required", []) or [])):
                    ent["text"] = shared["text"]
                # ALSO: cover alts 'text|...' or '...|text'
                elif any(("text|" in (r or "")) or ("|text" in (r or "")) for r in (spec.get("required", []) or [])):
                    ent["text"] = shared["text"]

            # NEW: Defensive fill for media.transcript when url missing
            if node["intent"] == "media.transcript":
                if ("url" not in ent or _is_missing_val(ent.get("url"))) and "url" not in bind:
                    if "url" in shared:
                        ent["url"] = shared["url"]
                    elif fallback_url:
                        ent["url"] = fallback_url

            node2=dict(node); node2["entities"]=ent; node2["bind"]=bind
            out.append(node2)
        return out

    def apply_phase2_policy(self, nodes: List[Dict[str,Any]], shared: Dict[str,Any], hints: Optional[Dict[str,Any]]=None) -> Tuple[List[Dict[str,Any]], Optional[Dict[str,Any]]]:
        clarify = None
        new_nodes=[]
        duration_hint = None
        if hints:
            duration_hint = hints.get("duration_minutes_hint")
        for nd in nodes:
            intent = nd["intent"]
            ent = dict(nd.get("entities", {}))
            bind = dict(nd.get("bind", {}))

            # infer provider
            if (("provider" not in ent) or _is_missing_val(ent.get("provider"))) and "provider" not in bind and "url" in ent:
                prov = detect_provider_from_url(ent["url"])
                if prov: ent["provider"] = prov

            # Provider allowlist control for media
            if intent.startswith(self.RISKY_URL_INTENT_PREFIXES):
                prov = ent.get("provider") or ""
                allow = prov in (self.policy.get("provider_allowlist") or [])
                if not allow:
                    fallback = self.policy.get("fallback_on_violation", "inline.tldr")
                    if fallback not in self.intent_set:
                        fallback = "text.summarize" if "text.summarize" in self.intent_set else "chat.general"
                    nd = dict(nd)
                    safe_entities = {}
                    if fallback in ("inline.tldr","text.summarize"):
                        safe_entities["text"] = f"User shared a link. Scraping disabled for provider '{prov or 'unknown'}'. Summarize using only the URL text (do not fetch): {ent.get('url','')}"
                    nd["intent"] = fallback
                    nd["entities"] = safe_entities
                    nd["bind"] = {}
                else:
                    # enforce caps where applicable
                    if intent == "media.transcript":
                        if "max_minutes" not in ent or _is_missing_val(ent.get("max_minutes")):
                            ent["max_minutes"] = self.caps["media_transcript_max_minutes"]
                        if duration_hint and duration_hint > self.caps["media_transcript_max_minutes"] and clarify is None:
                            clarify = {
                                "question": f"ویدیو طولانی‌تر از حد مجاز ({self.caps['media_transcript_max_minutes']} دقیقه) است. فقط {self.caps['media_transcript_max_minutes']} دقیقه اول را انجام بدهم؟",
                                "missing": ["confirm"],
                                "node_index": 0
                            }

            # Audio STT cap
            if intent == "audio.stt":
                if "max_minutes" not in ent:
                    ent["max_minutes"] = self.caps["audio_stt_max_minutes"]

            # NEW: scheduler safety net backfill from hints
            if hints and intent.startswith("sched."):
                for key in ("duration_minutes","datetime","date","time","attendee","booking_ref","date_range","week"):
                    if key in hints and (key not in ent or _is_missing_val(ent.get(key))):
                        ent[key] = hints[key]

            nd2=dict(nd); nd2["entities"]=ent; nd2["bind"]=bind
            new_nodes.append(nd2)
        return new_nodes, clarify

    # ─────────────── V7: pass‑through intent normalizer ───────────────
    def _normalize_intent(self, raw_intent: str, user_text: str = "") -> str:
        """
        Pass-through normalizer:
        - Keep LLM/SR output if it's a known intent.
        - Otherwise, fall back to 'chat.general' to keep the plan executable.
        """
        cand = (raw_intent or "").strip()
        if cand in self.intent_set:
            return cand
        return "chat.general" if "chat.general" in self.intent_set else cand
    # ──────────────────────────────────────────────────────────────────

    # ─────────────── V7: SR-only domain decision ───────────────
    def _node_decide_domain(self, state: RouterState) -> RouterState:
        """
        SR-only domain decision.
        If SR is confident, set a scoping hint (self._current_domain_hint) and return that domain.
        If SR abstains / is not confident, return outward default "core" with NO scoping hint.
        Also records a detailed trace in self._sr_trace["domain"].
        """
        env: MessageEnvelope = state["env"]
        text = self._normalize_fa_ar(env.text or "")

        # reset per turn
        self._domain_from_sr = False
        self._current_domain_hint = None

        # safe defaults for trace fields
        name1: Optional[str] = None
        score1: float = 0.0
        score2: float = 0.0
        margin: float = 0.0
        confident: bool = False

        if self.semantic_enabled and self._sr_domain and text.strip():
            # tolerant score reader (score/similarity/similarity_score)
            def _score(x) -> float:
                try:
                    v = (
                        getattr(x, "score", None)
                        or getattr(x, "similarity", None)
                        or getattr(x, "similarity_score", None)
                        or (x.get("score") if isinstance(x, dict) else None)
                        or (x.get("similarity") if isinstance(x, dict) else None)
                        or (x.get("similarity_score") if isinstance(x, dict) else None)
                    )
                    return float(v or 0.0)
                except Exception:
                    return 0.0

            def _name(x) -> Optional[str]:
                if isinstance(x, str):
                    return x
                if isinstance(x, dict):
                    return x.get("name") or x.get("route") or x.get("route_name")
                return (
                    getattr(x, "name", None)
                    or getattr(x, "route", None)
                    or getattr(x, "route_name", None)
                )

            try:
                picks = self._sr_domain(text=text, limit=2)
                if not isinstance(picks, list):
                    picks = [picks] if picks else []
            except Exception as e:
                if self._debug:
                    print("[DEBUG] SR domain call failed:", repr(e))
                picks = []

            if picks:
                name1 = _name(picks[0])
                score1 = _score(picks[0])
                if len(picks) > 1:
                    score2 = _score(picks[1])
                margin = score1 - score2
                confident = bool(
                    name1
                    and score1 >= float(self.semantic_domain_threshold)
                    and margin >= 0.03
                )

                # record trace regardless of confidence
                self._sr_trace["domain"] = {
                    "picked": name1 if confident else None,
                    "top": name1,
                    "top_score": score1,
                    "second_score": score2,
                    "margin": margin,
                    "confident": confident,
                }

                if confident:
                    self._current_domain_hint = name1
                    self._domain_from_sr = True
                    return {"domain": name1}

        # SR abstained / not confident → outward default only; NO scoping hint
        self._current_domain_hint = None
        self._domain_from_sr = False

        # ensure trace exists even when no picks or error
        if "domain" not in self._sr_trace or not isinstance(self._sr_trace["domain"], dict):
            self._sr_trace["domain"] = {
                "picked": None,
                "top": None,
                "top_score": 0.0,
                "second_score": 0.0,
                "margin": 0.0,
                "confident": False,
            }
        else:
            # update with last computed values (may still be defaults)
            self._sr_trace["domain"].update({
                "picked": None,
                "top": name1,
                "top_score": score1,
                "second_score": score2,
                "margin": margin,
                "confident": False,
            })

        return {"domain": "core"}

    # ────────────────────────────────────────────────────────────────

    def _decide_domain_llm(self, env: MessageEnvelope) -> Dict[str, Any]:
        # (unused in v7 path, retained for completeness)
        norm_text = self._normalize_fa_ar(env.text or "")
        if SystemMessage and HumanMessage:
            msgs = [
                SystemMessage(content=self.classify_system),
                HumanMessage(content=f"Text: {norm_text[:400]}\nMedia: {env.media_type}\nHasURL: {bool(env.urls)}\nGroup: {env.is_group}")
            ]
        else:
            msgs = [self.classify_system, norm_text]
        structured = self.llm.with_structured_output(DecideOut, method="json_schema", strict=True)
        try:
            out: DecideOut = structured.invoke(msgs, config={"callbacks": [UsageCallback(self._meter, "decide_domain")]})
            return out.model_dump()
        except Exception:
            return {"domain":"core","confidence":0.7,"reason":"fallback"}

    def _node_plan_pipeline(self, state: RouterState) -> RouterState:
        env: MessageEnvelope = state["env"]
        sess: SessionState = state["sess"]

        plan_plain = self._core_plan_pipeline(
            env,
            sess.pipeline,
            sess.user_prompts,
            is_direction_change(env.text),
            adapter_hints=state.get("adapter_hints") or {},
            sess=sess  # Patch B: pass session to enable sticky inputs
        )

        nodes_h = self._harmonize_with_hints(plan_plain.get("pipeline", []) or [], state.get("adapter_hints") or {})
        plan_plain["pipeline"] = nodes_h

        now_ts = int(time.time())
        nodes_raw = plan_plain.get("pipeline", []) or [{"task_id":"t1","intent":"chat.general","entities":{},"confidence":0.75,"bind":{}}]
        nodes=[]
        for nd in nodes_raw:
            nd2 = dict(nd)
            nd2.setdefault("bind", {})
            nd2.setdefault("_meta", {})
            nd2["_meta"].update({"origin_turn": sess.turn_no, "created_ts": now_ts})
            canon = self._normalize_intent(nd2.get("intent",""), getattr(env, "text", ""))
            nd2["intent"] = canon
            if canon == "image.generate":
                ent = dict(nd2.get("entities") or {})
                if not ent.get("prompt") and env.text: ent["prompt"] = env.text
                if not ent.get("size"):
                    m = self._guess_size_from_text(env.text or "")
                    if m: ent["size"] = m
                nd2["entities"] = ent
            if canon in ("text.translate","translate.text"):
                ent = dict(nd2.get("entities") or {})
                if not ent.get("target_lang"):
                    lg = self._guess_lang_to(env.text or "")
                    if lg: ent["target_lang"] = lg
                nd2["entities"] = ent
            nodes.append(nd2)
        return {"plan_raw": plan_plain, "nodes_raw": nodes}


    def _parse_time_range(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        # matches 00:00:05 to 00:00:25 or 5s to 25s
        t = (text or "").lower()
        m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?|\d{1,3}s)\s*(?:to|–|-)\s*(\d{1,2}:\d{2}(?::\d{2})?|\d{1,3}s)", t)
        if not m: return (None, None)
        def norm(x):
            if x.endswith("s"):
                s = int(x[:-1])
                return f"00:00:{s:02d}" if s < 60 else f"00:{s//60:02d}:{s%60:02d}"
            if re.match(r"^\d{1,2}:\d{2}$", x):  # mm:ss
                mm, ss = x.split(":")
                return f"00:{int(mm):02d}:{int(ss):02d}"
            if re.match(r"^\d{1,2}:\d{2}:\d{2}$", x):  # hh:mm:ss
                return x
            return None
        return norm(m.group(1)), norm(m.group(2))



    def _canonicalize_entities_for_intent(self, intent: str, entities: Dict[str,Any], user_text: str) -> Dict[str,Any]:
        ent = dict(entities or {})
        if intent in ("finance.crypto.price","finance.crypto.ohlc"):
            if not any(ent.get(k) for k in ("symbol","pair","asset")):
                sym = self._guess_symbol_from_text(user_text or "")
                if sym: ent["symbol"] = sym

        elif intent == "network.vpn.get":
            low = (user_text or "").lower()
            if "purpose" not in ent or _is_missing_val(ent.get("purpose")):
                if re.search(r"\bstream|netflix|نتفلیکس|استریم\b", low): ent["purpose"] = "streaming"
                elif re.search(r"\bgam(e|ing)\b|گیم", low): ent["purpose"] = "gaming"
                elif re.search(r"\bprivacy|secure|امنیت\b", low): ent["purpose"] = "privacy"
            if "region" not in ent or _is_missing_val(ent.get("region")):
                if re.search(r"\b(us|usa|america|آمریکا)\b", low): ent["region"] = "US"
                elif re.search(r"\b(uk|britain|england)\b", low): ent["region"] = "UK"
                elif re.search(r"\b(eu|europe|اروپا)\b", low): ent["region"] = "EU"
                elif re.search(r"\b(de|germany|آلمان)\b", low): ent["region"] = "DE"
                elif re.search(r"\b(nl|netherlands|هلند)\b", low): ent["region"] = "NL"

        elif intent == "core.language.set":
            low = (user_text or "").lower()
            if "language" not in ent or _is_missing_val(ent.get("language")):
                if re.search(r"\b(fa|farsi|persian)\b|فارسی", low): ent["language"] = "fa"
                elif re.search(r"\b(en|english)\b|انگلیسی", low): ent["language"] = "en"

        elif intent == "image.generate":
            if "prompt" not in ent and user_text: ent["prompt"] = user_text
            if "size" not in ent:
                sz = self._guess_size_from_text(user_text or "")
                if sz: ent["size"] = sz

        elif intent == "video.thumbnail":
            if "at" not in ent:
                tc = self._parse_timecode(user_text or "")
                if tc: ent["at"] = tc

        elif intent in ("text.translate","translate.text"):
            if "target_lang" not in ent:
                lg = self._guess_lang_to(user_text or "")
                if lg: ent["target_lang"] = lg

        # NEW: scheduler cancel fallback booking_ref extractor
        elif intent == "sched.cancel":
            if "booking_ref" not in ent or _is_missing_val(ent.get("booking_ref")):
                m = re.search(r"(?:booking|ref|code|کد)\s*([A-Za-z0-9\-]+)", (user_text or "").lower())
                if m: ent["booking_ref"] = m.group(1)
        elif intent == "sched.reschedule":
            if "booking_ref" not in ent or _is_missing_val(ent.get("booking_ref")):
                # accept: booking ABC123 | ref ABC123 | reference ABC123 | code ABC123 | کد ABC123
                m = re.search(r"(?:booking|ref(?:erence)?|code|کد)\s*[:\-#]?\s*([A-Za-z0-9_\-]+)",
                              (user_text or "").lower())
                if m:
                    ent["booking_ref"] = m.group(1)

        # NEW: music canonicalizers
        elif intent == "music.convert":
            low = (user_text or "").lower()
            if "format" not in ent:
                if "mp3" in low: ent["format"] = "mp3"
            if "bitrate" not in ent:
                m = re.search(r"(\d{2,3})\s*kbps", low)
                if m: ent["bitrate"] = f"{m.group(1)}kbps"
        elif intent == "music.preview":
            if "length_seconds" not in ent:
                m = re.search(r"(\d{1,3})\s*s(?:ec|)", (user_text or "").lower())
                if m: ent["length_seconds"] = m.group(1)

        elif intent == "video.thumbnail":
            # keep your existing 'at' inference
            if "at" not in ent:
                tc = self._parse_timecode(user_text or "")
                if tc: ent["at"] = tc

        elif intent == "video.gif":
            if "start" not in ent or "end" not in ent:
                s,e = self._parse_time_range(user_text or "")
                if s and e:
                    ent["start"], ent["end"] = s, e

        return ent

    def _node_merge_and_prepare(self, state: RouterState) -> RouterState:
        env: MessageEnvelope = state["env"]
        sess: SessionState = state["sess"]
        nodes = deepcopy(state["nodes_raw"])
        # Patch B: use sticky shared inputs
        shared = self.collect_shared_inputs(env, sess)

        if not is_direction_change(env.text) and sess.pipeline:
            prev_by_id = {n["task_id"]: n for n in sess.pipeline}
            merged=[]
            for nd in nodes:
                if nd["task_id"] in prev_by_id:
                    prev = prev_by_id[nd["task_id"]]
                    nd["entities"] = {**prev.get("entities",{}), **nd.get("entities",{})}
                    b=dict(prev.get("bind",{})); b.update(nd.get("bind",{})); nd["bind"]=b
                    m=dict(prev.get("_meta",{})); m.update(nd.get("_meta",{})); nd["_meta"]=m
                merged.append(nd)
            nodes = merged

        env_mod = new_turn_modality(env)
        pruned=[]
        for nd in nodes:
            if self.should_prune_node(nd, env_mod, sess.turn_no):
                tmp=dict(nd); tmp["entities"]=dict(tmp.get("entities",{})); tmp["bind"]=dict(tmp.get("bind",{}))
                self.fill_required_alternatives_node(tmp["entities"], tmp["bind"], shared, tmp["intent"])
                miss_tmp = self.missing_for_node(tmp["intent"], tmp["entities"], tmp["bind"])
                if miss_tmp:
                    continue
                nd=tmp
            pruned.append(nd)
        nodes = pruned
        # Prefer shared; backfill from session bag when shared is missing (kept; shared already merged)
        shared_with_bag = dict(shared)
        for k in ("file_id", "url", "provider", "text"):
            if k not in shared_with_bag and (sess.entity_bag or {}).get(k):
                shared_with_bag[k] = sess.entity_bag[k]

        bag_shared = dict(shared)
        for k in ("file_id","url","provider","text"):
            if k not in bag_shared and (sess.entity_bag or {}).get(k):
                bag_shared[k] = sess.entity_bag[k]
        nodes = self.autofill_inputs_pipeline(nodes, bag_shared)
        nodes = self.auto_bind_text(nodes)
        return {"nodes": nodes}

    def _node_apply_policy(self, state: RouterState) -> RouterState:
        env: MessageEnvelope = state["env"]
        sess: SessionState = state["sess"]
        nodes = deepcopy(state["nodes"])
        # Patch B: sticky shared inputs
        shared = self.collect_shared_inputs(env, sess)
        for k in ("file_id","url","provider","text"):
            if k not in shared and (sess.entity_bag or {}).get(k):
                shared[k] = sess.entity_bag[k]
        nodes, policy_clarify = self.apply_phase2_policy(nodes, shared, hints=state.get("adapter_hints"))

        if len(nodes) > 3:
            # If there are cross-task dependencies, do NOT condense — keep the chain intact.
            has_binds = any((nd.get("bind") or {}) for nd in nodes)
            if not has_binds:
                condensed = self._condense_pipeline(env, nodes)

                # Post-condense safety: ensure no bind points to a removed task.
                ids_after = {n.get("task_id") for n in condensed}
                broken = False
                for n in condensed:
                    for _, b in (n.get("bind") or {}).items():
                        if isinstance(b, dict) and b.get("from_task") and b["from_task"] not in ids_after:
                            broken = True
                            break
                    if broken:
                        break

                nodes = condensed if not broken else nodes

            now_ts = int(time.time())
            for nd in nodes:
                nd.setdefault("bind", {})
                nd.setdefault("_meta", {"origin_turn": sess.turn_no, "created_ts": now_ts})


        bag = dict(sess.entity_bag or {})
        bag.setdefault("timezone", self.defaults.get("timezone"))
        bag.setdefault("quality", self.defaults.get("media_quality"))
        bag.setdefault("voice", self.defaults.get("tts_voice"))

        node_statuses: List[NodeStatus] = []
        first_incomplete_idx: Optional[int] = None
        first_ready_idx: Optional[int] = None
        min_conf = 1.0

        for idx, nd in enumerate(nodes):
            ent0 = {**bag, **nd.get("entities", {})}
            ent0 = self._canonicalize_entities_for_intent(nd["intent"], ent0, env.text or "")
            ent = self._filter_entities_for_intent(nd["intent"], ent0)
            # slot-fill (regex → JSON spans) before we check for missing keys
            ent = self._slotfill_for_intent(nd["intent"], ent, env.text or "")

            bind = dict(nd.get("bind", {}))
            if "size" in (self.intent_specs.get(nd["intent"],{}).get("optional",[])) and "size" not in ent and "size" not in bind:
                ent["size"] = self.defaults.get("image_size")

            miss = self.missing_for_node(nd["intent"], ent, bind)
            ready = len(miss) == 0
            c = float(nd.get("confidence",0.8))
            min_conf = min(min_conf, c)
            node_statuses.append(NodeStatus(
                task_id=nd["task_id"], intent=nd["intent"],
                entities=ent, bind=bind, confidence=c,
                missing=miss, ready=ready
            ))
            if not ready and first_incomplete_idx is None:
                first_incomplete_idx = idx
            if ready and first_ready_idx is None:
                first_ready_idx = idx

        clarify = None
        next_node_index = 0
        if first_incomplete_idx is not None:
            cur = node_statuses[first_incomplete_idx]
            clarify = {
                "question": self.friendly_question(cur.intent, cur.missing, cur.entities),
                "missing": cur.missing[:1],
                "node_index": first_incomplete_idx
            }
            next_node_index = first_incomplete_idx
        else:
            next_node_index = first_ready_idx if first_ready_idx is not None else 0

        if policy_clarify and not clarify:
            clarify = policy_clarify
            next_node_index = clarify.get("node_index", 0)

        if node_statuses and all(ns.ready for ns in node_statuses):
            clarify = None
        elif clarify is None and self.clarify_mode == "when_low_conf":
            if min_conf < self.thresholds["low_conf_threshold"]:
                q = self._generate_exploratory_question(state["env"].text or "", [asdict(n) for n in node_statuses])
                clarify = {"question": q, "missing": ["intent"], "node_index": next_node_index}

        parts=[]
        for i, ns in enumerate(node_statuses, start=1):
            ready_flag = "✅" if ns.ready else f"(needs: {', '.join(ns.missing[:1])})"
            parts.append(f"{i}) {ns.intent} {ready_flag}")
        if clarify: parts.append(f"— Q: {clarify['question']}")
        reply_text = "Pipeline:\n" + "\n".join(parts) if parts else "I planned no tasks."

        expected_outputs = {ns.intent: self.intent_outputs.get(ns.intent, {}) for ns in node_statuses}
        ns_dicts = []
        for ns in node_statuses:
            ns_dicts.append({
                "task_id": ns.task_id,
                "intent": ns.intent,
                "entities": ns.entities,
                "bind": ns.bind,
                "confidence": ns.confidence,
                "missing": ns.missing,
                "ready": ns.ready
            })
        return {
            "nodes": nodes,
            "node_statuses": ns_dicts,
            "clarify": clarify,
            "next_node_index": next_node_index,
            "reply_text": reply_text,
            "expected_outputs": expected_outputs,
            "min_conf": min_conf
        }

    def _node_finalize(self, state: RouterState) -> RouterState:
        env: MessageEnvelope = state["env"]
        sess: SessionState = state["sess"]
        sess.last_domain = state.get("domain","core")
        sess.pipeline = []
        for ns, nd in zip(state.get("node_statuses", []), state.get("nodes", [])):
            sess.pipeline.append({
                "task_id": ns["task_id"],
                "intent": ns["intent"],
                "entities": ns["entities"],
                "bind": ns["bind"],
                "confidence": ns["confidence"],
                "_meta": nd.get("_meta", {})
            })
        sess.current_node_idx = state.get("next_node_index", 0)
        if state.get("node_statuses"):
            cur = state["node_statuses"][sess.current_node_idx]
            sess.last_intent = cur["intent"]
            for k in ("text","url","file_id","provider"):
                if k in cur["entities"]:
                    sess.entity_bag[k] = cur["entities"][k]

        # ▶▶ Persist raw env attachment too (even if not used by current node)
        if env.file_id:
            sess.entity_bag["file_id"] = env.file_id
        if env.urls:
            sess.entity_bag["url"] = env.urls[0]
            prov = detect_provider_from_url(env.urls[0])
            if prov:
                sess.entity_bag["provider"] = prov
        # ◀◀

        self._save_session(env.user_id, sess)
        return {"domain": state.get("domain","core")}


    def _load_session(self, session_id: str) -> SessionState:
        return deepcopy(self._sessions.get(session_id) or SessionState())
    def _save_session(self, session_id: str, st: SessionState):
        st.updated_at = int(time.time())
        self._sessions[session_id] = deepcopy(st)

    def should_prune_node(self, node: Dict[str,Any], env_modality: str, current_turn: int) -> bool:
        meta = node.get("_meta", {})
        try:
            origin_turn = int(meta.get("origin_turn", current_turn))
        except Exception:
            origin_turn = current_turn
        age = max(0, current_turn - origin_turn)
        modality = intent_modality(node.get("intent","text"))
        incompatible = (
            (env_modality == "image" and modality in ("text","voice")) or
            (env_modality == "voice" and modality in ("text","image")) or
            (env_modality == "text"  and modality in ("image","voice","url") and age >= 1)
        )
        too_old = age > self.thresholds.get("incomplete_max_age_turns", 2)
        return incompatible or too_old

    def route(self, text: str, session_id: str = "u1", extra_env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # clear previous turn’s SR info
        self._sr_trace = {"domain": {}, "intent": {}, "final_intent": None}
        self._meter.reset()
        self._slot_calls_this_turn = 0

        extra = extra_env or {}
        urls = extra.get("urls") or extract_urls(text or "")
        sanitized_text = sanitize_user_text(text or "")
        env = MessageEnvelope(
            user_id=session_id, chat_id=extra.get("chat_id","c1"),
            text=sanitized_text or "", media_type=extra.get("media_type","none"),
            file_id=extra.get("file_id"), urls=urls, lang_hint=extra.get("lang_hint", self.lang_hint),
            is_group=bool(extra.get("is_group", False))
        )
        sess = self._load_session(session_id)
        sess.turn_no += 1
        if env.text:
            sess.user_prompts.append(env.text)

        init_state: RouterState = {
            "env": env, "sess": sess,
            "settings": self.settings_raw, "routes": self.routes_raw,
            "adapter_hints": extra.get("adapter_hints") or {}
        }
        out: RouterState = self.graph.invoke(init_state)

        node_statuses = out.get("node_statuses", [])
        final_intent = None
        if node_statuses:
            final_intent = node_statuses[out.get("next_node_index", 0)].get("intent")

        return {
            "request_id": str(uuid.uuid4()),
            "domain": out.get("domain","core"),
            "nodes": node_statuses,
            "next_node_index": out.get("next_node_index", 0),
            "clarify": out.get("clarify"),
            "reply_text": out.get("reply_text", ""),
            "expected_outputs": out.get("expected_outputs", {}),
            "usage": self._meter.to_dict(),
            # expose SR decision info
            "sr_debug": {
                "domain": self._sr_trace.get("domain"),
                "intent": self._sr_trace.get("intent"),
                "final_intent": final_intent,
                "matched_sr_top": bool(
                    final_intent and (self._sr_trace.get("intent", {}) or {}).get("top") == final_intent
                ),
            },
        }

    # Utility methods used in prompts
    def _coerce_scalar(self, v: Any) -> Any:
        # Keep same behavior as v5 (coerce strings "true"/"false" and numbers if needed)
        if isinstance(v, str):
            vl = v.strip().lower()
            if vl == "true": return True
            if vl == "false": return False
            try:
                if "." in vl: return float(vl)
                return int(vl)
            except Exception:
                return v
        return v

    def _generate_exploratory_question(self, text: str, node_statuses: List[Dict[str, Any]]) -> str:
        # Minimal fallback question
        return "دقیق‌تر بفرمایید هدفتان چیست؟ سه گزینه: (۱) خلاصه (۲) ترجمه (۳) ادامه بده"




__all__ = [
    "ConfigLLMRouter",
    "intent_modality",
    "new_turn_modality",
    "is_direction_change",
]

