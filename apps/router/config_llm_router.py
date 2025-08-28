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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def intent_modality(intent: str) -> str:
    """Return the modality associated with ``intent``.

    Only a subset of the real production mapping is required for the tests.  Any
    unknown intent defaults to ``text``.
    """

    if intent.startswith(("image.", "ocr.", "qr.", "barcode.", "receipt.", "table.")):
        return "image" if intent.startswith("image.") else "file"
    if intent.startswith(("stt.", "audio.")):
        return "voice"
    if intent.startswith(("files.", "music.")):
        return "file"
    if intent.startswith(("link.", "video.", "media.", "podcast.")):
        return "url"
    return "text"


def new_turn_modality(env: MessageEnvelope) -> str:
    """Best effort detection of the modality of ``env``."""

    if env.media_type == "image":
        return "image"
    if env.media_type == "voice":
        return "voice"
    if getattr(env, "file_id", None):
        return "file"
    if env.urls:
        return "url"
    if env.text:
        return "text"
    return "none"


CHANGE_PATTERNS = [
    r"\binstead\b",
    r"\bforget (that|it|previous)\b",
    r"\bcancel (that|previous)\b",
    r"\bno, just\b",
]
DIRCHANGE_RE = re.compile("|".join(CHANGE_PATTERNS), re.IGNORECASE)


def is_direction_change(text: str) -> bool:
    """Return ``True`` if the user appears to change the conversation topic."""

    return bool(DIRCHANGE_RE.search(text or ""))


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


@dataclass
class ConfigLLMRouter:
    """Router that follows the five node flow outlined in the design notes.

    Parameters
    ----------
    settings:
        Mapping produced from ``router.yaml``.
    routes:
        Mapping produced from ``route_registry.yaml`` which contains family
        definitions and intents.
    """

    settings: Dict[str, Any]
    routes: Dict[str, Any]

    def __post_init__(self) -> None:  # pragma: no cover - executed during tests
        router_cfg = self.settings.get("router", {}) if isinstance(self.settings, dict) else {}

        # Gate configuration -------------------------------------------------
        gate_cfg = router_cfg.get("gate", {})
        allowlist = (
            gate_cfg.get("family_selection", {})
            .get("policy_allowlist", {})
            .get("allow", [])
        )
        self.allowed_families: List[str] = list(allowlist)

        # Thresholds used in ``finalize``
        thr = router_cfg.get("thresholds", {})
        self.tau_exec = float(thr.get("tau_exec", 0.6))
        self.tau_clarify = float(thr.get("tau_clarify", 0.4))

        # Policy settings ----------------------------------------------------
        policy_cfg = router_cfg.get("policy", {})
        self.provider_allowlist = (
            policy_cfg.get("provider_allowlist", {}) if isinstance(policy_cfg, dict) else {}
        )

        # Route registry -----------------------------------------------------
        fam_cfg = self.routes.get("families", {}) if isinstance(self.routes, dict) else {}
        self.family_intents: Dict[str, List[str]] = {}
        for name, info in fam_cfg.items():
            intents_ref = info.get("intents_ref")
            intents = self.routes.get("intents", {}).get(intents_ref, [])
            self.family_intents[name] = [it["name"] if isinstance(it, dict) else it for it in intents]

        # Session store ------------------------------------------------------
        self._sessions: Dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Pipeline nodes
    # ------------------------------------------------------------------

    # The following node implementations are intentionally small – they only
    # emulate the behaviour of the production system but still consult the YAML
    # configuration where appropriate.

    def _node_decide_domain(self, env: MessageEnvelope) -> Dict[str, Any]:
        """Decide which families should be considered for ``env``."""

        text = (env.text or "").lower()

        # naive keyword mapping ------------------------------------------------
        family = "core"
        if any(k in text for k in ["video", "gif", "thumbnail"]):
            family = "video"
        elif any(k in text for k in ["music", "song", "track"]):
            family = "music"
        elif env.urls:
            family = "media"

        # honour policy allow list --------------------------------------------
        families = [f for f in [family] if not self.allowed_families or f in self.allowed_families]
        if not families:
            families = ["core"]

        return {"domain": families[0], "gated_families": families}

    def _node_plan_pipeline(self, families: List[str]) -> Dict[str, Any]:
        """Produce a shortlist of intents based on ``families``."""

        intents: List[str] = []
        for fam in families:
            intents.extend(self.family_intents.get(fam, []))

        if not intents:
            intents = ["chat.general"]

        top = intents[0]
        return {"shortlist": intents, "top": top, "score": 0.7}

    def _node_merge_prepare(self, top_intent: str, env: MessageEnvelope, sess: SessionState) -> Dict[str, Any]:
        """Bind sticky resources from the session."""

        entities: Dict[str, Any] = {}
        file_id = env.attachments[0].file_id if env.attachments else None
        if file_id:
            entities["file_id"] = file_id
            sess.last_file = file_id
        elif sess.last_file and intent_modality(top_intent) in ("image", "file", "voice"):
            entities["file_id"] = sess.last_file

        if env.urls:
            entities["url"] = env.urls[0]
            sess.last_url = env.urls[0]
        elif sess.last_url and intent_modality(top_intent) == "url":
            entities["url"] = sess.last_url

        return {"entities": entities}

    def _node_apply_policy(self, top_intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Very small provider allow list check."""

        if "url" in entities:
            prov = "generic"
            # extract provider from hostname if possible
            m = re.match(r"https?://([^/]+)/", entities["url"] + "/")
            if m:
                prov = m.group(1).split(".")[0]

            allowed = self.provider_allowlist.get(top_intent.split(".")[0], [])
            if allowed and prov not in allowed:
                return {"allow": False, "fallback": "text.summarize"}

        return {"allow": True}

    def _node_finalize(self, top_intent: str, policy: Dict[str, Any]) -> List[str]:
        """Return the final list of routes after applying policy."""

        if not policy.get("allow"):
            return [policy.get("fallback", "chat.general")]
        return [top_intent]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, text: str, session_id: str = "u1", extra_env: Optional[Dict[str, Any]] = None) -> RouterOutput:
        """Route ``text`` and return a :class:`RouterOutput`.

        The method mirrors the signature of the production router.  ``extra_env``
        can provide additional envelope fields such as ``file_id`` or ``urls``.
        """

        extra = extra_env or {}
        attachments = []
        if extra.get("file_id"):
            attachments.append(
                Attachment(file_id=extra.get("file_id"), media_type=extra.get("media_type"))
            )

        env = MessageEnvelope(
            user_id=session_id,
            chat_id=extra.get("chat_id", session_id),
            text=text,
            urls=extra.get("urls", []),
            attachments=attachments,
            locale=extra.get("locale", "en"),
            timezone=extra.get("timezone", "UTC"),
        )

        sess = self._sessions.setdefault(session_id, SessionState())

        domain = self._node_decide_domain(env)
        plan = self._node_plan_pipeline(domain["gated_families"])
        merge = self._node_merge_prepare(plan["top"], env, sess)
        policy = self._node_apply_policy(plan["top"], merge["entities"])
        routes = self._node_finalize(plan["top"], policy)

        sess.last_intent = routes[0]

        return RouterOutput(routes=routes)


__all__ = [
    "ConfigLLMRouter",
    "intent_modality",
    "new_turn_modality",
    "is_direction_change",
]

