"""Chat adapter service.

This module exposes a tiny :class:`ChatAdapter` class that mirrors the public
API of the much larger production component.  The adapter normalises incoming
messages and delegates to the orchestrator which in turn talks to the router.

Only a very small subset of the real behaviour is implemented – just enough
for the tests in this kata.  The structure however matches the architecture
described in the configuration files allowing the stand‑ins to be replaced by
fully fledged implementations without changing the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from lib.contracts.envelope import MessageEnvelope
from lib.config.chat_adapter_loader import AdapterConfig, load_chat_adapter_config

from .service import ChatMemory, normalize_message


@dataclass
class ChatAdapter:
    """Light‑weight chat adapter used in the examples.

    Parameters
    ----------
    orchestrator:
        Instance of :class:`apps.orchestrator.Orchestrator` that turns
        normalised envelopes into AOR structures.  The object is kept
        deliberately small so the focus of the exercises can remain on the
        wiring rather than complex business logic.
    """

    orchestrator: Any
    config: AdapterConfig | None = field(default=None)
    config_path: str = "config/chat_adapter.yaml"

    def __post_init__(self) -> None:
        if self.config is None and Path(self.config_path).exists():
            self.config = load_chat_adapter_config(self.config_path)
        self._memory: Dict[str, ChatMemory] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _mem(self, chat_id: str) -> ChatMemory:
        mem = self._memory.get(chat_id)
        if mem is None:
            mem = self._memory[chat_id] = ChatMemory()
        ttl = int(self.config.sticky.get("ttl_turns", 0)) if self.config else 0
        mem.decay(ttl)
        return mem

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_user_update(
        self,
        chat_id: str,
        user_id: str,
        *,
        text: str = "",
        attachments: List[Dict[str, Any]] | None = None,
        locale: str | None = None,
        timezone: str | None = None,
    ) -> Any:
        """Normalise the message and forward it to the orchestrator.

        The return value is whatever structure the orchestrator produces.  In
        the real project this would include dispatch decisions and telemetry
        data.  For the purposes of the unit tests we simply return the
        orchestrator result directly.
        """

        mem = self._mem(chat_id)
        msg = {
            "text": text,
            "attachments": attachments or [],
            "locale": locale,
            "timezone": timezone,
        }
        envelope_dict = normalize_message(msg, self.config, mem)
        envelope = MessageEnvelope(**envelope_dict)
        return self.orchestrator.process_user_input(chat_id, envelope.text)


__all__ = ["ChatAdapter"]
