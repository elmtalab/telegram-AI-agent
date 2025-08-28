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
from typing import Any

from lib.contracts.envelope import MessageEnvelope
from lib.config.yaml_loader import load_yaml

from .service import normalize_message


@dataclass
class ChatAdapter:
    """Light‑weight chat adapter used in the examples.

    Parameters
    ----------
    orchestrator:
        Instance of :class:`apps.orchestrator.Orchestrator` that turns normalised
        envelopes into AOR structures.  The object is kept deliberately small so
        the focus of the exercises can remain on the wiring rather than complex
        business logic.
    """

    orchestrator: Any
    config: Any | None = field(default=None)
    config_path: str = "config/chat_adapter.yaml"

    def __post_init__(self) -> None:
        if self.config is None and Path(self.config_path).exists():
            self.config = load_yaml(self.config_path)

    def handle_user_update(self, chat_id: str, user_id: str, *, text: str = "") -> Any:
        """Normalise the message and forward it to the orchestrator.

        The return value is whatever structure the orchestrator produces.  In
        the real project this would include dispatch decisions and telemetry
        data.  For the purposes of the unit tests we simply return the
        orchestrator result directly.
        """

        envelope_dict = normalize_message({"text": text})
        envelope = MessageEnvelope(message=envelope_dict.get("text", ""))
        return self.orchestrator.process_user_input(chat_id, envelope.message)


__all__ = ["ChatAdapter"]

