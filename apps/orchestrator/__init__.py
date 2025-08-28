"""Orchestrator service.

The orchestrator receives a plan skeleton from the router and wraps it in an
``AOR`` (Action Oriented Response) object.  Only minimal functionality is
implemented which keeps the module light weight while still demonstrating the
flow through the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lib.contracts.aor import AOR
from lib.config.yaml_loader import load_yaml

from . import builder, idempotency


@dataclass
class Orchestrator:
    """Compile router output into an :class:`~lib.contracts.aor.AOR` instance."""

    router: Any
    config: Any | None = field(default=None)
    config_path: str = "config/orchestrator.yaml"

    def __post_init__(self) -> None:
        if self.config is None and Path(self.config_path).exists():
            self.config = load_yaml(self.config_path)

    def process_user_input(self, session_id: str, text: str) -> AOR:
        """Run the routing stage and build an ``AOR`` structure."""

        plan = self.router.route(text)
        nodes = builder.build_nodes(plan.model_dump())
        ensured = idempotency.ensure_idempotent(nodes)
        return AOR(steps=ensured.get("routes", []))


__all__ = ["Orchestrator"]

