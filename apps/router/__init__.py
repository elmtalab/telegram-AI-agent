"""Routing service.

The real project contains a highly sophisticated router that performs
modality detection, semantic ranking and policy checks.  For the unit tests in
this kata we only require a very small subset of this functionality.  The
:class:`Router` class below acts as a façade around the placeholder modules in
this package such as :mod:`gate` and :mod:`models`.

Keeping the public method :meth:`route` compatible with the production code
allows the HTTP layer and orchestrator to be exercised without pulling in any
heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lib.contracts.router_output import RouterOutput
from lib.config.yaml_loader import load_yaml

from . import gate


@dataclass
class Router:
    """Minimal router facade used in tests.

    Parameters
    ----------
    config: optional mapping loaded from ``router.yaml``.  The values are not
        used directly but are stored to demonstrate how the real component would
        be configured.
    """

    config: Any | None = field(default=None)
    config_path: str = "config/router.yaml"

    def __post_init__(self) -> None:
        """Load configuration from ``config_path`` if no mapping was provided."""

        if self.config is None and Path(self.config_path).exists():
            self.config = load_yaml(self.config_path)

    def route(self, text: str) -> RouterOutput:
        """Return a :class:`RouterOutput` for the given ``text``.

        The gate check is invoked for completeness but its return value is not
        interpreted – the stub gate always allows the request.  A single
        ``chat.general`` route is emitted which is sufficient for the higher
        level components and unit tests.
        """

        gate.gate_request({"text": text})
        # The configuration may define a default list of routes.  If not present
        # we fall back to a single ``chat.general`` entry.
        routes = self.config.get("default_routes", ["chat.general"]) if isinstance(self.config, dict) else ["chat.general"]
        return RouterOutput(routes=routes)


__all__ = ["Router"]

