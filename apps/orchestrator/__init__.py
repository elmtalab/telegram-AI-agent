"""Orchestrator service.

This module provides :class:`AOROrchestrator` which compiles router output
into an Action Oriented Response (AOR) structure.  The orchestrator reads its
runtime configuration from ``config/orchestrator.yaml`` allowing the behaviour
to be adjusted without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from lib.config.yaml_loader import load_yaml
from lib.utils.helpers import _utcnow_iso

# Re-export the idempotency helper module for backwards compatibility.
from . import idempotency  # noqa: F401


DEFAULT_SCHEMA_VERSION = "aor.v1"


@dataclass
class AOROrchestrator:
    """Compile router output into a full AOR dictionary."""

    router: Any
    config: Dict[str, Any] | None = field(default=None)
    config_path: str = "config/orchestrator.yaml"
    schema_version: str = field(init=False, default=DEFAULT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if self.config is None and Path(self.config_path).exists():
            self.config = load_yaml(self.config_path)
        if isinstance(self.config, dict):
            ver = (
                self.config.get("orchestrator", {})
                .get("aor", {})
                .get("schema", {})
                .get("version")
            )
            if ver:
                self.schema_version = f"aor.v{ver}"

    @staticmethod
    def _nodes_to_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for ns in nodes or []:
            task_id = ns.get("task_id")
            for dest, src in (ns.get("bind") or {}).items():
                if isinstance(src, dict) and src.get("from_task"):
                    edges.append(
                        {
                            "from": src["from_task"],
                            "output": src.get("key", "text"),
                            "to": task_id,
                            "input": dest,
                        }
                    )
        return edges

    def _build_aor(
        self,
        session_id: str,
        text: str,
        attachment: Optional[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        nodes = plan.get("nodes", []) or []
        awaiting_input = plan.get("clarify") is not None
        aor_nodes: List[Dict[str, Any]] = []
        for ns in nodes:
            aor_nodes.append(
                {
                    "task_id": ns.get("task_id"),
                    "intent": ns.get("intent"),
                    "input_declared": dict(ns.get("entities", {}) or {}),
                    "bind": dict(ns.get("bind", {}) or {}),
                    "ready": bool(ns.get("ready", False)),
                    "missing": list(ns.get("missing", []) or []),
                    "confidence": float(ns.get("confidence", 0.0)),
                }
            )
        edges = self._nodes_to_edges(nodes)

        aor: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "request": {
                "request_id": plan.get("request_id"),
                "session_id": session_id,
                "submitted_at": _utcnow_iso(),
                "input": {"text": text, "attachment": (attachment or None)},
                "metadata": plan.get("metadata", {}),
            },
            "status": "waiting_input" if awaiting_input else "planned",
            "final": False,
            "plan": {
                "plan_id": str(uuid.uuid4()),
                "domain": plan.get("domain", "core"),
                "nodes": aor_nodes,
                "edges": edges,
                "router_reply_text": plan.get("reply_text", ""),
                "expected_outputs": plan.get("expected_outputs", {}),
                "router_usage": plan.get("usage", {}),
                "sr_debug": plan.get("sr_debug"),
            },
            "execution": {
                "progress": 0.0,
                "steps": [],
                "outputs_keys": {},
                "final_output": None,
            },
            "usage": plan.get("usage", {}),
            "errors": [],
        }

        if awaiting_input:
            q = plan.get("clarify", {}).get("question")
            miss = plan.get("clarify", {}).get("missing", [])
            aor["next_action"] = {
                "type": "ask_user",
                "awaiting_key": (miss[0] if miss else None),
                "question": q,
            }
        else:
            aor["next_action"] = {
                "type": "enqueue",
                "reason": "plan_ready_no_execution",
            }

        return aor

    def process_user_input(
        self,
        user_id: str,
        text: str,
        attachment: Optional[Dict[str, Any]] = None,
        extra_env: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        extra_env = dict(extra_env or {})
        if attachment and attachment.get("file_id"):
            extra_env["file_id"] = attachment["file_id"]
            extra_env["media_type"] = attachment.get("media_type", "document")
        try:
            plan = self.router.route(
                text or "",
                session_id=user_id,
                extra_env=extra_env,
            )
        except TypeError:  # Fallback to simple router used in tests
            plan = self.router.route(text or "")
        if hasattr(plan, "model_dump"):
            plan = plan.model_dump()
        if "adapter_hints" in extra_env:
            plan.setdefault("metadata", {})["adapter_hints"] = extra_env.get(
                "adapter_hints"
            )
        return self._build_aor(user_id, text, attachment, plan)


# Backwards compatible alias
Orchestrator = AOROrchestrator


__all__ = ["AOROrchestrator", "Orchestrator", "idempotency"]

