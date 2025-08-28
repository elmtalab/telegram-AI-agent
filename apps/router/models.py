"""Pydantic models and JSON schemas used by the router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, TYPE_CHECKING, Literal

try:  # Pydantic v2 provides ``ConfigDict``
    from pydantic import BaseModel, Field, ConfigDict
    _HAS_CONFIGDICT = True
except Exception:  # pragma: no cover - fall back to v1 style
    from pydantic import BaseModel, Field  # type: ignore
    _HAS_CONFIGDICT = False

from lib.contracts.envelope import MessageEnvelope


class RouterConfig(BaseModel):
    name: str


class _BaseStrictModel(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - for pydantic v1
        class Config:
            extra = "forbid"


class KV(_BaseStrictModel):
    key: str
    value: str = ""


class BindRef(_BaseStrictModel):
    dest: str
    from_task: str
    key: Optional[str] = "text"


class PipelineItem(_BaseStrictModel):
    task_id: str
    intent: str
    entities: List[KV] = Field(default_factory=list)
    bind: List[BindRef] = Field(default_factory=list)
    confidence: float = 0.7


class Clarify(_BaseStrictModel):
    question: str
    missing: List[str] = Field(default_factory=list)
    node_index: int = 0


class PlanOut(_BaseStrictModel):
    pipeline: List[PipelineItem]
    clarify: Optional[Clarify] = None


class DecideOut(_BaseStrictModel):
    domain: Literal["core", "media", "images", "tutor", "scheduler", "utilities", "other"]
    confidence: float
    reason: Optional[str] = None


class SlotSpan(_BaseStrictModel):
    key: str               # one of the allowed field names for the route
    text: str              # exact substring
    start: int             # inclusive
    end: int               # exclusive
    normalized: Optional[str] = None  # optional post-normalization


class SlotPack(_BaseStrictModel):
    items: List[SlotSpan] = Field(default_factory=list)


if TYPE_CHECKING:  # to avoid circular import with config_llm_router.SessionState
    from apps.router.config_llm_router import SessionState


class RouterState(TypedDict, total=False):
    env: MessageEnvelope
    sess: "SessionState"
    settings: Dict[str, Any]
    routes: Dict[str, Any]
    domain: str
    plan_raw: Dict[str, Any]
    nodes_raw: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    node_statuses: List[Dict[str, Any]]
    clarify: Optional[Dict[str, Any]]
    next_node_index: int
    reply_text: str
    expected_outputs: Dict[str, Any]
    min_conf: float
    error: Optional[str]
    adapter_hints: Optional[Dict[str, Any]]

