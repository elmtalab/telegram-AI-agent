"""MessageEnvelope model used by the chat adapter."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """Representation of an attachment in the envelope."""

    file_id: str | None = None
    mime: str | None = None
    media_type: str | None = None
    size: int | None = None
    sha256: str | None = None


class MessageEnvelope(BaseModel):
    """Normalised message passed from the adapter to the orchestrator."""

    text: str = ""
    urls: List[str] = Field(default_factory=list)
    attachments: List[Attachment] = Field(default_factory=list)
    locale: str = "en-US"
    timezone: str = "UTC"
    session_snapshot: Dict[str, Any] = Field(default_factory=dict)
    adapter_hints: Dict[str, Any] = Field(default_factory=dict)
