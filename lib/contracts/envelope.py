"""MessageEnvelope model."""
from pydantic import BaseModel

class MessageEnvelope(BaseModel):
    message: str
