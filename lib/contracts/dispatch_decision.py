"""DispatchDecision model."""
from pydantic import BaseModel

class DispatchDecision(BaseModel):
    decision: str
