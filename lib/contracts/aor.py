"""AOR model."""
from pydantic import BaseModel

class AOR(BaseModel):
    steps: list[str]
