"""RouterOutput model."""
from pydantic import BaseModel

class RouterOutput(BaseModel):
    routes: list[str]
