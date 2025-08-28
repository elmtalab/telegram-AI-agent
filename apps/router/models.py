"""Pydantic models for router configuration."""
from pydantic import BaseModel

class RouterConfig(BaseModel):
    name: str
