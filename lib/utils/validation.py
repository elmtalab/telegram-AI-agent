"""Validation helpers."""

def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
