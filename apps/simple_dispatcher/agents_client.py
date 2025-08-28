"""HTTP client to agents."""

def call_agent(name: str, payload: dict) -> dict:
    return {"agent": name, "payload": payload}
