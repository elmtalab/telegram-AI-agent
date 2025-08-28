"""HTTP wrapper around the lightweight router implementation.

The endpoint defined here simply delegates to :class:`apps.router.Router`.  The
class is intentionally tiny but its public interface mirrors the real system so
the rest of the stack can be exercised end‑to‑end without relying on the
monolithic reference implementation.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from apps.router import Router


router = Router()
app = FastAPI()


class RouteRequest(BaseModel):
    """Incoming payload for the routing endpoint."""

    text: str = ""


@app.post("/route")
async def route(req: RouteRequest):
    """Return a router output for the provided text."""

    return router.route(req.text)
