"""Expose the orchestrator over HTTP.

The implementation delegates to the light‑weight :class:`apps.orchestrator.Orchestrator`
which itself relies on :class:`apps.router.Router`.  Both classes are minimal
but mirror the public API of the production services.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from apps.orchestrator import Orchestrator
from apps.router import Router


router = Router()
orchestrator = Orchestrator(router)
app = FastAPI()


class AORRequest(BaseModel):
    text: str = ""


@app.post("/aor")
async def aor(req: AORRequest):
    """Return an AOR structure for the supplied text."""

    return orchestrator.process_user_input("session", req.text)
