"""HTTP entry point for the chat adapter.

The adapter glues together the router and orchestrator to form the public
interface used by clients.  This version uses the light‑weight stand‑ins from
the :mod:`apps` packages rather than the monolithic prototype.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from apps.chat_adapter import ChatAdapter
from apps.orchestrator import AOROrchestrator
from apps.router import Router


router = Router()
orchestrator = AOROrchestrator(router)
adapter = ChatAdapter(orchestrator, router)
app = FastAPI()


class IngestRequest(BaseModel):
    text: str = ""


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Ingest a message envelope and return the resulting AOR."""

    return adapter.handle_user_update("chat", "user", text=req.text)
