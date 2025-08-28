from fastapi import FastAPI

app = FastAPI()

@app.post("/ingest")
async def ingest():
    """Ingest a message envelope."""
    return {"status": "ok"}
