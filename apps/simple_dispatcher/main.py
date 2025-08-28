from fastapi import FastAPI

app = FastAPI()

@app.post("/enqueue")
async def enqueue():
    """Enqueue a task."""
    return {"status": "queued"}
