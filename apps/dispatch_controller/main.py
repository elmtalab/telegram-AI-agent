from fastapi import FastAPI

app = FastAPI()

@app.post("/dispatch")
async def dispatch():
    """Return a dispatch decision."""
    return {"decision": "ask"}
