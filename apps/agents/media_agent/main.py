from fastapi import FastAPI

app = FastAPI()

@app.post("/media")
async def media_handler():
    """Handle media requests."""
    return {"result": "ok"}
