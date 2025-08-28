from fastapi import FastAPI

app = FastAPI()

@app.post("/route")
async def route():
    """Return a router output."""
    return {"routes": []}
