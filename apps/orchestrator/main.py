from fastapi import FastAPI

app = FastAPI()

@app.post("/aor")
async def aor():
    """Return AOR."""
    return {"aor": []}
