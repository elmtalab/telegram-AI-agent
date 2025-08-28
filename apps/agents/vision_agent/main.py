from fastapi import FastAPI

app = FastAPI()

@app.post("/vision")
async def vision_handler():
    return {"result": "ok"}
