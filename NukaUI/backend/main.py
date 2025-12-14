from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

state = {
    "active": False,
    "current_reps": 0,
    "target_reps": 10,
}

@app.get("/status")
def get_status():
    return state

@app.post("/rep")
def add_rep():
    if state["active"]:
        state["current_reps"] += 1
    return {"ok": True}

@app.post("/start")
def start_alarm():
    state["active"] = True
    state["current_reps"] = 0
    return {"ok": True}

@app.post("/stop")
def stop_alarm():
    state["active"] = False
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_json(state)
        await asyncio.sleep(0.5)
