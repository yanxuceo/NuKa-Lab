from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import requests

AGENT_BASE = "http://127.0.0.1:8008"
MOTION_MJPEG = "http://127.0.0.1:8010/mjpeg"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Agent proxy (READ ONLY)
# ----------------------

def agent_status():
    r = requests.get(f"{AGENT_BASE}/status", timeout=1.0)
    r.raise_for_status()
    return r.json()

@app.get("/status")
def status():
    return agent_status()

# ----------------------
# WebSocket → UI
# ----------------------

clients: set[WebSocket] = set()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    try:
        # send snapshot immediately
        await ws.send_text(json.dumps(agent_status()))

        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)

@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_loop())

async def broadcast_loop():
    last = None
    while True:
        await asyncio.sleep(0.2)
        try:
            cur = agent_status()
        except Exception:
            cur = {"active": False, "error": "agent_offline"}

        if cur != last:
            last = cur
            msg = json.dumps(cur)
            dead = []
            for ws in clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for d in dead:
                clients.discard(d)


@app.get("/mjpeg")
def mjpeg():
    """
    Proxy MJPEG stream from nukamotion (8010) to UI.
    """
    try:
        r = requests.get(
            MOTION_MJPEG,
            stream=True,
            timeout=(3.0, 3600.0),
        )
        r.raise_for_status()
    except Exception as e:
        return {"error": "motion_stream_unavailable", "detail": str(e)}

    def gen():
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            r.close()

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
