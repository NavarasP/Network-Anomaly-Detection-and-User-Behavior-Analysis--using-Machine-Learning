from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List, Optional
from datetime import datetime
import asyncio
import json
from fastapi.responses import StreamingResponse

app = FastAPI(title="Anomaly Detection API")

MODEL_DIR_CANDIDATES = [
    os.path.join(os.getcwd(), "models"),
    os.path.join(os.getcwd(), "backend", "models"),
    "/data/models",
]

model = None
scaler_net = None
scaler_user = None
pca = None

# In-memory ring buffer for recent events
RECENT_EVENTS: list = []
RECENT_EVENTS_MAX = 500

class Features(BaseModel):
    network: List[float]
    user: List[float]


def _find_and_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_models():
    global model, scaler_net, scaler_user, pca
    for base in MODEL_DIR_CANDIDATES:
        try:
            if not os.path.exists(base):
                continue
            scaler_net = _find_and_load(os.path.join(base, "scaler_net.joblib"))
            scaler_user = _find_and_load(os.path.join(base, "scaler_user.joblib"))
            pca = _find_and_load(os.path.join(base, "pca_net.joblib"))
            # rf model saved as rf_model.joblib in notebook
            model = _find_and_load(os.path.join(base, "rf_model.joblib")) or _find_and_load(os.path.join(base, "rf_ueba_net.joblib"))
            # if anything found, break
            if scaler_net is not None and scaler_user is not None and model is not None:
                print("Loaded models from", base)
                return
        except Exception:
            continue


@app.on_event("startup")
def startup_event():
    load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score")
def score(features: Features):
    if model is None or scaler_net is None or scaler_user is None:
        raise HTTPException(status_code=503, detail="Model or scalers not available on server")

    # Validate incoming feature vector lengths when scalers are available
    if scaler_net is not None:
        try:
            expected_net = scaler_net.scale_.shape[0]
            if len(features.network) != expected_net:
                raise HTTPException(status_code=400, detail=f"Expected network vector length {expected_net}, got {len(features.network)}")
        except AttributeError:
            pass

    if scaler_user is not None:
        try:
            expected_user = scaler_user.scale_.shape[0]
            if len(features.user) != expected_user:
                raise HTTPException(status_code=400, detail=f"Expected user vector length {expected_user}, got {len(features.user)}")
        except AttributeError:
            pass

    net = np.array(features.network).reshape(1, -1)
    user = np.array(features.user).reshape(1, -1)

    try:
        net_scaled = scaler_net.transform(net)
        user_scaled = scaler_user.transform(user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling failed: {e}")

    if pca is not None:
        net_enc = pca.transform(net_scaled)
    else:
        net_enc = net_scaled

    x = np.hstack([net_enc, user_scaled])

    try:
        prob = model.predict_proba(x)[:, 1]
        score = float(prob[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"score": score}


def _gen_event():
    """Synchronous helper to generate a single mock event dict."""
    event_types = ["Normal Activity", "Suspicious Login Detected", "Anomaly in Network Traffic"]
    weights = [0.8, 0.15, 0.05]
    r = np.random.rand()
    acc = 0.0
    chosen = event_types[0]
    for i, et in enumerate(event_types):
        acc += weights[i]
        if r <= acc:
            chosen = et
            break

    severity = "Critical" if "Anomaly" in chosen else "Warning" if "Suspicious" in chosen else "Normal"
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event": chosen,
        "ip": f"192.168.1.{np.random.randint(2,255)}",
        "severity": severity,
    }


def _push_event(ev: dict):
    """Push event into in-memory ring buffer."""
    RECENT_EVENTS.insert(0, ev)
    if len(RECENT_EVENTS) > RECENT_EVENTS_MAX:
        RECENT_EVENTS.pop()


async def event_generator(interval: float = 1.0):
    """Async generator that yields server-sent events indefinitely.

    It uses the same event distribution as the frontend simulator. Each yielded chunk
    is a text/event-stream formatted message with a JSON payload under the `data:` field.
    """
    try:
        while True:
            ev = _gen_event()
            _push_event(ev)
            payload = json.dumps(ev)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # client disconnected
        return


@app.get("/stream")
async def stream(interval: float = 1.0):
    """SSE endpoint that streams live mock logs as JSON via Server-Sent Events.

    Query params:
    - interval: float seconds between messages (default 1.0)
    """
    headers = {"Cache-Control": "no-cache", "Content-Type": "text/event-stream"}
    return StreamingResponse(event_generator(interval=interval), headers=headers)


@app.get("/logs")
def get_logs(limit: int = 50):
    """Return recent events from the in-memory buffer. """
    return {"count": len(RECENT_EVENTS), "events": RECENT_EVENTS[:limit]}
