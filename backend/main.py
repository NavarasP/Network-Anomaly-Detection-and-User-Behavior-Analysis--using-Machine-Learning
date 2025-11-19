from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import asyncio
import json
from typing import List, Dict, Optional

# -------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------
app = FastAPI(title="Anomaly Detection API")

# Use __file__ to get the correct backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
LOG_FILE = os.path.join(BACKEND_DIR, "network.log")

model = None
scaler = None

RECENT_EVENTS = []
RECENT_MAX = 500


# -------------------------------------------------------
# SCHEMA
# -------------------------------------------------------
class Features(BaseModel):
    features: List[float]


# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
def load_models():
    global model, scaler

    scaler_path = os.path.join(MODEL_DIR, "scaler_nslkdd.joblib")
    model_path = os.path.join(MODEL_DIR, "rf_encoded.joblib")

    if not os.path.exists(scaler_path):
        print("❌ scaler_nslkdd.joblib missing")
        return

    if not os.path.exists(model_path):
        print("❌ rf_encoded.joblib missing")
        return

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    print("✅ RandomForest + Scaler Loaded (NO TensorFlow)")


@app.on_event("startup")
def startup():
    load_models()


# -------------------------------------------------------
# CORS
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# -------------------------------------------------------
# /score ENDPOINT (RF ONLY)
# -------------------------------------------------------
@app.post("/score")
def score(features: Features):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    x = np.array(features.features).reshape(1, -1)

    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling failed: {e}")

    try:
        # Adapt feature size if model expects a different number of features
        expected = getattr(model, "n_features_in_", x_scaled.shape[1])
        if x_scaled.shape[1] != expected:
            if x_scaled.shape[1] > expected:
                x_input = x_scaled[:, :expected]
            else:
                # pad zeros to the right
                pad = np.zeros((x_scaled.shape[0], expected - x_scaled.shape[1]))
                x_input = np.hstack([x_scaled, pad])
        else:
            x_input = x_scaled

        prob = float(model.predict_proba(x_input)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"score": prob}


# -------------------------------------------------------
# LOG PARSER (timestamp, attack_type, 41 KDD features)
# -------------------------------------------------------
def parse_log(line: str) -> Optional[Dict]:
    """Parse log line: timestamp,attack_type,<41 KDD features>"""
    try:
        parts = line.strip().split(',')
        if len(parts) < 43:  # timestamp + attack_type + 41 features
            return None
        
        return {
            "timestamp": parts[0],
            "attack_type": parts[1],
            "features": [float(x) if x.replace('.','',1).replace('-','',1).isdigit() else 0.0 for x in parts[2:43]]
        }
    except Exception:
        return None


# -------------------------------------------------------
# FEATURE BUILDER (41 KDD features already in log)
# -------------------------------------------------------
def build_features(log: Dict) -> List[float]:
    return log["features"]  # Already have all 41 KDD features


# -------------------------------------------------------
# SCORING FOR STREAMING
# -------------------------------------------------------
def score_log(log: Dict) -> float:
    x = np.array(build_features(log)).reshape(1, -1)
    x_scaled = scaler.transform(x)
    expected = getattr(model, "n_features_in_", x_scaled.shape[1])
    if x_scaled.shape[1] != expected:
        if x_scaled.shape[1] > expected:
            x_input = x_scaled[:, :expected]
        else:
            pad = np.zeros((x_scaled.shape[0], expected - x_scaled.shape[1]))
            x_input = np.hstack([x_scaled, pad])
    else:
        x_input = x_scaled
    return float(model.predict_proba(x_input)[:, 1][0])


# -------------------------------------------------------
# SSE STREAM ENDPOINT
# -------------------------------------------------------
async def tail_log(interval: float = 0.5):
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "a").close()

    with open(LOG_FILE, "r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                parsed = parse_log(line)
                if parsed:
                    score = score_log(parsed)
                    severity = "Critical" if score > 0.7 else "Warning" if score > 0.4 else "Normal"
                    
                    # Extract some key features for display
                    feats = parsed["features"]
                    src_bytes = feats[4] if len(feats) > 4 else 0
                    dst_bytes = feats[5] if len(feats) > 5 else 0

                    event = {
                        "timestamp": parsed["timestamp"],
                        "attack_type": parsed["attack_type"],
                        "user_id": parsed["attack_type"],  # Use attack_type as identifier
                        "event": f"Traffic: {parsed['attack_type']}",
                        "src_ip": f"src_bytes={int(src_bytes)}",
                        "score": score,
                        "severity": severity,
                        "raw_log": line.strip(),
                        "top_features": [],  # Can add feature importance later
                        "user_behavior": [f"Attack type: {parsed['attack_type']}"] if parsed['attack_type'] != 'normal' else []
                    }

                    RECENT_EVENTS.insert(0, event)
                    if len(RECENT_EVENTS) > RECENT_MAX:
                        RECENT_EVENTS.pop()

                    yield f"data: {json.dumps(event)}\n\n"
            else:
                await asyncio.sleep(interval)


@app.get("/stream")
async def stream(interval: float = 0.5):
    headers = {"Content-Type": "text/event-stream"}
    return StreamingResponse(tail_log(interval), headers=headers)


# -------------------------------------------------------
# RECENT LOGS
# -------------------------------------------------------
@app.get("/logs")
def logs(limit: int = 50):
    return {"events": RECENT_EVENTS[:limit], "count": len(RECENT_EVENTS)}
