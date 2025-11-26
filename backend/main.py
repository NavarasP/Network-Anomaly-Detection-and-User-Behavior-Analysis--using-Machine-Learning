from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import numpy as np
import onnxruntime as ort
import os
import asyncio
import json
from typing import List, Dict, Optional

# -------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------
app = FastAPI(title="Anomaly Detection API")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
LOG_FILE = os.path.join(BACKEND_DIR, "network.log")

scaler = None
rf_model = None
encoder_session = None

RECENT_EVENTS = []
RECENT_MAX = 500

# -------------------------------------------------------
# API SCHEMA
# -------------------------------------------------------
class Features(BaseModel):
    features: List[float]


# -------------------------------------------------------
# LOAD MODELS (Scaler, ONNX Encoder, RF)
# -------------------------------------------------------
def load_models():
    global scaler, rf_model, encoder_session

    scaler_path = os.path.join(MODEL_DIR, "scaler_nslkdd.joblib")
    encoder_path = os.path.join(MODEL_DIR, "stacked_encoder.onnx")
    rf_path = os.path.join(MODEL_DIR, "rf_encoded.joblib")

    # Validate files
    if not os.path.exists(scaler_path):
        print("❌ Missing scaler_nslkdd.joblib")
        return
    if not os.path.exists(encoder_path):
        print("❌ Missing stacked_encoder.onnx")
        return
    if not os.path.exists(rf_path):
        print("❌ Missing rf_encoded.joblib")
        return

    # Load models
    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    encoder_session = ort.InferenceSession(encoder_path)

    print("✅ Loaded: Scaler + ONNX Encoder + RF Model (NO TensorFlow)")


@app.on_event("startup")
def startup_event():
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
    return {
        "status": "ok",
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder_session is not None,
        "model_loaded": rf_model is not None,
    }


# -------------------------------------------------------
# ONNX ENCODER WRAPPER
# -------------------------------------------------------
def encode_with_onnx(x_scaled: np.ndarray):
    """Run NDAE encoder ONNX model"""
    global encoder_session

    input_name = encoder_session.get_inputs()[0].name
    output_name = encoder_session.get_outputs()[0].name

    result = encoder_session.run(
        [output_name],
        {input_name: x_scaled.astype(np.float32)}
    )
    return result[0]


# -------------------------------------------------------
# SCORE ENDPOINT
# -------------------------------------------------------
@app.post("/score")
def score(features: Features):
    if scaler is None or rf_model is None or encoder_session is None:
        raise HTTPException(status_code=503, detail="Model not fully loaded")

    x = np.array(features.features).reshape(1, -1)

    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling failed: {e}")

    # Encode using ONNX NDAE encoder
    code = encode_with_onnx(x_scaled)

    try:
        prob = float(rf_model.predict_proba(code)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"score": prob}


# -------------------------------------------------------
# LOG PARSER (timestamp, type, 41 features)
# -------------------------------------------------------
def parse_log(line: str) -> Optional[Dict]:
    """Expected: timestamp,attack_type,<41 KDD features>"""
    try:
        parts = line.strip().split(',')
        if len(parts) < 43:
            return None

        timestamp = parts[0]
        attack = parts[1]

        # Extract 41 NSL-KDD features
        features = []
        for v in parts[2:43]:
            try:
                features.append(float(v))
            except:
                features.append(0.0)

        return {
            "timestamp": timestamp,
            "attack_type": attack,
            "features": features
        }

    except Exception:
        return None


# -------------------------------------------------------
# STREAM SCORING
# -------------------------------------------------------
def score_log_entry(entry: Dict) -> float:
    x = np.array(entry["features"]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    code = encode_with_onnx(x_scaled)
    return float(rf_model.predict_proba(code)[:, 1][0])


# -------------------------------------------------------
# USER BEHAVIOR DETECTION
# -------------------------------------------------------
def detect_user_behavior(features: List[float], score: float, attack_type: str) -> List[str]:
    behaviors = []

    duration = features[0]
    protocol_type = features[1]
    service = features[2]
    flag = features[3]
    src_bytes = features[4]
    dst_bytes = features[5]
    land = features[6]
    wrong_fragment = features[7]
    urgent = features[8]
    hot = features[9]
    num_failed_logins = features[10]
    logged_in = features[11]
    num_compromised = features[12]
    root_shell = features[13]
    su_attempted = features[14]
    num_root = features[15]
    num_file_creations = features[16]
    num_shells = features[17]
    num_access_files = features[18]
    num_outbound_cmds = features[19]
    is_host_login = features[20]
    is_guest_login = features[21]
    count = features[22]
    srv_count = features[23]
    serror_rate = features[24]
    srv_serror_rate = features[25]
    rerror_rate = features[26]
    srv_rerror_rate = features[27]
    same_srv_rate = features[28]
    diff_srv_rate = features[29]
    srv_diff_host_rate = features[30]
    dst_host_count = features[31]
    dst_host_srv_count = features[32]
    dst_host_same_srv_rate = features[33]
    dst_host_diff_srv_rate = features[34]
    dst_host_same_src_port_rate = features[35]
    dst_host_srv_diff_host_rate = features[36]
    dst_host_serror_rate = features[37]
    dst_host_srv_serror_rate = features[38]
    dst_host_rerror_rate = features[39]
    dst_host_srv_rerror_rate = features[40]

    # Now add meaningful behavior logic

    if num_failed_logins > 0:
        behaviors.append("Multiple Failed Logins")

    if root_shell > 0:
        behaviors.append("Root Shell Access")

    if su_attempted > 0:
        behaviors.append("SU Command Attempted")

    if num_file_creations > 5:
        behaviors.append("Excessive File Creation")

    if num_compromised > 0:
        behaviors.append("Compromised Indicators")

    if is_guest_login > 0:
        behaviors.append("Guest Login Detected")

    if rerror_rate > 0.5:
        behaviors.append("High Rerror Rate")

    if same_srv_rate > 0.5:
        behaviors.append("High Same-SRV Rate")

    if src_bytes > 10000 or dst_bytes > 10000:
        behaviors.append("Large Data Transfer")

    if not behaviors:
        if score > 0.7:
            behaviors.append(f"Attack Detected ({attack_type})")
        elif score > 0.4:
            behaviors.append("Suspicious Activity")
        else:
            behaviors.append("Normal Activity")

    return behaviors



# -------------------------------------------------------
# SSE STREAM
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
                    score = score_log_entry(parsed)
                    sev = "Critical" if score > 0.7 else "Warning" if score > 0.4 else "Normal"
                    user_behaviors = detect_user_behavior(parsed["features"], score, parsed["attack_type"])

                    # Frontend-ready structured event
                    event = {
                        "timestamp": parsed["timestamp"],
                        "user_id": parsed["attack_type"],
                        "attack_type": parsed["attack_type"],
                        "event": f"Activity detected: {parsed['attack_type']}",
                        "score": score,
                        "severity": sev,
                        "user_behavior": user_behaviors,
                        "src_ip": f"10.0.0.{abs(hash(parsed['attack_type'])) % 255}",
                        "dst_ip": f"192.168.1.{int(parsed['features'][2]) % 255}",
                        "raw_log": line.strip()
                    }

                    RECENT_EVENTS.insert(0, event)
                    if len(RECENT_EVENTS) > RECENT_MAX:
                        RECENT_EVENTS.pop()

                    yield f"data: {json.dumps(event)}\n\n"
            else:
                await asyncio.sleep(interval)


@app.get("/stream")
async def stream(interval: float = 0.5):
    return StreamingResponse(
        tail_log(interval),
        media_type="text/event-stream"
    )


# -------------------------------------------------------
# RECENT EVENTS
# -------------------------------------------------------
@app.get("/logs")
def logs(limit: int = 50):
    return {
        "events": RECENT_EVENTS[:limit],
        "count": len(RECENT_EVENTS)
    }
