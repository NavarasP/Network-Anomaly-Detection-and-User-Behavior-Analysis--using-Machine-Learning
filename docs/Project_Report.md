# Network Anomaly Detection and User Behavior Analysis — Project Report

Date: November 26, 2025
Repository: `Network-Anomaly-Detection-and-User-Behavior-Analysis--using-Machine-Learning`

## Summary
This project delivers a real-time network anomaly detection and user behavior analysis system. The backend (FastAPI) ingests log lines, scales and encodes features with an ONNX encoder, scores anomalies via a RandomForest model, and streams results to a Next.js frontend dashboard using Server-Sent Events (SSE). Utilities exist to convert NSL-KDD datasets to the backend log format and to retrain models.

## Architecture Overview
- Backend API: FastAPI (`backend/main.py`)
  - Loads scaler, ONNX encoder, and RF classifier from `backend/models/`
  - Provides endpoints: `/health`, `/score`, `/stream` (SSE), `/logs`
  - Tails `backend/network.log` and streams scored events to the frontend
- Data & Model Utilities
  - NSL-KDD converter to `network.log`: `backend/kdd_to_network_log.py`
  - Retraining pipeline (UEBA fusion): `backend/retrain_on_kdd.py`
  - Keras → ONNX conversion: `backend/encoder_to_onnx.py`
- Frontend Dashboard: Next.js (`frontend/`)
  - Consumes `/stream` SSE for live updates and `/logs` for history
- ML Prototypes & Notebook
  - Synthetic pipeline and UEBA fusion: `ai product.py`
  - Model experimentation: `notebooks/model.ipynb`

## Backend (FastAPI)
File: `backend/main.py`

- Model Artifacts (expected in `backend/models/`):
  - `scaler_nslkdd.joblib` — Scaler for the 41 NSL-KDD features
  - `stacked_encoder.onnx` — Encoder (stacked NDAE) exported to ONNX
  - `rf_encoded.joblib` — RandomForest classifier trained on encoder outputs

- Endpoints:
  - `GET /health`: Returns status and which models are loaded
  - `POST /score`:
    - Request: `{ "features": [f1, f2, ..., f41] }`
    - Flow: scale → ONNX encode → RF predict_proba → `{ "score": <float 0..1> }`
  - `GET /stream`: SSE endpoint streaming scored events as JSON (`text/event-stream`)
  - `GET /logs?limit=50`: Returns recent events buffer (up to 500 kept in memory)

- Log Tailing & Event Structure:
  - Reads `backend/network.log` continuously; each line parsed as:
    - `timestamp,attack_type,<41 features>` (total 43 comma-separated values)
  - For each parsed line, computes anomaly score and severity:
    - Severity: `Critical` (>0.7), `Warning` (>0.4), else `Normal`
  - Event JSON emitted in SSE includes:
    - `timestamp`, `user_id` (alias of `attack_type`), `attack_type`, `event` (string),
      `score`, `severity`, `user_behavior` (list of flags), `src_ip`, `dst_ip`, `raw_log`

- User Behavior Detection (heuristics):
  - Flags based on NSL-KDD-like fields, e.g., `num_failed_logins`, `root_shell`, `su_attempted`,
    `num_file_creations`, `num_compromised`, `is_guest_login`, `rerror_rate`, `same_srv_rate`, `src_bytes/dst_bytes`
  - If no specific flags, severity-based generic tags: `Attack Detected`, `Suspicious Activity`, or `Normal Activity`

- Dependencies (`backend/requirements.txt`):
  - `fastapi`, `uvicorn[standard]`, `joblib`, `scikit-learn`, `numpy`, `pandas`

## Data Integration & Utilities

### NSL-KDD → `network.log`
File: `backend/kdd_to_network_log.py`
- Reads `data/nsl-kdd/KDDTest+.txt`
- For each line: builds `timestamp,attack_type,<41 features>` and appends to `backend/network.log`
- CLI options:
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python kdd_to_network_log.py --delay 1.0 --limit 100 --start 0
```
- Output example:
```
14:07:31,neptune,0,tcp,http,SF,181,5450,0,0,0,0,0,0,0,0,0,0,0,0,0,0,510,12,0.07,0.05,0.0,0.0,0.02,0.03,0.00,37,26,0.86,0.06,0.00,0.00,0.00,0.00,0.00,0.00
```

### Retraining Pipeline (UEBA Fusion)
File: `backend/retrain_on_kdd.py`
- Loads `data/nsl-kdd/KDDTrain+.txt`
- Engineer network features (duration, src/dst bytes, fragments, counts) with one-hot proto/service/flag
- Synthesize user features: `login_hour`, `avg_login_hour`, `device_count`, `new_device_flag`, `sensitive_access`
- Scale network + apply PCA; scale user; combine; train `RandomForest`
- Saves artifacts to `backend/models/`:
  - `scaler_net.joblib`, `scaler_user.joblib`, `pca_net.joblib`, `rf_ueba_net.joblib`

Note: These artifact names differ from the backend runtime expectation (`scaler_nslkdd.joblib`, `stacked_encoder.onnx`, `rf_encoded.joblib`). To use retrained models with the current backend:
- Either adapt `backend/main.py` to load `rf_ueba_net.joblib` and corresponding scalers/PCA, or
- Produce compatible artifacts: a unified `scaler_nslkdd.joblib`, an ONNX encoder, and an RF model trained on encoder codes (`rf_encoded.joblib`).

### Keras → ONNX Conversion
File: `backend/encoder_to_onnx.py`
- Loads `models/stacked_encoder.keras` and exports to `models/stacked_encoder.onnx` using `tf2onnx`
- Example run:
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python encoder_to_onnx.py
```

## Frontend (Next.js)
Directory: `frontend/`

- Entry: `frontend/pages/index.js`
- Behavior:
  - On mount, fetches `http://localhost:8000/logs?limit=40` for recent history
  - Subscribes to `http://localhost:8000/stream` via `EventSource` for live events
  - Renders alert banner (Warnings/Critical), severity distribution, log table, and details panel
  - Explainability panel shows `user_behavior` flags; placeholder support for `top_features`

- Commands:
```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm install
npm run dev
```
- Config: `frontend/package.json` (Next 13.4.x, React 18)
- Note: `frontend/README.md` references `/score` usage, but the current UI uses `/logs` and `/stream`.

## ML Prototypes & Notebook

### `ai product.py`
- Generates synthetic dataset with network+user features
- Trains autoencoder-based encoder (Keras) + RandomForest on encoded+user features
- Saves models to `/mnt/data/models/`:
  - `encoder_model.keras`, `rf_ueba_net.joblib`, and two scalers
- Includes `score_event()` helper and simulated streaming alerts

### `notebooks/model.ipynb`
- Notebook for experimentation (training/saving stacked encoder, RF, etc.)
- Intended to produce artifacts compatible with the backend ONNX pipeline

## Setup & Run

### Backend
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Ensure `backend/models/` contains:
- `scaler_nslkdd.joblib`
- `stacked_encoder.onnx`
- `rf_encoded.joblib`

### Frontend
```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm install
npm run dev
```
Open `http://localhost:3000`

### Generate Logs (NSL-KDD)
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python kdd_to_network_log.py --delay 1.0 --limit 100
```
The backend will stream events to the frontend in real-time.

## API Reference

### `POST /score`
Request:
```json
{
  "features": [0.0, 1.0, 2.0, 3.0, ..., 40.0]
}
```
Response:
```json
{ "score": 0.734 }
```
Errors:
- 503 if models not loaded
- 400 if scaling fails
- 500 if prediction fails

### `GET /health`
```json
{ "status": "ok", "scaler_loaded": true, "encoder_loaded": true, "model_loaded": true }
```

### `GET /logs?limit=50`
Returns recent events buffer (max 500 kept).

### `GET /stream`
SSE stream; each message line begins with `data: {...}\n\n` with the event JSON.

## Log Format
Backend expects CSV lines:
```
<timestamp>,<attack_type>,<41 features>
```
- `<timestamp>`: e.g., `14:23:01`
- `<attack_type>`: NSL-KDD label (`normal`, `neptune`, etc.)
- 41 features: NSL-KDD feature vector as floats/ints

Generated by `backend/kdd_to_network_log.py` from NSL-KDD test set.

## Known Gaps & Recommendations
- Artifact mismatch:
  - Retrain pipeline saves `rf_ueba_net.joblib` + split scalers + PCA
  - Runtime expects `rf_encoded.joblib` + single scaler + ONNX encoder
  - Recommendation: align artifact contracts or add a loader path in `backend/main.py` to support both.

- Documentation drift:
  - `docs/QUICKSTART.md` references `generate_logs.py` and an alternate log format; file is not present in the repo. Prefer `kdd_to_network_log.py` or add the missing generator.
  - `frontend/README.md` mentions `/score`; current UI uses `/logs`+`/stream`. Update README for consistency.

- Feature mapping assumptions:
  - `detect_user_behavior()` indexes NSL-KDD fields directly from the 41-element vector. Ensure the order matches NSL-KDD canonical feature list to avoid misinterpretation.

- Persistence & scale:
  - `RECENT_EVENTS` is in-memory only and capped at 500. For production, consider a DB or Redis buffer and log rotation.

- Security:
  - Add authentication for API endpoints and CORS hardening.

## Future Work
- Unify the training pipeline to produce ONNX-compatible encoders and RF models that match backend expectations
- Add SHAP-based explainability for top contributing features per event
- Support real log sources (syslog/IDS/NetFlow) and multi-source ingestion
- Add Docker files and CI for reproducible deployments
- Expand frontend with filters, charts, and per-user drilldowns

## Appendix: Commands (Windows PowerShell)
```powershell
# Backend: start API
cd "E:\Duk\S3\AI Pro\backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Frontend: start dashboard
cd "E:\Duk\S3\AI Pro\frontend"
npm install
npm run dev

# Generate NSL-KDD logs
cd "E:\Duk\S3\AI Pro\backend"
python kdd_to_network_log.py --delay 1.0 --limit 100
```
