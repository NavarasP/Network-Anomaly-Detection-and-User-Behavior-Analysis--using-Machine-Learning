# Network Anomaly Detection and User Behavior Analysis

This project provides a real-time system that ingests NSL-KDD-like network logs, detects anomalies using a stacked encoder (ONNX) plus RandomForest classifier, and surfaces human-readable user behavior flags in a live dashboard.

## Project Overview
- **Backend (FastAPI):** `backend/main.py`
  - Loads scaler, ONNX encoder, and RF classifier from `backend/models/`
  - Tails `backend/network.log`, scores events, and streams via SSE
  - Endpoints: `/health`, `/score`, `/stream`, `/logs`
- **Data Utilities:**
  - NSL-KDD converter: `backend/kdd_to_network_log.py`
  - Encoder export (Keras → ONNX): `backend/encoder_to_onnx.py`
  - Retraining pipeline (UEBA fusion demo): `backend/retrain_on_kdd.py`
- **Frontend (Next.js):** `frontend/`
  - Subscribes to `/stream` and renders live events with severity and behavior flags
- **Docs & Notebook:**
  - Detailed report: `docs/Project_Report.md`
  - Experiments: `notebooks/model.ipynb`

## Installation

### Backend (Windows PowerShell)
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place model artifacts in `backend/models/`:
- `scaler_nslkdd.joblib`
- `stacked_encoder.onnx`
- `rf_encoded.joblib`

### Frontend
```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm install
```

## Running Backend + Frontend

### Start Backend API
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend Dashboard
```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm run dev
```
Open `http://localhost:3000`.

### Generate/Append Logs (NSL-KDD)
```powershell
cd "E:\Duk\S3\AI Pro\backend"
python kdd_to_network_log.py --delay 1.0 --limit 100
```
This reads `data/nsl-kdd/KDDTest+.txt` and appends to `backend/network.log`. The backend will stream events to the UI.

## API Endpoints

### `GET /health`
Returns backend readiness:
```json
{ "status": "ok", "scaler_loaded": true, "encoder_loaded": true, "model_loaded": true }
```

### `POST /score`
Request body:
```json
{ "features": [f1, f2, ..., f41] }
```
Response:
```json
{ "score": 0.734 }
```

### `GET /logs?limit=50`
Returns recent events buffer (in-memory, capped at 500):
```json
{ "events": [ { "timestamp": "...", "score": 0.12, "severity": "Normal" } ], "count": 42 }
```

### `GET /stream`
Server-Sent Events stream. Each message line:
```
data: {"timestamp":"...","score":0.87,"severity":"Critical", ...}

```

## Screenshots
Add screenshots to `docs/assets/` and reference them here, for example:

![Dashboard Overview](assets/dashboard_overview.png)
![Live Alerts](assets/live_alerts.png)

> Note: Create the `docs/assets/` folder and place images to make these links work.

## Contributors
- `@NavarasP` — Maintainer
- Contributions welcome via PRs

## Notes
- If `/health` reports models missing or `/score` returns 503, ensure the expected artifacts exist in `backend/models/`.
- Frontend endpoints are hardcoded to `http://localhost:8000` in `frontend/pages/index.js`; update there if deploying elsewhere.
