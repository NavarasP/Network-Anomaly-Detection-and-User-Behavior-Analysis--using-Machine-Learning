# Quick Start: Real-Time Log Analysis System

## Architecture Overview

The system now operates in three components:

1. **Log Generator** (`backend/generate_logs.py`) - Simulates network logs
2. **Backend API** (`backend/main.py`) - Tails logs, predicts anomalies, streams to frontend
3. **Frontend Dashboard** (`frontend/`) - Displays live logs with explainability

---

## Step 1: Start the Backend

```powershell
cd "E:\Duk\S3\AI Pro\backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The backend will:
- Load ML models from `backend/models/`
- Create `network.log` if it doesn't exist
- Start tailing the log file for new entries
- Stream parsed + predicted events via `/stream` endpoint

---

## Step 2: Start the Frontend

```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm run dev
```

Open http://localhost:3000 in your browser. The dashboard will:
- Connect to backend SSE stream
- Show "Listening for logs" status
- Display empty table initially (no logs yet)

---

## Step 3: Generate Logs

Open a **new terminal** and run:

```powershell
cd "E:\Duk\S3\AI Pro\backend"
python generate_logs.py --count 20 --delay 2
```

This will append 20 log lines to `network.log` with a 2-second delay between each.

### Generator Options

```powershell
# Generate 10 logs instantly
python generate_logs.py --count 10 --burst

# Generate 100 logs slowly (1.5s apart)
python generate_logs.py --count 100 --delay 1.5

# Continuous generation (run in background)
while ($true) { python generate_logs.py --count 5 --delay 3; Start-Sleep -Seconds 1 }
```

---

## What You'll See

### Frontend Dashboard

1. **Alert Banner** (top): Shows recent Critical/Warning alerts with scores
2. **Analyzing Indicator**: Flashes briefly when processing new logs
3. **Log Table**: 
   - Raw CSV log text
   - User ID
   - Anomaly score (0-1)
   - Severity (Normal/Warning/Critical)
   - Click any row to see details
4. **Right Panel**:
   - Stats cards (Total/Normal/Warning/Critical counts)
   - **Log Details** section:
     - User behavior flags (new device, sensitive access, off-hours login)
     - Top contributing features with importance scores
   - System status

### Backend Logs

Watch the backend terminal for:
```
INFO:     127.0.0.1:xxxxx - "GET /stream HTTP/1.1" 200 OK
```

Each new log line is parsed, scored, and streamed.

---

## Log Format

`network.log` uses CSV format:

```
timestamp,user_id,duration,src_bytes,dst_bytes,wrong_fragment,urgent,count_same_dst,srv_count,protocol_type,service,flag,login_hour,avg_login_hour,device_count,new_device_flag,sensitive_access,src_ip,dst_ip
```

Example line:
```
14:23:01,user_42,5.2,450.3,230.1,0,0,3,5,0,2,1,14.5,12.0,2,0,0,192.168.1.42,10.0.1.5
```

See `backend/log_format.md` for full specification.

---

## Explainability Features

### User Behavior Flags

The system detects:
- **New device** (new_device_flag = 1)
- **Sensitive resource access** (sensitive_access = 1)
- **Off-hours login** (|login_hour - avg_login_hour| > 6)

### Feature Importance

Top 5 contributing features are shown with:
- Feature name
- Feature value (scaled)
- Model importance weight
- Contribution to anomaly score

---

## Troubleshooting

### "Model or scalers not available"
- Check that `backend/models/` contains:
  - `scaler_net.joblib`
  - `scaler_user.joblib`
  - `rf_ueba_net.joblib` or `rf_model.joblib`
  - Optionally: `pca_net.joblib`

### "Hydration error" in frontend
- Already fixed: initial state is deterministic
- Clear browser cache if persists

### No logs appearing
- Verify `generate_logs.py` is writing to `backend/network.log`
- Check backend terminal for parse errors
- Ensure backend started before generating logs

### Scikit-learn version warnings
- Models trained with sklearn 1.7.2, runtime uses 1.6.1
- Warnings are non-fatal; consider `pip install scikit-learn==1.7.2` to match

---

## Production Considerations

1. **Persistence**: Add database or log rotation for `RECENT_EVENTS` buffer
2. **Real Logs**: Replace `generate_logs.py` with real syslog/firewall/IDS feeds
3. **Authentication**: Add API keys or OAuth to backend endpoints
4. **Scaling**: Use Redis for event buffer, deploy with Docker/K8s
5. **Model Updates**: Retrain periodically with new labeled data

---

## Quick Test Sequence

```powershell
# Terminal 1: Backend
cd "E:\Duk\S3\AI Pro\backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd "E:\Duk\S3\AI Pro\frontend"
npm run dev

# Terminal 3: Generate logs
cd "E:\Duk\S3\AI Pro\backend"
python generate_logs.py --count 30 --delay 1

# Browser: Open http://localhost:3000
# Watch logs appear in real-time with predictions!
```

---

**Status**: ✅ System ready. Start all three components and watch live anomaly detection with explainability!
