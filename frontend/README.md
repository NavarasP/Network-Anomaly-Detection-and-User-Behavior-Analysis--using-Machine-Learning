# Frontend (Next.js) Dashboard

This UI displays real-time network logs and anomaly scores streamed from the backend.

## Prerequisites
- Backend running at `http://localhost:8000` (see `docs/Project_Report.md`)
- Recent events available via `GET /logs` and live updates via `GET /stream` (SSE)

## Run
```powershell
cd "E:\Duk\S3\AI Pro\frontend"
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

## How It Works
- On mount, the page fetches history from `http://localhost:8000/logs?limit=40`.
- It subscribes to Server-Sent Events from `http://localhost:8000/stream` for live updates.
- Events include `timestamp`, `user_id`, `attack_type`, `score`, `severity`, `user_behavior`, and `raw_log`.

## Configuration
- Backend URL is currently hardcoded to `http://localhost:8000` in `pages/index.js`.
- To change, update the fetch and `EventSource` endpoints in `frontend/pages/index.js`.

## Notes
- The UI avoids random SSR state and populates data only on the client (after mount) to prevent hydration issues.
- If no events appear, ensure the backend is running and `backend/network.log` is being appended (use `backend/kdd_to_network_log.py`).
