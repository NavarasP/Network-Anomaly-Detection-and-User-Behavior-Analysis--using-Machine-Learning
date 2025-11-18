Project scaffold: FastAPI backend + Next.js frontend

Backend (FastAPI)
- entry: backend/main.py
- models: loadable from /mnt/data/models or ./models
- run: from project root or inside backend

  # create venv and install
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -r backend/requirements.txt
  uvicorn backend.main:app --reload --port 8000

Frontend (Next.js)
- entry: frontend/pages/index.js
- run:
  cd frontend
  npm install
  npm run dev

Notes
- The backend expects model artifacts (scaler_net.joblib, scaler_user.joblib, rf_model.joblib or rf_ueba_net.joblib) to be present in one of these paths: ./models, ./backend/models, /mnt/data/models
- If model artifacts are not present the /health endpoint will report model_loaded=false and /score will return 503.
