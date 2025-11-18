# Prototype implementation for Network Anomaly Detection + User-Behavior fusion
# - Synthetic dataset generation (replaceable with real NSL-KDD / your logs)
# - Stacked non-symmetric autoencoder (encoder-only) for feature extraction (Keras/TensorFlow)
# - RandomForest classifier on encoded features + user-behavior features
# - Simulated streaming scoring loop that produces alerts with explainability info
# - Saves trained models to /mnt/data for download

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
from datetime import datetime, timedelta

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 1) Synthetic dataset creation
# ---------------------------
def generate_synthetic_dataset(n_samples=20000, n_users=200):
    # Network-like features (inspired by KDD features)
    duration = np.random.exponential(scale=1.0, size=n_samples) * 10  # seconds
    src_bytes = np.random.exponential(scale=300, size=n_samples)
    dst_bytes = np.random.exponential(scale=300, size=n_samples)
    wrong_fragment = np.random.poisson(0.01, n_samples)
    urgent = np.random.poisson(0.005, n_samples)
    count_same_dst = np.random.poisson(2, n_samples)
    srv_count = np.random.poisson(5, n_samples)
    protocol_type = np.random.choice([0,1,2], size=n_samples)  # tcp/udp/icmp encoded as ints
    service = np.random.choice(range(10), size=n_samples)  # categorical service id
    flag = np.random.choice(range(6), size=n_samples)  # flag id
    
    # User behavior features (per event map to a user)
    user_ids = np.random.choice([f"user_{i}" for i in range(n_users)], size=n_samples)
    # create per-user baseline stats
    user_baseline = {}
    for i in range(n_users):
        uid = f"user_{i}"
        user_baseline[uid] = {
            "avg_login_hour": np.random.uniform(8,18),  # typical active hours
            "device_count": np.random.randint(1,5),
            "sensitive_access_rate": np.random.beta(1.2, 8)  # probability of accessing sensitive resources
        }
    # For each event, produce user-behavior features (sometimes anomalous)
    avg_login_hour = np.array([user_baseline[u]["avg_login_hour"] for u in user_ids])
    device_count = np.array([user_baseline[u]["device_count"] for u in user_ids])
    sensitive_access_prob = np.array([user_baseline[u]["sensitive_access_rate"] for u in user_ids])
    
    # Current login hour for the event (may deviate)
    login_hour = (avg_login_hour + np.random.normal(scale=4.0, size=n_samples)) % 24
    # new device flag: occurs rarely, more likely if device_count small
    new_device_flag = (np.random.rand(n_samples) < 0.02).astype(int)
    # user anomalous flag (simulated ground truth) depends on combination
    # We'll create anomalies: large outbound bytes + login at odd hour + sensitive access
    sensitive_access = (np.random.rand(n_samples) < sensitive_access_prob).astype(int)
    
    # Label generation (0 normal, 1 anomaly)
    # Base anomaly probability
    base_prob = 0.02
    anomaly_score = (
        (src_bytes > 1500).astype(int)*0.4 +
        (dst_bytes > 1500).astype(int)*0.3 +
        (np.abs(login_hour - avg_login_hour) > 6).astype(int)*0.2 +
        new_device_flag*0.2 +
        sensitive_access*0.2 +
        (count_same_dst > 10).astype(int)*0.2
    )
    prob = base_prob + 0.6 * (anomaly_score.clip(0,1))
    labels = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "wrong_fragment": wrong_fragment,
        "urgent": urgent,
        "count_same_dst": count_same_dst,
        "srv_count": srv_count,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "user_id": user_ids,
        "avg_login_hour": avg_login_hour,
        "login_hour": login_hour,
        "device_count": device_count,
        "new_device_flag": new_device_flag,
        "sensitive_access": sensitive_access,
        "label": labels
    })
    return df

df = generate_synthetic_dataset(n_samples=12000, n_users=250)
print("Dataset shape:", df.shape)
df.head()

# Use ace_tools.display_dataframe_to_user if available (it will be used automatically by the environment)
try:
    from ace_tools import display_dataframe_to_user
    display_dataframe_to_user("sample_dataset", df.head(200))
except Exception:
    pass

# ---------------------------
# 2) Feature engineering
# ---------------------------
# numeric features for autoencoder
network_features = ["duration","src_bytes","dst_bytes","wrong_fragment","urgent","count_same_dst","srv_count"]
cat_features = ["protocol_type","service","flag"]
user_features = ["login_hour","avg_login_hour","device_count","new_device_flag","sensitive_access"]

# One-hot encode small categorical features (protocol_type, flag) and service as embedding via one-hot (small)
df_enc = pd.get_dummies(df, columns=["protocol_type","flag","service"], prefix=["proto","flag","svc"])
feature_cols = network_features + [c for c in df_enc.columns if c.startswith(("proto_","flag_","svc_"))]  # network for autoencoder/encoder

X_net = df_enc[feature_cols].values
X_user = df[user_features].values
y = df["label"].values

# Standardize network features (important for autoencoder)
scaler_net = StandardScaler()
X_net_scaled = scaler_net.fit_transform(X_net)

# Save scaler
os.makedirs("/mnt/data/models", exist_ok=True)
joblib.dump(scaler_net, "/mnt/data/models/scaler_net.joblib")

# ---------------------------
# 3) Non-symmetric deep autoencoder (encoder-only)
# ---------------------------
input_dim = X_net_scaled.shape[1]
encoding_dim = max(8, input_dim // 3)

# Build a simple encoder (non-symmetric: no decoder for fast encoding)
encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
x = layers.Dense(128, activation="relu")(encoder_inputs)
x = layers.Dense(64, activation="relu")(x)
encoded = layers.Dense(encoding_dim, activation="relu", name="encoded_vector")(x)
encoder = keras.Model(encoder_inputs, encoded, name="encoder_model")
encoder.summary()

# Train a full autoencoder that has a decoder to allow reconstruction loss (but we will keep encoder)
# Full autoencoder for training stability
ae_input = keras.Input(shape=(input_dim,), name="ae_input")
ae_x = layers.Dense(128, activation="relu")(ae_input)
ae_x = layers.Dense(64, activation="relu")(ae_x)
ae_encoded = layers.Dense(encoding_dim, activation="relu")(ae_x)
ae_x = layers.Dense(64, activation="relu")(ae_encoded)
ae_x = layers.Dense(128, activation="relu")(ae_x)
ae_decoded = layers.Dense(input_dim, activation="linear")(ae_x)
autoencoder = keras.Model(ae_input, ae_decoded, name="autoencoder_full")
autoencoder.compile(optimizer="adam", loss="mse")

# Train autoencoder quickly (small epochs for demo)
history = autoencoder.fit(X_net_scaled, X_net_scaled, epochs=10, batch_size=256, validation_split=0.1, verbose=1)
# After training, extract encoder by copying layers
# Build encoder from trained autoencoder
encoder = keras.Model(autoencoder.input, autoencoder.layers[3].output)  # depends on layer indices above
encoded_X = encoder.predict(X_net_scaled, batch_size=512)

# Save encoder
encoder.save("/mnt/data/models/encoder_model.keras", include_optimizer=False)

print("Encoded feature shape:", encoded_X.shape)

# ---------------------------
# 4) Train RandomForest on encoded network features + user features
# ---------------------------
# Normalize/scale user features
scaler_user = StandardScaler()
X_user_scaled = scaler_user.fit_transform(X_user)
joblib.dump(scaler_user, "/mnt/data/models/scaler_user.joblib")

# Combined feature vector
X_combined = np.hstack([encoded_X, X_user_scaled])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Save classifier
joblib.dump(clf, "/mnt/data/models/rf_ueba_net.joblib")

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print("\nClassification report (RandomForest on encoded+user features):\n")
print(classification_report(y_test, y_pred, digits=4))
try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", auc)
except Exception as e:
    print("AUC error:", e)

# ---------------------------
# 5) Simulated streaming scoring + alerting
# ---------------------------
# Function to score an event (single row) using stored models: scaler_net, encoder, scaler_user, clf
def load_models_from_disk():
    scaler_net = joblib.load("/mnt/data/models/scaler_net.joblib")
    scaler_user = joblib.load("/mnt/data/models/scaler_user.joblib")
    clf = joblib.load("/mnt/data/models/rf_ueba_net.joblib")
    encoder = tf.keras.models.load_model("/mnt/data/models/encoder_model.keras", compile=False)
    return scaler_net, scaler_user, encoder, clf

scaler_net_, scaler_user_, encoder_, clf_ = load_models_from_disk()

def score_event(event_row):
    # event_row: pd.Series containing raw event fields (must match df columns)
    # preprocess network features similarly
    net = event_row[feature_cols].values.reshape(1,-1)
    net_scaled = scaler_net_.transform(net)
    encoded = encoder_.predict(net_scaled)
    user = event_row[user_features].values.reshape(1,-1)
    user_scaled = scaler_user_.transform(user)
    combined = np.hstack([encoded, user_scaled])
    proba = clf_.predict_proba(combined)[0,1]
    pred = int(proba > 0.5)
    # simple explainability: contributions via tree feature importances (mapped back approximately)
    # We compute SHAP-like approx by multiplying features by feature_importances where possible
    importances = clf_.feature_importances_
    contrib = combined.flatten() * importances[:combined.shape[1]]
    top_idx = np.argsort(-np.abs(contrib))[:5]
    top_feats = [{"feature_index": int(i), "value": float(combined.flatten()[i]), "importance": float(importances[i]), "score_contrib": float(contrib[i])} for i in top_idx]
    return {"score": float(proba), "alert": bool(pred), "top_contributors": top_feats}


# Simulate streaming: pick random events and score them
stream_sample = df_enc.sample(20, random_state=1).reset_index(drop=True)
alerts = []
for i, row in stream_sample.iterrows():
    res = score_event(row)
    out = {
        "event_index": i,
        "user_id": row["user_id"],
        "score": res["score"],
        "alert": res["alert"],
        "top_contributors": res["top_contributors"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    if res["alert"]:
        alerts.append(out)

print(f"\nSimulated stream produced {len(alerts)} alerts out of {len(stream_sample)} events.")
if alerts:
    alerts_df = pd.DataFrame(alerts)
    try:
        display_dataframe_to_user("simulated_alerts", alerts_df)
    except Exception:
        print(alerts_df.to_dict(orient='records'))

# Save a small metadata file describing model and how to replace dataset
meta = {
    "note": "Prototype models saved. Replace synthetic dataset with NSL-KDD/real logs and retrain for production.",
    "models": {
        "encoder": "/mnt/data/models/encoder_model.keras",
        "rf": "/mnt/data/models/rf_ueba_net.joblib",
        "scaler_net": "/mnt/data/models/scaler_net.joblib",
        "scaler_user": "/mnt/data/models/scaler_user.joblib"
    },
    "feature_columns_network": feature_cols,
    "feature_columns_user": user_features
}
with open("/mnt/data/models/README_model_metadata.json","w") as f:
    json.dump(meta, f, indent=2)

print("\nModels and metadata saved to /mnt/data/models/. You can download them from the UI.\n")
