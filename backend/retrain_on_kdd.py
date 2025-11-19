#!/usr/bin/env python3
"""
Retrain model on NSL-KDD dataset to match real attack patterns.

This script reads the KDD training data, extracts features matching our format,
trains the RandomForest model, and saves all artifacts.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Paths
TRAIN_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "nsl-kdd", "KDDTrain+.txt")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_kdd_data(filepath, limit=None):
    """Load and parse KDD data into feature vectors."""
    print(f"Loading data from {filepath}...")
    
    data = []
    labels = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            parts = line.strip().split(',')
            if len(parts) < 42:
                continue
            
            try:
                # Extract features matching our network.log format
                duration = float(parts[0])
                protocol_str = parts[1]
                service_str = parts[2]
                flag_str = parts[3]
                src_bytes = float(parts[4])
                dst_bytes = float(parts[5])
                wrong_fragment = int(parts[7])
                urgent = int(parts[8])
                count_same_dst = int(parts[22])
                srv_count = int(parts[23])
                attack_type = parts[40]
                
                # Map protocol
                protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
                protocol_type = protocol_map.get(protocol_str, 0)
                
                # Map service and flag
                service = hash(service_str) % 10
                flag = hash(flag_str) % 6
                
                # Network features (7 base + one-hot encoding)
                net_base = [duration, src_bytes, dst_bytes, wrong_fragment, urgent, count_same_dst, srv_count]
                
                # One-hot encode
                proto_oh = [0.0] * 3
                proto_oh[protocol_type] = 1.0
                svc_oh = [0.0] * 10
                svc_oh[service] = 1.0
                flag_oh = [0.0] * 6
                flag_oh[flag] = 1.0
                
                net_features = net_base + proto_oh + svc_oh + flag_oh  # 26 features
                
                # Synthesize user features (realistic distributions)
                login_hour = np.random.uniform(0, 24)
                avg_login_hour = np.random.uniform(8, 18)
                device_count = np.random.randint(1, 5)
                new_device_flag = 1 if attack_type != "normal" and np.random.rand() < 0.3 else 0
                sensitive_access = 1 if attack_type != "normal" and np.random.rand() < 0.4 else 0
                
                user_features = [login_hour, avg_login_hour, device_count, new_device_flag, sensitive_access]
                
                # Label: 1 for attack, 0 for normal
                label = 0 if attack_type == "normal" else 1
                
                data.append(net_features + user_features)
                labels.append(label)
                
            except Exception as e:
                print(f"Skipping line {i}: {e}")
                continue
    
    print(f"Loaded {len(data)} samples")
    print(f"  Normal: {labels.count(0)}")
    print(f"  Attacks: {labels.count(1)}")
    
    return np.array(data), np.array(labels)


def main():
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Training file not found: {TRAIN_FILE}")
        print("Please ensure KDDTrain+.txt is in data/nsl-kdd/")
        return
    
    # Load data (use full dataset to get both classes)
    X, y = load_kdd_data(TRAIN_FILE, limit=None)
    
    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n⚖️ Class distribution: {dict(zip(unique, counts))}")
    
    if len(unique) < 2:
        print("❌ Need both normal and attack samples.")
        return
    
    # Split features
    X_net = X[:, :26]  # Network features
    X_user = X[:, 26:]  # User features
    
    print("\n📊 Training pipeline...")
    
    # Scale network features
    scaler_net = StandardScaler()
    X_net_scaled = scaler_net.fit_transform(X_net)
    
    # Apply PCA to network features
    pca = PCA(n_components=8, random_state=42)
    X_net_pca = pca.fit_transform(X_net_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Scale user features
    scaler_user = StandardScaler()
    X_user_scaled = scaler_user.fit_transform(X_user)
    
    # Combine
    X_combined = np.hstack([X_net_pca, X_user_scaled])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, stratify=y, random_state=42)
    
    # Train RandomForest
    print("\n🌲 Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\n📈 Evaluation:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Only compute proba if binary classification
    if len(np.unique(y)) == 2:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    
    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler_net, os.path.join(MODEL_DIR, "scaler_net.joblib"))
    joblib.dump(scaler_user, os.path.join(MODEL_DIR, "scaler_user.joblib"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_net.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_ueba_net.joblib"))
    
    print(f"\n✅ Models saved to {MODEL_DIR}")
    print("\nRestart the backend to load the new models:")
    print("  python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload")


if __name__ == "__main__":
    main()
