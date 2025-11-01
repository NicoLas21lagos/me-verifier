# train.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import json
import os

def load_data():
    """Cargar embeddings y labels"""
    embeddings = np.load('data/embeddings.npy')
    labels = np.load('data/labels.npy')
    return embeddings, labels

def train_model():
    """Entrenar modelo de clasificación"""
    print("Cargando datos...")
    X, y = load_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar características
    print("Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    print("Entrenando modelo...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['not_me', 'me']))
    
    # Guardar modelo y scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Guardar métricas
    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X.shape[1]
    }
    
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Modelo guardado en models/model.joblib")
    return model, scaler, metrics

if __name__ == "__main__":
    train_model()