# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import joblib
import json
import os

def evaluate_model():
    """Evaluar modelo y generar reportes"""
    # Cargar datos y modelo
    X, y = np.load('data/embeddings.npy'), np.load('data/labels.npy')
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # Escalar datos
    X_scaled = scaler.transform(X)
    
    # Predecir
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['not_me', 'me'], 
                yticklabels=['not_me', 'me'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.savefig('reports/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.savefig('reports/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Evaluación completada. Gráficos guardados en reports/")

if __name__ == "__main__":
    os.makedirs('reports', exist_ok=True)
    evaluate_model()