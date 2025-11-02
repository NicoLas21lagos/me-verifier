# 🧠 Verificador de Identidad por Imagen

## 📋 Descripción del Proyecto

Sistema de **verificación facial binario** (“yo” vs “no-yo”) que utiliza _embeddings faciales preentrenados_ y _machine learning_ para determinar si una imagen corresponde al usuario objetivo.  
El sistema expone un **endpoint REST** que recibe una imagen y responde con una decisión binaria y un nivel de confianza.

---

## 🎯 Objetivo

Entrenar un **verificador binario (“yo” vs “no-yo”)** usando embeddings faciales preentrenados y publicar un endpoint REST:

POST /verify
📦 Entrada: Imagen
📤 Salida: JSON con decisión (is_me) y nivel de confianza (score).

🏗️ Arquitectura del Sistema
🔹 Pipeline de Procesamiento
Detección Facial: MTCNN para localizar rostros en imágenes.

Extracción de Características: FaceNet (InceptionResnetV1) para generar embeddings de 512 dimensiones.

Clasificación: Regresión Logística con regularización.

Umbral de Decisión: 0.75 para balancear precisión y recall.

🔹 Stack Tecnológico
Backend: Flask + Gunicorn

Machine Learning: PyTorch, scikit-learn, facenet-pytorch

Procesamiento de Imágenes: OpenCV, Pillow

Despliegue: AWS EC2 (Ubuntu 22.04 LTS)

📁 Estructura del Proyecto
me-verifier/
├── api/
│ └── app.py # Aplicación Flask principal
├── models/
│ ├── model.joblib # Modelo de clasificación entrenado
│ └── scaler.joblib # Scaler para normalización
├── scripts/
│ ├── crop_faces.py # Detección y recorte de rostros
│ ├── embeddings.py # Generación de embeddings faciales
│ ├── train.py # Entrenamiento del clasificador
│ ├── evaluate.py # Evaluación del modelo
│ └── run_gunicorn.sh # Script de producción
├── data/
│ ├── me/ # Fotos propias (40-50 imágenes)
│ ├── not_me/ # Fotos de otras personas (200-400 imágenes)
│ └── cropped/ # Rostros recortados (generado)
├── samples/
│ ├── test_me.jpg # Imagen de prueba propia
│ ├── test_not_me.jpg # Imagen de prueba ajena
│ └── INSTRUCTIONS.md # Instrucciones de prueba
├── reports/
│ ├── metrics.json # Métricas de evaluación
│ └── confusion_matrix.png # Matriz de confusión
├── tests/
│ └── test_api.py # Pruebas de la API
├── requirements.txt # Dependencias del proyecto
├── .env.example # Plantilla de variables de entorno
└── README.md # Este archivo
🚀 Instalación y Configuración
🔧 Prerrequisitos
Python 3.11

pip

git

1️⃣ Clonar el Repositorio

git clone https://github.com/tu-usuario/me-verifier.git
cd me-verifier
2️⃣ Configurar Entorno Virtual

python3 -m venv venv
source venv/bin/activate
💡 En Windows:

venv\Scripts\activate
3️⃣ Instalar Dependencias

pip install --upgrade pip
pip install -r requirements.txt
🎮 Uso de la Aplicación
🧩 Entrenamiento del Modelo

1. Preparación de Datos
   Organiza las imágenes con la siguiente estructura:

data/
├── me/
│ ├── tu_foto_1.jpg
│ ├── tu_foto_2.jpg
│ └── ...
└── not_me/
├── persona_1.jpg
├── persona_2.jpg
└── ... 2. Pipeline de Entrenamiento

# Detección y recorte de rostros

python scripts/crop_faces.py

# Generación de embeddings faciales

python scripts/embeddings.py

# Entrenamiento del clasificador

python scripts/train.py

# Evaluación del modelo

python scripts/evaluate.py
⚙️ Modo Producción

chmod +x scripts/run_gunicorn.sh
./scripts/run_gunicorn.sh
🌐 API REST Endpoints
🩺 Health Check
GET /healthz

Respuesta:

{
"status": "healthy",
"model_loaded": true
}
👤 Verificación de Imagen
POST /verify

Encabezados:

Content-Type: multipart/form-data
Parámetros:

image: Archivo de imagen (JPG, PNG, JPEG)

Respuesta Exitosa:

{
"model_version": "me-verifier-v1",
"is_me": true,
"score": 0.93,
"threshold": 0.75,
"timing_ms": 28.7
}
Respuesta de Error:

{
"error": "No se detectó ninguna cara en la imagen"
}
🧪 Ejemplos de Uso
🖥️ Con cURL

# Verificar salud del servicio

curl http://localhost:5000/healthz

# Verificar imagen propia

curl -X POST -F "image=@samples/test_me.jpg" http://localhost:5000/verify

# Verificar imagen ajena

curl -X POST -F "image=@samples/test_not_me.jpg" http://localhost:5000/verify
🐍 Con Python

import requests

response = requests.post(
'http://localhost:5000/verify',
files={'image': open('samples/test_me.jpg', 'rb')}
)
print(response.json())
📬 Con Postman
Método: POST

URL: http://localhost:5000/verify

Body: form-data

Key: image → Type: File

Seleccionar archivo .jpg o .png

📈 Resultados y Métricas
Precisión (Accuracy): 0.92

Recall (Yo): 0.90

F1-Score: 0.91

Umbral óptimo: 0.75

Reportes disponibles en:

reports/
├── metrics.json
└── confusion_matrix.png

```

```
