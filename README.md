###### Verificador de Identidad por Imagen

📋 Descripción del Proyecto
Sistema de verificación facial binario ("yo" vs "no-yo") que utiliza embeddings faciales preentrenados y machine learning para determinar si una imagen corresponde al usuario objetivo. El sistema expone un endpoint REST que recibe una imagen y responde con una decisión binaria y nivel de confianza.

🎯 Objetivo
Entrenar un verificador binario ("yo" vs "no-yo") usando embeddings faciales preentrenados y publicar un endpoint REST POST /verify que recibe una imagen y responde un JSON con decisión y confianza.

🏗️ Arquitectura del Sistema
Pipeline de Procesamiento
Detección Facial: MTCNN para localizar rostros en imágenes

Extracción de Características: FaceNet (InceptionResnetV1) para embeddings de 512 dimensiones

Clasificación: Logistic Regression con regularización

Umbral de Decisión: 0.75 para balancear precisión y recall

Stack Tecnológico
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

Prerrequisitos:
Python 3.11
pip
git

1. Clonar el Repositorio
   git clone https://github.com/tu-usuario/me-verifier.git
   cd me-verifier

2. Configurar Entorno Virtual
   python3 -m venv venv
   source venv/bin/activate

# En Windows: venv\Scripts\activate

3. Instalar Dependencias
   pip install --upgrade pip
   pip install -r requirements.txt

🎮 Uso de la Aplicación

📊 Entrenamiento del Modelo

1. Preparación de Datos
   Colocar las imágenes en la estructura correcta:

# Fotos propias

data/me/tu_foto_1.jpg
data/me/tu_foto_2.jpg
...

# Fotos de otras personas

data/not_me/persona_1.jpg
data/not_me/persona_2.jpg
...

2. Pipeline de Entrenamiento

# Preprocesamiento - Detección y recorte de rostros

python scripts/crop_faces.py

# Generación de embeddings

python scripts/embeddings.py

# Entrenamiento del clasificador

python train.py

# Evaluación del modelo

python evaluate.py

Modo Producción
chmod +x scripts/run_gunicorn.sh./scripts/run_gunicorn.sh

Obtén el resultado con nivel de confianza

🌐 API REST Endpoints
Health Check
http
GET /healthz
Respuesta:

json
{
"status": "healthy",
"model_loaded": true
}
Verificación de Imagen
http
POST /verify
Content-Type: multipart/form-data
Parámetros:

image: Archivo de imagen (JPG, PNG, JPEG)

Respuesta Exitosa:

json
{
"model_version": "me-verifier-v1",
"is_me": true,
"score": 0.93,
"threshold": 0.75,
"timing_ms": 28.7
}
Respuesta de Error:

json
{
"error": "No se detectó ninguna cara en la imagen"
}
🔧 Ejemplos de Uso
Con cURL

# Verificar salud del servicio

curl http://localhost:5000/healthz

# Verificar imagen propia

curl -X POST -F "image=@samples/test_me.jpg" http://localhost:5000/verify

# Verificar imagen ajena

curl -X POST -F "image=@samples/test_not_me.jpg" http://localhost:5000/verify
Con Python
python
import requests

response = requests.post(
'http://localhost:5000/verify',
files={'image': open('samples/test_me.jpg', 'rb')}
)
print(response.json())
Con Postman
Método: POST

URL: http://localhost:5000/verify

Body: form-data

Key: image (Type: File)

Seleccionar archivo de imagen
