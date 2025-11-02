# api/app.py
import os
import time
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_MB', 5)) * 1024 * 1024

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelos globalmente
class FaceVerifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.classifier = joblib.load(os.getenv('MODEL_PATH', 'models/model.joblib'))
        self.scaler = joblib.load(os.getenv('SCALER_PATH', 'models/scaler.joblib'))
        self.threshold = float(os.getenv('THRESHOLD', 0.75))
    
    def verify_face(self, image_path):
        """Verificar si la cara en la imagen es la persona objetivo"""
        start_time = time.time()
        
        try:
            # Cargar y procesar imagen
            image = Image.open(image_path).convert('RGB')
            
            # Detectar cara
            face = self.mtcnn(image)
            if face is None:
                return {
                    'error': 'No se detectó ninguna cara en la imagen',
                    'timing_ms': (time.time() - start_time) * 1000
                }
            
            # Obtener embedding
            face_tensor = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedder(face_tensor)
            embedding_np = embedding.cpu().numpy().flatten().reshape(1, -1)
            
            # Escalar y predecir
            embedding_scaled = self.scaler.transform(embedding_np)
            probability = self.classifier.predict_proba(embedding_scaled)[0, 1]
            
            # Aplicar umbral
            is_me = probability >= self.threshold
            
            timing_ms = (time.time() - start_time) * 1000
            
            return {
                'is_me': bool(is_me),
                'score': float(probability),
                'threshold': self.threshold,
                'timing_ms': timing_ms,
                'model_version': 'me-verifier-v1'
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return {
                'error': f'Error procesando imagen: {str(e)}',
                'timing_ms': (time.time() - start_time) * 1000
            }

# Inicializar verificador
verifier = FaceVerifier()

@app.route('/')
def index():
    """Página principal con interfaz web"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_image():
    """Endpoint para verificación de imágenes"""
    start_time = time.time()
    
    # Verificar que se envió una imagen
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó imagen'}), 400
    
    file = request.files['image']
    
    # Verificar que el archivo tiene nombre
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    # Verificar tipo de archivo
    if not (file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
        return jsonify({'error': 'Solo se permiten archivos PNG, JPG o JPEG'}), 400
    
    try:
        # Guardar archivo temporal
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Verificar cara
        result = verifier.verify_face(temp_path)
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        
        # Devolver resultado
        if 'error' in result:
            return jsonify(result), 400
        else:
            return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"Error general: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/healthz', methods=['GET'])
def health_check():
    """Endpoint de salud"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)