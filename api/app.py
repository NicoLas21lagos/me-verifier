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

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_MB', 5)) * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVerifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
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
        
        model_path = os.getenv('MODEL_PATH', 'models/model.joblib')
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.joblib')
        
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado en: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = float(os.getenv('THRESHOLD', 0.75))
        
        logger.info("✅ Modelos cargados correctamente")
    
    def verify_face(self, image_path):
        """Verificar si la cara en la imagen es la persona objetivo"""
        start_time = time.time()
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            face = self.mtcnn(image)
            if face is None:
                return {
                    'error': 'No se detectó ninguna cara en la imagen',
                    'timing_ms': (time.time() - start_time) * 1000
                }
            
            face_tensor = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedder(face_tensor)
            embedding_np = embedding.cpu().numpy().flatten().reshape(1, -1)
            
            embedding_scaled = self.scaler.transform(embedding_np)
            probability = self.classifier.predict_proba(embedding_scaled)[0, 1]
            
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

try:
    verifier = FaceVerifier()
    logger.info("✅ FaceVerifier inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando FaceVerifier: {e}")
    verifier = None

@app.route('/')
def index():
    """Página principal con interfaz web"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_image():
    """Endpoint para verificación de imágenes"""
    start_time = time.time()
    
    if verifier is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó imagen'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if not (file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
        return jsonify({'error': 'Solo se permiten archivos PNG, JPG o JPEG'}), 400
    
    try:
        filename = secure_filename(file.filename)
        
        temp_dir = os.path.join(os.environ.get('TEMP', os.getcwd()), 'face_verifier_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        if not os.path.exists(temp_path):
            return jsonify({'error': 'No se pudo guardar el archivo temporal'}), 500
        
        result = verifier.verify_face(temp_path)
        
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"No se pudo eliminar archivo temporal: {cleanup_error}")
        
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
    status = 'healthy' if verifier is not None else 'unhealthy'
    return jsonify({'status': status, 'model_loaded': verifier is not None})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    if verifier is None:
        logger.error("❌ No se pudo inicializar el modelo. Verifica los archivos del modelo.")
    else:
        logger.info("🚀 Servicio iniciado correctamente")
    
    app.run(host='0.0.0.0', port=port, debug=debug)