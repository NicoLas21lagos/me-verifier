# scripts/crop_faces.py
import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np

def crop_faces(input_dir, output_dir, target_size=(160, 160)):
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar MTCNN para detección facial
    mtcnn = MTCNN(
        image_size=target_size[0],
        margin=20,
        keep_all=False,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Procesar cada imagen
    for person_dir in os.listdir(input_dir):
        person_input_path = os.path.join(input_dir, person_dir)
        person_output_path = os.path.join(output_dir, person_dir)
        os.makedirs(person_output_path, exist_ok=True)
        
        if os.path.isdir(person_input_path):
            for img_name in os.listdir(person_input_path):
                img_path = os.path.join(person_input_path, img_name)
                
                try:
                    # Cargar imagen
                    img = Image.open(img_path).convert('RGB')
                    
                    # Detectar y recortar rostro
                    face = mtcnn(img, save_path=os.path.join(person_output_path, img_name))
                    
                    if face is not None:
                        print(f"Rostro detectado en: {img_path}")
                    else:
                        print(f"No se detectó rostro en: {img_path}")
                        
                except Exception as e:
                    print(f"Error procesando {img_path}: {str(e)}")

if __name__ == "__main__":
    # Procesar fotos propias
    crop_faces('data/me', 'data/cropped/me')
    
    # Procesar fotos de otras personas
    crop_faces('data/not_me', 'data/cropped/not_me')