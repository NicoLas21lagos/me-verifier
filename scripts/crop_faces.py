# scripts/crop_faces.py
import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np

def setup_directories():
    """Crear directorios necesarios"""
    os.makedirs('data/cropped/me', exist_ok=True)
    os.makedirs('data/cropped/not_me', exist_ok=True)

def initialize_mtcnn():
    """Inicializar detector MTCNN"""
    return MTCNN(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

def process_images(input_dir, output_dir, mtcnn):
    """Procesar imágenes y extraer rostros"""
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path).convert('RGB')
                
                faces = mtcnn(image)
                
                if faces is not None:
                    if isinstance(faces, torch.Tensor):
                        faces = [faces] if faces.dim() == 3 else faces
                        
                        for i, face in enumerate(faces):
                            face_pil = Image.fromarray(
                                (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            )
                            
                            output_path = os.path.join(
                                output_dir, 
                                f"{os.path.splitext(filename)[0]}_{i}.jpg"
                            )
                            face_pil.save(output_path)
                            count += 1
                            
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
    
    print(f"Procesadas {count} caras en {output_dir}")

def main():
    setup_directories()
    mtcnn = initialize_mtcnn()
    
    print("Procesando imágenes propias...")
    process_images('data/me', 'data/cropped/me', mtcnn)
    
    print("Procesando imágenes de otros...")
    process_images('data/not_me', 'data/cropped/not_me', mtcnn)
    
    print("Preprocesamiento completado!")

if __name__ == "__main__":
    main()