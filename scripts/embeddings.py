# scripts/embeddings.py
import os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import pandas as pd
from torchvision import transforms

class FaceEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_embedding(self, image_path):
        """Obtener embedding de una imagen"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None

def generate_embeddings():
    embedder = FaceEmbedder()
    data = []
    
    print("Generando embeddings propios...")
    for filename in os.listdir('data/cropped/me'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join('data/cropped/me', filename)
            embedding = embedder.get_embedding(path)
            if embedding is not None:
                data.append({
                    'filename': filename,
                    'embedding': embedding,
                    'label': 1,
                    'class': 'me'
                })
    
    print("Generando embeddings de otros...")
    for filename in os.listdir('data/cropped/not_me'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join('data/cropped/not_me', filename)
            embedding = embedder.get_embedding(path)
            if embedding is not None:
                data.append({
                    'filename': filename,
                    'embedding': embedding,
                    'label': 0,
                    'class': 'not_me'
                })
    
    if data:
        embeddings = np.array([item['embedding'] for item in data])
        labels = np.array([item['label'] for item in data])
        filenames = [item['filename'] for item in data]
        classes = [item['class'] for item in data]
        
        np.save('data/embeddings.npy', embeddings)
        np.save('data/labels.npy', labels)
        
        df = pd.DataFrame({
            'filename': filenames,
            'class': classes,
            'label': labels
        })
        df.to_csv('data/metadata.csv', index=False)
        
        print(f"Embeddings generados: {len(data)} total")
        print(f" - Yo: {sum(labels)}")
        print(f" - No-yo: {len(labels) - sum(labels)}")

if __name__ == "__main__":
    generate_embeddings()