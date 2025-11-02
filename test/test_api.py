# tests/test_api.py
import requests
import json
import os

def test_api():
    """Probar la API localmente"""
    base_url = "http://localhost:5000"
    
    # Probar health check
    print("Probando health check...")
    try:
        response = requests.get(f"{base_url}/healthz")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Error en health check: {e}")
        return
    
    # Probar con imagen propia
    print("\nProbando con imagen propia...")
    if os.path.exists('samples/selfie.jpg'):
        with open('samples/selfie.jpg', 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/verify", files=files)
            print(f"Imagen propia: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
    
    # Probar con imagen de otra persona
    print("\nProbando con imagen ajena...")
    if os.path.exists('samples/other.jpg'):
        with open('samples/other.jpg', 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/verify", files=files)
            print(f"Imagen ajena: {response.status_code}")
            print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()