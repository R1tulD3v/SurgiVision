import requests
import torch
import numpy as np
from PIL import Image
import cv2
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed

def test_external_images():
    """Test with external medical images"""
    detector = Spleen3DAnomalyDetectorFixed("../models/best_spleen_3d_autoencoder.pth")
    
    # You can download sample medical images or use your uploaded ones
    print("Testing external images...")
    
    # Example: Create a simple normal pattern
    normal_test = np.ones((64, 64, 64)) * 0.3  # Uniform tissue
    volume_tensor = torch.FloatTensor(normal_test[np.newaxis, np.newaxis, ...])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume_tensor = volume_tensor.to(device)
    
    with torch.no_grad():
        reconstructed = detector.model(volume_tensor)
        error = torch.mean((volume_tensor - reconstructed) ** 2).item()
    
    print(f"Uniform normal pattern error: {error:.6f}")
    
    # Test with noise
    noisy_test = normal_test + np.random.normal(0, 0.5, (64, 64, 64))
    noisy_test = np.clip(noisy_test, 0, 1)
    volume_tensor_noisy = torch.FloatTensor(noisy_test[np.newaxis, np.newaxis, ...]).to(device)
    
    with torch.no_grad():
        reconstructed_noisy = detector.model(volume_tensor_noisy)
        error_noisy = torch.mean((volume_tensor_noisy - reconstructed_noisy) ** 2).item()
    
    print(f"Noisy pattern error: {error_noisy:.6f}")

if __name__ == "__main__":
    test_external_images()
