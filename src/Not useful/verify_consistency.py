import torch
import numpy as np
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed

def verify_training_consistency():
    """Verify training data processing is consistent"""
    print("🔬 VERIFYING TRAINING DATA CONSISTENCY")
    print("="*60)
    
    detector = Spleen3DAnomalyDetectorFixed("../models/best_spleen_3d_autoencoder.pth")
    
    # Test same volume multiple times
    volume_idx = 0  # Test first volume
    volume_name = detector.preprocessor.image_files[volume_idx].name
    
    print(f"Testing volume: {volume_name}")
    print("Running same volume 3 times to check consistency...")
    
    results = []
    for run in range(3):
        result = detector.detect_anomaly_from_training_file(volume_idx, threshold=0.010)
        if result:
            error = result['reconstruction_error']
            results.append(error)
            print(f"Run {run+1}: Error = {error:.8f}")
    
    if len(results) == 3:
        if np.allclose(results, results[0], rtol=1e-6):
            print("✅ CONSISTENT: Same volume gives same results")
        else:
            print("❌ INCONSISTENT: Same volume gives different results!")
            print("This indicates randomness in preprocessing or model")
    
    # Now check what the raw training process would see
    print(f"\n🔍 CHECKING RAW TRAINING DATA PROCESSING:")
    
    volume_path = detector.preprocessor.image_files[volume_idx]
    mask_path = detector.preprocessor.label_files[volume_idx]
    
    print(f"Image path: {volume_path}")
    print(f"Mask path: {mask_path}")
    
    # Process exactly like training would
    volume, mask = detector.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
    
    if volume is not None:
        print(f"Processed volume shape: {volume.shape}")
        print(f"Processed volume range: {volume.min():.4f} to {volume.max():.4f}")
        
        # Apply spleen mask like training
        spleen_mask = mask > 0
        masked_volume = volume.copy()
        masked_volume[~spleen_mask] = 0
        
        print(f"Spleen voxels: {np.sum(spleen_mask):,}")
        print(f"Masked volume range: {masked_volume.min():.4f} to {masked_volume.max():.4f}")
        
        # Run through model exactly like training
        volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volume_tensor = volume_tensor.to(device)
        
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            direct_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        print(f"Direct model error: {direct_error:.8f}")
        
        if len(results) > 0:
            print(f"Streamlit method error: {results[0]:.8f}")
            if abs(direct_error - results[0]) < 1e-6:
                print("✅ MATCH: Direct and Streamlit methods identical")
            else:
                print("❌ MISMATCH: Different processing paths!")

if __name__ == "__main__":
    verify_training_consistency()
