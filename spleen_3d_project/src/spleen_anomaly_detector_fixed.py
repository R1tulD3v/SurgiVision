import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from spleen_3d_model import Spleen3DAutoencoder
from spleen_preprocessing import SpleenDataPreprocessor

class Spleen3DAnomalyDetectorFixed:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        self.model = Spleen3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
        
        print(f"✅ Loaded model from {model_path}")
        print(f"Using device: {self.device}")
    
    def calculate_threshold_correctly(self):
        """Calculate threshold using EXACT same preprocessing as training"""
        print("Calculating threshold using training preprocessing...")
        
        normal_errors = []
        
        # Use first 10 training volumes with EXACT same preprocessing
        for i in range(min(10, len(self.preprocessor.image_files))):
            volume_path = self.preprocessor.image_files[i]
            mask_path = self.preprocessor.label_files[i]
            
            try:
                # Use EXACT same preprocessing as training
                volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                # Create spleen-only volume (EXACTLY like training dataset)
                spleen_mask = mask > 0
                masked_volume = volume.copy()
                masked_volume[~spleen_mask] = 0  # Zero out non-spleen regions
                
                volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                # Calculate reconstruction error
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    normal_errors.append(error)
                
                print(f"  Volume {i+1}: reconstruction error = {error:.6f}")
                
            except Exception as e:
                print(f"  Volume {i+1}: Error - {e}")
                continue
        
        if normal_errors:
            mean_error = np.mean(normal_errors)
            std_error = np.std(normal_errors)
            threshold = mean_error + 3.0 * std_error  # Increased to 3 sigma
            
            print(f"\nCorrected Threshold Calculation:")
            print(f"Normal errors: {[f'{e:.6f}' for e in normal_errors]}")
            print(f"Mean error: {mean_error:.6f}")
            print(f"Std error: {std_error:.6f}")
            print(f"Threshold (mean + 3*std): {threshold:.6f}")
            
            return threshold, normal_errors
        else:
            print("❌ Could not calculate threshold")
            return 0.02, []
    
    def detect_anomaly_from_training_file(self, volume_idx, threshold=None):
        """Test detection on a training file using exact preprocessing"""
        if threshold is None:
            threshold = 0.02
        
        if volume_idx >= len(self.preprocessor.image_files):
            print(f"❌ Index {volume_idx} out of range")
            return None
        
        volume_path = self.preprocessor.image_files[volume_idx]
        mask_path = self.preprocessor.label_files[volume_idx]
        
        print(f"Analyzing training volume {volume_idx}: {volume_path.name}")
        
        try:
            # Use EXACT same preprocessing as training
            volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
            if volume is None:
                return None
            
            # Create spleen-only volume (same as training)
            spleen_mask = mask > 0
            masked_volume = volume.copy()
            masked_volume[~spleen_mask] = 0
            
            volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            # Model inference
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            # Anomaly detection
            is_anomaly = reconstruction_error > threshold
            confidence = reconstruction_error / threshold if threshold > 0 else 0
            
            result = {
                'volume_idx': volume_idx,
                'file_path': str(volume_path),
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'reconstruction_error': reconstruction_error,
                'threshold': threshold,
                'spleen_voxels': np.sum(spleen_mask)
            }
            
            print(f"Reconstruction Error: {reconstruction_error:.6f}")
            print(f"Threshold: {threshold:.6f}")
            print(f"Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ NORMAL'}")
            print(f"Confidence: {confidence:.2f}x threshold")
            print(f"Spleen voxels: {np.sum(spleen_mask)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing volume {volume_idx}: {e}")
            return None
    
    def create_synthetic_anomaly(self, volume_idx):
        """Create synthetic anomaly by adding bright lesion to normal spleen"""
        if volume_idx >= len(self.preprocessor.image_files):
            return None
        
        volume_path = self.preprocessor.image_files[volume_idx]
        mask_path = self.preprocessor.label_files[volume_idx]
        
        # Load and preprocess normal volume
        volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
        if volume is None:
            return None
        
        # Create synthetic anomaly
        anomalous_volume = volume.copy()
        spleen_mask = mask > 0
        
        # Add bright lesion in spleen region
        spleen_coords = np.where(spleen_mask)
        if len(spleen_coords[0]) > 0:
            # Pick random location in spleen
            idx = np.random.randint(len(spleen_coords[0]))
            center_x, center_y, center_z = spleen_coords[0][idx], spleen_coords[1][idx], spleen_coords[2][idx]
            
            # Add spherical bright lesion
            for x in range(max(0, center_x-3), min(64, center_x+4)):
                for y in range(max(0, center_y-3), min(64, center_y+4)):
                    for z in range(max(0, center_z-3), min(64, center_z+4)):
                        if (x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2 <= 9:  # sphere
                            anomalous_volume[x, y, z] = min(1.0, anomalous_volume[x, y, z] + 0.4)
        
        return anomalous_volume, mask

def test_corrected_detection():
    """Test corrected anomaly detection"""
    print("=== Testing CORRECTED 3D Spleen Anomaly Detection ===")
    
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize corrected detector
    detector = Spleen3DAnomalyDetectorFixed(model_path)
    
    # Calculate corrected threshold
    threshold, normal_errors = detector.calculate_threshold_correctly()
    
    print(f"\n=== Testing Normal Volume (should be NORMAL) ===")
    result_normal = detector.detect_anomaly_from_training_file(0, threshold)
    
    print(f"\n=== Testing Another Normal Volume ===") 
    result_normal2 = detector.detect_anomaly_from_training_file(5, threshold)
    
    # Test synthetic anomaly
    print(f"\n=== Testing Synthetic Anomaly ===")
    anomalous_volume, mask = detector.create_synthetic_anomaly(0)
    if anomalous_volume is not None:
        # Test synthetic anomaly
        spleen_mask = mask > 0
        masked_anomaly = anomalous_volume.copy()
        masked_anomaly[~spleen_mask] = 0
        
        volume_tensor = torch.FloatTensor(masked_anomaly[np.newaxis, np.newaxis, ...]).to(detector.device)
        
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        is_anomaly = error > threshold
        confidence = error / threshold
        
        print(f"Synthetic anomaly reconstruction error: {error:.6f}")
        print(f"Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ NORMAL'}")
        print(f"Confidence: {confidence:.2f}x threshold")
    
    # Summary
    if result_normal and result_normal2:
        normal_fps = sum([1 for r in [result_normal, result_normal2] if r['is_anomaly']])
        print(f"\n📊 SUMMARY:")
        print(f"Normal volumes tested: 2")
        print(f"False positives: {normal_fps}")
        print(f"Corrected threshold: {threshold:.6f}")
        
        if normal_fps == 0:
            print("✅ FALSE POSITIVE ISSUE FIXED!")
        else:
            print("⚠️  Still some false positives - may need threshold adjustment")

if __name__ == "__main__":
    test_corrected_detection()
