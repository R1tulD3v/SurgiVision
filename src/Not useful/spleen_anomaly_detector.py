import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from spleen_3d_model import Spleen3DAutoencoder
from spleen_preprocessing import SpleenDataPreprocessor

class Spleen3DAnomalyDetector:
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
    
    def calculate_threshold(self):
        """Calculate anomaly threshold from training data"""
        print("Calculating anomaly threshold from normal spleen volumes...")
        
        normal_errors = []
        
        # Test on first 10 training volumes (known normal)
        for i in range(min(10, len(self.preprocessor.image_files))):
            volume_path = self.preprocessor.image_files[i]
            mask_path = self.preprocessor.label_files[i]
            
            try:
                # Preprocess
                volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                # Create spleen-only volume (normal tissue)
                spleen_volume = volume * (mask > 0)
                volume_tensor = torch.FloatTensor(spleen_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
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
            threshold = mean_error + 2.5 * std_error  # 2.5 sigma threshold
            
            print(f"\nThreshold Calculation:")
            print(f"Normal errors: {normal_errors}")
            print(f"Mean error: {mean_error:.6f}")
            print(f"Std error: {std_error:.6f}")
            print(f"Threshold (mean + 2.5*std): {threshold:.6f}")
            
            return threshold, normal_errors
        else:
            print("❌ Could not calculate threshold - no valid normal volumes")
            return 0.02, []  # Default threshold
    
    def detect_anomaly_from_file(self, ct_path, threshold=None):
        """Detect anomaly in spleen CT file"""
        if threshold is None:
            threshold = 0.02  # Default threshold
        
        print(f"Analyzing: {Path(ct_path).name}")
        
        try:
            # Load CT volume
            ct_img = nib.load(ct_path)
            ct_volume = ct_img.get_fdata()
            
            print(f"Original CT shape: {ct_volume.shape}")
            
            # For detection, we need to segment spleen first (simplified approach)
            # In real application, you'd use spleen segmentation model
            # For now, we'll process the whole cropped region
            
            # Basic preprocessing
            processed_volume = self.preprocess_for_detection(ct_volume)
            
            if processed_volume is None:
                return None
            
            volume_tensor = torch.FloatTensor(processed_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            # Model inference
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            # Anomaly detection
            is_anomaly = reconstruction_error > threshold
            confidence = reconstruction_error / threshold if threshold > 0 else 0
            
            # Generate error map
            error_volume = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
            
            result = {
                'file_path': ct_path,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'reconstruction_error': reconstruction_error,
                'threshold': threshold,
                'error_volume': error_volume,
                'processed_volume': processed_volume,
                'original_shape': ct_volume.shape
            }
            
            print(f"Reconstruction Error: {reconstruction_error:.6f}")
            print(f"Threshold: {threshold:.6f}")
            print(f"Result: {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ NORMAL'}")
            print(f"Confidence: {confidence:.2f}x threshold")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {ct_path}: {e}")
            return None
    
    def preprocess_for_detection(self, volume):
        """Simplified preprocessing for detection"""
        try:
            # Basic intensity windowing for spleen
            volume_windowed = np.clip(volume, -200, 300)
            volume_norm = (volume_windowed + 200) / 500
            
            # Find approximate spleen region (center-right abdomen)
            z_mid = volume.shape[2] // 2
            y_center = volume.shape[1] // 2
            x_center = int(volume.shape[0] * 0.4)  # Right side
            
            # Crop around likely spleen location
            crop_size = 100
            x_min = max(0, x_center - crop_size//2)
            x_max = min(volume.shape[0], x_center + crop_size//2)
            y_min = max(0, y_center - crop_size//2)
            y_max = min(volume.shape[1], y_center + crop_size//2)
            z_min = max(0, z_mid - 30)
            z_max = min(volume.shape[2], z_mid + 30)
            
            cropped = volume_norm[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Resize to model input size
            from scipy import ndimage
            zoom_factors = [64/cropped.shape[i] for i in range(3)]
            resized = ndimage.zoom(cropped, zoom_factors, order=1)
            
            return resized
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

def test_anomaly_detection():
    """Test anomaly detection system"""
    print("=== Testing 3D Spleen Anomaly Detection ===")
    
    # Check if trained model exists
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please complete training first!")
        return
    
    # Initialize detector
    detector = Spleen3DAnomalyDetector(model_path)
    
    # Calculate threshold
    threshold, normal_errors = detector.calculate_threshold()
    
    # Test on a training volume (should be normal)
    test_file = "../data/Task09_Spleen/imagesTr/spleen_10.nii.gz"
    if Path(test_file).exists():
        print(f"\n=== Testing on Known Normal Volume ===")
        result = detector.detect_anomaly_from_file(test_file, threshold)
        
        if result:
            print("✅ Detection test completed!")
        else:
            print("❌ Detection test failed")
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    test_anomaly_detection()
