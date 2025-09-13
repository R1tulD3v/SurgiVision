import torch
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed

class RobustImageProcessor:
    def __init__(self, model_path):
        self.detector = Spleen3DAnomalyDetectorFixed(model_path)
        
        # Calculate statistics from training data for normalization
        self.training_stats = self.calculate_training_statistics()
        
    def calculate_training_statistics(self):
        """Calculate mean/std from training data for normalization"""
        print("📊 Calculating training data statistics...")
        
        all_values = []
        
        # Sample from multiple training volumes
        for i in range(min(5, len(self.detector.preprocessor.image_files))):
            volume_path = self.detector.preprocessor.image_files[i]
            mask_path = self.detector.preprocessor.label_files[i]
            
            volume, mask = self.detector.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
            if volume is not None:
                # Only use spleen regions
                spleen_mask = mask > 0
                spleen_values = volume[spleen_mask]
                all_values.extend(spleen_values.flatten())
        
        if all_values:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            print(f"Training data - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            return {'mean': mean_val, 'std': std_val}
        else:
            return {'mean': 0.3, 'std': 0.1}  # Default values
    
    def process_external_2d_image(self, image_file):
        """Process external 2D image with domain adaptation"""
        try:
            # Load image
            image = Image.open(image_file)
            if image.mode != 'L':
                image = image.convert('L')
            
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Resize to 64x64
            img_resized = cv2.resize(img_array, (64, 64))
            
            # Create 3D volume by replicating slices
            volume_3d = np.stack([img_resized] * 64, axis=2)
            
            # CRITICAL: Normalize to match training data distribution
            current_mean = np.mean(volume_3d[volume_3d > 0])
            current_std = np.std(volume_3d[volume_3d > 0])
            
            if current_std > 0:
                # Standardize and rescale to match training data
                volume_normalized = (volume_3d - current_mean) / current_std
                volume_final = volume_normalized * self.training_stats['std'] + self.training_stats['mean']
                
                # Clip to reasonable range
                volume_final = np.clip(volume_final, 0, 1)
            else:
                # Fallback for uniform images
                volume_final = np.full_like(volume_3d, self.training_stats['mean'])
            
            return volume_final
            
        except Exception as e:
            print(f"Error processing 2D image: {e}")
            return None
    
    def process_external_3d_nifti(self, nifti_file):
        """Process external NIfTI with domain adaptation"""
        try:
            import tempfile
            import nibabel as nib
            import os
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
                tmp_file.write(nifti_file.getvalue())
                temp_path = tmp_file.name
            
            # Load NIfTI
            nii_img = nib.load(temp_path)
            volume_data = nii_img.get_fdata()
            
            # Basic preprocessing (similar to training)
            volume_windowed = np.clip(volume_data, -200, 300)
            volume_norm = (volume_windowed + 200) / 500
            
            # Extract center region
            center_x, center_y, center_z = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
            crop_size = 80
            
            x_start = max(0, center_x - crop_size//2)
            x_end = min(volume_norm.shape[0], center_x + crop_size//2)
            y_start = max(0, center_y - crop_size//2) 
            y_end = min(volume_norm.shape[1], center_y + crop_size//2)
            z_start = max(0, center_z - 20)
            z_end = min(volume_norm.shape[2], center_z + 20)
            
            cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Resize to 64x64x64
            zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
            resized_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
            
            # CRITICAL: Domain adaptation normalization
            current_mean = np.mean(resized_volume[resized_volume > 0.1])
            current_std = np.std(resized_volume[resized_volume > 0.1])
            
            if current_std > 0 and not np.isnan(current_mean):
                # Normalize to match training distribution
                volume_standardized = (resized_volume - current_mean) / current_std
                volume_final = volume_standardized * self.training_stats['std'] + self.training_stats['mean']
                volume_final = np.clip(volume_final, 0, 1)
            else:
                # Fallback
                volume_final = resized_volume
            
            # Clean up
            os.unlink(temp_path)
            
            return volume_final
            
        except Exception as e:
            print(f"Error processing 3D NIfTI: {e}")
            return None
    
    def detect_with_domain_adaptation(self, processed_volume, adaptive_threshold=True):
        """Run detection with optional adaptive thresholding"""
        if processed_volume is None:
            return None
        
        # Create tensor
        volume_tensor = torch.FloatTensor(processed_volume[np.newaxis, np.newaxis, ...])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volume_tensor = volume_tensor.to(device)
        
        # Run model
        with torch.no_grad():
            reconstructed = self.detector.model(volume_tensor)
            reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        # Use adaptive threshold based on image characteristics
        if adaptive_threshold:
            # For external images, use higher threshold
            base_threshold = 0.009551  # Recommended from validation
            
            # Check if this looks like medical data
            volume_characteristics = {
                'mean_intensity': np.mean(processed_volume),
                'std_intensity': np.std(processed_volume),
                'non_zero_ratio': np.sum(processed_volume > 0.1) / processed_volume.size
            }
            
            # Adjust threshold based on how "medical-like" the image appears
            if (0.2 < volume_characteristics['mean_intensity'] < 0.6 and 
                volume_characteristics['non_zero_ratio'] > 0.3):
                # Looks medical-like, use standard threshold
                threshold = base_threshold
            else:
                # Doesn't look like training data, use higher threshold
                threshold = base_threshold * 3.0  # More lenient
        else:
            threshold = 0.009551
        
        # Determine anomaly
        is_anomaly = reconstruction_error > threshold
        confidence = reconstruction_error / threshold
        error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reconstruction_error': reconstruction_error,
            'threshold': threshold,
            'processed_volume': processed_volume,
            'error_map': error_map,
            'domain_adapted': True
        }

def test_robust_processing():
    """Test the robust processing"""
    print("🧪 TESTING ROBUST DOMAIN-ADAPTIVE PROCESSING")
    print("="*60)
    
    processor = RobustImageProcessor("../models/best_spleen_3d_autoencoder.pth")
    
    # Test 1: Create test patterns that should be normal
    print("\n📋 Test 1: Synthetic Normal Pattern")
    normal_pattern = np.full((64, 64, 64), processor.training_stats['mean'])
    result1 = processor.detect_with_domain_adaptation(normal_pattern)
    
    if result1:
        print(f"Error: {result1['reconstruction_error']:.6f}")
        print(f"Threshold: {result1['threshold']:.6f}")  
        print(f"Result: {'ANOMALY' if result1['is_anomaly'] else 'NORMAL'}")
    
    # Test 2: Pattern with training-like statistics
    print("\n📋 Test 2: Training-like Pattern")
    training_like = np.random.normal(
        processor.training_stats['mean'], 
        processor.training_stats['std'], 
        (64, 64, 64)
    )
    training_like = np.clip(training_like, 0, 1)
    result2 = processor.detect_with_domain_adaptation(training_like)
    
    if result2:
        print(f"Error: {result2['reconstruction_error']:.6f}")
        print(f"Threshold: {result2['threshold']:.6f}")
        print(f"Result: {'ANOMALY' if result2['is_anomaly'] else 'NORMAL'}")
    
    print(f"\n💡 Training stats - Mean: {processor.training_stats['mean']:.4f}, Std: {processor.training_stats['std']:.4f}")

if __name__ == "__main__":
    test_robust_processing()
