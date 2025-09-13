import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from spleen_preprocessing import SpleenDataPreprocessor
from spleen_3d_model import Spleen3DAutoencoder

class MedicalAnomalyCreator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def create_spleen_cyst(self, base_volume, spleen_mask):
        """Create hypodense cyst (dark region) - common spleen pathology"""
        anomalous_volume = base_volume.copy()
        spleen_coords = np.where(spleen_mask)
        
        if len(spleen_coords[0]) > 50:
            # Large cyst (hypodense - darker than normal tissue)
            center_idx = len(spleen_coords[0]) // 3
            cx, cy, cz = spleen_coords[0][center_idx], spleen_coords[1][center_idx], spleen_coords[2][center_idx]
            
            # Create large spherical cyst (12x12x12 voxels)
            for x in range(max(0, cx-6), min(64, cx+7)):
                for y in range(max(0, cy-6), min(64, cy+7)):
                    for z in range(max(0, cz-6), min(64, cz+7)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 36:  # Sphere radius 6
                            # Make it very hypodense (much darker)
                            anomalous_volume[x, y, z] = max(0.0, anomalous_volume[x, y, z] - 0.7)
                            
        return anomalous_volume, "Large spleen cyst (hypodense lesion)"
    
    def create_spleen_infarct(self, base_volume, spleen_mask):
        """Create spleen infarct (wedge-shaped hypodense area)"""
        anomalous_volume = base_volume.copy()
        spleen_coords = np.where(spleen_mask)
        
        if len(spleen_coords[0]) > 100:
            # Wedge-shaped infarct
            center_idx = len(spleen_coords[0]) // 2
            cx, cy, cz = spleen_coords[0][center_idx], spleen_coords[1][center_idx], spleen_coords[2][center_idx]
            
            # Create wedge shape (triangular infarct)
            for x in range(max(0, cx-8), min(64, cx+9)):
                for y in range(max(0, cy-8), min(64, cy+9)):
                    for z in range(max(0, cz-4), min(64, cz+5)):
                        # Wedge condition
                        if (x-cx) >= 0 and (y-cy)**2 + (z-cz)**2 <= (x-cx+1)**2:
                            anomalous_volume[x, y, z] = max(0.0, anomalous_volume[x, y, z] - 0.6)
                            
        return anomalous_volume, "Spleen infarct (wedge-shaped hypodense region)"
    
    def create_spleen_laceration(self, base_volume, spleen_mask):
        """Create spleen laceration (linear hypodense tear)"""
        anomalous_volume = base_volume.copy()
        spleen_coords = np.where(spleen_mask)
        
        if len(spleen_coords[0]) > 80:
            # Linear laceration across spleen
            center_idx = len(spleen_coords[0]) // 2
            cx, cy, cz = spleen_coords[0][center_idx], spleen_coords[1][center_idx], spleen_coords[2][center_idx]
            
            # Create linear tear
            for i in range(-10, 11):
                x = cx + i
                for j in range(-2, 3):
                    y = cy + j
                    for k in range(-2, 3):
                        z = cz + k
                        if 0 <= x < 64 and 0 <= y < 64 and 0 <= z < 64:
                            if spleen_mask[x, y, z]:  # Only in spleen
                                anomalous_volume[x, y, z] = max(0.0, anomalous_volume[x, y, z] - 0.8)
                                
        return anomalous_volume, "Spleen laceration (linear hypodense tear)"
    
    def create_hyperdense_mass(self, base_volume, spleen_mask):
        """Create hyperdense mass (very bright lesion)"""
        anomalous_volume = base_volume.copy()
        spleen_coords = np.where(spleen_mask)
        
        if len(spleen_coords[0]) > 60:
            # Very bright mass
            center_idx = len(spleen_coords[0]) // 4
            cx, cy, cz = spleen_coords[0][center_idx], spleen_coords[1][center_idx], spleen_coords[2][center_idx]
            
            # Large bright mass (10x10x10)
            for x in range(max(0, cx-5), min(64, cx+6)):
                for y in range(max(0, cy-5), min(64, cy+6)):
                    for z in range(max(0, cz-5), min(64, cz+6)):
                        distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                        if distance <= 25:  # Sphere
                            # Make it VERY bright
                            anomalous_volume[x, y, z] = min(1.0, anomalous_volume[x, y, z] + 0.8)
                            
        return anomalous_volume, "Hyperdense mass (very bright lesion)"
    
    def create_multiple_metastases(self, base_volume, spleen_mask):
        """Create multiple small metastases"""
        anomalous_volume = base_volume.copy()
        spleen_coords = np.where(spleen_mask)
        
        if len(spleen_coords[0]) > 200:
            # 5-6 small metastases
            np.random.seed(123)  # Reproducible
            for met in range(5):
                # Random location in spleen
                idx = np.random.randint(len(spleen_coords[0]))
                cx, cy, cz = spleen_coords[0][idx], spleen_coords[1][idx], spleen_coords[2][idx]
                
                # Small bright lesion
                for x in range(max(0, cx-3), min(64, cx+4)):
                    for y in range(max(0, cy-3), min(64, cy+4)):
                        for z in range(max(0, cz-3), min(64, cz+4)):
                            distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
                            if distance <= 9:  # Small sphere
                                anomalous_volume[x, y, z] = min(1.0, anomalous_volume[x, y, z] + 0.7)
                                
        return anomalous_volume, "Multiple spleen metastases"
    
    def create_all_pathologies(self, base_index=5):
        """Create comprehensive pathological test cases"""
        if base_index >= len(self.preprocessor.image_files):
            return []
        
        base_scan = self.preprocessor.image_files[base_index]
        mask_scan = self.preprocessor.label_files[base_index]
        
        volume, mask = self.preprocessor.preprocess_spleen_volume(base_scan, mask_scan)
        if volume is None:
            return []
        
        spleen_mask = mask > 0
        pathological_cases = []
        
        # Create each type of pathology
        pathology_creators = [
            self.create_spleen_cyst,
            self.create_spleen_infarct, 
            self.create_spleen_laceration,
            self.create_hyperdense_mass,
            self.create_multiple_metastases
        ]
        
        for creator in pathology_creators:
            try:
                anomalous_volume, description = creator(volume, spleen_mask)
                pathological_cases.append({
                    'volume': anomalous_volume,
                    'mask': mask,
                    'description': description,
                    'expected': 'anomaly'
                })
            except Exception as e:
                print(f"Failed to create pathology: {e}")
                continue
        
        return pathological_cases

class EnhancedAnomalyTester:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = Spleen3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor
        self.preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
        self.anomaly_creator = MedicalAnomalyCreator(self.preprocessor)
        
        print(f"✅ Enhanced tester loaded on {self.device}")
    
    def calculate_adaptive_threshold(self):
        """Calculate threshold with multiple sensitivity levels"""
        print("Calculating adaptive thresholds...")
        
        normal_errors = []
        for i in range(min(15, len(self.preprocessor.image_files))):
            try:
                volume_path = self.preprocessor.image_files[i]
                mask_path = self.preprocessor.label_files[i]
                
                volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                if volume is None:
                    continue
                
                spleen_mask = mask > 0
                masked_volume = volume.copy()
                masked_volume[~spleen_mask] = 0
                
                volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                
                with torch.no_grad():
                    reconstructed = self.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    normal_errors.append(error)
                
            except:
                continue
        
        if normal_errors:
            mean_error = np.mean(normal_errors)
            std_error = np.std(normal_errors)
            
            thresholds = {
                'conservative': mean_error + 4.0 * std_error,  # Very specific
                'balanced': mean_error + 3.0 * std_error,      # Balanced
                'sensitive': mean_error + 2.0 * std_error,     # More sensitive
            }
            
            print(f"Normal errors range: {min(normal_errors):.6f} - {max(normal_errors):.6f}")
            print(f"Mean: {mean_error:.6f}, Std: {std_error:.6f}")
            print(f"Thresholds - Conservative: {thresholds['conservative']:.6f}")
            print(f"           - Balanced: {thresholds['balanced']:.6f}")
            print(f"           - Sensitive: {thresholds['sensitive']:.6f}")
            
            return thresholds, normal_errors
        
        return {'balanced': 0.015}, []
    
    def test_pathological_cases(self, threshold):
        """Test enhanced pathological cases"""
        print(f"\n=== Testing Enhanced Pathological Cases ===")
        print(f"Using threshold: {threshold:.6f}")
        
        # Create realistic pathologies
        pathological_cases = self.anomaly_creator.create_all_pathologies()
        
        results = []
        correct_detections = 0
        
        for i, case in enumerate(pathological_cases):
            print(f"\n🩺 Case {i+1}: {case['description']}")
            
            # Test pathological case
            spleen_mask = case['mask'] > 0
            masked_volume = case['volume'].copy()
            masked_volume[~spleen_mask] = 0
            
            volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...]).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(volume_tensor)
                error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            is_anomaly = error > threshold
            confidence = error / threshold
            
            if is_anomaly:
                correct_detections += 1
                status = "✅ DETECTED"
            else:
                status = "❌ MISSED"
            
            print(f"   Reconstruction Error: {error:.6f}")
            print(f"   Result: {status}")
            print(f"   Confidence: {confidence:.2f}x threshold")
            
            results.append({
                'description': case['description'],
                'error': error,
                'detected': is_anomaly,
                'confidence': confidence
            })
        
        detection_rate = correct_detections / len(pathological_cases) * 100 if pathological_cases else 0
        print(f"\n📊 Pathological Detection Rate: {correct_detections}/{len(pathological_cases)} ({detection_rate:.1f}%)")
        
        return results, detection_rate
    
    def comprehensive_evaluation(self):
        """Run comprehensive evaluation with multiple thresholds"""
        print("=" * 60)
        print("🔬 COMPREHENSIVE ENHANCED ANOMALY EVALUATION")
        print("=" * 60)
        
        # Calculate adaptive thresholds
        thresholds, normal_errors = self.calculate_adaptive_threshold()
        
        best_threshold = None
        best_score = 0
        
        for threshold_name, threshold_value in thresholds.items():
            print(f"\n{'='*50}")
            print(f"🎯 TESTING WITH {threshold_name.upper()} THRESHOLD: {threshold_value:.6f}")
            print(f"{'='*50}")
            
            # Test pathological cases
            results, detection_rate = self.test_pathological_cases(threshold_value)
            
            # Calculate false positive rate on normals (should be 0%)
            normal_fps = 0
            normal_tested = 0
            for i in range(min(5, len(self.preprocessor.image_files))):
                try:
                    volume_path = self.preprocessor.image_files[i]
                    mask_path = self.preprocessor.label_files[i]
                    
                    volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                    if volume is None:
                        continue
                        
                    spleen_mask = mask > 0
                    masked_volume = volume.copy()
                    masked_volume[~spleen_mask] = 0
                    
                    volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...]).to(self.device)
                    
                    with torch.no_grad():
                        reconstructed = self.model(volume_tensor)
                        error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    
                    if error > threshold_value:
                        normal_fps += 1
                    normal_tested += 1
                    
                except:
                    continue
            
            fp_rate = (normal_fps / normal_tested * 100) if normal_tested > 0 else 0
            
            # Overall score (balance between detection rate and low false positives)
            score = detection_rate * (1 - fp_rate/100)
            
            print(f"\n📊 RESULTS FOR {threshold_name.upper()} THRESHOLD:")
            print(f"   Anomaly Detection Rate: {detection_rate:.1f}%")
            print(f"   False Positive Rate: {fp_rate:.1f}%")
            print(f"   Overall Score: {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold_value
        
        print(f"\n🏆 RECOMMENDED THRESHOLD: {best_threshold:.6f}")
        print(f"🎯 EXPECTED PERFORMANCE: {best_score:.1f}% accuracy")
        
        return best_threshold

def main():
    """Run enhanced anomaly testing"""
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if not Path(model_path).exists():
        print("❌ Model not found!")
        return
    
    # Run enhanced testing
    tester = EnhancedAnomalyTester(model_path)
    recommended_threshold = tester.comprehensive_evaluation()
    
    print(f"\n✅ Enhanced testing completed!")
    print(f"💡 Use threshold {recommended_threshold:.6f} for hackathon demo")
    
    # Save recommended threshold
    with open("../models/recommended_threshold.txt", "w") as f:
        f.write(f"{recommended_threshold:.6f}")
    
    print("🔧 Threshold saved to models/recommended_threshold.txt")

if __name__ == "__main__":
    main()
