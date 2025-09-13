import torch
import numpy as np
from pathlib import Path
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed
from enhanced_anomaly_creator import MedicalAnomalyCreator

class ModelValidator:
    def __init__(self):
        self.detector = Spleen3DAnomalyDetectorFixed("../models/best_spleen_3d_autoencoder.pth")
        self.anomaly_creator = MedicalAnomalyCreator(self.detector.preprocessor)
        
    def test_known_normals(self, threshold=0.008756):
        """Test on training data that should be NORMAL"""
        print("🧪 TESTING ON KNOWN NORMAL TRAINING VOLUMES")
        print("="*60)
        
        normal_results = []
        false_positives = 0
        
        # Test first 10 training volumes (should all be NORMAL)
        for i in range(min(10, len(self.detector.preprocessor.image_files))):
            print(f"\nTesting Volume {i+1}:")
            
            result = self.detector.detect_anomaly_from_training_file(i, threshold)
            
            if result:
                normal_results.append(result)
                
                if result['is_anomaly']:
                    false_positives += 1
                    status = "❌ FALSE POSITIVE"
                else:
                    status = "✅ CORRECT (Normal)"
                
                print(f"  File: {self.detector.preprocessor.image_files[i].name}")
                print(f"  Error: {result['reconstruction_error']:.6f}")
                print(f"  Threshold: {threshold:.6f}")
                print(f"  Confidence: {result['confidence']:.3f}x")
                print(f"  Result: {status}")
        
        fp_rate = false_positives / len(normal_results) * 100 if normal_results else 0
        print(f"\n📊 NORMAL VOLUME RESULTS:")
        print(f"False Positive Rate: {false_positives}/{len(normal_results)} ({fp_rate:.1f}%)")
        
        return normal_results, fp_rate
    
    def test_synthetic_anomalies(self, threshold=0.008756):
        """Test on synthetic anomalies that should be DETECTED"""
        print("\n🦠 TESTING ON SYNTHETIC ANOMALIES")
        print("="*60)
        
        pathological_cases = self.anomaly_creator.create_all_pathologies(base_index=5)
        
        detected_count = 0
        anomaly_results = []
        
        for i, case in enumerate(pathological_cases):
            print(f"\nTesting Synthetic Case {i+1}: {case['description']}")
            
            # Prepare volume for analysis
            spleen_mask = case['mask'] > 0
            masked_volume = case['volume'].copy()
            masked_volume[~spleen_mask] = 0
            
            volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            volume_tensor = volume_tensor.to(device)
            
            # Run detection
            with torch.no_grad():
                reconstructed = self.detector.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            
            is_anomaly = reconstruction_error > threshold
            confidence = reconstruction_error / threshold
            
            if is_anomaly:
                detected_count += 1
                status = "✅ CORRECTLY DETECTED"
            else:
                status = "❌ FALSE NEGATIVE (MISSED)"
            
            print(f"  Error: {reconstruction_error:.6f}")
            print(f"  Threshold: {threshold:.6f}")
            print(f"  Confidence: {confidence:.3f}x")
            print(f"  Result: {status}")
            
            anomaly_results.append({
                'description': case['description'],
                'error': reconstruction_error,
                'detected': is_anomaly,
                'confidence': confidence
            })
        
        detection_rate = detected_count / len(pathological_cases) * 100 if pathological_cases else 0
        print(f"\n📊 ANOMALY DETECTION RESULTS:")
        print(f"Detection Rate: {detected_count}/{len(pathological_cases)} ({detection_rate:.1f}%)")
        
        return anomaly_results, detection_rate
    
    def analyze_error_distributions(self):
        """Analyze reconstruction error distributions"""
        print("\n📈 ANALYZING ERROR DISTRIBUTIONS")
        print("="*60)
        
        normal_errors = []
        anomaly_errors = []
        
        # Get normal errors
        for i in range(min(15, len(self.detector.preprocessor.image_files))):
            try:
                result = self.detector.detect_anomaly_from_training_file(i, 0.1)  # High threshold
                if result:
                    normal_errors.append(result['reconstruction_error'])
            except:
                continue
        
        # Get anomaly errors
        pathological_cases = self.anomaly_creator.create_all_pathologies(base_index=8)
        for case in pathological_cases:
            try:
                spleen_mask = case['mask'] > 0
                masked_volume = case['volume'].copy()
                masked_volume[~spleen_mask] = 0
                
                volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                volume_tensor = volume_tensor.to(device)
                
                with torch.no_grad():
                    reconstructed = self.detector.model(volume_tensor)
                    error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    anomaly_errors.append(error)
            except:
                continue
        
        if normal_errors and anomaly_errors:
            normal_mean = np.mean(normal_errors)
            normal_std = np.std(normal_errors)
            anomaly_mean = np.mean(anomaly_errors)
            anomaly_std = np.std(anomaly_errors)
            
            print(f"Normal Errors:")
            print(f"  Mean: {normal_mean:.6f}")
            print(f"  Std: {normal_std:.6f}")
            print(f"  Range: {min(normal_errors):.6f} - {max(normal_errors):.6f}")
            
            print(f"\nAnomaly Errors:")
            print(f"  Mean: {anomaly_mean:.6f}")
            print(f"  Std: {anomaly_std:.6f}")
            print(f"  Range: {min(anomaly_errors):.6f} - {max(anomaly_errors):.6f}")
            
            # Calculate separation
            separation = (anomaly_mean - normal_mean) / (normal_std + anomaly_std)
            print(f"\nSeparation Metric: {separation:.3f}")
            
            if separation > 2.0:
                print("✅ EXCELLENT separation between normal and anomaly errors")
            elif separation > 1.0:
                print("⚠️  MODERATE separation - threshold tuning needed")
            else:
                print("❌ POOR separation - model may need retraining")
            
            # Suggest optimal threshold
            optimal_threshold = normal_mean + 2.5 * normal_std
            print(f"\n💡 SUGGESTED THRESHOLD: {optimal_threshold:.6f}")
            
            return normal_errors, anomaly_errors, optimal_threshold
        
        return [], [], 0.008756
    
    def test_threshold_sensitivity(self):
        """Test different threshold values"""
        print("\n🎛️  THRESHOLD SENSITIVITY ANALYSIS")
        print("="*60)
        
        thresholds = [0.005, 0.008, 0.010, 0.015, 0.020, 0.025]
        
        for threshold in thresholds:
            print(f"\nTesting Threshold: {threshold:.6f}")
            
            # Quick test on 5 normal + 3 synthetic anomalies
            normal_fps = 0
            for i in range(5):
                try:
                    result = self.detector.detect_anomaly_from_training_file(i, threshold)
                    if result and result['is_anomaly']:
                        normal_fps += 1
                except:
                    continue
            
            # Test synthetic anomalies
            pathological_cases = self.anomaly_creator.create_all_pathologies(base_index=6)
            anomaly_detections = 0
            for case in pathological_cases[:3]:
                try:
                    spleen_mask = case['mask'] > 0
                    masked_volume = case['volume'].copy()
                    masked_volume[~spleen_mask] = 0
                    
                    volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    volume_tensor = volume_tensor.to(device)
                    
                    with torch.no_grad():
                        reconstructed = self.detector.model(volume_tensor)
                        error = torch.mean((volume_tensor - reconstructed) ** 2).item()
                    
                    if error > threshold:
                        anomaly_detections += 1
                except:
                    continue
            
            fp_rate = normal_fps / 5 * 100
            detection_rate = anomaly_detections / 3 * 100
            
            print(f"  False Positive Rate: {fp_rate:.1f}%")
            print(f"  Anomaly Detection Rate: {detection_rate:.1f}%")
            print(f"  Balance Score: {detection_rate - fp_rate:.1f}")

def main():
    """Run comprehensive model validation"""
    print("🔬 COMPREHENSIVE MODEL VALIDATION")
    print("="*60)
    
    try:
        validator = ModelValidator()
        
        # Test 1: Known normal volumes
        normal_results, fp_rate = validator.test_known_normals()
        
        # Test 2: Synthetic anomalies  
        anomaly_results, detection_rate = validator.test_synthetic_anomalies()
        
        # Test 3: Error distribution analysis
        normal_errors, anomaly_errors, optimal_threshold = validator.analyze_error_distributions()
        
        # Test 4: Threshold sensitivity
        validator.test_threshold_sensitivity()
        
        # Summary
        print(f"\n🎯 VALIDATION SUMMARY:")
        print("="*60)
        print(f"False Positive Rate: {fp_rate:.1f}% (should be <10%)")
        print(f"Anomaly Detection Rate: {detection_rate:.1f}% (should be >80%)")
        print(f"Recommended Threshold: {optimal_threshold:.6f}")
        
        if fp_rate < 10 and detection_rate > 80:
            print("✅ MODEL IS WORKING CORRECTLY!")
        elif fp_rate > 50:
            print("❌ MODEL FLAGGING TOO MANY FALSE POSITIVES")
            print("💡 Try increasing threshold or check preprocessing")
        elif detection_rate < 50:
            print("❌ MODEL MISSING TOO MANY ANOMALIES") 
            print("💡 Try decreasing threshold or retrain model")
        else:
            print("⚠️  MODEL NEEDS THRESHOLD TUNING")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
