import torch
import numpy as np
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed

def diagnose_threshold_issue():
    """Diagnose what's wrong with threshold detection"""
    print("🔬 DIAGNOSING THRESHOLD ISSUE")
    print("="*60)
    
    detector = Spleen3DAnomalyDetectorFixed("../models/best_spleen_3d_autoencoder.pth")
    
    # Test multiple volumes and get their actual errors
    errors = []
    volume_names = []
    
    print("Testing first 10 training volumes:")
    
    for i in range(min(10, len(detector.preprocessor.image_files))):
        result = detector.detect_anomaly_from_training_file(i, threshold=1.0)  # High threshold to see all errors
        
        if result:
            error = result['reconstruction_error']
            errors.append(error)
            volume_name = detector.preprocessor.image_files[i].name
            volume_names.append(volume_name)
            
            print(f"Volume {i+1}: {volume_name}")
            print(f"  Error: {error:.6f}")
    
    if errors:
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        print(f"\n📊 ERROR STATISTICS:")
        print(f"Mean Error: {mean_error:.6f}")
        print(f"Std Dev: {std_error:.6f}")
        print(f"Min Error: {min_error:.6f}")
        print(f"Max Error: {max_error:.6f}")
        
        # Calculate suggested thresholds
        conservative_threshold = max_error + std_error  # Above highest normal
        optimal_threshold = mean_error + 3*std_error    # 3-sigma rule
        
        print(f"\n🎯 SUGGESTED THRESHOLDS:")
        print(f"Conservative (0% false positives): {conservative_threshold:.6f}")
        print(f"Optimal (3-sigma rule): {optimal_threshold:.6f}")
        print(f"Your current threshold: 0.009000")
        
        # Test each suggested threshold
        print(f"\n🧪 TESTING SUGGESTED THRESHOLDS:")
        
        for threshold_name, threshold_value in [
            ("Conservative", conservative_threshold),
            ("Optimal", optimal_threshold),
            ("Current", 0.009000)
        ]:
            false_positives = sum(1 for error in errors if error > threshold_value)
            fp_rate = false_positives / len(errors) * 100
            
            print(f"{threshold_name} ({threshold_value:.6f}): {false_positives}/{len(errors)} FP ({fp_rate:.1f}%)")
    
    return errors, volume_names

def find_working_threshold():
    """Find a threshold that actually works"""
    print(f"\n🔍 FINDING WORKING THRESHOLD:")
    
    detector = Spleen3DAnomalyDetectorFixed("../models/best_spleen_3d_autoencoder.pth")
    
    # Test different thresholds
    thresholds_to_test = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    
    for threshold in thresholds_to_test:
        false_positives = 0
        total_tested = 0
        
        for i in range(min(10, len(detector.preprocessor.image_files))):
            result = detector.detect_anomaly_from_training_file(i, threshold=threshold)
            if result:
                if result['is_anomaly']:
                    false_positives += 1
                total_tested += 1
        
        fp_rate = false_positives / total_tested * 100 if total_tested > 0 else 0
        
        print(f"Threshold {threshold:.6f}: {false_positives}/{total_tested} FP ({fp_rate:.1f}%)")
        
        if fp_rate == 0:
            print(f"✅ WORKING THRESHOLD FOUND: {threshold:.6f}")
            return threshold
    
    print("❌ No working threshold found in tested range")
    return None

if __name__ == "__main__":
    errors, names = diagnose_threshold_issue()
    working_threshold = find_working_threshold()
    
    if working_threshold:
        print(f"\n🎯 RECOMMENDED ACTION:")
        print(f"Use threshold: {working_threshold:.6f} in your Streamlit demo")
    else:
        print(f"\n⚠️ ISSUE DETECTED:")
        print("Your model may need retraining or there's a data processing inconsistency")
