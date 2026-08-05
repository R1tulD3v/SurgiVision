from pathlib import Path

import numpy as np
import torch

import config
import inference
from spleen_3d_model import Spleen3DAutoencoder
from spleen_preprocessing import SpleenDataPreprocessor


class Spleen3DAnomalyDetectorFixed:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained model
        self.model = Spleen3DAutoencoder()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = SpleenDataPreprocessor(config.DATA_ROOT)

        print(f"✅ Loaded model from {model_path}")
        print(f"Using device: {self.device}")

    def calibrate_threshold(self, indices=None, sigma=None):
        """Calibrate the decision threshold on a set of *normal* volumes.

        threshold = mean + sigma * std of reconstruction errors over ``indices``.

        Pass a held-out set (e.g. the validation split from data_splits) to
        avoid leakage. If ``indices`` is None, the first 10 volumes are used for
        backward compatibility -- note those overlap the training set, which is
        optimistic; prefer passing validation indices.
        """
        if sigma is None:
            sigma = config.THRESHOLD_SIGMA
        n = len(self.preprocessor.image_files)
        if indices is None:
            indices = list(range(min(10, n)))

        normal_errors = []
        for i in indices:
            if i < 0 or i >= n:
                continue
            volume_path = self.preprocessor.image_files[i]
            mask_path = self.preprocessor.label_files[i]
            try:
                volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                if volume is None:
                    continue
                masked_volume = inference.apply_spleen_mask(volume, mask)
                error = inference.reconstruction_error(self.model, masked_volume, self.device)
                normal_errors.append(error)
                print(f"  Volume {i}: reconstruction error = {error:.6f}")
            except Exception as e:
                print(f"  Volume {i}: Error - {e}")
                continue

        if normal_errors:
            mean_error = float(np.mean(normal_errors))
            std_error = float(np.std(normal_errors))
            threshold = mean_error + sigma * std_error

            print(f"\nThreshold calibration on {len(normal_errors)} normal volumes:")
            print(f"Mean error: {mean_error:.6f} | Std: {std_error:.6f}")
            print(f"Threshold (mean + {sigma}*std): {threshold:.6f}")
            return threshold, normal_errors

        print("❌ Could not calibrate threshold; using config default")
        return config.DEFAULT_THRESHOLD, []

    def calculate_threshold_correctly(self):
        """Backward-compatible wrapper that calibrates on the first 10 volumes.

        Deprecated: those volumes overlap the training set. Prefer
        ``calibrate_threshold(validation_indices)``.
        """
        return self.calibrate_threshold(indices=None)

    def detect_anomaly_from_training_file(self, volume_idx, threshold=None):
        """Detect an anomaly on a dataset volume using the training preprocessing."""
        if threshold is None:
            threshold = config.DEFAULT_THRESHOLD

        if volume_idx >= len(self.preprocessor.image_files):
            print(f"❌ Index {volume_idx} out of range")
            return None

        volume_path = self.preprocessor.image_files[volume_idx]
        mask_path = self.preprocessor.label_files[volume_idx]

        print(f"Analyzing volume {volume_idx}: {volume_path.name}")

        try:
            volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
            if volume is None:
                return None

            spleen_mask = mask > 0
            masked_volume = inference.apply_spleen_mask(volume, mask)
            reconstruction_error = inference.reconstruction_error(
                self.model, masked_volume, self.device
            )
            decision = inference.decide_anomaly(reconstruction_error, threshold)

            result = {
                'volume_idx': volume_idx,
                'file_path': str(volume_path),
                'is_anomaly': decision['is_anomaly'],
                'confidence': decision['confidence'],
                'reconstruction_error': reconstruction_error,
                'threshold': threshold,
                'spleen_voxels': int(np.sum(spleen_mask)),
            }

            print(f"Reconstruction Error: {reconstruction_error:.6f}")
            print(f"Threshold: {threshold:.6f}")
            print(f"Result: {'🚨 ANOMALY DETECTED' if decision['is_anomaly'] else '✅ NORMAL'}")
            print(f"Confidence: {decision['confidence']:.2f}x threshold")

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

    model_path = config.AUTOENCODER_PATH
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Initialize corrected detector
    detector = Spleen3DAnomalyDetectorFixed(model_path)

    # Calculate corrected threshold
    threshold, normal_errors = detector.calculate_threshold_correctly()

    print("\n=== Testing Normal Volume (should be NORMAL) ===")
    result_normal = detector.detect_anomaly_from_training_file(0, threshold)

    print("\n=== Testing Another Normal Volume ===")
    result_normal2 = detector.detect_anomaly_from_training_file(5, threshold)

    # Test synthetic anomaly
    print("\n=== Testing Synthetic Anomaly ===")
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
        print("\n📊 SUMMARY:")
        print("Normal volumes tested: 2")
        print(f"False positives: {normal_fps}")
        print(f"Corrected threshold: {threshold:.6f}")

        if normal_fps == 0:
            print("✅ FALSE POSITIVE ISSUE FIXED!")
        else:
            print("⚠️  Still some false positives - may need threshold adjustment")

if __name__ == "__main__":
    test_corrected_detection()
