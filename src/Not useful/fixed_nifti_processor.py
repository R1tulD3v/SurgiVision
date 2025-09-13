import streamlit as st
import torch
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os
from PIL import Image
import cv2
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed
from enhanced_anomaly_creator import MedicalAnomalyCreator

def process_nifti_file_fixed(uploaded_file, detector, threshold=0.009551):
    """Fixed NIfTI file processing for both .nii and .nii.gz"""
    try:
        # Get original filename and extension
        original_name = uploaded_file.name
        file_extension = Path(original_name).suffix.lower()
        
        print(f"Processing file: {original_name}")
        print(f"Detected extension: {file_extension}")
        
        # Handle both .nii and .nii.gz files
        if file_extension == '.nii':
            # Uncompressed NIfTI file
            suffix = '.nii'
        elif file_extension == '.gz':
            # Check if it's .nii.gz
            if original_name.lower().endswith('.nii.gz'):
                suffix = '.nii.gz'
            else:
                suffix = '.gz'
        else:
            # Default to .nii.gz
            suffix = '.nii.gz'
        
        # Save with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        print(f"Saved to temporary file: {temp_path}")
        
        # Try to load the NIfTI file
        try:
            nii_img = nib.load(temp_path)
            volume_data = nii_img.get_fdata()
            print(f"✅ Successfully loaded NIfTI file: {volume_data.shape}")
        except Exception as load_error:
            print(f"Error loading with nibabel: {load_error}")
            # Clean up and return error
            os.unlink(temp_path)
            return None
        
        # Basic preprocessing
        volume_windowed = np.clip(volume_data, -200, 300)
        volume_norm = (volume_windowed + 200) / 500
        
        print(f"Intensity range after windowing: {volume_norm.min():.3f} to {volume_norm.max():.3f}")
        
        # Extract center region (spleen area)
        center_x, center_y, center_z = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
        crop_size = 80
        
        x_start = max(0, center_x - crop_size//2)
        x_end = min(volume_norm.shape[0], center_x + crop_size//2)
        y_start = max(0, center_y - crop_size//2) 
        y_end = min(volume_norm.shape[1], center_y + crop_size//2)
        z_start = max(0, center_z - 20)
        z_end = min(volume_norm.shape[2], center_z + 20)
        
        cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
        print(f"Cropped volume shape: {cropped_volume.shape}")
        
        # Resize to model input size (64x64x64)
        from scipy import ndimage
        zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
        resized_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
        
        print(f"Resized volume shape: {resized_volume.shape}")
        print(f"Resized intensity range: {resized_volume.min():.3f} to {resized_volume.max():.3f}")
        
        # Run through model
        volume_tensor = torch.FloatTensor(resized_volume[np.newaxis, np.newaxis, ...])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volume_tensor = volume_tensor.to(device)
        
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        
        # Determine anomaly
        is_anomaly = reconstruction_error > threshold
        confidence = reconstruction_error / threshold
        error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reconstruction_error': reconstruction_error,
            'threshold': threshold,
            'original_shape': volume_data.shape,
            'processed_volume': resized_volume,
            'error_map': error_map,
            'image_type': '3D'
        }
        
    except Exception as e:
        print(f"❌ Error processing NIfTI file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_nifti_processing():
    """Test function to debug NIfTI processing"""
    print("🧪 Testing NIfTI file processing...")
    
    # This would be called when you upload a file
    # For testing, create a dummy uploaded file object
    
    print("Ready to test with uploaded spleen.nii file")

if __name__ == "__main__":
    test_nifti_processing()
