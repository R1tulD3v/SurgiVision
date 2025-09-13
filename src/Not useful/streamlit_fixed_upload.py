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

# Page configuration
st.set_page_config(
    page_title="SpleenGuard AI - Fixed Upload",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_anomaly_detector():
    """Load the trained anomaly detector"""
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if Path(model_path).exists():
        detector = Spleen3DAnomalyDetectorFixed(model_path)
        return detector, True
    else:
        return None, False

def process_nifti_file_fixed(uploaded_file, detector, threshold=0.009551):
    """Fixed NIfTI processing for both .nii and .nii.gz files"""
    try:
        st.info(f"📋 Processing file: {uploaded_file.name}")
        
        # Determine correct file extension
        original_name = uploaded_file.name.lower()
        
        if original_name.endswith('.nii.gz'):
            suffix = '.nii.gz'
        elif original_name.endswith('.nii'):
            suffix = '.nii'
        elif original_name.endswith('.gz'):
            suffix = '.nii.gz'  # Assume it's compressed NIfTI
        else:
            suffix = '.nii'  # Default
        
        st.info(f"🔧 Using file format: {suffix}")
        
        # Save uploaded file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        st.info(f"💾 Saved to temporary location")
        
        # Load NIfTI file
        try:
            nii_img = nib.load(temp_path)
            volume_data = nii_img.get_fdata()
            st.success(f"✅ Successfully loaded volume: {volume_data.shape}")
        except Exception as load_error:
            st.error(f"❌ Failed to load NIfTI file: {load_error}")
            os.unlink(temp_path)
            return None
        
        # Show file information
        st.info(f"""
        📊 **File Information:**
        - Original shape: {volume_data.shape}
        - Data type: {volume_data.dtype}
        - Value range: {volume_data.min():.1f} to {volume_data.max():.1f}
        - File size: {uploaded_file.size / (1024*1024):.1f} MB
        """)
        
        # Preprocess for model
        with st.spinner("🔄 Preprocessing for AI analysis..."):
            
            # CT intensity windowing
            volume_windowed = np.clip(volume_data, -200, 300)
            volume_norm = (volume_windowed + 200) / 500
            
            # Extract center region (likely spleen area)
            center_x = volume_norm.shape[0] // 2
            center_y = volume_norm.shape[1] // 2  
            center_z = volume_norm.shape[2] // 2
            
            crop_size = 80
            x_start = max(0, center_x - crop_size//2)
            x_end = min(volume_norm.shape[0], center_x + crop_size//2)
            y_start = max(0, center_y - crop_size//2)
            y_end = min(volume_norm.shape[1], center_y + crop_size//2)
            z_start = max(0, center_z - 20)
            z_end = min(volume_norm.shape[2], center_z + 20)
            
            cropped_volume = volume_norm[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Resize to model input
            from scipy import ndimage
            zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
            processed_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
        
        st.success(f"✅ Preprocessing completed: {processed_volume.shape}")
        
        # Run AI analysis
        with st.spinner("🧠 Running AI anomaly detection..."):
            volume_tensor = torch.FloatTensor(processed_volume[np.newaxis, np.newaxis, ...])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            volume_tensor = volume_tensor.to(device)
            
            with torch.no_grad():
                reconstructed = detector.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        
        # Analysis results
        is_anomaly = reconstruction_error > threshold
        confidence = reconstruction_error / threshold
        error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reconstruction_error': reconstruction_error,
            'threshold': threshold,
            'original_shape': volume_data.shape,
            'processed_volume': processed_volume,
            'error_map': error_map,
            'original_file': uploaded_file.name
        }
        
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None

def create_3d_visualization(volume, title="3D Volume"):
    """Create 3D plot"""
    sampled = volume[::2, ::2, ::2]
    z, y, x = np.mgrid[0:sampled.shape[0], 0:sampled.shape[1], 0:sampled.shape[2]]
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()  
    values_flat = sampled.flatten()
    
    mask = values_flat > 0.1
    if np.sum(mask) == 0:
        mask = values_flat > 0.05
    
    x_filtered = x_flat[mask]
    y_filtered = y_flat[mask]
    z_filtered = z_flat[mask]
    values_filtered = values_flat[mask]
    
    fig = go.Figure(data=go.Scatter3d(
        x=x_filtered, y=y_filtered, z=z_filtered,
        mode='markers',
        marker=dict(size=2, color=values_filtered, colorscale='Viridis', opacity=0.6),
        name='Tissue'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        width=600, height=500
    )
    
    return fig

def main():
    st.markdown("# 🩺 SpleenGuard AI - Fixed NIfTI Upload")
    st.markdown("### Test your spleen.nii file here")
    
    detector, model_loaded = load_anomaly_detector()
    
    if not model_loaded:
        st.error("❌ Model not found!")
        return
    
    st.markdown("### 📤 Upload Your Spleen NIfTI File")
    
    uploaded_file = st.file_uploader(
        "Choose your spleen.nii file",
        type=['nii', 'gz'],
        help="Upload your spleen.nii or spleen.nii.gz file"
    )
    
    if uploaded_file is not None:
        st.success(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
        
        if st.button("🔍 Analyze Spleen", type="primary"):
            result = process_nifti_file_fixed(uploaded_file, detector)
            
            if result:
                # Show results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result['is_anomaly']:
                        st.error("🚨 ANOMALY DETECTED")
                    else:
                        st.success("✅ NORMAL PATTERN") 
                
                with col2:
                    st.metric("Reconstruction Error", f"{result['reconstruction_error']:.6f}")
                
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.2f}x threshold")
                
                # Visualization
                st.markdown("### 🫘 3D Visualization")
                fig = create_3d_visualization(result['processed_volume'], "Your Spleen Volume")
                st.plotly_chart(fig, use_container_width=True)
                
                # Details
                st.markdown("### 📊 Analysis Details")
                st.info(f"""
                **File:** {result['original_file']}
                **Original Shape:** {result['original_shape']}
                **Processed Shape:** 64×64×64 voxels
                **Reconstruction Error:** {result['reconstruction_error']:.6f}
                **Threshold:** {result['threshold']:.6f}
                **Classification:** {'🚨 ANOMALY' if result['is_anomaly'] else '✅ NORMAL'}
                """)
            else:
                st.error("❌ Failed to process your file")

if __name__ == "__main__":
    main()
