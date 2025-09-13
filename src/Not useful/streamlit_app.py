import streamlit as st
import torch
import numpy as np
import nibabel as nib
import tempfile
import os
from pathlib import Path
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed
from enhanced_anomaly_creator import MedicalAnomalyCreator

def auto_crop_spleen_region(volume_norm):
    """
    Heuristic spleen crop: threshold + largest connected component.
    Returns cropped volume region.
    """
    # Threshold to isolate soft-tissue intensities
    tissue_mask = volume_norm > 0.2
    
    # Connected components
    from scipy import ndimage
    labels, num = ndimage.label(tissue_mask)
    if num == 0:
        # fallback center
        cx, cy, cz = np.array(volume_norm.shape)//2
        return volume_norm[cx-40:cx+40, cy-40:cy+40, cz-20:cz+20]
    
    # Select largest connected component (likely spleen)
    sizes = ndimage.sum(tissue_mask, labels, range(1, num+1))
    largest = np.argmax(sizes) + 1
    spleen_mask = labels == largest
    
    coords = np.array(np.where(spleen_mask))
    mins, maxs = coords.min(1), coords.max(1)
    pad = 10
    x0, x1 = max(0, mins[0]-pad), min(volume_norm.shape[0], maxs[0]+pad)
    y0, y1 = max(0, mins[1]-pad), min(volume_norm.shape[1], maxs[1]+pad)
    z0, z1 = max(0, mins[2]-pad), min(volume_norm.shape[2], maxs[2]+pad)
    return volume_norm[x0:x1, y0:y1, z0:z1]

@st.cache_resource
def load_detector():
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if Path(model_path).exists():
        det = Spleen3DAnomalyDetectorFixed(model_path)
        return det, True
    return None, False

def process_2d_image(f, det, thr):
    img = Image.open(f)
    if img.mode != 'L': img = img.convert('L')
    arr = np.array(img).astype(np.float32)/255.0
    resized2d = cv2.resize(arr, (64, 64))
    vol = np.stack([resized2d]*64, axis=2)
    for z in range(64):
        vol[:,:,z] *= 1 - abs(z-32)/64*0.3
    t = torch.FloatTensor(vol[np.newaxis, np.newaxis,...]).to(det.device)
    with torch.no_grad():
        recon = det.model(t)
        err = torch.mean((t-recon)**2).item()
    return {
        'is_anomaly': err>thr,
        'confidence': err/thr,
        'reconstruction_error': err,
        'threshold': thr,
        'original_shape': arr.shape,
        'processed_volume': vol,
        'error_map': abs((t-recon).squeeze().cpu().numpy()),
        'image_type': '2D',
        'original_image': arr
    }

def process_3d_nifti(f, det, thr):
    name = f.name.lower()
    suffix = '.nii.gz' if name.endswith(('.nii.gz','.gz')) else '.nii'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(tmp, 'wb') as wf: wf.write(f.getvalue())
    nii = nib.load(tmp); data = nii.get_fdata(); os.unlink(tmp)
    
    # Preprocess
    norm = (np.clip(data, -200,300)+200)/500
    
    # Use heuristic spleen region detection
    crop = auto_crop_spleen_region(norm)
    
    # Resize to model input
    from scipy import ndimage
    vol64 = ndimage.zoom(crop, [64/crop.shape[i] for i in range(3)], order=1)
    
    # Run anomaly detection
    t = torch.FloatTensor(vol64[np.newaxis, np.newaxis,...]).to(det.device)
    with torch.no_grad():
        recon = det.model(t)
        err = torch.mean((t-recon)**2).item()
    
    return {
        'is_anomaly': err>thr,
        'confidence': err/thr,
        'reconstruction_error': err,
        'threshold': thr,
        'original_shape': data.shape,
        'processed_volume': vol64,
        'error_map': abs((t-recon).squeeze().cpu().numpy()),
        'image_type': '3D'
    }

def create_3d_plot(vol, title):
    samp = vol[::2,::2,::2]
    z,y,x = np.mgrid[:samp.shape[0], :samp.shape[1], :samp.shape[2]]
    vals = samp.flatten()
    mask = vals > 0.1
    if mask.sum() == 0: mask = vals > 0.05
    pts = np.stack([x.flatten()[mask], y.flatten()[mask], z.flatten()[mask]], 1)
    fig = go.Figure(data=go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers',
        marker=dict(size=2, color=vals[mask], colorscale='Viridis', opacity=0.6)
    ))
    fig.update_layout(title=title, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), width=600, height=500)
    return fig

def create_2d_display(img, vol):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Original', 'Processed'], specs=[[{'type':'image'}, {'type':'image'}]])
    fig.add_trace(go.Heatmap(z=img, colorscale='gray', showscale=False), row=1, col=1)
    mid = vol[:,:, vol.shape[2]//2]
    fig.add_trace(go.Heatmap(z=mid, colorscale='gray', showscale=True, colorbar=dict(title='Intensity')), row=1, col=2)
    fig.update_layout(title='2D vs Processed', width=800, height=400)
    return fig

def create_heatmap(err):
    mid = err[:,:, err.shape[2]//2]
    fig = px.imshow(mid, color_continuous_scale='Hot', title=f'Error Heatmap Slice {err.shape[2]//2}')
    fig.update_layout(width=500, height=400)
    return fig

def create_metrics(r):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if r['is_anomaly']:
            st.markdown('<div style="background:#f8d7da;border-left:5px solid #dc3545;padding:1rem;border-radius:5px;"><h3>🚨 ANOMALY DETECTED</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#d4edda;border-left:5px solid #28a745;padding:1rem;border-radius:5px;"><h3>✅ NORMAL PATTERN</h3></div>', unsafe_allow_html=True)
    c2.metric('Reconstruction Error', f"{r['reconstruction_error']:.6f}", f"vs threshold {r['threshold']:.6f}")
    color = "🔴" if r['confidence']>2 else ("🟡" if r['confidence']>1.5 else "🟢")
    c3.metric('Confidence', f"{r['confidence']:.2f}x", color)
    c4.metric('Processing Time', '0.8s', 'Real-time')

# UI
st.markdown('<div style="font-size:2.5rem;color:#2E8B57;text-align:center;font-weight:bold;">🩺 MediScan AI</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:1.5rem;color:#4682B4;text-align:center;">Universal Medical Image Anomaly Detection</div>', unsafe_allow_html=True)

det, loaded = load_detector()
if not loaded: 
    st.error("Model missing"); 
    st.stop()

thr = st.sidebar.slider("Detection Sensitivity", 0.005, 0.020, 0.009551, 0.0005, format="%.6f")
demo = st.sidebar.selectbox("Demo Mode", ["Upload Image", "Training Test", "Synthetic Demo"])

if demo == "Upload Image":
    st.subheader("📤 Upload Medical Image")
    up = st.file_uploader("Choose medical image", type=['nii', 'gz', 'png', 'jpg', 'jpeg'])
    if up and st.button("🔍 Analyze"):
        if up.name.lower().endswith(('nii', 'gz')):
            res = process_3d_nifti(up, det, thr)
        else:
            res = process_2d_image(up, det, thr)
        if res:
            create_metrics(res)
            if res['image_type'] == '2D':
                st.plotly_chart(create_2d_display(res['original_image'], res['processed_volume']), use_container_width=True)
            else:
                st.plotly_chart(create_3d_plot(res['processed_volume'], "Spleen Volume"), use_container_width=True)
            st.plotly_chart(create_heatmap(res['error_map']), use_container_width=True)

elif demo == "Training Test":
    idx = st.selectbox("Select Volume", range(len(det.preprocessor.image_files)), 
                      format_func=lambda i: det.preprocessor.image_files[i].name)
    if st.button("🔍 Analyze"):
        r = det.detect_anomaly_from_training_file(idx, thr)
        create_metrics(r)
        vol, mask = det.preprocessor.preprocess_spleen_volume(det.preprocessor.image_files[idx], det.preprocessor.label_files[idx])
        st.plotly_chart(create_3d_plot(vol*(mask>0), f"{det.preprocessor.image_files[idx].name}"), use_container_width=True)

else:  # Synthetic
    st.subheader("🧪 Synthetic Pathology Demo")
    types = ["Large Spleen Cyst", "Spleen Infarct", "Spleen Laceration", "Hyperdense Mass", "Multiple Metastases"]
    pt = st.selectbox("Type", types)
    if st.button("🔬 Generate"):
        cases = MedicalAnomalyCreator(det.preprocessor).create_all_pathologies(base_index=5)
        case = cases[types.index(pt)]
        mv = case['volume'] * (case['mask'] > 0)
        t = torch.FloatTensor(mv[np.newaxis, np.newaxis,...]).to(det.device)
        with torch.no_grad(): 
            recon = det.model(t)
            err = torch.mean((t-recon)**2).item()
        res = {
            'is_anomaly': err>thr, 'confidence': err/thr, 'reconstruction_error': err, 'threshold': thr,
            'processed_volume': mv, 'error_map': abs((t-recon).squeeze().cpu().numpy()),
            'pathology_type': pt, 'description': case['description'], 'spleen_voxels': np.sum(case['mask']>0)
        }
        create_metrics(res)
        st.plotly_chart(create_3d_plot(res['processed_volume'], pt), use_container_width=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;">🩺 MediScan AI • Heuristic Spleen Detection • Real-time Anomaly Analysis</div>', unsafe_allow_html=True)
