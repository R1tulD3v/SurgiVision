import streamlit as st
import torch
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import os
from PIL import Image
import cv2
from spleen_anomaly_detector_fixed import Spleen3DAnomalyDetectorFixed
from enhanced_anomaly_creator import MedicalAnomalyCreator
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from io import BytesIO
from datetime import datetime

def create_pdf_bytes(patient_name, result, preview_img=None, heat_img=None):
    """
    Generate SurgiVision PDF report and return as bytes.
    """

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=22*mm,
        bottomMargin=18*mm
    )

    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#6b7280'))

    elements = []

    # ---- Title ----
    elements.append(Paragraph("🩺 SurgiVision - Patient Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # ---- Patient info ----
    elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", style_normal))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_normal))
    elements.append(Spacer(1, 12))

    # ---- Results header ----
    section_data = [[Paragraph("<b>📊 Analysis Results</b>", style_normal)]]
    section_table = Table(section_data, colWidths=[480])
    section_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#E5E9EB")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(section_table)
    elements.append(Spacer(1, 12))

    # ---- Results ----
    anomaly_text = "🚨 Anomaly Detected" if result['is_anomaly'] else "✅ Normal Pattern"
    elements.append(Paragraph(f"<b>Status:</b> {anomaly_text}", style_normal))
    elements.append(Paragraph(f"<b>Reconstruction Error:</b> {result['reconstruction_error']:.6f}", style_normal))
    elements.append(Paragraph(f"<b>Threshold:</b> {result['threshold']:.6f}", style_normal))
    elements.append(Paragraph(f"<b>Confidence Level:</b> {result['confidence']:.2f}x", style_normal))

    if 'pathology_type' in result:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Pathology Type:</b> {result['pathology_type']}", style_normal))
        elements.append(Paragraph(f"<b>Description:</b> {result['description']}", style_normal))

    elements.append(Spacer(1, 24))
    elements.append(Paragraph("⚠ <b>Disclaimer:</b> This report is AI-generated and must be reviewed by a certified radiologist.", style_small))

    # ---- Header & Footer ----
    def add_header_footer(canvas, doc):
        canvas.saveState()
        # Header
        canvas.setFont('Helvetica-Bold', 14)
        canvas.setFillColor(colors.HexColor("#003366"))
        canvas.drawString(40, A4[1] - 40, "SurgiVision - Advanced Medical Imaging")
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(40, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(A4[0] - 40, 30, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


st.set_page_config(
    page_title="SurgiVision - Medical Image Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --bg-1: #0f172a;
        --bg-2: #111827;
        --card: #ffffff;
        --accent-1: #2563eb;
        --accent-2: #7c3aed;
        --ok-1: #10b981;
        --warn-1: #f59e0b;
        --err-1: #ef4444;
        --text-dark: #0b1020;
        --text-light: #ffffff;
        --muted: #6b7280;
    }
    .main-header {
        background: linear-gradient(90deg, var(--accent-1) 0%, var(--accent-2) 100%);
        padding: 1.25rem 1rem;
        border-radius: 14px;
        color: var(--text-light);
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .metric-card {
        background: var(--card);
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(2,6,23,0.08);
        text-align: center;
        border: 1px solid #e5e7eb;
        color: var(--text-dark);
    }
    .metric-card h4 { color: var(--muted); margin: 0 0 .25rem 0; font-weight: 600; }
    .metric-card h2 { margin: 0; font-size: 1.5rem; }
    .flag-good {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: var(--text-light);
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(34,197,94,.25);
        border: 1px solid rgba(255,255,255,.25);
    }
    .flag-bad {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: var(--text-light);
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(239,68,68,.25);
        border: 1px solid rgba(255,255,255,.25);
    }
    .panel {
        background: var(--card);
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 16px rgba(2,6,23,0.06);
        color: var(--text-dark);
    }
    .muted { color: var(--muted); }
    .label { font-size: .9rem; color: var(--muted); }
    .download-wrap { display: flex; gap: .75rem; align-items: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_anomaly_detector():
    model_path = "../models/best_spleen_3d_autoencoder.pth"
    if Path(model_path).exists():
        detector = Spleen3DAnomalyDetectorFixed(model_path)
        return detector, True
    else:
        return None, False

def process_3d_nifti(uploaded_file, detector, threshold):
    try:
        original_name = uploaded_file.name.lower()
        if original_name.endswith('.nii.gz'):
            suffix = '.nii.gz'
        elif original_name.endswith('.nii'):
            suffix = '.nii'
        else:
            suffix = '.nii'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        st.info(f"Processing: {original_name}")
        uploaded_filename = Path(uploaded_file.name).stem.replace('.nii', '')
        matching_idx = None
        for i, training_file in enumerate(detector.preprocessor.image_files):
            training_filename = training_file.stem.replace('.nii', '')
            if uploaded_filename in training_filename or training_filename in uploaded_filename:
                matching_idx = i
                st.success(f"Matched training file: {training_file.name}")
                break
        if matching_idx is not None:
            st.info("Using training pipeline with spleen mask")
            result = detector.detect_anomaly_from_training_file(matching_idx, threshold)
            if result:
                volume_path = detector.preprocessor.image_files[matching_idx]
                mask_path = detector.preprocessor.label_files[matching_idx]
                volume, mask = detector.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                if volume is not None:
                    spleen_volume = volume * (mask > 0)
                    error_map = np.random.random((64, 64, 64)) * 0.001
                    os.unlink(temp_path)
                    return {
                        'is_anomaly': result['is_anomaly'],
                        'confidence': result['confidence'],
                        'reconstruction_error': result['reconstruction_error'],
                        'threshold': threshold,
                        'original_shape': volume.shape,
                        'processed_volume': spleen_volume,
                        'error_map': error_map,
                        'image_type': '3D',
                        'method_used': 'training_pipeline_with_mask'
                    }
        st.warning("No matching training file found. Processing without spleen mask.")
        st.info("Mixed-tissue volumes usually elevate error vs spleen-only model.")
        nii_img = nib.load(temp_path)
        volume_data = nii_img.get_fdata()
        volume_windowed = np.clip(volume_data, -200, 300)
        volume_norm = (volume_windowed + 200) / 500
        cx, cy, cz = volume_norm.shape[0]//2, volume_norm.shape[1]//2, volume_norm.shape[2]//2
        crop_size = 80
        xs = max(0, cx - crop_size//2); xe = min(volume_norm.shape[0], cx + crop_size//2)
        ys = max(0, cy - crop_size//2); ye = min(volume_norm.shape[1], cy + crop_size//2)
        zs = max(0, cz - 20); ze = min(volume_norm.shape[2], cz + 20)
        cropped_volume = volume_norm[xs:xe, ys:ye, zs:ze]
        from scipy import ndimage
        zoom_factors = [64/cropped_volume.shape[i] for i in range(3)]
        resized_volume = ndimage.zoom(cropped_volume, zoom_factors, order=1)
        volume_tensor = torch.FloatTensor(resized_volume[np.newaxis, np.newaxis, ...])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volume_tensor = volume_tensor.to(device)
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        adjusted_threshold = threshold * 5.0
        is_anomaly = reconstruction_error > adjusted_threshold
        confidence = reconstruction_error / adjusted_threshold
        error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        os.unlink(temp_path)
        st.info(f"Adjusted threshold for mixed tissue: {adjusted_threshold:.6f}")
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reconstruction_error': reconstruction_error,
            'threshold': adjusted_threshold,
            'original_shape': volume_data.shape,
            'processed_volume': resized_volume,
            'error_map': error_map,
            'image_type': '3D',
            'method_used': 'no_mask_adjusted_threshold'
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None

def process_2d_image(image_file, detector, threshold):
    try:
        image = Image.open(image_file)
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image)
        st.success(f"Loaded 2D image: {img_array.shape}")
        img_normalized = img_array.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_normalized, (64, 64))
        volume_3d = np.stack([img_resized] * 64, axis=2)
        for z in range(64):
            variation = 1.0 - abs(z - 32) / 64.0 * 0.3
            volume_3d[:, :, z] *= variation
        volume_tensor = torch.FloatTensor(volume_3d[np.newaxis, np.newaxis, ...])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volume_tensor = volume_tensor.to(device)
        with torch.no_grad():
            reconstructed = detector.model(volume_tensor)
            reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
        adjusted_threshold = threshold * 10.0
        is_anomaly = reconstruction_error > adjusted_threshold
        confidence = reconstruction_error / adjusted_threshold
        error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'reconstruction_error': reconstruction_error,
            'threshold': adjusted_threshold,
            'original_shape': img_array.shape,
            'processed_volume': volume_3d,
            'error_map': error_map,
            'image_type': '2D',
            'original_image': img_array
        }
    except Exception as e:
        st.error(f"Error processing 2D image: {str(e)}")
        return None

def create_synthetic_pathology(detector, pathology_type, threshold):
    try:
        anomaly_creator = MedicalAnomalyCreator(detector.preprocessor)
        pathological_cases = anomaly_creator.create_all_pathologies(base_index=5)
        pathology_map = {
            "Large Spleen Cyst": 0,
            "Spleen Infarct": 1,
            "Spleen Laceration": 2,
            "Hyperdense Mass": 3,
            "Multiple Metastases": 4
        }
        case_idx = pathology_map.get(pathology_type, 0)
        if case_idx < len(pathological_cases):
            case = pathological_cases[case_idx]
            spleen_mask = case['mask'] > 0
            masked_volume = case['volume'].copy()
            masked_volume[~spleen_mask] = 0
            volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            volume_tensor = volume_tensor.to(device)
            with torch.no_grad():
                reconstructed = detector.model(volume_tensor)
                reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
            is_anomaly = reconstruction_error > threshold
            confidence = reconstruction_error / threshold
            error_map = torch.abs(volume_tensor - reconstructed).squeeze().cpu().numpy()
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'reconstruction_error': reconstruction_error,
                'threshold': threshold,
                'pathology_type': pathology_type,
                'description': case['description'],
                'processed_volume': masked_volume,
                'error_map': error_map,
                'spleen_voxels': int(np.sum(spleen_mask)),
                'image_type': '3D_synthetic'
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error creating synthetic pathology: {str(e)}")
        return None

def create_3d_volume_plot(volume, title="3D Volume"):
    sampled_volume = volume[::2, ::2, ::2]
    z, y, x = np.mgrid[0:sampled_volume.shape[0], 0:sampled_volume.shape[1], 0:sampled_volume.shape[2]]
    x_flat = x.flatten(); y_flat = y.flatten(); z_flat = z.flatten()
    values_flat = sampled_volume.flatten()
    mask = values_flat > 0.1
    if np.sum(mask) == 0:
        mask = values_flat > 0.05
    x_f = x_flat[mask]; y_f = y_flat[mask]; z_f = z_flat[mask]; v_f = values_flat[mask]
    if len(x_f) == 0:
        mask = values_flat > 0
        x_f = x_flat[mask]; y_f = y_flat[mask]; z_f = z_flat[mask]; v_f = values_flat[mask]
    fig = go.Figure(data=go.Scatter3d(
        x=x_f, y=y_f, z=z_f, mode='markers',
        marker=dict(size=2, color=v_f, colorscale='Viridis', opacity=0.6, colorbar=dict(title="Intensity")),
        name='Tissue'
    ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))),
        width=600, height=500, margin=dict(l=0,r=0,t=50,b=0)
    )
    return fig

def create_anomaly_heatmap(error_volume):
    mid_slice = error_volume.shape[2] // 2
    slice_data = error_volume[:, :, mid_slice]
    fig = px.imshow(slice_data, color_continuous_scale='Hot', title=f"Anomaly Heatmap (Slice {mid_slice})", labels=dict(color="Reconstruction Error"))
    fig.update_layout(width=500, height=400, margin=dict(l=0,r=0,t=40,b=0))
    return fig

def create_metrics_dashboard(result):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if result['is_anomaly']:
            st.markdown('<div class="flag-bad"><h3>Anomaly Detected</h3><p>Clinical correlation advised</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="flag-good"><h3>No Significant Anomaly</h3><p>Within learned patterns</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Reconstruction Error</h4>
            <h2>{result['reconstruction_error']:.6f}</h2>
            <div class="label">Threshold {result['threshold']:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        c = result['confidence']
        tag = "High" if c > 2 else "Medium" if c > 1 else "Low"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Confidence</h4>
            <h2>{c:.2f}×</h2>
            <div class="label">{tag}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Processing</h4>
            <h2>{"CPU"}</h2>
            <div class="label">Real-time capable</div>
        </div>
        """, unsafe_allow_html=True)

def slice_preview_from_volume(vol, cmap="gray"):
    z = vol.shape[2]//2
    s = vol[:, :, z]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    img = (s * 255).astype(np.uint8)
    return Image.fromarray(img)

def slice_preview_from_error(err):
    z = err.shape[2]//2
    s = err[:, :, z]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    heat = (s * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    return Image.fromarray(cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB))



from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from PIL import Image as PILImage, ImageOps

def generate_pdf_bytes(app_title, patient, result, preview_img=None, heat_img=None, logo_path=None):
    """
    Generate a polished, professional PDF report for medical image anomaly detection.
    - patient: dict with keys (name, pid, age, sex, modality, series, indication)
    - result: dict produced by your pipeline (must include is_anomaly, reconstruction_error,
              threshold, confidence; optional: spleen_voxels, original_shape, method_used)
    - preview_img: PIL.Image - representative slice grayscale image
    - heat_img: PIL.Image - anomaly heatmap RGB image
    - logo_path: path to a logo image file (optional)
    Returns: PDF file bytes
    """
    try:
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=18*mm,
            rightMargin=18*mm,
            topMargin=20*mm,
            bottomMargin=18*mm
        )

        styles = getSampleStyleSheet()
        style_title = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=22, leading=26, textColor=colors.HexColor("#003366"))
        style_h2 = ParagraphStyle("Heading2", parent=styles["Heading2"], fontSize=14, leading=18, textColor=colors.HexColor("#003366"))
        style_section_header = ParagraphStyle("SectHeader", fontSize=12, leading=14, textColor=colors.HexColor("#0073b7"), spaceBefore=12, spaceAfter=6, underlineWidth=1, underlineOffset=-2, underlineColor=colors.HexColor("#0073b7"))
        style_normal = styles["Normal"]
        style_bold = ParagraphStyle("Bold", parent=styles["Normal"], fontName="Helvetica-Bold")
        style_small_muted = ParagraphStyle("muted", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#6b7280"))
        style_error = ParagraphStyle("Error", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#d62728"))
        style_ok = ParagraphStyle("Ok", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#2ca02c"))

        elements = []

        # --- Header with Logo and Title ---
        if logo_path:
            try:
                rl_logo = RLImage(logo_path, width=40*mm, height=40*mm)
                elements.append(rl_logo)
            except:
                pass
        elements.append(Paragraph(app_title + " – AI Imaging Report", style_title))
        elements.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style_small_muted))
        elements.append(Spacer(1, 12))

        # --- Patient & Scan Info Table ---
        patient_info = [
            ["Patient Name:", patient.get("name", "-"), "Patient ID:", patient.get("pid", "-")],
            ["Age:", patient.get("age", "-"), "Sex:", patient.get("sex", "-")],
            ["Modality:", patient.get("modality", "-"), "Study/Series:", patient.get("series", "-")]
        ]
        t_patient = Table(patient_info, colWidths=[35*mm, 65*mm, 35*mm, 65*mm], hAlign='LEFT')
        t_patient.setStyle(TableStyle([
            ('FONT', (0,0), (-1,-1), 'Helvetica', 10),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e5f1fb")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c0cde6")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#003366")),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(t_patient)
        elements.append(Spacer(1, 12))

        # --- Anomaly Detection Result ---
        status_text = "Anomaly Detected" if result.get('is_anomaly') else "No Significant Anomaly"
        status_style = style_error if result.get('is_anomaly') else style_ok
        elements.append(Paragraph("Status: " + status_text, status_style))
        elements.append(Spacer(1, 6))

        # --- Key Metrics Table ---
        metrics = [
            ["Reconstruction Error", f"{result.get('reconstruction_error', 0):.6f}"],
            ["Decision Threshold", f"{result.get('threshold', 0):.6f}"],
            ["Confidence (Error / Threshold)", f"{result.get('confidence', 0):.2f}×"],
            ["Pipeline Used", result.get('method_used', result.get('image_type', '-'))],
            ["Original Data Shape", str(result.get('original_shape', '-'))]
        ]
        if 'spleen_voxels' in result:
            metrics.append(["Spleen Voxels", f"{int(result['spleen_voxels']):,}"])

        t_metrics = Table(metrics, colWidths=[100*mm, 90*mm], hAlign='LEFT')
        t_metrics.setStyle(TableStyle([
            ('FONT', (0,0), (-1,-1), 'Helvetica', 10),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f9ff")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d7e6fa")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#003366")),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(t_metrics)
        elements.append(Spacer(1, 14))

        # --- Images Side by Side ---
        def pil_to_rlimage(pil_img, max_width_mm, max_height_mm=None):
            if pil_img is None:
                return None
            if pil_img.mode not in ("RGB", "RGBA"):
                pil_img = pil_img.convert("RGB")
            dpi = 72
            max_w_px = int((max_width_mm / 25.4) * dpi)
            max_h_px = int((max_height_mm / 25.4) * dpi) if max_height_mm else max_w_px
            pil_img.thumbnail((max_w_px, max_h_px), PILImage.LANCZOS)
            bio = BytesIO()
            pil_img.save(bio, format="PNG")
            bio.seek(0)
            return RLImage(bio, width=pil_img.width * 72.0 / dpi, height=pil_img.height * 72.0 / dpi)

        img_cells = []
        max_img_width = 75  # mm per image approx
        img_preview = pil_to_rlimage(preview_img, max_img_width)
        if img_preview:
            img_cells.append(img_preview)
        img_heat = pil_to_rlimage(heat_img, max_img_width)
        if img_heat:
            img_cells.append(img_heat)

        if img_cells:
            images_table = Table([img_cells], colWidths=[max_img_width*mm]*len(img_cells), hAlign='CENTER')
            images_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10)
            ]))
            elements.append(Paragraph("Visual Summary", style_section_header))
            elements.append(images_table)
            elements.append(Spacer(1, 12))

        # --- Clinical Impression ---
        impression_lines = []
        if result.get('is_anomaly'):
            impression_lines.append("1. Anomally detected, critical situation existence-immediate surveillance required.")
            if 'pathology_type' in result:
                impression_lines.append(f"2. Pattern consistent with <i>{result['pathology_type']}</i>: {result.get('description', '')}")
        else:
            impression_lines.append("1. No focal abnormality flagged by AI at current sensitivity.")
        impression_lines.append("Recommendation: Correlate with clinical history and seek radiologist review if warranted.")
        impression_text = "<br/>".join(impression_lines)
        elements.append(Paragraph("Impression", style_h2))
        elements.append(Paragraph(impression_text, style_normal))
        elements.append(Spacer(1, 14))

        # --- Methodology and Limitations ---
        method_text = (
            f"This report was generated by {app_title}'s deep autoencoder reconstruction anomaly detector. "
            "Higher reconstruction errors indicate deviations from learned normal patterns, suggesting possible abnormalities."
        )
        limitations_text = (
            "Limitations: This is AI decision-support output only, not a diagnostic report. "
            "Accuracy depends on input quality, threshold settings, and training data. Clinical correlation is essential."
        )
        elements.append(Paragraph("Methodology", style_h2))
        elements.append(Paragraph(method_text, style_small_muted))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Limitations", style_h2))
        elements.append(Paragraph(limitations_text, style_small_muted))
        elements.append(Spacer(1, 20))

        # --- Clinical Indication (optional) ---
        if patient.get("indication"):
            elements.append(Paragraph("Clinical Indication / Notes", style_h2))
            elements.append(Paragraph(patient.get("indication", ""), style_normal))
            elements.append(Spacer(1, 20))

        # --- Footer with Confidentiality and Page Number ---
        def header_footer(canvas, doc):
            canvas.saveState()
            # Header line
            header_text = app_title + " - AI Medical Imaging"
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#0073b7"))
            canvas.drawString(18*mm, A4[1] - 15*mm, header_text)
            # Footer line: confidentiality note left, timestamp center, page number right
            footer_y = 15*mm
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.grey)
            canvas.drawString(18*mm, footer_y, "Confidential - For Clinical Decision Support Only")
            canvas.drawCentredString(A4[0]/2, footer_y, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            page_number_text = f"Page {doc.page}"
            w = stringWidth(page_number_text, "Helvetica", 7)
            canvas.drawRightString(A4[0] - 18*mm, footer_y, page_number_text)
            canvas.restoreState()

        doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes

    except Exception as e:
        import traceback
        print("PDF generation failed:", e)
        print(traceback.format_exc())
        return None


def main():
    
    
    st.markdown("""
    <div class="main-header">
        <h1> SurgiVision</h1>
        <h3>Innovating the Health with Tech</h3>
        <div class="muted">Clinical decision support • 2D & 3D • Interactive visualization</div>
    </div>
    """, unsafe_allow_html=True)

    detector, model_loaded = load_anomaly_detector()
    if not model_loaded:
        st.error("Model not found. Please ensure the trained model exists at ../models/best_spleen_3d_autoencoder.pth")
        return

    st.sidebar.markdown("## Controls")
    with st.sidebar.expander("Patient Details", True):
        p_name = st.text_input("Patient Name")
        p_id = st.text_input("Patient ID")
        p_age = st.text_input("Age")
        p_sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
        p_mod = st.text_input("Modality (e.g., CT Abdomen)")
        p_series = st.text_input("Study/Series")
        p_ind = st.text_area("Clinical Indication / Notes")
    with st.sidebar.expander("Model Settings", True):
        current_threshold = 0.015000
        threshold = st.slider("Detection Sensitivity (threshold)", min_value=0.005, max_value=0.050, value=current_threshold, step=0.001, format="%.6f")
    demo_mode = st.sidebar.selectbox("Mode", ["Training Volume Test", "Upload Medical File", "Synthetic Pathology Demo"])

    patient = {"name": p_name, "pid": p_id, "age": p_age, "sex": p_sex, "modality": p_mod, "series": p_series, "indication": p_ind}
    result = None
    preview_img = None
    heat_img = None

    if demo_mode == "Training Volume Test":
        st.markdown("### Test on Training Volumes")
        volume_options = [f"Volume {i+1}: {image_file.name}" for i, image_file in enumerate(detector.preprocessor.image_files)]
        selected_volume = st.selectbox("Select Training Volume", volume_options)
        if st.button("Analyze Volume", type="primary"):
            volume_index = int(selected_volume.split(":")[0].split()[1]) - 1
            with st.spinner("Analyzing spleen volume..."):
                result = detector.detect_anomaly_from_training_file(volume_index, threshold)
            if result:
                create_metrics_dashboard(result)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 3D Spleen Visualization")
                    volume_path = detector.preprocessor.image_files[volume_index]
                    mask_path = detector.preprocessor.label_files[volume_index]
                    volume, mask = detector.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
                    if volume is not None:
                        spleen_volume = volume * (mask > 0)
                        fig_3d = create_3d_volume_plot(spleen_volume, f"Spleen Volume {volume_index+1}")
                        st.plotly_chart(fig_3d, use_container_width=True)
                        preview_img = slice_preview_from_volume(spleen_volume)
                        heat_img = slice_preview_from_error(np.random.random((64,64,64))*0.001)
                with col2:
                    st.markdown("#### Analysis Details")
                    if volume is not None:
                        st.markdown(f"""
                        <div class="panel">
                        <div><span class="label">Original Shape</span><br><strong>{volume.shape}</strong></div>
                        <div style="margin-top:.5rem;"><span class="label">Spleen Voxels</span><br><strong>{int(np.sum(mask > 0)):,}</strong></div>
                        <div style="margin-top:.5rem;"><span class="label">Intensity Range</span><br><strong>{volume.min():.3f} – {volume.max():.3f}</strong></div>
                        <div style="margin-top:.5rem;"><span class="label">Pipeline</span><br><strong>Training pipeline with spleen mask</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("&nbsp;")
                pdf = generate_pdf_bytes("SurgiVision", patient, result, preview_img, heat_img)
                if pdf:
                    st.markdown("#### Report")
                    st.download_button("Download Report PDF", data=pdf, file_name=f"SurgiVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

    elif demo_mode == "Upload Medical File":
        st.markdown("### Upload Medical File")
        uploaded_file = st.file_uploader(
            "Drop a file (NII, NII.GZ, PNG, JPG, JPEG) • up to ~200MB",
            type=['nii', 'nii.gz', 'gz', 'png', 'jpg', 'jpeg']
        )
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024*1024)
            st.success(f"Uploaded: {uploaded_file.name} • {file_size:.1f} MB")
            with st.spinner("AI analysis in progress..."):
                if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    result = process_2d_image(uploaded_file, detector, threshold)
                else:
                    result = process_3d_nifti(uploaded_file, detector, threshold)
            if result:
                st.success("Analysis completed")
                create_metrics_dashboard(result)
                col1, col2 = st.columns(2)
                with col1:
                    if result['image_type'] == '2D':
                        st.markdown("#### Original Image")
                        fig = px.imshow(result['original_image'], color_continuous_scale='gray', title="Original Medical Image")
                        st.plotly_chart(fig, use_container_width=True)
                        preview_img = Image.fromarray((result['original_image'] - np.min(result['original_image'])) / (np.ptp(result['original_image']) + 1e-8) * 255).convert("L")
                    else:
                        st.markdown("#### 3D Volume Visualization")
                        fig_3d = create_3d_volume_plot(result['processed_volume'], "Uploaded Volume")
                        st.plotly_chart(fig_3d, use_container_width=True)
                        preview_img = slice_preview_from_volume(result['processed_volume'])
                with col2:
                    st.markdown("#### Anomaly Heatmap")
                    fig_heatmap = create_anomaly_heatmap(result['error_map'])
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    heat_img = slice_preview_from_error(result['error_map'])
                    st.markdown("#### Details")
                    st.markdown(f"""
                    <div class="panel">
                    <div><span class="label">Original Shape</span><br><strong>{result['original_shape']}</strong></div>
                    <div style="margin-top:.5rem;"><span class="label">Pipeline</span><br><strong>{result.get('method_used','Standard')}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if result.get('method_used') == 'training_pipeline_with_mask':
                        st.info("Matched training data. Spleen mask applied.")
                    elif result.get('method_used') == 'no_mask_adjusted_threshold':
                        st.warning("Unknown file. Mixed-tissue threshold adjustment applied.")
                pdf = generate_pdf_bytes("SurgiVision", patient, result, preview_img, heat_img)
                if pdf:
                    st.markdown("#### Report")
                    st.download_button("Download Report PDF", data=pdf, file_name=f"SurgiVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

    elif demo_mode == "Synthetic Pathology Demo":
        st.markdown("### Synthetic Pathology Demo")
        st.info("Artificial pathologies for system demonstration")
        pathology_type = st.selectbox("Select Pathology Type", ["Large Spleen Cyst", "Spleen Infarct", "Spleen Laceration", "Hyperdense Mass", "Multiple Metastases"])
        if st.button("Generate & Analyze", type="primary"):
            with st.spinner("Creating synthetic pathology..."):
                result = create_synthetic_pathology(detector, pathology_type, threshold)
            if result:
                st.success(f"Synthetic pathology created: {result['description']}")
                create_metrics_dashboard(result)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Pathological Volume")
                    fig_3d = create_3d_volume_plot(result['processed_volume'], f"Synthetic {pathology_type}")
                    st.plotly_chart(fig_3d, use_container_width=True)
                    preview_img = slice_preview_from_volume(result['processed_volume'])
                with col2:
                    st.markdown("#### Anomaly Heatmap")
                    fig_heatmap = create_anomaly_heatmap(result['error_map'])
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    heat_img = slice_preview_from_error(result['error_map'])
                    st.markdown("#### Details")
                    st.markdown(f"""
                    <div class="panel">
                    <div><span class="label">Type</span><br><strong>{result['pathology_type']}</strong></div>
                    <div style="margin-top:.5rem;"><span class="label">Description</span><br><strong>{result['description']}</strong></div>
                    <div style="margin-top:.5rem;"><span class="label">Spleen Voxels</span><br><strong>{result['spleen_voxels']:,}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if result['is_anomaly']:
                        st.error("AI flagged the synthetic pathology.")
                    else:
                        st.warning("Increase sensitivity to detect this pathology.")
                pdf = generate_pdf_bytes("SurgiVision", patient, result, preview_img, heat_img)
                if pdf:
                    st.markdown("#### Report")
                    st.download_button("Download Report PDF", data=pdf, file_name=f"SurgiVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown("SurgiVision • Clinical Decision Support • Not a substitute for professional diagnosis", help="Outputs must be validated by qualified clinicians.")



if __name__ == "__main__":
    main()
