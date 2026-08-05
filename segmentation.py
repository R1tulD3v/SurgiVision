"""Automatic spleen segmentation using a MONAI pretrained UNet.

Removes the ground-truth-mask dependency: given a raw CT, predict the spleen
mask so the anomaly pipeline works on unseen uploads (instead of over-flagging
whole-abdomen scans). Uses the official `spleen_ct_segmentation` bundle's
network + preprocessing.

The model weights are a large binary kept out of git; download once with:
    python -m monai.bundle download --name spleen_ct_segmentation --bundle_dir <dir>
and point SURGIVISION_SEGMENTATION_MODEL_PATH at the bundle's models/model.pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config

# Network + preprocessing parameters from the bundle's configs/inference.json.
_ROI = (96, 96, 96)
_PIXDIM = (1.5, 1.5, 2.0)
_HU_MIN, _HU_MAX = -57, 164


class SpleenSegmenter:
    """Wraps the MONAI UNet; call :meth:`segment` with a NIfTI path."""

    def __init__(self, model_path: Optional[Path | str] = None, device=None):
        from monai.networks.nets import UNet

        self.device = device or torch.device("cpu")
        path = Path(model_path) if model_path else config.SEGMENTATION_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {path}")

        self.net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        ).to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()

    def segment(self, nifti_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mask, hu_volume)`` in a shared RAS/1.5x1.5x2.0 grid.

        ``mask`` is a binary spleen mask; ``hu_volume`` is the resampled CT in
        Hounsfield units on the same grid (so no inverse resampling is needed to
        crop the spleen for the autoencoder).
        """
        from monai.inferers import SlidingWindowInferer
        from monai.transforms import (
            EnsureChannelFirst,
            LoadImage,
            Orientation,
            ScaleIntensityRange,
            Spacing,
        )

        img = LoadImage(image_only=True)(str(nifti_path))
        img = EnsureChannelFirst()(img)
        img = Orientation(axcodes="RAS")(img)
        img = Spacing(pixdim=_PIXDIM, mode="bilinear")(img)  # [1, H, W, D] HU

        net_in = ScaleIntensityRange(
            a_min=_HU_MIN, a_max=_HU_MAX, b_min=0.0, b_max=1.0, clip=True
        )(img)
        inferer = SlidingWindowInferer(roi_size=_ROI, sw_batch_size=1, overlap=0.5)
        with torch.no_grad():
            logits = inferer(net_in.unsqueeze(0).to(self.device), self.net)
        pred = torch.argmax(logits, dim=1)[0]  # [H, W, D]

        mask = (pred > 0).cpu().numpy().astype(np.uint8)
        hu = np.asarray(img[0].cpu().numpy(), dtype=np.float32)
        return mask, hu
