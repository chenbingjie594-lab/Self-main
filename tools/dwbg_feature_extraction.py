"""YOLO detection-head-input feature extraction for DWBG-v2.

Features are captured with a *pre-hook* on the actual Detect module, i.e. the
P3/P4/P5 tensors immediately before the detection head.  No NMS box feature is
used.  Importing this module does not require Ultralytics until an extractor is
constructed.
"""
from __future__ import annotations

import numpy as np

try:
    from .dwbg_v2_utils import clip_roi_xyxy
except ImportError:
    from dwbg_v2_utils import clip_roi_xyxy


class DetectInputExtractor:
    def __init__(self, weights, device, imgsz):
        import torch
        from ultralytics import YOLO
        self.torch, self.imgsz = torch, int(imgsz)
        self.model = YOLO(str(weights))
        modules = list(self.model.model.modules())
        detect = next((m for m in reversed(modules) if m.__class__.__name__.lower() == "detect"), None)
        if detect is None:
            raise RuntimeError("Could not locate YOLO Detect head")
        self.features = None
        def capture(module, inputs):
            values = inputs[0] if inputs and isinstance(inputs[0], (list, tuple)) else inputs
            self.features = [value.detach() for value in values]
        self.handle = detect.register_forward_pre_hook(capture)
        self.device = device

    def close(self):
        self.handle.remove()

    def encode(self, image_path, bbox_xyxy, image_width, image_height):
        """Run one image and return concat(L2(pool(P3)), L2(pool(P4)), L2(pool(P5)))."""
        import torch.nn.functional as F
        self.features = None
        self.model.predict(source=str(image_path), imgsz=self.imgsz, device=self.device,
                           conf=0.001, rect=False, verbose=False, stream=False)
        if not self.features or len(self.features) != 3:
            raise RuntimeError("Detect pre-hook did not capture three feature maps")
        # Ultralytics letterboxes the source into a square imgsz canvas before
        # the P3/P4/P5 maps are produced.  Map original-pixel GT/candidate
        # coordinates through that same resize+padding transform first.
        ratio = min(self.imgsz / float(image_width), self.imgsz / float(image_height))
        pad_x = (self.imgsz - image_width * ratio) / 2.0
        pad_y = (self.imgsz - image_height * ratio) / 2.0
        x0, y0, x1, y1 = map(float, bbox_xyxy)
        letterboxed_box = [x0 * ratio + pad_x, y0 * ratio + pad_y,
                           x1 * ratio + pad_x, y1 * ratio + pad_y]
        vectors = []
        for fmap in self.features:
            if fmap.ndim != 4 or fmap.shape[0] != 1:
                raise RuntimeError(f"Unexpected feature shape: {tuple(fmap.shape)}")
            _, _, map_h, map_w = fmap.shape
            x0, y0, x1, y1 = clip_roi_xyxy(letterboxed_box, self.imgsz, self.imgsz, map_w, map_h)
            pooled = F.adaptive_avg_pool2d(fmap[0:1, :, y0:y1, x0:x1], (1, 1)).flatten()
            pooled = F.normalize(pooled, p=2, dim=0, eps=1e-8)
            vectors.append(pooled.float().cpu().numpy())
        result = np.concatenate(vectors).astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite P3/P4/P5 feature")
        return result
