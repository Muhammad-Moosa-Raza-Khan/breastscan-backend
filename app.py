"""
MedScan API — Production Backend
MedFormerULTRA architecture.
FIXED v4 — correct modality detection, colour-preserving preprocessing,
            fixed GradCAM lesion overlay, all 500 errors resolved.
"""

import os
import gc
import base64
import traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from scipy.fftpack import dct
from flask import Flask, request, jsonify

# =====================================================================
# 1. APP INIT
# =====================================================================
app = Flask(__name__)
application = app
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

MODEL_PATHS = {
    'mammogram_mias': 'models/mammo_mias.pth',
    'mammogram_dmid': 'models/mammo_dmid.pth',
    'ultrasound':     'models/ultrasound_model.pth',
    'histopathology': 'models/hist_model.pth',
    'mri':            'models/mri_model.pth',
}

DATASET_CONFIGS = {
    'histopathology': {'num_classes': 2, 'labels': ['Benign',  'Malignant']},
    'mri':            {'num_classes': 2, 'labels': ['Healthy', 'Sick']},
    'mammogram_dmid': {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
    'mammogram_mias': {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
    'ultrasound':     {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
}

VALID_SCAN_TYPES = set(DATASET_CONFIGS.keys()) | {'mammogram', 'auto'}

# Modalities that have natural colour content — must NOT convert to gray
COLOUR_MODALITIES = {'histopathology', 'ultrasound'}

active_model_type = None
loaded_model      = None

# =====================================================================
# 2. ARCHITECTURE — MedFormerULTRA
# =====================================================================

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class ExpertBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MoE(nn.Module):
    def __init__(self, channels, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ExpertBlock(channels) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, num_experts), nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.gate(x)
        stacked = torch.stack([e(x) for e in self.experts], dim=1)
        w       = weights.view(-1, self.num_experts, 1, 1, 1)
        return (stacked * w).sum(dim=1), weights


class MedFormerULTRA(nn.Module):
    FEATURE_DIM = 256
    NUM_EXPERTS = 3

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=False, features_only=True)
        with torch.no_grad():
            feats = self.backbone(torch.randn(1, 3, 224, 224))
        self.feat_dims = [feats[-2].shape[1], feats[-1].shape[1]]

        cd = self.FEATURE_DIM
        self.proj1 = nn.Sequential(nn.Conv2d(self.feat_dims[0], cd, 1),
                                   nn.BatchNorm2d(cd), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(self.feat_dims[1], cd, 1),
                                   nn.BatchNorm2d(cd), nn.ReLU(inplace=True))
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.moe       = MoE(cd, self.NUM_EXPERTS)
        self.attention = EfficientAttention(cd, num_heads=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(cd + 64, 512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(0.4))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.aux_classifier = nn.Linear(cd, num_classes)

    def forward(self, x, freq, return_all=False):
        feats  = self.backbone(x)
        f1, f2 = feats[-2], feats[-1]
        f1, f2 = self.proj1(f1), self.proj2(f2)
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        feat, gate_w = self.moe(f1 + f2)
        B, C, H, W   = feat.shape
        feat_down    = F.adaptive_avg_pool2d(feat, (H//2, W//2))
        feat_att     = self.attention(feat_down.flatten(2).transpose(1,2))
        feat_spatial = F.interpolate(
            feat_att.transpose(1,2).reshape(B,C,H//2,W//2),
            size=(H,W), mode='bilinear', align_corners=False)
        feat   = feat + feat_spatial
        pooled = self.global_pool(feat).flatten(1)
        combined   = torch.cat([pooled, self.freq_encoder(freq).flatten(1)], dim=1)
        fused_feat = self.fusion(combined)
        logits     = self.classifier(fused_feat)
        aux_logits = self.aux_classifier(pooled)
        if return_all:
            return logits, aux_logits, gate_w
        return logits, aux_logits


# =====================================================================
# 3. FREQUENCY EXTRACTOR
# =====================================================================

def _dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def extract_freq_tensor(img_rgb_f32, target_size=320):
    gray = cv2.cvtColor((img_rgb_f32 * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dct_f = _dct2(gray)
    h, w  = dct_f.shape
    low   = dct_f.copy()
    low[int(h*0.3):, :] = 0
    low[:, int(w*0.3):] = 0
    freq3 = np.stack([low, low, low], axis=2)
    freq3 = cv2.resize(freq3, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    vmax  = np.abs(freq3).max()
    if vmax > 0:
        freq3 /= vmax
    return torch.from_numpy(freq3.transpose(2, 0, 1)).float()


# =====================================================================
# 4. TRANSFORMS & PREPROCESSING
# =====================================================================

IMG_SIZE = 320

spatial_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_inputs(img_rgb_uint8, scan_type):
    """
    FIX — CRITICAL: histopathology and ultrasound are COLOUR images.
    Converting them to grayscale destroys H&E staining information and
    ultrasound tissue contrast. Use LAB-space CLAHE to enhance brightness
    while preserving colour channels.

    Mammogram and MRI are truly grayscale → grayscale CLAHE is correct.
    """
    if scan_type in COLOUR_MODALITIES:
        # LAB CLAHE — preserves colour
        img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enh   = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        # Grayscale CLAHE — correct for mammogram / MRI
        gray     = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    spatial = spatial_transform(enhanced_rgb)
    freq    = extract_freq_tensor(enhanced_rgb, IMG_SIZE)
    return spatial.unsqueeze(0), freq.unsqueeze(0)


# =====================================================================
# 5. MODEL LOADER
# =====================================================================

def load_model(scan_type):
    """Load model; cache one at a time. Returns (model, labels)."""
    global loaded_model, active_model_type

    if active_model_type == scan_type and loaded_model is not None:
        return loaded_model, DATASET_CONFIGS[scan_type]['labels']

    if loaded_model is not None:
        print(f" * [GATEKEEPER] Unloading {active_model_type}…")
        del loaded_model
        loaded_model = None
        gc.collect()

    path   = MODEL_PATHS.get(scan_type)
    config = DATASET_CONFIGS.get(scan_type)

    if not path:
        raise RuntimeError(f"No model path configured for '{scan_type}'.")
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}. "
                           f"Please ensure the model is uploaded to the server.")

    print(f" * [GATEKEEPER] Loading {scan_type} ({config['num_classes']} classes)…")
    model      = MedFormerULTRA(num_classes=config['num_classes'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    state_dict = (checkpoint['model_state_dict']
                  if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                  else checkpoint)

    if isinstance(checkpoint, dict):
        print(f"   epoch={checkpoint.get('epoch','?')}  "
              f"best_acc={checkpoint.get('best_acc','?')}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:    print(f"   [WARN] Missing   ({len(missing)}): {missing[:3]}")
    if unexpected: print(f"   [WARN] Unexpect  ({len(unexpected)}): {unexpected[:3]}")
    if not missing and not unexpected:
        print("   [OK] Perfect weight match.")

    model.eval()
    loaded_model      = model
    active_model_type = scan_type
    print(f"   [OK] {scan_type} ready.")
    return model, config['labels']


# Pre-load default
try:
    load_model('mammogram_mias')
except Exception as e:
    print(f"[STARTUP] {e}")


# =====================================================================
# 6. AUTO SCAN-TYPE DETECTION — v4 (H&E hue pixel ratio, centre-crop)
# =====================================================================

def _he_pixel_pct(img_bgr):
    """
    Percentage of pixels that fall in H&E hue ranges (pink + purple)
    with sufficient saturation. Robust to pale/faded slides.
    Uses centre 84% of the image to ignore UI chrome.
    """
    h, w  = img_bgr.shape[:2]
    crop  = img_bgr[int(h*0.08):int(h*0.92), int(w*0.05):int(w*0.95)]
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue   = hsv[:, :, 0]
    sat   = hsv[:, :, 1]
    # Pink  (eosin):    hue 0–15  or  165–180 in OpenCV (0–180 scale)
    # Purple (hematoxylin): hue 120–160
    pink   = ((hue <= 15) | (hue >= 165)) & (sat > 25)
    purple = ((hue >= 120) & (hue <= 160)) & (sat > 25)
    return float((pink | purple).mean() * 100)


def auto_detect_scan_type(img_bgr):
    """
    Priority:
      1. Histopathology — H&E hue pixel ratio > 8 % (centre-cropped)
      2. MRI            — bright centre vs dark border
      3. Ultrasound     — high Laplacian speckle variance
      4. Mammogram      — fallback
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
               if len(img_bgr.shape) == 3 else img_bgr.copy()

        # 1. Histopathology
        he_pct = _he_pixel_pct(img_bgr)
        if he_pct > 8.0:
            print(f"   [AUTO] histopathology  (H&E pct={he_pct:.1f}%)")
            return 'histopathology'

        # 2. MRI — bright centre, dark border
        h4, w4  = gray.shape[0]//4, gray.shape[1]//4
        border  = float(np.mean([gray[0].mean(), gray[-1].mean(),
                                 gray[:, 0].mean(), gray[:, -1].mean()]))
        centre  = float(gray[h4:3*h4, w4:3*w4].mean())
        c_ratio = (centre - border) / (centre + 1e-6)
        if c_ratio > 0.30:
            print(f"   [AUTO] mri  (centre_ratio={c_ratio:.2f})")
            return 'mri'

        # 3. Ultrasound — speckle noise → high Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var > 500:
            print(f"   [AUTO] ultrasound  (lap={lap_var:.0f})")
            return 'ultrasound'

        # 4. Fallback
        print(f"   [AUTO] mammogram_dmid  (fallback, he={he_pct:.1f}% lap={lap_var:.0f})")
        return 'mammogram_dmid'

    except Exception as e:
        print(f"   [AUTO] Exception: {e} — fallback mammogram_dmid")
        return 'mammogram_dmid'


# =====================================================================
# 7. VALIDATION
# =====================================================================

_BLUR_THRESH = {
    'mammogram_mias': 8,  'mammogram_dmid': 8,
    'mri': 12,            'ultrasound': 20,
    'histopathology': 20,
}
_CONTRAST_THRESH = {
    'mammogram_mias': 6,  'mammogram_dmid': 6,
    'mri': 8,             'ultrasound': 8,
    'histopathology': 8,
}


def validate_medical_image(img_bgr, scan_type):
    try:
        h, w = img_bgr.shape[:2]
        if h < 128 or w < 128:
            return False, "Resolution too low (min 128×128)."
        if h > 4096 or w > 4096:
            return False, "Image too large (max 4096×4096)."

        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                  if len(img_bgr.shape) == 3 else img_bgr.copy()
        mean_px = float(np.mean(gray))

        if mean_px < 4:
            return False, "Image is blank/black. Please retake the scan."
        if mean_px > 251:
            return False, "Image is overexposed. Please retake."

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < _BLUR_THRESH.get(scan_type, 20):
            return False, (f"Image appears too blurry (sharpness={lap_var:.1f}, "
                           f"min={_BLUR_THRESH.get(scan_type, 20)}). "
                           "Please upload a sharper scan.")

        if float(np.std(gray)) < _CONTRAST_THRESH.get(scan_type, 8):
            return False, "Insufficient contrast. Please use a properly acquired scan."

        # Colour check only for grayscale modalities
        if len(img_bgr.shape) == 3 and scan_type not in COLOUR_MODALITIES:
            ch_means  = [float(img_bgr[:, :, c].mean()) for c in range(3)]
            if float(np.std(ch_means)) > 35:
                return False, (f"This does not look like a {scan_type.replace('_',' ')} scan. "
                               "Please upload a proper grayscale medical image.")

        # Screenshot heuristic — use centre crop to avoid dataset text labels
        h_c, w_c = gray.shape
        gray_center = gray[int(h_c*0.05):int(h_c*0.95), int(w_c*0.05):int(w_c*0.95)]
        edges = cv2.Canny(gray_center, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=120,
                                minLineLength=int(w_c * 0.75), maxLineGap=10)
        if lines is not None and len(lines) > 6:
            return False, "Image appears to be a screenshot. Please crop to the scan only."

        # Solid-colour check
        blocks = _block_variance(gray)
        if blocks.size > 0 and np.sum(blocks < 5) / blocks.size > 0.85:
            return False, "Image appears to be a solid colour — not a medical scan."

        return True, None
    except Exception:
        return True, None


def _block_variance(gray, bs=32):
    h, w   = gray.shape
    rows, cols = h // bs, w // bs
    v = [float(np.var(gray[r*bs:(r+1)*bs, c*bs:(c+1)*bs]))
         for r in range(rows) for c in range(cols)]
    return np.array(v) if v else np.array([100.0])


# =====================================================================
# 8. MAMMOGRAM PREPROCESSING
# =====================================================================

def preprocess_mammogram(img_bgr):
    try:
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                   if len(img_bgr.shape) == 3 else img_bgr.copy()
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception:
        return img_bgr


def crop_to_breast_tissue(img_bgr):
    try:
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                    if len(img_bgr.shape) == 3 else img_bgr
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        k         = np.ones((5, 5), np.uint8)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
        cnts, _   = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c          = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad        = max(5, min(w, h) // 20)
            hi, wi     = img_bgr.shape[:2]
            return img_bgr[max(0,y-pad):min(hi,y+h+pad),
                           max(0,x-pad):min(wi,x+w+pad)]
        return img_bgr
    except Exception:
        return img_bgr


def get_tissue_mask_advanced(img_bgr):
    try:
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                    if len(img_bgr.shape) == 3 else img_bgr
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k      = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
        return cv2.erode(thresh, np.ones((7, 7), np.uint8), iterations=1)
    except Exception:
        return np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255


# =====================================================================
# 9. CV ABNORMALITY ANALYSIS (mammogram only)
# =====================================================================

def comprehensive_abnormality_analysis(img_bgr):
    try:
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                   if len(img_bgr.shape) == 3 else img_bgr
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        _, bright = cv2.threshold(enhanced, 210, 255, cv2.THRESH_BINARY)
        cnts, _   = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions, total_area = [], 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area <= 30: continue
            perim = cv2.arcLength(cnt, True)
            circ  = (4*np.pi*area/perim**2) if perim > 0 else 0
            mask  = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mInt  = cv2.mean(enhanced, mask=mask)[0]
            score = (3 if area>100 else 2 if area>50 else 1)
            score += (2 if mInt>230 else 1 if mInt>220 else 0)
            score += (1 if circ>0.6 else 0)
            if score >= 3:
                regions.append({'area': area, 'circularity': circ,
                                'intensity': mInt, 'score': score})
                total_area += area

        n = len(regions)
        if n == 0:
            sev, conf, cat = 0.0, 0.9, "Normal"
        elif n == 1:
            r = regions[0]
            sev, cat, conf = ((0.8, "Malignant", 0.7)
                              if r['area'] > 500 or r['intensity'] > 235
                              else (0.5, "Benign", 0.7))
        else:
            sev, cat, conf = ((0.9, "Malignant", 0.8)
                              if any(r['area'] > 300 for r in regions)
                              else (0.6, "Benign", 0.8))

        return sev, conf, {
            'suspicious_regions': n,
            'total_area':   int(total_area),
            'max_intensity': max(r['intensity'] for r in regions) if regions else 0,
            'cv_prediction': cat,
        }
    except Exception:
        return 0.0, 0.0, {}


# =====================================================================
# 10. GRAD-CAM
# =====================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, m, i, o):    self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients   = go[0].detach()
    def remove_hooks(self):          [h.remove() for h in self._hooks]

    def __call__(self, spatial, freq, class_idx=None):
        try:
            self.model.eval()
            logits, _ = self.model(spatial, freq)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward(retain_graph=True)
            if self.gradients is None or self.activations is None:
                return None
            grads   = self.gradients.cpu().numpy()[0]   # (C, H, W)
            acts    = self.activations.cpu().numpy()[0]  # (C, H, W)
            weights = np.mean(grads, axis=(1, 2))        # (C,)
            cam     = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        except Exception as e:
            print(f"[GradCAM] {e}")
            traceback.print_exc()
            return None


def get_target_layer(model):
    """Return last Conv2d with ≥ 256 channels in backbone, else largest in whole model."""
    try:
        best = None
        for m in model.backbone.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 256:
                best = m
        if best: return best
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 128:
                best = m
        if best: return best
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d): return m
    except Exception:
        pass
    return None


def generate_heatmap_overlay(img_bgr, cam, prediction_label, scan_type):
    """
    FIX: For colour modalities (histo/ultrasound) the tissue mask must be
    all-True (no black-border masking). Also fixed: cam is resized to match
    img_bgr dimensions before element-wise operations.
    """
    try:
        if cam is None:
            return None

        img    = img_bgr.copy()
        h_img, w_img = img.shape[:2]

        # Refine cam with image intensity
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
                  if len(img.shape) == 3 else img
        gray_rs = cv2.resize(gray.astype(np.float32) / 255.0, (w_img, h_img))
        cam_rs  = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        cam_rs  = np.maximum(cam_rs, 0)
        if cam_rs.max() > 0: cam_rs /= cam_rs.max()

        intensity = np.power(gray_rs, 0.7)
        refined   = cam_rs * 0.7 + intensity * 0.3
        if refined.max() > 0: refined /= refined.max()

        # Tissue mask
        if 'mammogram' in scan_type:
            tissue = get_tissue_mask_advanced(img) > 0
        else:
            tissue = np.ones((h_img, w_img), dtype=bool)

        # Threshold
        thr = (0.25 if prediction_label in ('Malignant', 'Sick') else
               0.30 if prediction_label == 'Benign' else 0.35)
        if scan_type in COLOUR_MODALITIES:
            thr = max(thr - 0.05, 0.15)

        masked = refined * tissue
        masked[masked < thr] = 0
        if masked.max() > 0: masked /= masked.max()
        masked = cv2.GaussianBlur(masked, (15, 15), 0)

        colored   = cv2.applyColorMap(np.uint8(255 * masked), cv2.COLORMAP_JET)
        overlay   = img.copy()
        heat_mask = (masked > 0.01) & tissue
        if np.any(heat_mask):
            overlay[heat_mask] = cv2.addWeighted(
                colored[heat_mask], 0.65, img[heat_mask], 0.35, 0)

        _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[Heatmap] {e}")
        traceback.print_exc()
        return None


# =====================================================================
# 11. DOCTOR EXTRAS
# =====================================================================

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    result = {'lesion_count': 0, 'lesions': [],
              'total_lesion_area_pct': 0.0, 'overall_morphology': 'N/A'}
    try:
        if cam is None: return result
        h_img, w_img = img_bgr.shape[:2]
        cam_rs = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        if cam_rs.max() > 0: cam_rs /= cam_rs.max()

        binary = (cam_rs > 0.45).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
               if len(img_bgr.shape) == 3 else img_bgr
        total_px = h_img * w_img
        total_area, lesions = 0, []

        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < 60: continue
            x, y, w, h = cv2.boundingRect(cnt)
            perim    = cv2.arcLength(cnt, True)
            circ     = (4*np.pi*area/perim**2) if perim > 0 else 0
            hull     = cv2.convexHull(cnt)
            hull_a   = cv2.contourArea(hull)
            solidity = area / hull_a if hull_a > 0 else 0
            ar       = w / h if h > 0 else 1.0
            mask_r   = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask_r, [cnt], -1, 255, -1)
            mInt     = float(cv2.mean(gray, mask=mask_r)[0])
            stdInt   = float(np.std(gray[mask_r > 0])) if mask_r.any() else 0.0
            lesions.append({
                'lesion_id':          i + 1,
                'area_pixels':        int(area),
                'area_percentage':    round(area / total_px * 100, 2),
                'bounding_box':       {'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
                'circularity':        round(float(circ), 3),
                'solidity':           round(float(solidity), 3),
                'aspect_ratio':       round(float(ar), 3),
                'mean_intensity':     round(mInt, 2),
                'std_intensity':      round(stdInt, 2),
                'irregularity_score': round(1 - float(circ), 3),
                'morphology':         _classify_morphology(circ, solidity, ar),
                'suspicion_level':    _suspicion_level(circ, solidity, mInt),
            })
            total_area += area

        lesions.sort(key=lambda l: l['area_pixels'], reverse=True)
        result.update({
            'lesion_count':          len(lesions),
            'lesions':               lesions,
            'total_lesion_area_pct': round(total_area / total_px * 100, 2),
            'overall_morphology':    (lesions[0]['morphology']
                                     if lesions else 'No detectable lesion'),
        })
    except Exception as e:
        print(f"[Lesion] {e}")
    return result


def _classify_morphology(c, s, ar):
    if c > 0.80 and s > 0.90: return "Round / Oval — Low suspicion"
    if c > 0.60 and s > 0.80: return "Oval with smooth margins — Low-intermediate suspicion"
    if c < 0.40 or  s < 0.65: return "Irregular / Spiculated — High suspicion"
    if ar > 1.5 or  ar < 0.67: return "Elongated — Moderate suspicion"
    return "Lobular — Moderate suspicion"


def _suspicion_level(c, s, mInt):
    sc = (2 if c < 0.5 else 0) + (2 if s < 0.7 else 0) + (1 if mInt > 200 else 0)
    return "High" if sc >= 4 else "Moderate" if sc >= 2 else "Low"


def compute_texture_features(img_bgr, cam=None):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
               if len(img_bgr.shape) == 3 else img_bgr.copy()
        h_img, w_img = gray.shape
        if cam is not None:
            cam_rs = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
            roi = gray[(cam_rs > 0.45)]
        else:
            roi = gray.flatten()
        if len(roi) == 0: roi = gray.flatten()
        f    = roi.astype(np.float64)
        mu   = float(np.mean(f))
        std  = float(np.std(f))
        skew = float(np.mean(((f-mu)/std)**3)) if std > 0 else 0.0
        kurt = float(np.mean(((f-mu)/std)**4) - 3) if std > 0 else 0.0
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist / hist.sum(); hist = hist[hist > 0]
        ent  = float(-np.sum(hist * np.log2(hist)))
        return {'mean_intensity': round(mu,2), 'std_deviation': round(std,2),
                'skewness': round(skew,4),     'kurtosis':      round(kurt,4),
                'entropy':  round(ent,4),       'dynamic_range': int(gray.max())-int(gray.min())}
    except Exception as e:
        print(f"[Texture] {e}"); return {}


def compute_risk_score(final_pred, final_conf, cv_sev, lesion_data, scan_type):
    try:
        score, reasoning = 1, []
        if final_pred in ('Normal', 'Healthy'):
            score = 1 if final_conf >= 0.70 else 2
            reasoning.append("AI predicts no significant abnormality.")
        elif final_pred == 'Benign':
            score = 3 if final_conf < 0.65 else 2
            reasoning.append("AI detects likely benign finding.")
        elif final_pred in ('Malignant', 'Sick'):
            score = 5 if final_conf >= 0.85 else 4 if final_conf >= 0.70 else 3
            reasoning.append("AI detects suspicious/malignant pattern.")

        if cv_sev >= 0.8 and score < 4:
            score += 1
            reasoning.append("CV analysis found high-intensity suspicious regions.")
        elif cv_sev >= 0.6 and score < 3:
            score += 1
            reasoning.append("CV analysis found moderate imaging abnormality.")

        hi = [l for l in lesion_data.get('lesions', []) if l.get('suspicion_level')=='High']
        if hi and score < 4:
            score = min(score+1, 5)
            reasoning.append(f"{len(hi)} high-suspicion lesion(s) detected.")

        score  = min(score, 5)
        labels = {1:"Negative — No significant finding",
                  2:"Benign finding — Routine follow-up",
                  3:"Probably benign — Short-interval follow-up (6 months)",
                  4:"Suspicious — Tissue sampling recommended",
                  5:"Highly suggestive of malignancy — Biopsy required"}
        intervals = {1:"Routine annual screening", 2:"Routine annual screening",
                     3:"Follow-up imaging in 6 months",
                     4:"Biopsy / specialist referral within 2 weeks",
                     5:"Urgent biopsy required"}
        return {'risk_score': score, 'risk_category': f"Category {score}",
                'risk_label': labels.get(score,""),
                'recommended_interval': intervals.get(score,""),
                'reasoning': reasoning}
    except Exception as e:
        print(f"[RiskScore] {e}"); return {}


def generate_annotated_image(img_bgr, cam, prediction_label):
    try:
        annotated = img_bgr.copy()
        if cam is None: return None
        h_img, w_img = img_bgr.shape[:2]
        cam_rs = cv2.resize(cam, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        if cam_rs.max() > 0: cam_rs /= cam_rs.max()

        binary = (cam_rs > 0.45).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = {'Malignant':(0,0,255), 'Sick':(0,0,255),
                  'Benign':(0,165,255),
                  'Normal':(0,255,0),   'Healthy':(0,255,0)
                  }.get(prediction_label, (255,255,0))
        for i, cnt in enumerate(cnts):
            if cv2.contourArea(cnt) < 60: continue
            cv2.drawContours(annotated, [cnt], -1, colour, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x,y), (x+w,y+h), colour, 1)
            cv2.putText(annotated, f"R{i+1}", (x, max(15, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[Annotated] {e}"); return None


def _clinical_notes(pred, conf, lesion_data, texture_data, scan_type):
    notes = [f"Scan type: {scan_type.replace('_',' ').title()}",
             f"Primary finding: {pred} (confidence {conf:.0%})"]
    if conf < 0.60:
        notes.append("Warning: Low AI confidence — independent radiologist review strongly advised.")
    lc = lesion_data.get('lesion_count', 0)
    if lc > 0:
        notes.append(f"{lc} region(s) of interest identified.")
        for l in lesion_data.get('lesions', []):
            notes.append(f"  Region {l['lesion_id']}: {l['morphology']} | "
                         f"Area {l['area_percentage']:.1f}% | Suspicion: {l['suspicion_level']}")
    else:
        notes.append("No distinct lesion boundaries detected in attention map.")
    if texture_data:
        notes.append(f"Texture: entropy={texture_data.get('entropy','N/A')}, "
                     f"std={texture_data.get('std_deviation','N/A')}, "
                     f"skewness={texture_data.get('skewness','N/A')}")
    notes.append("Note: AI output is decision-support only. "
                 "Clinical judgement and histopathological confirmation are required.")
    return notes


# =====================================================================
# 12. INFERENCE HELPER
# =====================================================================

def run_inference(model, labels, spatial_t, freq_t):
    with torch.no_grad():
        logits, _ = model(spatial_t, freq_t)
        probs     = F.softmax(logits, dim=1)
        arr       = probs[0].cpu().numpy()
        conf, idx = torch.max(probs, dim=1)
    return labels[idx.item()], float(conf.item()), arr


# =====================================================================
# 13. PREDICT ROUTE
# =====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send image with key "file".'}), 400

    file      = request.files['file']
    mode      = request.form.get('mode', 'patient').lower()
    scan_type = request.form.get('scan_type', 'auto').strip().lower()

    if scan_type not in VALID_SCAN_TYPES:
        return jsonify({'error': f'Unknown scan_type "{scan_type}". '
                                 f'Valid: {sorted(VALID_SCAN_TYPES)}'}), 400

    try:
        raw = file.read()
        if not raw:
            return jsonify({'error': 'Uploaded file is empty.'}), 400
        arr     = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Cannot decode image. '
                                     'Please upload a valid JPEG, PNG, or BMP file.'}), 400

        # ── Step 1: resolve concrete model key ────────────────────────────
        if scan_type in ('auto', 'mammogram'):
            detected_scan = auto_detect_scan_type(img_bgr)
        else:
            detected_scan = scan_type

        if detected_scan == 'mammogram':
            detected_scan = 'mammogram_dmid'

        print(f" * [ROUTE] requested={scan_type!r}  detected={detected_scan!r}")

        # ── Step 2: validate ──────────────────────────────────────────────
        ok, reason = validate_medical_image(img_bgr, detected_scan)
        if not ok:
            return jsonify({'error': 'Invalid image', 'message': reason,
                            'action': 'Please retake or re-upload the scan.'}), 422

        # ── Step 3: preprocessing ─────────────────────────────────────────
        if 'mammogram' in detected_scan:
            img_cropped = crop_to_breast_tissue(img_bgr)
            img_proc    = preprocess_mammogram(img_cropped)
        else:
            img_cropped = img_bgr
            img_proc    = img_bgr

        img_resized = cv2.resize(img_proc, (IMG_SIZE, IMG_SIZE))
        img_rgb_res = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Pass scan type so colour modalities keep their colour channels
        spatial_t, freq_t = prepare_inputs(img_rgb_res, detected_scan)

        cv_sev, cv_details = 0.0, {}

        # ── Step 4: inference ─────────────────────────────────────────────
        if 'mammogram' in detected_scan and scan_type in ('auto', 'mammogram'):
            # Compare MIAS vs DMID
            try:
                model_m, labels_m = load_model('mammogram_mias')
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503
            pred_m, conf_m, probs_m = run_inference(model_m, labels_m, spatial_t, freq_t)

            try:
                model_d, labels_d = load_model('mammogram_dmid')
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503
            pred_d, conf_d, probs_d = run_inference(model_d, labels_d, spatial_t, freq_t)

            if conf_d >= conf_m:
                ai_pred, ai_score = pred_d, conf_d
                all_probs, win_labels = probs_d, labels_d
                winning_model = 'DMID'
            else:
                ai_pred, ai_score = pred_m, conf_m
                all_probs, win_labels = probs_m, labels_m
                winning_model = 'MIAS'
                try: load_model('mammogram_mias')
                except RuntimeError as e: return jsonify({'error': str(e)}), 503
        else:
            # Single model — direct load
            try:
                model_x, labels_x = load_model(detected_scan)
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503
            ai_pred, ai_score, all_probs = run_inference(
                model_x, labels_x, spatial_t, freq_t)
            win_labels    = labels_x
            winning_model = detected_scan

        print(f" * [INFER] model={winning_model}  pred={ai_pred}  conf={ai_score:.3f}")

        # ── Step 5: CV override (mammogram ONLY) ─────────────────────────
        srt    = np.sort(all_probs)[::-1]
        margin = float(srt[0] - srt[1]) if len(all_probs) > 1 else 1.0

        override   = False
        final_pred = ai_pred
        final_conf = float(ai_score)

        if 'mammogram' in detected_scan:
            cv_sev, cv_conf, cv_details = comprehensive_abnormality_analysis(img_cropped)
            cv_pred = cv_details.get('cv_prediction', 'Normal')
            if ai_pred == 'Normal' and cv_sev > 0.75 and margin < 0.20 and cv_conf > 0.75:
                override, final_pred, final_conf = True, cv_pred, cv_conf
                print(f" * [CV-OVERRIDE] Normal→{cv_pred}")
            elif ai_pred == 'Benign' and cv_sev > 0.90 and margin < 0.15 and cv_conf > 0.80:
                override, final_pred, final_conf = True, 'Malignant', cv_conf
                print(f" * [CV-OVERRIDE] Benign→Malignant")

        # ── Step 6: recommendation ────────────────────────────────────────
        if final_pred in ('Malignant', 'Sick'):
            rec = "HIGH RISK: Immediate biopsy and specialist consultation required."
        elif final_pred == 'Benign':
            rec = "MEDIUM RISK: Close monitoring with follow-up imaging in 3–6 months."
        else:
            rec = "LOW RISK: Continue routine annual screening."

        # ── Step 7: Grad-CAM ──────────────────────────────────────────────
        heatmap_b64   = None
        annotated_b64 = None
        cam_mask      = None
        lesion_data   = {}
        texture_data  = {}
        risk_data     = {}

        target_layer = get_target_layer(loaded_model)
        if target_layer is not None:
            try:
                final_idx = (win_labels.index(final_pred)
                             if final_pred in win_labels else 0)
                gcam     = GradCAM(loaded_model, target_layer)
                cam_mask = gcam(spatial_t, freq_t, class_idx=final_idx)
                gcam.remove_hooks()
                if cam_mask is not None:
                    heatmap_b64 = generate_heatmap_overlay(
                        img_resized, cam_mask, final_pred, detected_scan)
            except Exception as e:
                print(f"[GradCAM-run] {e}")
                traceback.print_exc()

        # ── Step 8: doctor extras ─────────────────────────────────────────
        if mode == 'doctor':
            lesion_data  = detect_lesion_boundaries(img_resized, cam_mask, detected_scan)
            texture_data = compute_texture_features(img_resized, cam_mask)
            risk_data    = compute_risk_score(final_pred, final_conf,
                                             cv_sev, lesion_data, detected_scan)
            if cam_mask is not None:
                annotated_b64 = generate_annotated_image(img_resized, cam_mask, final_pred)

        # ── Step 9: response ──────────────────────────────────────────────
        response = {
            'result':         final_pred,
            'confidence':     f"{final_conf:.2f}",
            'heatmap':        heatmap_b64,
            'recommendation': rec,
            'analysis_details': {
                'scan_type_detected': detected_scan,
                'winning_model':      winning_model,
                'ai_prediction':      ai_pred,
                'ai_confidence':      f"{ai_score:.2f}",
                'all_class_probs': {
                    win_labels[i]: round(float(all_probs[i]), 4)
                    for i in range(len(win_labels))
                },
                'cv_severity':      f"{cv_sev:.2f}",
                'cv_details':       cv_details,
                'override_active':  override,
                'certainty_margin': f"{margin:.2f}",
            },
        }

        if mode == 'doctor':
            response['doctor_report'] = {
                'annotated_image':  annotated_b64,
                'lesion_analysis':  lesion_data,
                'texture_features': texture_data,
                'risk_score':       risk_data,
                'clinical_notes':   _clinical_notes(
                    final_pred, float(final_conf),
                    lesion_data, texture_data, detected_scan),
            }

        return jsonify(response)

    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': 'Internal server error. Please try again.'}), 500


# =====================================================================
# 14. UTILITY ROUTES
# =====================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status':       'online',
        'active_model': active_model_type,
        'img_size':     IMG_SIZE,
        'endpoints': {
            'POST /predict': {
                'scan_type': ('auto | mammogram | mammogram_mias | '
                              'mammogram_dmid | ultrasound | histopathology | mri'),
                'mode': 'patient (default) | doctor',
            }
        },
    }), 200


@app.route('/models', methods=['GET'])
def model_status():
    return jsonify({
        name: {'path': path, 'exists': os.path.exists(path),
               'active': (name == active_model_type)}
        for name, path in MODEL_PATHS.items()
    }), 200


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum is 15 MB.'}), 413

@app.errorhandler(400)
def bad_req(e):
    return jsonify({'error': str(e)}), 400

@app.errorhandler(500)
def server_err(e):
    gc.collect()
    return jsonify({'error': 'Internal server error.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
