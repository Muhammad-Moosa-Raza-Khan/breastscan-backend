"""
MedScan API — Production Backend
Real MedFormerULTRA architecture integrated from training code.
FIXED VERSION — all bugs corrected.
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
from PIL import Image
from flask import Flask, request, jsonify

# =====================================================================
# 1. APP INIT
# =====================================================================
app = Flask(__name__)
application = app
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024   # 15 MB hard cap

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

active_model_type = None
loaded_model      = None
detected_classes  = []

# =====================================================================
# 2. REAL ARCHITECTURE — MedFormerULTRA
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.gate(x)
        stacked = torch.stack([e(x) for e in self.experts], dim=1)
        w       = weights.view(-1, self.num_experts, 1, 1, 1)
        return (stacked * w).sum(dim=1), weights


class MedFormerULTRA(nn.Module):
    FEATURE_DIM  = 256
    NUM_EXPERTS  = 3

    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=False, features_only=True)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
        self.feat_dims = [feats[-2].shape[1], feats[-1].shape[1]]

        cd = self.FEATURE_DIM
        self.proj1 = nn.Sequential(
            nn.Conv2d(self.feat_dims[0], cd, 1),
            nn.BatchNorm2d(cd), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(
            nn.Conv2d(self.feat_dims[1], cd, 1),
            nn.BatchNorm2d(cd), nn.ReLU(inplace=True))

        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.moe       = MoE(cd, self.NUM_EXPERTS)
        self.attention = EfficientAttention(cd, num_heads=4)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(
            nn.Linear(cd + 64, 512),
            nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.aux_classifier = nn.Linear(cd, num_classes)

    def forward(self, x, freq, return_all=False):
        feats    = self.backbone(x)
        f1, f2   = feats[-2], feats[-1]
        f1       = self.proj1(f1)
        f2       = self.proj2(f2)
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        fused    = f1 + f2

        feat, gate_weights = self.moe(fused)

        B, C, H, W  = feat.shape
        feat_down   = F.adaptive_avg_pool2d(feat, (H // 2, W // 2))
        feat_flat   = feat_down.flatten(2).transpose(1, 2)
        feat_att    = self.attention(feat_flat)
        feat_spatial = feat_att.transpose(1, 2).reshape(B, C, H // 2, W // 2)
        feat_spatial = F.interpolate(feat_spatial, size=(H, W),
                                     mode='bilinear', align_corners=False)
        feat = feat + feat_spatial

        pooled    = self.global_pool(feat).flatten(1)
        freq_feat = self.freq_encoder(freq).flatten(1)
        combined  = torch.cat([pooled, freq_feat], dim=1)
        fused_feat = self.fusion(combined)

        logits     = self.classifier(fused_feat)
        aux_logits = self.aux_classifier(pooled)

        if return_all:
            return logits, aux_logits, gate_weights
        return logits, aux_logits


# =====================================================================
# 3. FREQUENCY FEATURE EXTRACTOR
# =====================================================================

def _dct2(img_gray_f32):
    return dct(dct(img_gray_f32.T, norm='ortho').T, norm='ortho')


def extract_freq_tensor(img_rgb_f32, target_size=320):
    gray = cv2.cvtColor(
        (img_rgb_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    dct_feat = _dct2(gray)
    h, w     = dct_feat.shape
    ch, cw   = int(h * 0.3), int(w * 0.3)
    low_freq = dct_feat.copy()
    low_freq[ch:, :] = 0
    low_freq[:, cw:] = 0

    freq3 = np.stack([low_freq, low_freq, low_freq], axis=2)
    freq3 = cv2.resize(freq3, (target_size, target_size),
                       interpolation=cv2.INTER_LINEAR)

    vmax = np.abs(freq3).max()
    if vmax > 0:
        freq3 = freq3 / vmax

    return torch.from_numpy(freq3.transpose(2, 0, 1)).float()


# =====================================================================
# 4. TRANSFORMS
# =====================================================================

IMG_SIZE = 320

spatial_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_inputs(img_rgb_uint8):
    gray_u8   = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray_u8)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    spatial = spatial_transform(enhanced_rgb)
    freq    = extract_freq_tensor(enhanced_rgb, IMG_SIZE)

    return spatial.unsqueeze(0), freq.unsqueeze(0)


# =====================================================================
# 5. MODEL LOADER
# =====================================================================

def load_pytorch_model(scan_type):
    global loaded_model, active_model_type, detected_classes

    if active_model_type == scan_type and loaded_model is not None:
        return True

    if loaded_model is not None:
        print(f" * [GATEKEEPER] Unloading {active_model_type}…")
        del loaded_model
        loaded_model = None
        gc.collect()

    path   = MODEL_PATHS.get(scan_type)
    config = DATASET_CONFIGS.get(scan_type)

    if not path or not os.path.exists(path):
        print(f" ! [ERROR] Model file not found: {path}")
        return False

    detected_classes = config['labels']
    print(f" * [GATEKEEPER] Loading {scan_type} ({config['num_classes']} classes)…")

    try:
        model      = MedFormerULTRA(num_classes=config['num_classes'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   [INFO] Checkpoint — epoch: {checkpoint.get('epoch','?')}, "
                  f"best_acc: {checkpoint.get('best_acc','?')}")
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   [WARN] Missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"   [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        if not missing and not unexpected:
            print("   [OK] All weights loaded — zero mismatches.")

        model.eval()
        loaded_model      = model
        active_model_type = scan_type

        healthy, preds = _verify_model_diversity(model, config['num_classes'])
        if not healthy:
            print(f"   [WARN] Same prediction for all test inputs {preds}.")
        else:
            print(f"   [OK] Diversity check passed: {preds}")

        return True

    except Exception as e:
        print(f" ! [FATAL] Cannot load {scan_type}: {e}")
        traceback.print_exc()
        loaded_model      = None
        active_model_type = None
        return False


def _verify_model_diversity(model, num_classes):
    preds = []
    try:
        with torch.no_grad():
            for fill in [0.0, 0.5, None]:
                x = (torch.zeros(1,3,IMG_SIZE,IMG_SIZE) if fill == 0.0 else
                     torch.ones(1,3,IMG_SIZE,IMG_SIZE)*0.5 if fill == 0.5 else
                     torch.rand(1,3,IMG_SIZE,IMG_SIZE))
                f = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
                logits, _ = model(x, f)
                preds.append(torch.argmax(logits, dim=1).item())
        return len(set(preds)) > 1, preds
    except Exception:
        return True, preds


load_pytorch_model('mammogram_dmid')


# =====================================================================
# 6. MEDICAL IMAGE VALIDATOR — FIXED
# =====================================================================

def validate_medical_image(img_cv2, scan_type):
    """
    FIX 1: Separate blur/contrast thresholds per modality.
    Mammograms are naturally low-contrast so they need looser thresholds.
    Histopathology images are naturally colourful so colour check is skipped.
    """
    try:
        h, w = img_cv2.shape[:2]
        if h < 128 or w < 128:
            return False, "Resolution too low. Please upload a minimum 128x128 scan."
        if h > 4096 or w > 4096:
            return False, "Image too large (max 4096×4096). Resize before uploading."

        gray    = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                   if len(img_cv2.shape) == 3 else img_cv2.copy())
        mean_px = float(np.mean(gray))

        if mean_px < 5:
            return False, "Image is blank/black. Please retake the scan."
        if mean_px > 250:
            return False, "Image is overexposed/all-white. Please retake with correct exposure."

        # ── Per-modality blur threshold ──────────────────────────────
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if scan_type in ('mammogram_mias', 'mammogram_dmid'):
            blur_threshold     = 10   # mammograms are naturally soft
            contrast_threshold = 8
        elif scan_type == 'mri':
            blur_threshold     = 15
            contrast_threshold = 10
        else:
            blur_threshold     = 30
            contrast_threshold = 12

        if laplacian_var < blur_threshold:
            return False, "Image is too blurry. Please use a sharper scan."
        if float(np.std(gray)) < contrast_threshold:
            return False, "Insufficient contrast. Please use a properly acquired scan."

        # ── Colour check: skip for histopathology (always colourful) ─
        if len(img_cv2.shape) == 3 and scan_type != 'histopathology':
            ch_means  = [float(img_cv2[:, :, c].mean()) for c in range(3)]
            color_var = float(np.std(ch_means))
            if color_var > 30:
                return False, (f"This does not look like a {scan_type.replace('_',' ')} scan. "
                               "Please upload a proper grayscale medical image.")

        # ── Screenshot detection ─────────────────────────────────────
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=120,
                                minLineLength=int(w*0.75), maxLineGap=10)
        if lines is not None and len(lines) > 6:
            return False, "Image appears to be a screenshot. Please crop to the scan only."

        # ── Solid colour check ───────────────────────────────────────
        blocks = _block_variance(gray)
        if blocks.size > 0 and np.sum(blocks < 5) / blocks.size > 0.85:
            return False, "Image appears to be a solid colour — not a medical scan."

        return True, None
    except Exception:
        return True, None


def _block_variance(gray, bs=32):
    h, w = gray.shape
    rows, cols = h // bs, w // bs
    v = [float(np.var(gray[r*bs:(r+1)*bs, c*bs:(c+1)*bs]))
         for r in range(rows) for c in range(cols)]
    return np.array(v) if v else np.array([100.0])


# =====================================================================
# 7. AUTO SCAN-TYPE DETECTION — FIXED
# =====================================================================

def auto_detect_scan_type(img_cv2):
    """
    FIX 2: More reliable modality detection.

    Priority order:
      1. Histopathology  → strong colour saturation (H&E staining is pink/purple)
      2. MRI             → bright centre, dark borders, moderate texture
      3. Ultrasound      → high Laplacian variance (speckle noise)
      4. Mammogram       → fallback
    """
    try:
        gray = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape) == 3 else img_cv2.copy())

        # ── 1. Histopathology: check colour saturation ──────────────
        if len(img_cv2.shape) == 3:
            img_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
            mean_sat = float(img_hsv[:, :, 1].mean())
            # H&E slides have saturation typically > 60; grayscale scans < 20
            if mean_sat > 50:
                print(f"   [AUTO-DETECT] Histopathology (saturation={mean_sat:.1f})")
                return 'histopathology'

        # ── 2. MRI: bright centre relative to dark border ───────────
        h4, w4 = gray.shape[0]//4, gray.shape[1]//4
        border  = float(np.mean([gray[0].mean(), gray[-1].mean(),
                                  gray[:,0].mean(), gray[:,-1].mean()]))
        centre  = float(gray[h4:3*h4, w4:3*w4].mean())
        centre_ratio = (centre - border) / (centre + 1)
        if centre_ratio > 0.30:
            print(f"   [AUTO-DETECT] MRI (centre_ratio={centre_ratio:.2f})")
            return 'mri'

        # ── 3. Ultrasound: high speckle / Laplacian variance ────────
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var > 600:
            print(f"   [AUTO-DETECT] Ultrasound (laplacian={lap_var:.0f})")
            return 'ultrasound'

        # ── 4. Mammogram fallback ────────────────────────────────────
        print(f"   [AUTO-DETECT] Mammogram (fallback, laplacian={lap_var:.0f})")
        return 'mammogram_dmid'

    except Exception as e:
        print(f"   [AUTO-DETECT] Exception: {e} — falling back to mammogram_dmid")
        return 'mammogram_dmid'


# =====================================================================
# 8. MAMMOGRAM PREPROCESSING
# =====================================================================

def preprocess_mammogram(img_cv2):
    try:
        gray     = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                    if len(img_cv2.shape) == 3 else img_cv2.copy())
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception:
        return img_cv2


def crop_to_breast_tissue(img_cv2):
    try:
        gray      = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                     if len(img_cv2.shape) == 3 else img_cv2)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        k         = np.ones((5,5), np.uint8)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
        cnts, _   = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c        = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad      = max(5, min(w,h)//20)
            hi, wi   = img_cv2.shape[:2]
            return img_cv2[max(0,y-pad):min(hi,y+h+pad),
                           max(0,x-pad):min(wi,x+w+pad)]
        return img_cv2
    except Exception:
        return img_cv2


def get_tissue_mask_advanced(img_cv2):
    try:
        gray      = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                     if len(img_cv2.shape) == 3 else img_cv2)
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=3)
        return cv2.erode(thresh, np.ones((7,7), np.uint8), iterations=1)
    except Exception:
        return np.ones(img_cv2.shape[:2], dtype=np.uint8) * 255


# =====================================================================
# 9. CV ABNORMALITY ANALYSIS
# =====================================================================

def comprehensive_abnormality_analysis(img_cv2):
    try:
        gray    = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                   if len(img_cv2.shape) == 3 else img_cv2)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
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
                regions.append({'area':area,'circularity':circ,'intensity':mInt,'score':score})
                total_area += area

        n = len(regions)
        if n == 0:       sev, conf, cat = 0.0, 0.9, "Normal"
        elif n == 1:
            r = regions[0]
            if r['area']>500 or r['intensity']>235: sev,cat,conf = 0.8,"Malignant",0.7
            else:                                    sev,cat,conf = 0.5,"Benign",   0.7
        else:
            if any(r['area']>300 for r in regions): sev,cat,conf = 0.9,"Malignant",0.8
            else:                                    sev,cat,conf = 0.6,"Benign",   0.8

        return sev, conf, {
            'suspicious_regions': n,
            'total_area':         int(total_area),
            'max_intensity':      max(r['intensity'] for r in regions) if regions else 0,
            'cv_prediction':      cat,
        }
    except Exception:
        return 0.0, 0.0, {}


# =====================================================================
# 10. GRAD-CAM
# =====================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, m, i, o):    self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients   = go[0].detach()
    def remove_hooks(self): [h.remove() for h in self._hooks]

    def __call__(self, spatial, freq, class_idx=None):
        try:
            self.model.eval()
            logits, _ = self.model(spatial, freq)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[:, class_idx].backward(retain_graph=True)
            if self.gradients is None or self.activations is None:
                return None
            grads   = self.gradients.cpu().numpy()[0]
            acts    = self.activations.cpu().numpy()[0]
            weights = np.mean(grads, axis=(1,2))
            cam     = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        except Exception as e:
            print(f"[GradCAM] {e}")
            return None


def get_target_layer(model):
    try:
        if hasattr(model, 'backbone'):
            best = None
            for m in model.backbone.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels >= 256:
                    best = m
            if best: return best
        best = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 128:
                best = m
        if best: return best
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d): return m
    except Exception:
        pass
    return None


def refine_gradcam(cam, img_cv2):
    try:
        gray = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape) == 3 else img_cv2)
        intensity = cv2.resize(
            np.power(gray.astype(np.float32)/255.0, 0.7), (IMG_SIZE, IMG_SIZE))
        refined = cam * 0.7 + intensity * 0.3
        return refined / refined.max() if refined.max() > 0 else refined
    except Exception:
        return cam


def generate_heatmap_overlay(img_cv2, cam, prediction_label, scan_type):
    try:
        img    = img_cv2.copy()
        cam    = refine_gradcam(cam, img)
        tissue = (get_tissue_mask_advanced(img) > 0
                  if 'mammogram' in scan_type
                  else np.ones(img.shape[:2], dtype=bool))
        masked = cam * tissue
        thr    = (0.25 if prediction_label == 'Malignant' else
                  0.30 if prediction_label in ('Benign','Sick') else 0.35)
        if scan_type in ('histopathology','mri'):
            thr = max(thr - 0.05, 0.15)
        masked[masked < thr] = 0
        if masked.max() > 0: masked = masked / masked.max()
        masked    = cv2.GaussianBlur(masked, (15,15), 0)
        colored   = cv2.applyColorMap(np.uint8(255*masked), cv2.COLORMAP_JET)
        overlay   = img.copy()
        heat_mask = (masked > 0.01) & tissue
        if np.any(heat_mask):
            overlay[heat_mask] = cv2.addWeighted(
                colored[heat_mask], 0.65, img[heat_mask], 0.35, 0)
        _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[Heatmap] {e}")
        return None


# =====================================================================
# 11. DOCTOR EXTRAS
# =====================================================================

def detect_lesion_boundaries(img_cv2, cam, scan_type):
    result = {'lesion_count':0,'lesions':[],'total_lesion_area_pct':0.0,'overall_morphology':'N/A'}
    try:
        if cam is None: return result
        binary = (cam > 0.45).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray    = (cv2.cvtColor(img_cv2,cv2.COLOR_BGR2GRAY)
                   if len(img_cv2.shape)==3 else img_cv2)
        total_px, total_area, lesions = img_cv2.shape[0]*img_cv2.shape[1], 0, []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < 60: continue
            x,y,w,h = cv2.boundingRect(cnt)
            perim    = cv2.arcLength(cnt,True)
            circ     = (4*np.pi*area/perim**2) if perim>0 else 0
            hull     = cv2.convexHull(cnt)
            solidity = area/cv2.contourArea(hull) if cv2.contourArea(hull)>0 else 0
            ar       = w/h if h>0 else 1.0
            mask_r   = np.zeros(gray.shape,np.uint8)
            cv2.drawContours(mask_r,[cnt],-1,255,-1)
            mInt     = float(cv2.mean(gray,mask=mask_r)[0])
            stdInt   = float(np.std(gray[mask_r>0])) if mask_r.any() else 0.0
            lesions.append({
                'lesion_id':         i+1,
                'area_pixels':       int(area),
                'area_percentage':   round(area/total_px*100, 2),
                'bounding_box':      {'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
                'circularity':       round(float(circ),3),
                'solidity':          round(float(solidity),3),
                'aspect_ratio':      round(float(ar),3),
                'mean_intensity':    round(mInt,2),
                'std_intensity':     round(stdInt,2),
                'irregularity_score':round(1-float(circ),3),
                'morphology':        _classify_morphology(circ,solidity,ar),
                'suspicion_level':   _suspicion_level(circ,solidity,mInt),
            })
            total_area += area
        lesions.sort(key=lambda l:l['area_pixels'], reverse=True)
        result.update({
            'lesion_count':          len(lesions),
            'lesions':               lesions,
            'total_lesion_area_pct': round(total_area/total_px*100, 2),
            'overall_morphology':    lesions[0]['morphology'] if lesions else 'No detectable lesion',
        })
    except Exception as e:
        print(f"[Lesion] {e}")
    return result


def _classify_morphology(c, s, ar):
    if c>0.80 and s>0.90:   return "Round / Oval — Low suspicion"
    if c>0.60 and s>0.80:   return "Oval with smooth margins — Low-intermediate suspicion"
    if c<0.40 or s<0.65:    return "Irregular / Spiculated — High suspicion"
    if ar>1.5 or ar<0.67:   return "Elongated — Moderate suspicion"
    return "Lobular — Moderate suspicion"


def _suspicion_level(c, s, mInt):
    sc = (2 if c<0.5 else 0) + (2 if s<0.7 else 0) + (1 if mInt>200 else 0)
    return "High" if sc>=4 else "Moderate" if sc>=2 else "Low"


def compute_texture_features(img_cv2, cam=None):
    try:
        gray = (cv2.cvtColor(img_cv2,cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape)==3 else img_cv2.copy())
        roi  = gray[((cam>0.45).astype(np.uint8))>0] if (
            cam is not None and (cam>0.45).any()) else gray.flatten()
        if len(roi)==0: roi = gray.flatten()
        f   = roi.astype(np.float64)
        mu  = float(np.mean(f)); std = float(np.std(f))
        skew = float(np.mean(((f-mu)/std)**3)) if std>0 else 0.0
        kurt = float(np.mean(((f-mu)/std)**4)-3) if std>0 else 0.0
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist = hist/hist.sum(); hist = hist[hist>0]
        ent  = float(-np.sum(hist*np.log2(hist)))
        return {
            'mean_intensity': round(mu,2),  'std_deviation': round(std,2),
            'skewness': round(skew,4),       'kurtosis':      round(kurt,4),
            'entropy':  round(ent,4),        'dynamic_range': int(int(gray.max())-int(gray.min())),
        }
    except Exception as e:
        print(f"[Texture] {e}"); return {}


def compute_risk_score(final_pred, final_conf, cv_sev, lesion_data, scan_type):
    """
    FIX 3: Use final_pred (post-override) instead of raw ai_pred so that
    the BI-RADS category always matches the displayed diagnosis.
    """
    try:
        score, reasoning = 1, []
        if final_pred in ('Normal','Healthy'):
            score = 1 if final_conf >= 0.70 else 2
            reasoning.append("AI predicts no significant abnormality.")
        elif final_pred == 'Benign':
            score = 3 if final_conf < 0.65 else 2
            reasoning.append("AI detects likely benign finding.")
        elif final_pred in ('Malignant','Sick'):
            score = 5 if final_conf >= 0.85 else 4 if final_conf >= 0.70 else 3
            reasoning.append("AI detects suspicious/malignant pattern.")

        if cv_sev >= 0.8 and score < 4:
            score += 1; reasoning.append("CV analysis found high-intensity suspicious regions.")
        elif cv_sev >= 0.6 and score < 3:
            score += 1; reasoning.append("CV analysis found moderate imaging abnormality.")

        hi = [l for l in lesion_data.get('lesions',[]) if l.get('suspicion_level')=='High']
        if hi and score < 4:
            score = min(score+1, 5)
            reasoning.append(f"{len(hi)} high-suspicion lesion(s) detected.")

        score = min(score, 5)
        labels    = {1:"Negative — No significant finding",
                     2:"Benign finding — Routine follow-up",
                     3:"Probably benign — Short-interval follow-up (6 months)",
                     4:"Suspicious — Tissue sampling recommended",
                     5:"Highly suggestive of malignancy — Biopsy required"}
        intervals = {1:"Routine annual screening", 2:"Routine annual screening",
                     3:"Follow-up imaging in 6 months",
                     4:"Biopsy / specialist referral within 2 weeks",
                     5:"Urgent biopsy required"}
        return {'risk_score':score,'risk_category':f"Category {score}",
                'risk_label':labels.get(score,""),'recommended_interval':intervals.get(score,""),
                'reasoning':reasoning}
    except Exception as e:
        print(f"[RiskScore] {e}"); return {}


def generate_annotated_image(img_cv2, cam, prediction_label):
    try:
        annotated = img_cv2.copy()
        if cam is None: return None
        binary = (cam>0.45).astype(np.uint8)*255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        cnts,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = {'Malignant':(0,0,255),'Sick':(0,0,255),'Benign':(0,165,255),
                  'Normal':(0,255,0),'Healthy':(0,255,0)}.get(prediction_label,(255,255,0))
        for i,cnt in enumerate(cnts):
            if cv2.contourArea(cnt)<60: continue
            cv2.drawContours(annotated,[cnt],-1,colour,2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated,(x,y),(x+w,y+h),colour,1)
            cv2.putText(annotated,f"R{i+1}",(x,max(15,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,colour,2,cv2.LINE_AA)
        _,buf = cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,92])
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
        for l in lesion_data.get('lesions',[]):
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
# 12. PREDICT ROUTE — FIXED
# =====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send image with key "file".'}), 400

    file      = request.files['file']
    mode      = request.form.get('mode', 'patient').lower()
    scan_type = request.form.get('scan_type', 'mammogram_dmid').strip().lower()

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

        # ── Auto-detect BEFORE validation so we use the right thresholds ──
        if scan_type in ('auto', 'mammogram'):
            detected_scan = auto_detect_scan_type(img_bgr)
            print(f" * [AUTO-DETECT] -> {detected_scan}")
        else:
            detected_scan = scan_type

        # Resolve generic 'mammogram' alias
        if detected_scan == 'mammogram':
            detected_scan = 'mammogram_dmid'

        # ── Validate with correct scan type ───────────────────────────────
        ok, reason = validate_medical_image(img_bgr, detected_scan)
        if not ok:
            return jsonify({'error':'Invalid image','message':reason,
                            'action':'Please retake or re-upload the scan.'}), 422

        # ── Preprocessing ──────────────────────────────────────────────────
        if 'mammogram' in detected_scan:
            img_cropped = crop_to_breast_tissue(img_bgr)
            img_proc    = preprocess_mammogram(img_cropped)
        else:
            img_cropped = img_bgr
            img_proc    = img_bgr

        img_resized  = cv2.resize(img_proc, (IMG_SIZE, IMG_SIZE))
        img_rgb_res  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        spatial_t, freq_t = prepare_inputs(img_rgb_res)

        cv_sev     = 0.0
        cv_details = {}

        # ── FIX 4: Auto-mammogram compare — track classes per model ───────
        if scan_type in ('auto', 'mammogram') and 'mammogram' in detected_scan:
            print(" * [AUTO-COMPARE] MIAS…")
            if not load_pytorch_model('mammogram_mias'):
                return jsonify({'error':'MIAS model unavailable.'}), 503
            mias_classes = DATASET_CONFIGS['mammogram_mias']['labels'][:]
            with torch.no_grad():
                out_m, _ = loaded_model(spatial_t, freq_t)
                prob_m   = F.softmax(out_m, dim=1)
                conf_m, idx_m = torch.max(prob_m, 1)
                pred_m   = mias_classes[idx_m.item()]

            print(" * [AUTO-COMPARE] DMID…")
            if not load_pytorch_model('mammogram_dmid'):
                return jsonify({'error':'DMID model unavailable.'}), 503
            dmid_classes = DATASET_CONFIGS['mammogram_dmid']['labels'][:]
            with torch.no_grad():
                out_d, _ = loaded_model(spatial_t, freq_t)
                prob_d   = F.softmax(out_d, dim=1)
                conf_d, idx_d = torch.max(prob_d, 1)
                pred_d   = dmid_classes[idx_d.item()]

            if conf_d.item() >= conf_m.item():
                ai_pred      = pred_d
                ai_score     = conf_d.item()
                winning_model = 'DMID'
                all_probs    = prob_d[0].cpu().numpy()
                winning_classes = dmid_classes
                # Ensure DMID model stays loaded for GradCAM
                # (already loaded above)
            else:
                ai_pred      = pred_m
                ai_score     = conf_m.item()
                winning_model = 'MIAS'
                all_probs    = prob_m[0].cpu().numpy()
                winning_classes = mias_classes
                # Reload MIAS so GradCAM runs on the winning model
                load_pytorch_model('mammogram_mias')

            # Update global detected_classes to match the winning model
            detected_classes_local = winning_classes

        elif 'mammogram' in detected_scan and scan_type not in ('auto', 'mammogram'):
            # Explicit mammogram sub-type requested
            if not load_pytorch_model(detected_scan):
                return jsonify({'error':f'Model for "{detected_scan}" is unavailable.'}), 503
            with torch.no_grad():
                outputs, _ = loaded_model(spatial_t, freq_t)
                probs       = F.softmax(outputs, dim=1)
                all_probs   = probs[0].cpu().numpy()
                conf, cidx  = torch.max(probs, 1)
            ai_score = float(conf.item())
            ai_pred  = detected_classes[cidx.item()]
            winning_model = detected_scan
            detected_classes_local = detected_classes[:]

        else:
            # Non-mammogram modality
            if not load_pytorch_model(detected_scan):
                return jsonify({'error':f'Model for "{detected_scan}" is unavailable.'}), 503
            with torch.no_grad():
                outputs, _ = loaded_model(spatial_t, freq_t)
                probs       = F.softmax(outputs, dim=1)
                all_probs   = probs[0].cpu().numpy()
                conf, cidx  = torch.max(probs, 1)
            ai_score = float(conf.item())
            ai_pred  = detected_classes[cidx.item()]
            winning_model = detected_scan
            detected_classes_local = detected_classes[:]

        # ── CV override (mammogram only) ────────────────────────────────
        override   = False
        final_pred = ai_pred
        final_conf = float(ai_score)
        srt        = np.sort(all_probs)[::-1]
        margin     = float(srt[0]-srt[1]) if len(all_probs) > 1 else 1.0

        if 'mammogram' in detected_scan:
            cv_sev, cv_conf, cv_details = comprehensive_abnormality_analysis(img_cropped)
            cv_pred = cv_details.get('cv_prediction', 'Normal')
            if ai_pred == 'Normal' and cv_sev > 0.75 and margin < 0.20 and cv_conf > 0.75:
                override, final_pred, final_conf = True, cv_pred, cv_conf
            elif ai_pred == 'Benign' and cv_sev > 0.90 and margin < 0.15 and cv_conf > 0.80:
                override, final_pred, final_conf = True, 'Malignant', cv_conf

        # ── Recommendation ─────────────────────────────────────────────
        if final_pred in ('Malignant','Sick'):
            rec = "HIGH RISK: Immediate biopsy and specialist consultation required."
        elif final_pred == 'Benign':
            rec = "MEDIUM RISK: Close monitoring with follow-up imaging in 3-6 months."
        else:
            rec = "LOW RISK: Continue routine annual screening."

        # ── Grad-CAM (runs on currently loaded model = winning model) ──
        heatmap_b64   = None
        annotated_b64 = None
        cam_mask      = None
        lesion_data   = {}
        texture_data  = {}
        risk_data     = {}

        target_layer = get_target_layer(loaded_model)
        if target_layer is not None:
            try:
                final_idx = (detected_classes_local.index(final_pred)
                             if final_pred in detected_classes_local else 0)
                gcam     = GradCAM(loaded_model, target_layer)
                cam_mask = gcam(spatial_t, freq_t, class_idx=final_idx)
                gcam.remove_hooks()
                if cam_mask is not None:
                    heatmap_b64 = generate_heatmap_overlay(
                        img_resized, cam_mask, final_pred, detected_scan)
            except Exception as e:
                print(f"[GradCAM-run] {e}")

        # ── Doctor report ───────────────────────────────────────────────
        if mode == 'doctor':
            lesion_data   = detect_lesion_boundaries(img_resized, cam_mask, detected_scan)
            texture_data  = compute_texture_features(img_resized, cam_mask)
            # FIX 3: pass final_pred and final_conf (post-override)
            risk_data     = compute_risk_score(final_pred, final_conf,
                                               cv_sev, lesion_data, detected_scan)
            if cam_mask is not None:
                annotated_b64 = generate_annotated_image(img_resized, cam_mask, final_pred)

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
                    detected_classes_local[i]: round(float(all_probs[i]), 4)
                    for i in range(len(detected_classes_local))
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
# 13. UTILITY ROUTES
# =====================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status':       'online',
        'active_model': active_model_type,
        'classes':      detected_classes,
        'img_size':     IMG_SIZE,
        'endpoints': {
            'POST /predict': {
                'scan_type': 'auto|mammogram|mammogram_mias|mammogram_dmid|ultrasound|histopathology|mri',
                'mode':      'patient (default) | doctor',
            }
        },
    }), 200


@app.route('/models', methods=['GET'])
def model_status():
    return jsonify({
        name: {'path':path,'exists':os.path.exists(path),'active':(name==active_model_type)}
        for name,path in MODEL_PATHS.items()
    }), 200


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error':'File too large. Maximum is 15 MB.'}), 413

@app.errorhandler(400)
def bad_req(e):
    return jsonify({'error':str(e)}), 400

@app.errorhandler(500)
def server_err(e):
    gc.collect()
    return jsonify({'error':'Internal server error.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
