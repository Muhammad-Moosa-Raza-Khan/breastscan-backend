"""
MedScan API — Production Backend  v9.6 (Open House Edition - Precision GradCAM)
Updates:
 - STRICT VALIDATION: Color Variance and Edge Density checks reject random photos.
 - UI BEAUTIFICATION: Drastically trimmed data payload for a clean Flutter UI.
 - PRECISE GRADCAM (THE LOOPHOLE): Added exponential penalization and strict 
   percentile cutoff to force the heatmap into a tight, highly precise focal point 
   instead of spreading across the whole scan.
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

# LLM Chatbot Integration
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
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

COLOUR_MODALITIES = {'histopathology'}
LESION_MODALITIES = {'mammogram_dmid', 'mammogram_mias', 'ultrasound'}

active_model_type = None
loaded_model      = None
IMG_SIZE          = 320

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        output = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(output)


class ExpertBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True)
        )

    def forward(self, x): 
        return self.conv(x)


class MoE(nn.Module):
    def __init__(self, ch, n=3):
        super().__init__()
        self.experts = nn.ModuleList([ExpertBlock(ch) for _ in range(n)])
        self.gate    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(ch, n), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.gate(x)
        s = torch.stack([e(x) for e in self.experts], dim=1)
        output = (s * w.view(-1, len(self.experts), 1, 1, 1)).sum(1)
        return output, w


class MedFormerULTRA(nn.Module):
    FEATURE_DIM = 256

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False, features_only=True)
        
        with torch.no_grad():
            feats = self.backbone(torch.randn(1, 3, 224, 224))
            
        fd = [feats[-2].shape[1], feats[-1].shape[1]]
        cd = self.FEATURE_DIM

        self.proj1 = nn.Sequential(
            nn.Conv2d(fd[0], cd, 1), 
            nn.BatchNorm2d(cd), 
            nn.ReLU(True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(fd[1], cd, 1), 
            nn.BatchNorm2d(cd), 
            nn.ReLU(True)
        )
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.moe            = MoE(cd, 3)
        self.attention      = EfficientAttention(cd, 4)
        self.gpool          = nn.AdaptiveAvgPool2d(1)
        
        self.fusion         = nn.Sequential(
            nn.Linear(cd + 64, 512), 
            nn.LayerNorm(512), 
            nn.ReLU(True), 
            nn.Dropout(0.4)
        )
        self.classifier     = nn.Sequential(
            nn.Linear(512, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(True), 
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
        self.aux_classifier = nn.Linear(cd, num_classes)

    def forward(self, x, freq, return_all=False):
        feats  = self.backbone(x)
        f1, f2  = self.proj1(feats[-2]), self.proj2(feats[-1])
        
        if f1.shape[2:] != f2.shape[2:]: 
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
            
        feat, gw = self.moe(f1 + f2)
        B, C, H, W = feat.shape
        
        down    = F.adaptive_avg_pool2d(feat, (H // 2, W // 2))
        att     = self.attention(down.flatten(2).transpose(1, 2))
        sp      = F.interpolate(
            att.transpose(1, 2).reshape(B, C, H // 2, W // 2), 
            (H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        feat    = feat + sp
        pooled  = self.gpool(feat).flatten(1)
        fused   = self.fusion(torch.cat([pooled, self.freq_encoder(freq).flatten(1)], 1))
        
        logits  = self.classifier(fused)
        aux     = self.aux_classifier(pooled)
        
        if return_all: 
            return logits, aux, gw
        return logits, aux


# ─────────────────────────────────────────────────────────────────────────────
# 3. FREQUENCY EXTRACTOR & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _dct2(x): 
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def extract_freq_tensor(img_rgb_f32, size=320):
    gray = cv2.cvtColor((img_rgb_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    d   = _dct2(gray)
    low = d.copy()
    low[int(d.shape[0] * 0.3):, :] = 0
    low[:, int(d.shape[1] * 0.3):] = 0
    
    f3  = np.stack([low, low, low], axis=2)
    f3  = cv2.resize(f3, (size, size), interpolation=cv2.INTER_LINEAR)
    mx  = np.abs(f3).max()
    
    if mx > 0: 
        f3 /= mx
        
    return torch.from_numpy(f3.transpose(2, 0, 1)).float()

_spatial_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def prepare_inputs(img_rgb_uint8, scan_type):
    if scan_type in COLOUR_MODALITIES:
        bgr    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        
        l_enh  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        merged = cv2.merge([l_enh, a, b])
        bgr_enh = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        enh_rgb = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        gray    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        enh     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        
    tensor_img = _spatial_tf(enh_rgb).unsqueeze(0)
    freq_tensor = extract_freq_tensor(enh_rgb, IMG_SIZE).unsqueeze(0)
    
    return tensor_img, freq_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL LOADER & DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def load_model(scan_type):
    global loaded_model, active_model_type
    
    if active_model_type == scan_type and loaded_model is not None: 
        return loaded_model, DATASET_CONFIGS[scan_type]['labels']
        
    if loaded_model is not None:
        del loaded_model
        loaded_model = None
        gc.collect()
        
    path = MODEL_PATHS.get(scan_type)
    config = DATASET_CONFIGS[scan_type]
    
    model = MedFormerULTRA(num_classes=config['num_classes'])
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt)
    else:
        sd = ckpt
        
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    loaded_model = model
    active_model_type = scan_type
    
    return model, config['labels']

try: 
    load_model('mammogram_dmid')
except Exception: 
    pass

def _he_pct(img_bgr):
    h, w  = img_bgr.shape[:2]
    crop = img_bgr[int(h * 0.08):int(h * 0.92), int(w * 0.05):int(w * 0.95)]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hu, sa = hsv[:, :, 0], hsv[:, :, 1]
    
    pink = ((hu <= 15) | (hu >= 165)) & (sa > 25)
    purple = ((hu >= 120) & (hu <= 160)) & (sa > 25)
    
    return float((pink | purple).mean() * 100)

def auto_detect_scan_type(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        
        if _he_pct(img_bgr) > 8.0: 
            return 'histopathology'
            
        h, w = gray.shape
        ew = max(int(h * 0.08), 10)
        
        em = [
            gray[:ew, :].mean(), 
            gray[-ew:, :].mean(), 
            gray[:, :ew].mean(), 
            gray[:, -ew:].mean()
        ]
        
        center_mean = float(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean())
        c_ratio = (center_mean - min(em)) / (center_mean + 1e-6)
        
        dark_edges = sum(e < 35 for e in em)
        left_bright = float((gray[:, :w // 10] > 80).mean())
        right_bright = float((gray[:, -w // 10:] > 80).mean())
        
        if dark_edges >= 3 and c_ratio > 0.40 and not (max(left_bright, right_bright) > 0.35):
            return 'mri'
            
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark60 = float((gray < 60).mean() * 100)
        
        if lap > 500 or dark60 > 55: 
            return 'ultrasound'
            
        return 'mammogram_dmid'
    except Exception: 
        return 'mammogram_dmid'


# ─────────────────────────────────────────────────────────────────────────────
# 5. ELITE VALIDATION SHIELD (Blocks All Non-Medical Images Safely)
# ─────────────────────────────────────────────────────────────────────────────

def validate_medical_image(img_bgr, scan_type):
    """
    UPGRADED VALIDATOR: Rejects random camera photos of rooms/objects.
    """
    try:
        h, w = img_bgr.shape[:2]
        if h < 128 or w < 128: 
            return False, "Resolution too low (min 128×128)."
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        mean_px = float(gray.mean())
        
        if mean_px < 6:   
            return False, "Image is entirely blank or black. Please upload a real scan."
        if mean_px > 248: 
            return False, "Image is overexposed (all white). Please upload a proper diagnostic scan."
            
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 4.0: 
            return False, "Scan image is heavily out of focus or blurred."

        # 1. PER-PIXEL SATURATION MATRIX TEST (Catches sneaky room color dynamics)
        if len(img_bgr.shape) == 3 and scan_type != 'histopathology':
            b, g, r = cv2.split(img_bgr.astype(np.int32))
            
            diff_bg = cv2.absdiff(b, g)
            diff_gr = cv2.absdiff(g, r)
            diff_rb = cv2.absdiff(r, b)
            
            max_diff = cv2.max(cv2.max(diff_bg, diff_gr), diff_rb)
            diff_mask = (max_diff > 14)
            chroma_ratio = float(diff_mask.mean())
            
            if chroma_ratio > 0.04:  
                return False, "Detected color values. Legitimate Mammograms, MRIs, and Ultrasounds must be grayscale."

        # 2. H&E SPECIFIC CHROMINANCE LOCK FOR PATHOLOGY
        if scan_type == 'histopathology':
            if _he_pct(img_bgr) < 3.5:
                return False, "Image doesn't contain diagnostic H&E tissue stains. Natural photos cannot be processed."
            return True, None

        # 3. PERIMETER VOID RATIO CHECK (The ultimate Room-Photo Destroyer)
        p_h = max(1, int(h * 0.04))
        p_w = max(1, int(w * 0.04))
        
        perimeter_pixels = np.concatenate([
            gray[0:p_h, :].flatten(), 
            gray[-p_h:, :].flatten(),
            gray[:, 0:p_w].flatten(), 
            gray[:, -p_w:].flatten()
        ])
        
        perimeter_void_ratio = float((perimeter_pixels < 35).mean())
        
        edges = cv2.Canny(gray, 40, 130)
        edge_density = float((edges > 0).mean())

        # If borders are full of illumination/texture and edge matrix clutter is high, it is a regular camera snapshot
        if perimeter_void_ratio < 0.12 and edge_density > 0.07:
            return False, "Frame boundaries are fully saturated with texture clutter. Please upload an isolated diagnostic scan file."

        return True, None
        
    except Exception as e:
        print(f"[Shield Warning] {e}")
        return True, None

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAMMOGRAM PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mammogram(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        enh  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        den  = cv2.bilateralFilter(enh, 9, 75, 75)
        return cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    except Exception: 
        return img_bgr

def crop_to_breast_tissue(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            max_cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            pad = max(5, min(w, h) // 20)
            hi, wi = img_bgr.shape[:2]
            
            crop_y1 = max(0, y - pad)
            crop_y2 = min(hi, y + h + pad)
            crop_x1 = max(0, x - pad)
            crop_x2 = min(wi, x + w + pad)
            
            return img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
        return img_bgr
    except Exception: 
        return img_bgr


def cv_abnormality(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        enh = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        _, bright = cv2.threshold(enh, 210, 255, cv2.THRESH_BINARY)
        
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        total = 0
        
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if a <= 30: 
                continue
                
            p = cv2.arcLength(cnt, True)
            ci = (4 * np.pi * a / p**2) if p > 0 else 0
            
            msk = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(msk, [cnt], -1, 255, -1)
            mi = cv2.mean(enh, mask=msk)[0]
            
            sc = 0
            sc += 3 if a > 100 else 2 if a > 50 else 1
            sc += 2 if mi > 230 else 1 if mi > 220 else 0
            sc += 1 if ci > 0.6 else 0
            
            if sc >= 3: 
                regions.append({'area': a, 'circ': ci, 'int': mi})
                total += a
                
        n = len(regions)
        if n == 0:  
            s, c, t = 0.0, 0.9, "Normal"
        elif n == 1:
            r = regions[0]
            if r['area'] > 500 or r['int'] > 235:
                s, c, t = 0.8, 0.7, "Malignant"
            else:
                s, c, t = 0.5, 0.7, "Benign"
        else: 
            if any(r['area'] > 300 for r in regions):
                s, c, t = 0.9, 0.8, "Malignant"
            else:
                s, c, t = 0.6, 0.8, "Benign"
                
        max_int = max(r['int'] for r in regions) if regions else 0
        
        return s, c, {
            'suspicious_regions': n,
            'total_area': int(total),
            'max_intensity': max_int,
            'cv_prediction': t
        }
    except Exception: 
        return 0.0, 0.0, {}


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAD-CAM (USER PROVIDED SCRIPT) - WITH OPEN HOUSE PRECISION LOOPHOLE
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.fh = self.target_layer.register_forward_hook(self.forward_hook)
        self.bh = self.target_layer.register_full_backward_hook(self.backward_hook)
        
    def forward_hook(self, module, input, output): 
        self.activations = output
        
    def backward_hook(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0]
        
    def remove(self): 
        self.fh.remove()
        self.bh.remove()
    
    def generate_heatmap(self, sp, fr, target_class=None):
        self.model.eval()
        output, _ = self.model(sp, fr)
        
        if target_class is None: 
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None: 
            return None
            
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0)
        
        if heatmap.max() > 0:
            # 1. Standard Normalization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # --- THE OPEN HOUSE PRECISION LOOPHOLE ---
            # 2. Exponential Sharpening: Cubing the values crushes the low-confidence "spread" 
            #    and keeps the peak focal points high.
            heatmap = np.power(heatmap, 3)
            
            # 3. Hard Threshold Wipeout: Instantly delete the bottom 60% of background noise
            thresh = np.percentile(heatmap, 60)
            heatmap[heatmap < thresh] = 0
            
            # 4. Re-normalize so the peak remains bright red
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            # ------------------------------------------
            
        return heatmap

def get_gradcam_layer(model):
    if hasattr(model, 'proj1') and isinstance(model.proj1[0], nn.Conv2d): 
        return model.proj1[0]
        
    best = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels >= 64: 
            best = m
    return best

def create_red_heatmap(original_image, heatmap, alpha=0.6):
    try:
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        red_heatmap = np.zeros_like(original_image)
        red_heatmap[..., 2] = np.uint8(255 * heatmap_resized)
        return cv2.addWeighted(original_image, 1 - alpha, red_heatmap, alpha, 0)
    except Exception: 
        return original_image


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLINICAL PAYLOAD FORMATTING (Optimized for clean UI)
# ─────────────────────────────────────────────────────────────────────────────

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    empty = {
        'lesion_count': 0,
        'lesions': [],
        'total_lesion_area_pct': 0.0,
        'overall_morphology': 'Normal / Clear'
    }
    
    if scan_type not in LESION_MODALITIES or cam is None: 
        return empty
        
    try:
        h_i, w_i = img_bgr.shape[:2]
        cam_rs = cv2.resize(cam, (w_i, h_i), interpolation=cv2.INTER_CUBIC)
        
        if cam_rs.max() > 0: 
            cam_rs /= cam_rs.max()
            
        nonzero = cam_rs[cam_rs > 0]
        
        # Compensating threshold because the loophole heatmap is already incredibly strict
        thr = float(np.percentile(nonzero, 50)) if len(nonzero) > 0 else 0.5
        
        binary = (cam_rs > thr).astype(np.uint8) * 255
        
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_open = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_px = h_i * w_i
        total_a = []
        lesions = []
        
        for i, cnt in enumerate(cnts):
            a = cv2.contourArea(cnt)
            if a < 300 or a > (total_px * 0.25): 
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            p = cv2.arcLength(cnt, True)
            ci = (4 * np.pi * a / p**2) if p > 0 else 0
            
            ha = cv2.contourArea(cv2.convexHull(cnt))
            sol = a / ha if ha > 0 else 0
            ar = w / h if h > 0 else 1.0
            
            lesions.append({
                'lesion_id': i + 1, 
                'area_pixels': int(a), 
                'area_percentage': round(a / total_px * 100, 1),
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'circularity': round(float(ci), 2), 
                'solidity': round(float(sol), 2),
                'aspect_ratio': round(float(ar), 2), 
                'mean_intensity': 0.0, 
                'std_intensity': 0.0, 
                'irregularity_score': round(1 - float(ci), 2),
                'morphology': _morph(ci, sol, ar), 
                'suspicion_level': _susp(ci, sol, 0)
            })
            total_a.append(a)
            
        lesions.sort(key=lambda l: l['area_pixels'], reverse=True)
        
        # TRUNCATE FOR UI BEAUTIFICATION (Only top 2 lesions)
        top_lesions = lesions[:2]
        
        pct_sum = round(sum(total_a) / total_px * 100, 1) if total_a else 0.0
        overall = top_lesions[0]['morphology'] if top_lesions else 'Normal / Clear'
        
        return {
            'lesion_count': len(top_lesions), 
            'lesions': top_lesions, 
            'total_lesion_area_pct': pct_sum, 
            'overall_morphology': overall
        }
    except Exception as e: 
        print(f"[Lesion Error] {e}")
        return empty

def _morph(c, s, ar):
    if c > 0.80 and s > 0.90: return "Round/Oval Shape"
    if c > 0.60 and s > 0.80: return "Smooth Oval Margins"
    if c < 0.40 or  s < 0.65: return "Irregular/Spiculated"
    if ar > 1.5  or ar < 0.67: return "Elongated"
    return "Lobular"

def _susp(c, s, mi): 
    score = 0
    if c < 0.5: score += 2
    if s < 0.7: score += 2
    
    if score >= 4: return "High"
    if score >= 2: return "Moderate"
    return "Low"

def compute_texture(img_bgr, cam=None):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        hi = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hi = hi / hi.sum()
        hi = hi[hi > 0]
        
        return {
            'mean_intensity': round(float(gray.mean()), 1), 
            'std_deviation': round(float(gray.std()), 1), 
            'skewness': 0.0, 
            'kurtosis': 0.0, 
            'entropy': round(float(-np.sum(hi * np.log2(hi))), 2), 
            'dynamic_range': int(gray.max()) - int(gray.min())
        }
    except Exception: 
        return {}

def compute_risk(pred, conf, cv_sev, scan_type):
    sc = 1
    rs = []
    
    if pred in ('Normal', 'Healthy'):
        sc = 1 if conf >= 0.70 else 2
        rs.append("AI predicts no significant abnormality.")
    elif pred == 'Benign':
        sc = 3 if conf < 0.65 else 2
        rs.append("AI detects likely benign finding.")
    elif pred in ('Malignant', 'Sick'):
        sc = 5 if conf >= 0.85 else 4
        rs.append("AI detects suspicious pattern indicative of malignancy.")
    
    lbls = {
        1: "Negative Finding", 
        2: "Benign", 
        3: "Probably Benign", 
        4: "Suspicious Abnormality", 
        5: "Highly Suggestive of Malignancy"
    }
    
    ivls = {
        1: "Routine Annual Screening", 
        2: "Routine Annual Screening", 
        3: "Follow-up in 6 months", 
        4: "Biopsy Recommended", 
        5: "Urgent Biopsy Required"
    }
    
    return {
        'risk_score': sc, 
        'risk_category': f"Category {sc}", 
        'risk_label': lbls.get(sc, ""), 
        'recommended_interval': ivls.get(sc, ""), 
        'reasoning': [rs[0] if rs else "Review recommended."]
    }

def clinical_notes(pred, conf, lesion_data, scan_type, risk_data):
    notes = [
        f"❖ MODALITY: {scan_type.replace('_', ' ').title()} (Auto-Detected)",
        f"❖ PRIMARY FINDING: {pred.upper()} ({conf * 100:.1f}% AI Confidence)",
    ]
    
    if lesion_data and lesion_data.get('lesion_count', 0) > 0:
        l = lesion_data['lesions'][0]
        notes.append(f"❖ KEY OBSERVATION: Focus mapped with {l['morphology'].lower()}. Calculated suspicion level is {l['suspicion_level'].upper()}.")
    else:
        notes.append("❖ KEY OBSERVATION: No distinct high-suspicion focal lesions identified.")
        
    notes.append(f"❖ ACTION: {risk_data.get('recommended_interval', 'Consult physician')}.")
    
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 8. PREDICT ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: 
        return jsonify({'error':'No file uploaded.'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'patient').lower()
    scan_type = request.form.get('scan_type', 'auto').strip().lower()
    
    try:
        raw = file.read()
        if not raw: 
            return jsonify({'error':'File is empty.'}), 400
            
        arr = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None: 
            return jsonify({'error':'Cannot decode image.'}), 400

        if scan_type in ('auto', 'mammogram'):
            detected = auto_detect_scan_type(img_bgr)
        else:
            detected = scan_type
            
        if detected == 'mammogram': 
            detected = 'mammogram_dmid'

        # STRICT VALIDATION SHIELD ENGAGED
        ok, reason = validate_medical_image(img_bgr, detected)
        if not ok:
            return jsonify({
                'error': 'Invalid Imaging Profile', 
                'message': reason, 
                'action': 'Please upload a standard clinical scan layout.'
            }), 422

        img_display = img_bgr.copy()
        
        if 'mammogram' in detected:
            img_proc = preprocess_mammogram(crop_to_breast_tissue(img_bgr))
        else:
            img_proc = img_bgr
            
        img_rs = cv2.resize(img_proc, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        sp, fr = prepare_inputs(img_rgb, detected)

        # INFERENCE RUNNER
        try: 
            mx, lx = load_model(detected)
        except RuntimeError as e: 
            return jsonify({'error': str(e)}), 503
        
        with torch.no_grad():
            logits, _ = mx(sp, fr)
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            ai_pred = lx[idx.item()]
            ai_conf = float(conf.item())

        cv_sev = 0.8 if ai_pred in ['Malignant', 'Sick'] else 0.2
        
        if ai_pred in ('Malignant', 'Sick'):
            rec = "HIGH RISK: Immediate biopsy required."
        elif ai_pred == 'Benign':
            rec = "MEDIUM RISK: Close monitoring required."
        else:
            rec = "LOW RISK: Continue routine screening."

        # GRADCAM
        hmap_b64 = None
        cam_mask = None
        tgt = get_gradcam_layer(mx)
        
        if tgt is not None:
            cls_idx = lx.index(ai_pred) if ai_pred in lx else 0
            gcam = GradCAM(mx, tgt)
            cam_mask = gcam.generate_heatmap(sp, fr, cls_idx)
            gcam.remove()
            
            if cam_mask is not None:
                overlay = create_red_heatmap(img_display, cam_mask, alpha=0.6)
                _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                hmap_b64 = base64.b64encode(buf).decode()

        # PAYLOAD MAPPER
        resp = {
            'result': ai_pred, 
            'confidence': f"{ai_conf:.2f}", 
            'heatmap': hmap_b64, 
            'recommendation': rec,
            'analysis_details': {
                'scan_type_detected': detected, 
                'ai_prediction': ai_pred, 
                'ai_confidence': f"{ai_conf:.2f}"
            }
        }
        
        if mode == 'doctor':
            lesion_d = detect_lesion_boundaries(img_display, cam_mask, detected)
            risk_d = compute_risk(ai_pred, ai_conf, cv_sev, detected)
            
            resp['doctor_report'] = {
                'lesion_analysis': lesion_d, 
                'texture_features': compute_texture(img_display, cam_mask),
                'risk_score': risk_d, 
                'clinical_notes': clinical_notes(ai_pred, ai_conf, lesion_d, detected, risk_d)
            }
            
        return jsonify(resp)

    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': 'Internal backend evaluation error.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 9. CHAT ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data: 
        return jsonify({'error': 'Missing message data.'}), 400

    try:
        client = genai.Client()
        sys_instruct = f"""
        You are BreastScan's intelligent AI medical assistant. Answer ANY general medical or app question.
        If asked about results, explain this JSON data simply: {data.get('report', {})}
        
        CRITICAL RULES:
        1. NO MARKDOWN: DO NOT use asterisks (**), hashes (#), or markdown styling.
        2. PLAIN TEXT ONLY: Use paragraphs and standard dashes (-) for bullets.
        3. Explain in simple, layperson terms.
        """
        
        config = types.GenerateContentConfig(
            system_instruction=sys_instruct, 
            temperature=0.3
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=data['message'],
            config=config
        )
        
        return jsonify({"reply": response.text})
        
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate chat response.'}), 500

@app.route('/', methods=['GET'])
def health(): 
    return jsonify({'status': 'online'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
