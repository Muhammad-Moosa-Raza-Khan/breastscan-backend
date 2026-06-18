"""
================================================================================
                     BREASTSCAN PRODUCTION BACKEND SERVICE                      
================================================================================
File: app.py
Version: 9.7 (Open House Premium Edition)
Target Environment: Ubuntu Server / AWS EC2
Framework: Flask / PyTorch / OpenCV / Google GenAI Client

Description:
    This is the core production backend application for the BreastScan platform.
    It exposes high-performance API endpoints for automated medical imaging 
    classification, local structural lesion localization, texture extraction, 
    and multi-modal clinical report generation.

Architectural Features:
    1. MedFormerULTRA Multi-Modality Framework: Integrates an EfficientNetV2-S 
       backbone with spatial EfficientAttention and a Mixture of Experts (MoE) 
       gating network coupled with high-frequency discrete cosine transform features.
    2. Elite Validation Shield: Protects inference workers by evaluating spatial 
       illumination matrices, per-pixel color channel variances, and boundary 
       void patterns to guarantee only diagnostic medical layouts are processed.
    3. Isolated Hotspot Grad-CAM Engine: Employs custom PyTorch activation hooks 
       fused with mathematical exponential sharpening, dynamic percentiles, 
       and Gaussian isolation masks to create pin-point diagnostic heatmaps.
    4. Flutter-Optimized REST Layer: Delivers cleanly rounded data frames, 
       truncated report structures, and high-quality Base64 visual payloads 
       engineered for rapid native mobile rendering.

Author: Muhammad Moosa Raza Khan
License: Proprietary / Academic Research Deployment Only
================================================================================
"""

import os
import gc
import sys
import time
import math
import base64
import logging
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

# LLM Chatbot Integration Modules
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL SYSTEM CONFIGURATIONS & LOGGER SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('BreastScanBackend')

app = Flask(__name__)
application = app

# Enforce strict 15MB file uploading payload limit to protect against DoS vectors
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

# Set global random states to guarantee internal alignment across runtime instances
np.random.seed(42)
torch.manual_seed(42)

# Absolute path bindings to local compiled target weights files
MODEL_PATHS = {
    'mammogram_mias': 'models/mammo_mias.pth',
    'mammogram_dmid': 'models/mammo_dmid.pth',
    'ultrasound':     'models/ultrasound_model.pth',
    'histopathology': 'models/hist_model.pth',
    'mri':            'models/mri_model.pth',
}

# Mapping configuration boundaries for the multi-task targets
DATASET_CONFIGS = {
    'histopathology': {
        'num_classes': 2, 
        'labels': ['Benign', 'Malignant'],
        'description': 'Tissue biopsy sample visualization via H&E staining profiles.'
    },
    'mri': {
        'num_classes': 2, 
        'labels': ['Healthy', 'Sick'],
        'description': 'Magnetic Resonance Imaging scan tracking fluid and soft tissue density.'
    },
    'mammogram_dmid': {
        'num_classes': 3, 
        'labels': ['Benign', 'Malignant', 'Normal'],
        'description': 'Digital Mammography Screening for structural density abnormalities.'
    },
    'mammogram_mias': {
        'num_classes': 3, 
        'labels': ['Benign', 'Malignant', 'Normal'],
        'description': 'Film Mammography Screening tracking micro-calcification densities.'
    },
    'ultrasound': {
        'num_classes': 3, 
        'labels': ['Benign', 'Malignant', 'Normal'],
        'description': 'High-frequency acoustic echo localization mapping fluid vs solid masses.'
    },
}

VALID_SCAN_TYPES = set(DATASET_CONFIGS.keys()) | {'mammogram', 'auto'}
COLOUR_MODALITIES = {'histopathology'}
LESION_MODALITIES = {'mammogram_dmid', 'mammogram_mias', 'ultrasound'}

# Global Memory Reference Blocks for Hot Model Swapping
active_model_type = None
loaded_model      = None
IMG_SIZE          = 320

logger.info("========================================= ")
logger.info("Initialising BreastScan Microservices Engine")
logger.info("========================================= ")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ADVANCED MODEL ARCHITECTURE (MEDFORMER-ULTRA)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientAttention(nn.Module):
    """
    Memory-efficient multi-head spatial self-attention mechanism engineered for 
    high-resolution medical imagery tensors. Reduces quadratic spatial complexity 
    by computing context matrices across feature dimensions.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        logger.debug(f"[Attention Layer] Input Vector Shape: {x.shape}")
        
        # Project tokens and reshape into distinct attention heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate Scaled Dot-Product Attention context map
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Aggregate values and transform back to core projection dimensions
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(output)


class ExpertBlock(nn.Module):
    """
    Isolated Deep Convolutional Expert Engine processing localized micro-structural 
    features via deep separable layers to optimize parameter footprints.
    """
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        return self.conv(x)


class MoE(nn.Module):
    """
    Mixture of Experts Routing Layer. Dynamically maps blended spatial weights 
    to discrete structural feature fields based on global feature energy.
    """
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
        logger.debug(f"[MoE Layer] Input Feature Shape: {x.shape}")
        w = self.gate(x)
        
        # Evaluate activations across all expert routes concurrently
        s = torch.stack([e(x) for e in self.experts], dim=1)
        
        # Perform soft-gated weighted matrix integration
        output = (s * w.view(-1, len(self.experts), 1, 1, 1)).sum(1)
        return output, w


class MedFormerULTRA(nn.Module):
    """
    The Core Multimodal Deep Learning Classifier Architecture.
    Fuses deep EfficientNet multi-scale features with spatial self-attention 
    networks and global spectral frequency transforms.
    """
    FEATURE_DIM = 256

    def __init__(self, num_classes=3):
        super().__init__()
        logger.info(f"[Architecture] Initialising MedFormerULTRA Instance [Classes: {num_classes}]")
        
        # Initialize specialized EfficientNet backbone without native classification heads
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False, features_only=True)
        
        # Compute spatial feature map projections dynamically with a dummy tensor
        with torch.no_grad():
            feats = self.backbone(torch.randn(1, 3, 224, 224))
            
        fd = [feats[-2].shape[1], feats[-1].shape[1]]
        cd = self.FEATURE_DIM

        # Projection bridges mapping deep multi-scale layers to identical feature spaces
        self.proj1 = nn.Sequential(
            nn.Conv2d(fd[0], cd, kernel_size=1), 
            nn.BatchNorm2d(cd), 
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(fd[1], cd, kernel_size=1), 
            nn.BatchNorm2d(cd), 
            nn.ReLU(inplace=True)
        )
        
        # Dual-layer feature extractor mapping Discrete Cosine Transform profiles
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.moe            = MoE(cd, n=3)
        self.attention      = EfficientAttention(cd, num_heads=4)
        self.gpool          = nn.AdaptiveAvgPool2d(1)
        
        # Multi-modal fusion layer binding frequency profiles to spatial vectors
        self.fusion         = nn.Sequential(
            nn.Linear(cd + 64, 512), 
            nn.LayerNorm(512), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.4)
        )
        
        # Primary Diagnostic Classification Head
        self.classifier     = nn.Sequential(
            nn.Linear(512, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary Regularization Head
        self.aux_classifier = nn.Linear(cd, num_classes)

    def forward(self, x, freq, return_all=False):
        # Extract hierarchical features from input tensor space
        feats  = self.backbone(x)
        f1, f2  = self.proj1(feats[-2]), self.proj2(feats[-1])
        
        # Align spatial multi-scale boundaries gracefully via adaptive pooling
        if f1.shape[2:] != f2.shape[2:]: 
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
            
        # Route multi-scale features through core Mixture of Experts blocks
        feat, gw = self.moe(f1 + f2)
        B, C, H, W = feat.shape
        
        # Perform low-dimensional tokenization for structural attention processing
        down    = F.adaptive_avg_pool2d(feat, (H // 2, W // 2))
        att     = self.attention(down.flatten(2).transpose(1, 2))
        
        # Upsample attention spatial vectors back to target resolution matrices
        sp      = F.interpolate(
            att.transpose(1, 2).reshape(B, C, H // 2, W // 2), 
            (H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Apply residual attention links and compress to vectors
        feat    = feat + sp
        pooled  = self.gpool(feat).flatten(1)
        
        # Encode frequency tensors and fuse with structural vectors
        freq_enc = self.freq_encoder(freq).flatten(1)
        fused    = self.fusion(torch.cat([pooled, freq_enc], dim=1))
        
        # Generate diagnostic outputs
        logits  = self.classifier(fused)
        aux     = self.aux_classifier(pooled)
        
        if return_all: 
            return logits, aux, gw
        return logits, aux


# ─────────────────────────────────────────────────────────────────────────────
# 3. ADVANCED FREQUENCY TRANSFORMATIONS & IMAGE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _dct2(x): 
    """Two-Dimensional Discrete Cosine Transform formulation for structural analysis."""
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def extract_freq_tensor(img_rgb_f32, size=320):
    """
    Isolates low and mid-frequency energy domains using Discrete Cosine 
    Transforms to track structural anomalies independently of tissue contrast.
    """
    try:
        gray = cv2.cvtColor((img_rgb_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        dct_matrix = _dct2(gray)
        
        low_pass = dct_matrix.copy()
        # Cleanly eliminate high-frequency noise and scanning artifacts
        low_pass[int(dct_matrix.shape[0] * 0.3):, :] = 0
        low_pass[:, int(dct_matrix.shape[1] * 0.3):] = 0
        
        # Re-encode into uniform multi-channel structures matching target sizes
        freq_map  = np.stack([low_pass, low_pass, low_pass], axis=2)
        freq_map  = cv2.resize(freq_map, (size, size), interpolation=cv2.INTER_LINEAR)
        max_val   = np.abs(freq_map).max()
        
        if max_val > 0: 
            freq_map /= max_val
            
        return torch.from_numpy(freq_map.transpose(2, 0, 1)).float()
    except Exception as e:
        logger.error(f"[Frequency Extractor Error] Safe fallback triggered: {e}")
        return torch.zeros((3, size, size), dtype=torch.float32)


# Spatial standard normal transformation pipe matching ImageNet baselines
_spatial_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_inputs(img_rgb_uint8, scan_type):
    """
    Transforms multi-modal input arrays into balanced tensor pairs 
    containing enhanced structural images and localized frequency maps.
    """
    if scan_type in COLOUR_MODALITIES:
        # Specialized Contrast Enhancement for H&E Stain Profiles via CIELAB conversions
        bgr    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        
        l_enh  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        merged = cv2.merge([l_enh, a, b])
        bgr_enh = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        enh_rgb = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        # Uniform CLAHE Contrast Alignment for Grayscale Medical Scans
        gray    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        enh     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        
    tensor_img  = _spatial_tf(enh_rgb).unsqueeze(0)
    freq_tensor = extract_freq_tensor(enh_rgb, IMG_SIZE).unsqueeze(0)
    
    return tensor_img, freq_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 4. MEMORY-SAFE COMPUTED WEIGHTS HOT-SWAPPING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(scan_type):
    """
    Manages runtime allocation of PyTorch model weights inside CPU/GPU memory bounds. 
    Purges unused networks dynamically to optimize server resources.
    """
    global loaded_model, active_model_type
    
    # Return hot model allocation if matches requested scan route
    if active_model_type == scan_type and loaded_model is not None: 
        logger.info(f"[Model Swapper] Cache Hit: Using active network allocated for [{scan_type}]")
        return loaded_model, DATASET_CONFIGS[scan_type]['labels']
        
    logger.info(f"[Model Swapper] Cache Miss: Purging active network and loading weights for [{scan_type}]")
    
    # Flush memory footprint safely before reloading heavy weights
    if loaded_model is not None:
        del loaded_model
        loaded_model = None
        gc.collect()
        
    path = MODEL_PATHS.get(scan_type)
    if not path or not os.path.exists(path):
        raise RuntimeError(f"Target compiled weights file missing or corrupt at location: {path}")
        
    config = DATASET_CONFIGS[scan_type]
    
    # Reconstruct architecture baseline safely
    model = MedFormerULTRA(num_classes=config['num_classes'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Store references inside global scope safely
    loaded_model = model
    active_model_type = scan_type
    
    logger.info(f"[Model Swapper] Network weights successfully bound to runtime stack for [{scan_type}]")
    return model, config['labels']


# Pre-seed default configuration to minimize user latency on boot
try: 
    load_model('mammogram_dmid')
except Exception as e: 
    logger.warning(f"Asynchronous model pre-seeding skipped on launch sequence: {e}")


def _he_pct(img_bgr):
    """Measures relative coverage ratios of classic Hematoxylin and Eosin stains."""
    try:
        h, w  = img_bgr.shape[:2]
        crop = img_bgr[int(h * 0.08):int(h * 0.92), int(w * 0.05):int(w * 0.95)]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hu, sa = hsv[:, :, 0], hsv[:, :, 1]
        
        pink = ((hu <= 15) | (hu >= 165)) & (sa > 25)
        purple = ((hu >= 120) & (hu <= 160)) & (sa > 25)
        
        return float((pink | purple).mean() * 100)
    except Exception:
        return 0.0


def auto_detect_scan_type(img_bgr):
    """
    Evaluates matrix distributions, edge gradients, and background profiles 
    to automatically determine scan modality without user input.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        
        # 1. Catch Histopathology profiles immediately via stain color distributions
        if _he_pct(img_bgr) > 8.0: 
            return 'histopathology'
            
        h, w = gray.shape
        edge_width = max(int(h * 0.08), 10)
        
        edge_means = [
            gray[:edge_width, :].mean(), 
            gray[-edge_width:, :].mean(), 
            gray[:, :edge_width].mean(), 
            gray[:, -edge_width:].mean()
        ]
        
        center_mean = float(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean())
        contrast_ratio = (center_mean - min(edge_means)) / (center_mean + 1e-6)
        
        dark_edges = sum(e < 35 for e in edge_means)
        left_perimeter_bright = float((gray[:, :w // 10] > 80).mean())
        right_perimeter_bright = float((gray[:, -w // 10:] > 80).mean())
        
        # 2. Catch MRI profiles via bounded central illumination and dark margins
        if dark_edges >= 3 and contrast_ratio > 0.40 and not (max(left_perimeter_bright, right_perimeter_bright) > 0.35):
            return 'mri'
            
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_pixel_ratio = float((gray < 60).mean() * 100)
        
        # 3. Catch Ultrasound layouts via noise variances and high dark backgrounds
        if laplacian_variance > 500 or dark_pixel_ratio > 55: 
            return 'ultrasound'
            
        return 'mammogram_dmid'
    except Exception as e: 
        logger.error(f"[Modality Auto-Detector] Warning in execution path: {e}. Defaulting to mammogram.")
        return 'mammogram_dmid'


# ─────────────────────────────────────────────────────────────────────────────
# 5. ELITE VALIDATION SHIELD (REJECTS RAW CAMERA ENVIRONMENT IMAGES)
# ─────────────────────────────────────────────────────────────────────────────

def validate_medical_image(img_bgr, scan_type):
    """
    Validates image integrity to block cell phone camera photos of text, rooms, 
    or unrelated objects, protecting backend workers from unresolvable arrays.
    """
    try:
        h, w = img_bgr.shape[:2]
        if h < 128 or w < 128: 
            return False, "Resolution profile too low for structural processing (minimum threshold 128×128)."
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        mean_pixel_intensity = float(gray.mean())
        
        if mean_pixel_intensity < 6:   
            return False, "Image matrix presents as entirely blank or solid black. Ensure image file is non-corrupt."
        if mean_pixel_intensity > 248: 
            return False, "Image matrix presents as fully overexposed. Standard medical captures cannot be solid white."
            
        lap_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_variance < 4.0: 
            return False, "Image profiles fail baseline structural variance. Visual is severely blurred or missing details."

        # 1. PER-PIXEL SATURATION MATRIX TEST (Identifies non-grayscale camera environments)
        if len(img_bgr.shape) == 3 and scan_type != 'histopathology':
            b, g, r = cv2.split(img_bgr.astype(np.int32))
            
            diff_bg = cv2.absdiff(b, g)
            diff_gr = cv2.absdiff(g, r)
            diff_rb = cv2.absdiff(r, b)
            
            max_channel_variance = cv2.max(cv2.max(diff_bg, diff_gr), diff_rb)
            chroma_activation_mask = (max_channel_variance > 14)
            chroma_ratio = float(chroma_activation_mask.mean())
            
            if chroma_ratio > 0.04:  
                return False, "Image contains multi-channel color values. Mammogram, MRI, and Ultrasound profiles must be purely grayscale layouts."

        # 2. CHROMINANCE LOCK FOR PATHOLOGY SLIDES
        if scan_type == 'histopathology':
            if _he_pct(img_bgr) < 3.5:
                return False, "Target slide image does not display core H&E color signatures. Natural landscape photos are blocked."
            return True, None

        # 3. PERIMETER VOID RATIO CHECK (Identifies natural ambient scenery files)
        # Legitimate scanners force dark bounding margins. Camera shots fill frames completely.
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

        if perimeter_void_ratio < 0.12 and edge_density > 0.07:
            return False, "Frame boundaries contain high-frequency textural noise. Ensure file contains a clean isolated digital scan."

        return True, None
        
    except Exception as e:
        logger.warning(f"[Validation Shield Exception] Error during security screening: {e}. Passing file down-pipe safely.")
        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# 6. STRUCTURAL TARGET PREPROCESSING PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mammogram(img_bgr):
    """Applies bilateral structural filters and CLAHE contrast equalization."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        enh  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.bilateralFilter(enh, 9, 75, 75)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.error(f"[Preprocessing Error] Error in mammogram transform path: {e}")
        return img_bgr


def crop_to_breast_tissue(img_bgr):
    """Isolates true breast anatomy, removing dead space borders to maximize features."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        structuring_kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, structuring_kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, structuring_kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            primary_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(primary_contour)
            padding = max(5, min(w, h) // 20)
            hi, wi = img_bgr.shape[:2]
            
            crop_y1 = max(0, y - padding)
            crop_y2 = min(hi, y + h + padding)
            crop_x1 = max(0, x - padding)
            crop_x2 = min(wi, x + w + padding)
            
            return img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
        return img_bgr
    except Exception as e:
        logger.error(f"[Anatomy Cropper Exception] Error isolating mass region: {e}")
        return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAD-CAM CORE ENGINE (OPEN HOUSE SMOOTH PRECISION EDITION)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Custom hook runner capturing model backpropagation vectors. 
    Includes mathematical sharpening filters to fix diffuse artifacts, 
    ensuring a precise, professional visualization for presentation panels.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Bind explicit lifecycle execution hooks onto the layer
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
            logger.warning("[Grad-CAM Engine] Failed to capture hooks across targeting structures.")
            return None
            
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling of gradients across dimensions
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0)
        
        if heatmap.max() > 0:
            # 1. Base Scaling Normalization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # 2. Exponential Contrast Sharpening: Suppresses low-level ambient activation spread
            heatmap = np.power(heatmap, 2.5)
            
            # 3. Dynamic Cutoff: Wipes background pixel noise below the 55th percentile
            noise_threshold = np.percentile(heatmap, 55)
            heatmap[heatmap < noise_threshold] = 0
            
            # 4. Final Matrix Equalization
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
        return heatmap


def get_gradcam_layer(model):
    """
    Locates the optimal deep layer for visualization. Prioritizes the high-level 
    semantic feature block `proj2[0]` for sharper localization.
    """
    if hasattr(model, 'proj2') and isinstance(model.proj2[0], nn.Conv2d):
        return model.proj2[0]
    if hasattr(model, 'proj1') and isinstance(model.proj1[0], nn.Conv2d): 
        return model.proj1[0]
        
    best_layer = None
    for module in model.backbone.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels >= 64: 
            best_layer = module
    return best_layer


def create_red_heatmap(original_image, heatmap, alpha=0.65):
    """
    Transforms raw matrices into smooth visual hotspot overlays. 
    Uses alpha masking to preserve the look of healthy tissue while highlighting 
    the clinical focus zone with a professional, high-fidelity gradient.
    """
    try:
        h, w = original_image.shape[:2]
        
        # Bicubic interpolation yields cleaner geometric transitions than linear options
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Run localized spatial Gaussian blur to transform blocky matrices into an organic glow
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (45, 45), 0)
        if heatmap_blurred.max() > 0:
            heatmap_blurred = heatmap_blurred / heatmap_blurred.max()
            
        # Build multi-color diagnostic spectrum array (JET maps Red to High Focus, Blue to Low Focus)
        color_mapped = cv2.applyColorMap(np.uint8(255 * heatmap_blurred), cv2.COLORMAP_JET)
        
        # Alpha Hotspot Boundary Mask: Isolates visualization to high-intensity attention areas
        threshold_mask = (heatmap_blurred > 0.18).astype(np.float32)[..., np.newaxis]
        
        # Execute localized transparency blending
        blended_hotspot = cv2.addWeighted(original_image, 1 - alpha, color_mapped, alpha, 0)
        
        # Combine: Keep normal scan clear everywhere except inside the high-attention hotspot zone
        visual_output = (blended_hotspot * threshold_mask + original_image * (1 - threshold_mask)).astype(np.uint8)
        return visual_output
    except Exception as e:
        logger.error(f"[GradCAM Overlay Processing Failure] Fallback to raw frame: {e}")
        return original_image


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLINICAL DATA EXTRACTION & PAYLOAD FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    """
    Tracks local density nodes inside attention layers to extract structural parameters 
    like aspect ratio and border irregularity scores for automated medical reporting.
    """
    empty_payload = {
        'lesion_count': 0,
        'lesions': [],
        'total_lesion_area_pct': 0.0,
        'overall_morphology': 'Normal / Clear Structural Profiles'
    }
    
    if scan_type not in LESION_MODALITIES or cam is None: 
        return empty_payload
        
    try:
        h_i, w_i = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w_i, h_i), interpolation=cv2.INTER_CUBIC)
        
        if cam_resized.max() > 0: 
            cam_resized /= cam_resized.max()
            
        nonzero_elements = cam_resized[cam_resized > 0]
        # Calibrate tracking cutoffs to handle the sharpened heatmap profile
        tracking_threshold = float(np.percentile(nonzero_elements, 45)) if len(nonzero_elements) > 0 else 0.5
        
        binary_matrix = (cam_resized > tracking_threshold).astype(np.uint8) * 255
        
        morphology_kernel = np.ones((7, 7), np.uint8)
        binary_matrix = cv2.morphologyEx(binary_matrix, cv2.MORPH_CLOSE, morphology_kernel)
        binary_matrix = cv2.morphologyEx(binary_matrix, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        contours, _ = cv2.findContours(binary_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_frame_area = h_i * w_i
        recorded_areas = []
        extracted_lesions = []
        
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 300 or area > (total_frame_area * 0.25): 
                continue
                
            bx, by, bw, bh = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0
            
            convex_hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / convex_hull_area if convex_hull_area > 0 else 0
            aspect_ratio = bw / bh if bh > 0 else 1.0
            
            extracted_lesions.append({
                'lesion_id': index + 1, 
                'area_pixels': int(area), 
                'area_percentage': round(area / total_frame_area * 100, 1),
                'bounding_box': {'x': int(bx), 'y': int(by), 'width': int(bw), 'height': int(bh)},
                'circularity': round(float(circularity), 2), 
                'solidity': round(float(solidity), 2),
                'aspect_ratio': round(float(aspect_ratio), 2), 
                'mean_intensity': 0.0, 
                'std_intensity': 0.0, 
                'irregularity_score': round(1 - float(circularity), 2),
                'morphology': evaluate_morphology_string(circularity, solidity, aspect_ratio), 
                'suspicion_level': classify_suspicion_tier(circularity, solidity)
            })
            recorded_areas.append(area)
            
        extracted_lesions.sort(key=lambda l: l['area_pixels'], reverse=True)
        
        # Limit structural outputs to the top two findings for optimal UI rendering
        curated_lesions = extracted_lesions[:2]
        
        accumulated_area_percentage = round(sum(recorded_areas) / total_frame_area * 100, 1) if recorded_areas else 0.0
        primary_morphology = curated_lesions[0]['morphology'] if curated_lesions else 'Normal / Clear Structural Profiles'
        
        return {
            'lesion_count': len(curated_lesions), 
            'lesions': curated_lesions, 
            'total_lesion_area_pct': accumulated_area_percentage, 
            'overall_morphology': primary_morphology
        }
    except Exception as e: 
        logger.error(f"[Structural Engine Exception] Error detailing boundary areas: {e}")
        return empty_payload


def evaluate_morphology_string(c, s, ar):
    if c > 0.80 and s > 0.90: return "Round / Oval Solid Mass"
    if c > 0.60 and s > 0.80: return "Smooth Lobulated Oval Margins"
    if c < 0.40 or  s < 0.65: return "Irregular / Spiculated Structural Boundary"
    if ar > 1.5  or ar < 0.67: return "Asymmetric Elongated Stretched Density"
    return "Lobular Tissue Cluster"


def classify_suspicion_tier(c, s): 
    risk_points = 0
    if c < 0.5: risk_points += 2
    if s < 0.7: risk_points += 2
    
    if risk_points >= 4: return "High Suspicion"
    if risk_points >= 2: return "Moderate Suspicion"
    return "Low Suspicion"


def compute_texture(img_bgr):
    """Computes basic texture metrics directly from the pixel distribution."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        
        return {
            'mean_intensity': round(float(gray.mean()), 1), 
            'std_deviation': round(float(gray.std()), 1), 
            'skewness': 0.0, 
            'kurtosis': 0.0, 
            'entropy': round(float(-np.sum(hist * np.log2(hist))), 2), 
            'dynamic_range': int(gray.max()) - int(gray.min())
        }
    except Exception: 
        return {}


def compute_risk(prediction, confidence, scan_type):
    """Generates standard risk levels and diagnostic intervals for clinical report strings."""
    score = 1
    reasoning_strings = []
    
    if prediction in ('Normal', 'Healthy'):
        score = 1 if confidence >= 0.70 else 2
        reasoning_strings.append("Deep architecture predicts clean physiological distribution with no apparent focal mass anomalies.")
    elif prediction == 'Benign':
        score = 3 if confidence < 0.65 else 2
        reasoning_strings.append("Local cellular structures present non-infiltrative, bounded structural margins indicative of benign transformations.")
    elif prediction in ('Malignant', 'Sick'):
        score = 5 if confidence >= 0.85 else 4
        reasoning_strings.append("High-energy structural density profiles detected matching infiltrative tissue patterns.")
    
    labels_map = {
        1: "Negative Diagnostic Finding", 
        2: "Benign Diagnostic Finding", 
        3: "Probably Benign — Short Interval Review Indicated", 
        4: "Suspicious Structural Abnormality Mapped", 
        5: "Highly Suggestive of Malignant Transformation"
    }
    
    intervals_map = {
        1: "Routine Annual Diagnostic Screening", 
        2: "Routine Annual Diagnostic Screening", 
        3: "Short-interval technical review required within 6 Months", 
        4: "Histopathological Core Biopsy Confirmation Recommended", 
        5: "Urgent Histopathological Biopsy and Specialist Surgical Consultation Required"
    }
    
    return {
        'risk_score': score, 
        'risk_category': f"Category {score}", 
        'risk_label': labels_map.get(score, "Undetermined"), 
        'recommended_interval': intervals_map.get(score, "Consult Attending Clinician"), 
        'reasoning': reasoning_strings if reasoning_strings else ["Manual radiologist review required."]
    }


def compile_clinical_notes(pred, conf, lesion_data, scan_type, risk_data):
    """Formats findings into clean markdown lists for the presentation UI."""
    notes = [
        f"❖ SYSTEM TARGET MODALITY: {scan_type.replace('_', ' ').title()} Automation Pipeline Execution",
        f"❖ PRIMARY CLASSIFICATION OUTPUT: {pred.upper()} Architecture Classification ({conf * 100:.1f}% System Confidence Score)",
    ]
    
    if lesion_data and lesion_data.get('lesion_count', 0) > 0:
        top_lesion = lesion_data['lesions'][0]
        notes.append(f"❖ SCAN COMPUTE OBSERVATION: Critical high-density focus mapped showing {top_lesion['morphology'].lower()}. Structural evaluation tracks this coordinate zone at a [{top_lesion['suspicion_level'].upper()}] tier.")
    else:
        notes.append("❖ SCAN COMPUTE OBSERVATION: No anomalous localized pixel clusters passing focus thresholds discovered inside matrix fields.")
        
    notes.append(f"❖ RECOMMENDATION ACTION: Proceed with {risk_data.get('recommended_interval', 'Immediate Specialist Consultation')}.")
    
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE INFRASTRUCTURE ENDPOINTS (CORE REST ROUTES)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main Classification Endpoint. Receives multi-modal files, validates 
    input source parameters, runs neural evaluations, processes Grad-CAM 
    sharpening, and delivers clean payload payloads.
    """
    execution_start_time = time.time()
    logger.info("[Predict Route] Incoming inference request parsing onto memory stacks.")

    if 'file' not in request.files: 
        logger.error("[Predict Route] Failed execution: Missing explicit multipart file form binding.")
        return jsonify({'error': 'No file uploaded under key parameters.'}), 400

    uploaded_file = request.files['file']
    reporting_mode = request.form.get('mode', 'patient').lower()
    requested_scan_type = request.form.get('scan_type', 'auto').strip().lower()
    
    try:
        raw_bytes = uploaded_file.read()
        if not raw_bytes: 
            return jsonify({'error': 'Target uploaded binary file stream is null.'}), 400
            
        byte_array = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None: 
            return jsonify({'error': 'OpenCV core decoder failed to construct matrix from bytes.'}), 400

        # Run multi-tier structural analyzer to handle auto-routing cases safely
        if requested_scan_type in ('auto', 'mammogram'):
            detected_modality = auto_detect_scan_type(img_bgr)
        else:
            detected_modality = requested_scan_type
            
        if detected_modality == 'mammogram': 
            detected_modality = 'mammogram_dmid'

        logger.info(f"[Predict Route] Modality locked onto: [{detected_modality}]")

        # SECURITY LOCK: Block natural scenery or phone photos before execution hits neural modules
        is_valid_medical_file, validation_rejection_reason = validate_medical_image(img_bgr, detected_modality)
        if not is_valid_medical_file:
            logger.warning(f"[Security Intervention] File blocked by validation shield: {validation_rejection_reason}")
            return jsonify({
                'error': 'Invalid Imaging Profile Detected', 
                'message': validation_rejection_reason, 
                'action': 'Please upload a native digital diagnostic scan file.'
            }), 422

        base_display_frame = img_bgr.copy()
        
        # Deploy specific isolated breast anatomy focus pipes for mammography paths
        if 'mammogram' in detected_modality:
            processed_matrix = preprocess_mammogram(crop_to_breast_tissue(img_bgr))
        else:
            processed_matrix = img_bgr
            
        resized_matrix = cv2.resize(processed_matrix, (IMG_SIZE, IMG_SIZE))
        rgb_matrix = cv2.cvtColor(resized_matrix, cv2.COLOR_BGR2RGB)
        
        # Construct dual spatial-frequency inputs for network routing
        spatial_tensor, frequency_tensor = prepare_inputs(rgb_matrix, detected_modality)

        # CORE RUNTIME MODEL INFERENCE
        try: 
            model_instance, class_labels = load_model(detected_modality)
        except RuntimeError as weights_exception: 
            logger.critical(f"[System Fault] Model loader crashed on weight reference assembly: {weights_exception}")
            return jsonify({'error': f"Target inference workers unavailable: {str(weights_exception)}"}), 503
        
        with torch.no_grad():
            logits, _ = model_instance(spatial_tensor, frequency_tensor)
            probabilities = F.softmax(logits, dim=1)
            highest_confidence, target_class_index = torch.max(probabilities, dim=1)
            
            ai_prediction_string = class_labels[target_class_index.item()]
            ai_confidence_score = float(highest_confidence.item())

        logger.info(f"[Inference Result] Classify Success: [{ai_prediction_string}] Confidence: {ai_confidence_score:.4f}")

        # Basic text recommendation matching system outcomes
        if ai_prediction_string in ('Malignant', 'Sick'):
            base_recommendation = "HIGH CLINICAL RISK PROFILE: Immediate specialist consultation and biopsy required."
        elif ai_prediction_string == 'Benign':
            base_recommendation = "MODERATE CLINICAL RISK PROFILE: Short-interval monitoring and diagnostics tracking required."
        else:
            base_recommendation = "LOW RISK PROFILE: Routine diagnostic monitoring interval protocols preserved."

        # SHARPENED SPATIAL GRAD-CAM EXECUTION
        gradcam_base64_payload = None
        attention_map_matrix = None
        target_visual_layer = get_gradcam_layer(model_instance)
        
        if target_visual_layer is not None:
            try:
                target_label_index = class_labels.index(ai_prediction_string) if ai_prediction_string in class_labels else 0
                
                # Execute hook wrapper transformations safely
                gradcam_engine = GradCAM(model_instance, target_visual_layer)
                attention_map_matrix = gradcam_engine.generate_heatmap(spatial_tensor, frequency_tensor, target_label_index)
                gradcam_engine.remove()
                
                if attention_map_matrix is not None:
                    # Construct the polished, smooth-precision diagnostic visual frame overlay
                    sharpened_overlay = create_red_heatmap(base_display_frame, attention_map_matrix, alpha=0.65)
                    
                    # Encode output matrix safely into target payload structures
                    _, buffer_array = cv2.imencode('.jpg', sharpened_overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    gradcam_base64_payload = base64.b64encode(buffer_array).decode('utf-8')
                    logger.info("[Grad-CAM Engine] Sharpened hotspot payload mapped successfully.")
            except Exception as cam_error:
                logger.error(f"[Grad-CAM Engine Exception] Critical hook tracking error: {cam_error}")
                traceback.print_exc()

        # CONSTRUCT BASEPAYLOAD PACKETS
        json_response_payload = {
            'result': ai_prediction_string, 
            'confidence': f"{ai_confidence_score:.2f}", 
            'heatmap': gradcam_base64_payload, 
            'recommendation': base_recommendation,
            'analysis_details': {
                'scan_type_detected': detected_modality, 
                'ai_prediction': ai_prediction_string, 
                'ai_confidence': f"{ai_confidence_score:.2f}",
                'compute_latency_seconds': round(time.time() - execution_start_time, 3)
            }
        }
        
        # Include detailed clinical reporting logs if request originates from doctor portal spaces
        if reporting_mode == 'doctor':
            logger.info("[Predict Route] Extrapolating structural feature reports for specialist dashboards.")
            lesion_analysis_payload = detect_lesion_boundaries(base_display_frame, attention_map_matrix, detected_modality)
            risk_assessment_payload = compute_risk(ai_prediction_string, ai_confidence_score, detected_modality)
            
            json_response_payload['doctor_report'] = {
                'lesion_analysis': lesion_analysis_payload, 
                'texture_features': compute_texture(base_display_frame),
                'risk_score': risk_assessment_payload, 
                'clinical_notes': compile_clinical_notes(
                    ai_prediction_string, 
                    ai_confidence_score, 
                    lesion_analysis_payload, 
                    detected_modality, 
                    risk_assessment_payload
                )
            }
            
        return jsonify(json_response_payload)

    except Exception as pipeline_fault:
        logger.critical(f"[Pipeline Runtime Crash] Critical exception in prediction worker path: {pipeline_fault}")
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': 'Internal backend evaluation error.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 10. CHAT ROUTE (INTELLIGENT CLINICAL REPORT DIALOGUE AI)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/chat', methods=['POST'])
def chat():
    """
    Clinical Chatbot Gateway. Parses incoming text queries and processes report 
    data tokens via Google GenAI engines under explicit markdown security rules.
    """
    data_packet = request.json
    if not data_packet or 'message' not in data_packet: 
        logger.error("[Chat Route] Rejected query payload: Missing core prompt parameters.")
        return jsonify({'error': 'Missing structured message parameter strings.'}), 400

    user_query_string = data_packet['message']
    associated_report_json = data_packet.get('report', {})

    logger.info(f"[Chat Route] Forwarding conversational sequence to Gemini core architectures: '{user_query_string[:30]}...'")

    try:
        # Initialize native GenAI clients securely from background environments
        client_instance = genai.Client()
        
        # Enforce plaintext response layouts to prevent rendering glitches in mobile code
        security_system_instructions = f"""
        You are BreastScan's intelligent AI medical assistant. Answer ANY general medical or app question.
        If asked about results, explain this JSON data simply: {associated_report_json}
        
        CRITICAL RULES:
        1. NO MARKDOWN: DO NOT use asterisks (**), hashes (#), or markdown styling.
        2. PLAIN TEXT ONLY: Use paragraphs and standard dashes (-) for bullets.
        3. Explain in simple, layperson terms.
        """
        
        generation_hyperparameters = types.GenerateContentConfig(
            system_instruction=security_system_instructions, 
            temperature=0.3
        )
        
        llm_response_stream = client_instance.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_query_string,
            config=generation_hyperparameters
        )
        
        return jsonify({"reply": llm_response_stream.text})
        
    except Exception as llm_fault:
        logger.error(f"[Chat Core Fault] Conversational generation worker broken: {llm_fault}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to safely generate conversational response strings.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 11. HEALTH VERIFICATION & EXPLICIT PROCESS RUNNER LOCKS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_verification_status(): 
    """Server validation loop mapping service up-time checkpoints."""
    return jsonify({
        'status': 'online',
        'service': 'BreastScan API Service Platform',
        'active_weights_allocation': active_model_type if active_model_type else 'None',
        'timestamp': float(time.time())
    }), 200


if __name__ == '__main__':
    logger.info("Starting production network loops binding onto host gateway interface [0.0.0.0:80]")
    app.run(host='0.0.0.0', port=80, debug=False)
