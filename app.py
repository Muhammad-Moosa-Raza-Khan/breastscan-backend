import os
import io
import gc
import json
import base64
import traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# =====================================================================
# --- 1. INITIALIZE APP ---
# =====================================================================
app = Flask(__name__)
application = app

# Hard limit on upload size: 15 MB (protects free-tier RAM)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

UPLOAD_FOLDER = '/tmp'

MODEL_PATHS = {
    'mammogram_mias': 'models/mammo_mias.pth',
    'mammogram_dmid': 'models/mammo_dmid.pth',
    'ultrasound':     'models/ultrasound_model.pth',
    'histopathology': 'models/hist_model.pth',
    'mri':            'models/mri_model.pth'
}

DATASET_CONFIGS = {
    'histopathology': {'num_classes': 2, 'labels': ['Benign', 'Malignant']},
    'mri':            {'num_classes': 2, 'labels': ['Healthy', 'Sick']},
    'mammogram_dmid': {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']},
    'mammogram_mias': {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']},
    'ultrasound':     {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']}
}

# Valid scan types the API accepts
VALID_SCAN_TYPES = set(DATASET_CONFIGS.keys()) | {'mammogram'}

active_model_type = None
loaded_model      = None
detected_classes  = []

# =====================================================================
# --- 2. CUSTOM ARCHITECTURE (RAM-NET) — unchanged skeleton ---
# =====================================================================
class RAMNet(nn.Module):
    def __init__(self, num_classes):
        super(RAMNet, self).__init__()
        effnet = models.efficientnet_v2_s(weights=None)
        self.spatial_encoder = effnet.features

        # Paste your FULL __init__ here to match the saved .pth weights.
        # The skeleton below is kept so the server doesn't crash when
        # weights load with strict=False.
        self.temp_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # Paste your FULL forward() here.
        s_feat = self.spatial_encoder(x)
        return self.temp_classifier(s_feat)


# =====================================================================
# --- 3. DYNAMIC MODEL LOADER  (with health-check) ---
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
        model      = RAMNet(num_classes=config['num_classes'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # ── Handle checkpoint dict vs plain state_dict ──
        # Saved with torch.save({'model_state_dict': ..., 'epoch': ...}, path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch      = checkpoint.get('epoch', 'N/A')
            best_acc   = checkpoint.get('best_acc', 'N/A')
            print(f"   [INFO] Checkpoint — epoch: {epoch}, best_acc: {best_acc}")
        else:
            # Plain state_dict saved with torch.save(model.state_dict(), path)
            state_dict = checkpoint

        # Report any key mismatches so you can debug architecture gaps
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   [WARN] Missing keys ({len(missing)}): {missing[:5]}")
            print(f"   *** ACTION NEEDED: Paste your full RAMNet __init__ & forward() into this file ***")
        if unexpected:
            print(f"   [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        model.eval()
        loaded_model      = model
        active_model_type = scan_type

        # ── Health-check: make sure the model doesn't always pick one class ──
        healthy, sample_preds = _verify_model_diversity(model, config['num_classes'])
        if not healthy:
            print(f"   [WARN] Model outputs same class for all test inputs: {sample_preds}. "
                  f"Weights may not have loaded correctly — check architecture match.")

        return True
    except Exception as e:
        print(f" ! [FATAL] Cannot load {scan_type}: {e}")
        traceback.print_exc()
        loaded_model      = None
        active_model_type = None
        return False


def _verify_model_diversity(model, num_classes):
    """
    Run three synthetic inputs; warn if all produce the same argmax.
    Returns (is_diverse: bool, predictions: list).
    """
    preds = []
    try:
        with torch.no_grad():
            for t in [torch.zeros(1,3,224,224),
                      torch.ones(1,3,224,224)*0.5,
                      torch.rand(1,3,224,224)]:
                out = model(t)
                preds.append(torch.argmax(out, dim=1).item())
        return len(set(preds)) > 1, preds
    except Exception:
        return True, preds   # Don't block loading on error


# Pre-load default model at startup
load_pytorch_model('mammogram_dmid')


# =====================================================================
# --- 4. IMAGE TRANSFORM ---
# =====================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =====================================================================
# --- 5. MEDICAL IMAGE VALIDATOR (NEW) ---
# =====================================================================
def validate_medical_image(img_cv2, scan_type):
    """
    Comprehensive image quality + modality validation.
    Returns (is_valid: bool, rejection_reason: str | None).
    """
    try:
        h, w = img_cv2.shape[:2]

        # ── 5a. Minimum resolution ──
        if h < 128 or w < 128:
            return False, ("Image resolution is too low. "
                           "Please upload a minimum 128×128 medical scan.")

        # ── 5b. Maximum resolution guard (avoid OOM on free tier) ──
        if h > 4096 or w > 4096:
            return False, ("Image is extremely large. "
                           "Please resize to under 4096×4096 before uploading.")

        gray = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape) == 3 else img_cv2.copy())

        # ── 5c. Blank / black screen ──
        mean_px = float(np.mean(gray))
        if mean_px < 8:
            return False, ("Image appears to be blank or completely black. "
                           "Please retake or re-export the scan.")

        # ── 5d. All-white / overexposed ──
        if mean_px > 248:
            return False, ("Image is overexposed or all-white. "
                           "Please retake the scan with correct exposure.")

        # ── 5e. Blur detection (Laplacian variance) ──
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 30:
            return False, ("Image is too blurry or out of focus. "
                           "Please retake with a sharper scan.")

        # ── 5f. Insufficient contrast ──
        if float(np.std(gray)) < 12:
            return False, ("Image has insufficient contrast. "
                           "Please use a properly acquired medical scan.")

        # ── 5g. Modality-specific colour check ──
        if len(img_cv2.shape) == 3:
            b_mean = float(img_cv2[:, :, 0].mean())
            g_mean = float(img_cv2[:, :, 1].mean())
            r_mean = float(img_cv2[:, :, 2].mean())
            color_var = float(np.std([b_mean, g_mean, r_mean]))

            if scan_type == 'histopathology':
                # H&E slides are coloured — if near-grayscale, likely wrong modality
                if color_var < 5:
                    return False, ("Histopathology images should be H&E-stained "
                                   "(pink/purple). Please upload a proper tissue slide.")
            else:
                # Mammogram / MRI / Ultrasound must be near-grayscale
                if color_var > 30:
                    pretty = scan_type.replace('_', ' ').title()
                    return False, (f"This does not appear to be a {pretty} image. "
                                   f"Please upload a proper grayscale medical scan.")

        # ── 5h. Screenshot / UI element detection ──
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 2,
                                threshold=120,
                                minLineLength=int(w * 0.75),
                                maxLineGap=10)
        if lines is not None and len(lines) > 6:
            return False, ("Image appears to contain UI / screenshot elements. "
                           "Please crop and upload only the scan area.")

        # ── 5i. Mostly uniform regions (solid-colour photo / wallpaper) ──
        blocks = _compute_block_variance(gray, block_size=32)
        uniform_blocks = np.sum(blocks < 5)
        if uniform_blocks / blocks.size > 0.85:
            return False, ("Image appears to be a solid-colour or near-uniform image, "
                           "not a medical scan. Please upload the correct file.")

        return True, None

    except Exception as e:
        print(f"[VALIDATE] Error: {e}")
        # If validation itself crashes, allow the image through with a warning
        return True, None


def _compute_block_variance(gray, block_size=32):
    """Return array of local variance values for uniform-image detection."""
    h, w   = gray.shape
    rows   = h // block_size
    cols   = w // block_size
    variances = []
    for r in range(rows):
        for c in range(cols):
            block = gray[r*block_size:(r+1)*block_size,
                         c*block_size:(c+1)*block_size]
            variances.append(float(np.var(block)))
    return np.array(variances) if variances else np.array([100.0])


# =====================================================================
# --- 6. AUTO SCAN-TYPE DETECTION (heuristic, NEW) ---
# =====================================================================
def auto_detect_scan_type(img_cv2):
    """
    Heuristic modality detector when scan_type == 'auto'.
    Returns one of the DATASET_CONFIGS keys.
    """
    try:
        gray = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape) == 3 else img_cv2.copy())

        # Colourfulness → histopathology
        if len(img_cv2.shape) == 3:
            b_m = float(img_cv2[:, :, 0].mean())
            g_m = float(img_cv2[:, :, 1].mean())
            r_m = float(img_cv2[:, :, 2].mean())
            if float(np.std([b_m, g_m, r_m])) > 15:
                return 'histopathology'

        # Dark border + bright centre → MRI
        border_mean  = float(np.mean([gray[0, :].mean(), gray[-1, :].mean(),
                                      gray[:, 0].mean(),  gray[:, -1].mean()]))
        h4, w4       = gray.shape[0] // 4, gray.shape[1] // 4
        centre_mean  = float(gray[h4:3*h4, w4:3*w4].mean())
        dark_ratio   = (centre_mean - border_mean) / (centre_mean + 1)
        if dark_ratio > 0.35:
            return 'mri'

        # High-frequency speckle → ultrasound
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var > 800:
            return 'ultrasound'

        # Default to mammogram
        return 'mammogram_dmid'

    except Exception:
        return 'mammogram_dmid'


# =====================================================================
# --- 7. PREPROCESSING (unchanged) ---
# =====================================================================
def preprocess_mammogram(img_cv2):
    try:
        gray     = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                    if len(img_cv2.shape) == 3 else img_cv2.copy())
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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
        kernel    = np.ones((5, 5), np.uint8)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c        = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad      = max(5, min(w, h) // 20)
            hi, wi   = img_cv2.shape[:2]
            return img_cv2[max(0, y-pad):min(hi, y+h+pad),
                           max(0, x-pad):min(wi, x+w+pad)]
        return img_cv2
    except Exception:
        return img_cv2


def get_tissue_mask_advanced(img_cv2):
    try:
        gray      = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                     if len(img_cv2.shape) == 3 else img_cv2)
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel    = np.ones((5, 5), np.uint8)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.erode(thresh, np.ones((7, 7), np.uint8), iterations=1)
    except Exception:
        return np.ones(img_cv2.shape[:2], dtype=np.uint8) * 255


# =====================================================================
# --- 8. CV ABNORMALITY ANALYSIS (tuned thresholds, unchanged logic) ---
# =====================================================================
def comprehensive_abnormality_analysis(img_cv2):
    try:
        gray    = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                   if len(img_cv2.shape) == 3 else img_cv2)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, bright_mask = cv2.threshold(enhanced, 210, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        suspicious_regions, total_suspicious_area = [], 0
        for cnt in bright_contours:
            area = cv2.contourArea(cnt)
            if area > 30:
                x, y, w, h   = cv2.boundingRect(cnt)
                perimeter    = cv2.arcLength(cnt, True)
                circularity  = (4*np.pi*area/(perimeter**2)) if perimeter > 0 else 0
                mask         = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_int     = cv2.mean(enhanced, mask=mask)[0]

                region_score  = (3 if area > 100 else 2 if area > 50 else 1)
                region_score += (2 if mean_int > 230 else 1 if mean_int > 220 else 0)
                region_score += (1 if circularity > 0.6 else 0)

                if region_score >= 3:
                    suspicious_regions.append({
                        'area': area, 'circularity': circularity,
                        'intensity': mean_int, 'score': region_score
                    })
                    total_suspicious_area += area

        num_susp = len(suspicious_regions)
        if num_susp == 0:
            severity, confidence, category = 0.0, 0.9, "Normal"
        elif num_susp == 1:
            r = suspicious_regions[0]
            if r['area'] > 500 or r['intensity'] > 235:
                severity, category, confidence = 0.8, "Malignant", 0.7
            else:
                severity, category, confidence = 0.5, "Benign",    0.7
        else:
            if any(r['area'] > 300 for r in suspicious_regions):
                severity, category, confidence = 0.9, "Malignant", 0.8
            else:
                severity, category, confidence = 0.6, "Benign",    0.8

        return severity, confidence, {
            'suspicious_regions': num_susp,
            'total_area':         int(total_suspicious_area),
            'max_intensity':      max([r['intensity'] for r in suspicious_regions])
                                  if suspicious_regions else 0,
            'cv_prediction':      category
        }
    except Exception:
        return 0.0, 0.0, {}


# =====================================================================
# --- 9. ENHANCED GRAD-CAM (scan-type aware layer selection) ---
# =====================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient)
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, x, class_idx=None):
        try:
            self.model.eval()
            output = self.model(x)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[:, class_idx].backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                return None

            grads    = self.gradients.cpu().numpy()[0]   # (C, H, W)
            acts     = self.activations.cpu().numpy()[0] # (C, H, W)
            weights  = np.mean(grads, axis=(1, 2))       # (C,)
            cam      = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        except Exception as e:
            print(f"[GradCAM] Error: {e}")
            return None


def get_target_layer(model, scan_type='mammogram_dmid'):
    """
    Improved: prefer the last Conv2d inside the spatial encoder with
    large channel count (rich features). Falls back gracefully.
    """
    try:
        # 1st priority: last large conv inside spatial_encoder
        if hasattr(model, 'spatial_encoder'):
            best = None
            for m in model.spatial_encoder.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels >= 256:
                    best = m
            if best is not None:
                return best

        # 2nd priority: any conv ≥ 128 channels in the whole model
        best = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 128:
                best = m
        if best is not None:
            return best

        # Final fallback: very last conv anywhere
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                return m

    except Exception:
        pass
    return None


def refine_gradcam_with_intensity(gradcam_mask, original_img):
    try:
        gray         = (cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                        if len(original_img.shape) == 3 else original_img)
        intensity    = cv2.resize(
            np.power(gray.astype(np.float32) / 255.0, 0.7), (224, 224))
        refined      = (gradcam_mask * 0.7) + (intensity * 0.3)
        return refined / refined.max() if refined.max() > 0 else refined
    except Exception:
        return gradcam_mask


def generate_heatmap_overlay(original_cv2, heatmap, prediction_label, scan_type):
    """
    Improved heatmap overlay with scan-type specific thresholds.
    """
    try:
        img      = original_cv2.copy()
        heatmap  = refine_gradcam_with_intensity(heatmap, img)

        # Only apply tissue mask for mammograms; others use full image
        if 'mammogram' in scan_type:
            tissue_mask = get_tissue_mask_advanced(img)
            tissue_bool = tissue_mask > 0
        else:
            tissue_bool = np.ones(img.shape[:2], dtype=bool)

        masked_hm = heatmap * tissue_bool

        # Scan-type + prediction tuned threshold
        if prediction_label == 'Malignant':
            threshold = 0.25
        elif prediction_label in ('Benign', 'Sick'):
            threshold = 0.30
        else:
            threshold = 0.35

        # Histopathology and MRI benefit from a slightly lower threshold
        if scan_type in ('histopathology', 'mri'):
            threshold = max(threshold - 0.05, 0.15)

        masked_hm[masked_hm < threshold] = 0
        if masked_hm.max() > 0:
            masked_hm = masked_hm / masked_hm.max()

        masked_hm = cv2.GaussianBlur(masked_hm, (15, 15), 0)

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * masked_hm), cv2.COLORMAP_JET)
        overlay   = img.copy()
        heat_mask = (masked_hm > 0.01) & tissue_bool

        if np.any(heat_mask):
            overlay[heat_mask] = cv2.addWeighted(
                heatmap_colored[heat_mask], 0.65,
                img[heat_mask], 0.35, 0)

        _, buf = cv2.imencode('.jpg', overlay,
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        print(f"[Heatmap] Error: {e}")
        return None


# =====================================================================
# --- 10. DOCTOR-ONLY EXTRAS (NEW) ---
# =====================================================================

def detect_lesion_boundaries(img_cv2, gradcam_mask, scan_type):
    """
    Extract lesion contours from the Grad-CAM attention mask and compute
    shape/morphology features for each region of interest.
    """
    result = {
        'lesion_count':            0,
        'lesions':                 [],
        'total_lesion_area_pct':   0.0,
        'overall_morphology':      'N/A'
    }
    try:
        if gradcam_mask is None:
            return result

        # Threshold + clean
        binary = (gradcam_mask > 0.45).astype(np.uint8) * 255
        k      = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        gray        = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                       if len(img_cv2.shape) == 3 else img_cv2)
        total_px    = img_cv2.shape[0] * img_cv2.shape[1]
        total_area  = 0
        lesions     = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 60:           # Skip noise
                continue

            x, y, w, h   = cv2.boundingRect(cnt)
            perimeter    = cv2.arcLength(cnt, True)
            circularity  = ((4 * np.pi * area) / (perimeter ** 2)
                            if perimeter > 0 else 0)

            hull         = cv2.convexHull(cnt)
            hull_area    = cv2.contourArea(hull)
            solidity     = area / hull_area if hull_area > 0 else 0

            aspect_ratio = w / h if h > 0 else 1.0

            mask_roi = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
            mean_int = float(cv2.mean(gray, mask=mask_roi)[0])
            std_int  = float(np.std(gray[mask_roi > 0])) if mask_roi.any() else 0.0

            morphology = _classify_morphology(circularity, solidity, aspect_ratio)
            suspicion  = _suspicion_level(circularity, solidity, mean_int)

            lesions.append({
                'lesion_id':          i + 1,
                'area_pixels':        int(area),
                'area_percentage':    round(area / total_px * 100, 2),
                'bounding_box':       {'x': int(x), 'y': int(y),
                                       'width': int(w), 'height': int(h)},
                'circularity':        round(float(circularity), 3),
                'solidity':           round(float(solidity), 3),
                'aspect_ratio':       round(float(aspect_ratio), 3),
                'mean_intensity':     round(mean_int, 2),
                'std_intensity':      round(std_int, 2),
                'irregularity_score': round(1 - float(circularity), 3),
                'morphology':         morphology,
                'suspicion_level':    suspicion,
            })
            total_area += area

        lesions.sort(key=lambda l: l['area_pixels'], reverse=True)
        result['lesion_count']          = len(lesions)
        result['lesions']               = lesions
        result['total_lesion_area_pct'] = round(total_area / total_px * 100, 2)
        result['overall_morphology']    = (
            lesions[0]['morphology'] if lesions else 'No detectable lesion')

    except Exception as e:
        print(f"[Lesion] Error: {e}")

    return result


def _classify_morphology(circularity, solidity, aspect_ratio):
    if circularity > 0.80 and solidity > 0.90:
        return "Round / Oval — Low suspicion"
    elif circularity > 0.60 and solidity > 0.80:
        return "Oval with smooth margins — Low-intermediate suspicion"
    elif circularity < 0.40 or solidity < 0.65:
        return "Irregular / Spiculated — High suspicion"
    elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
        return "Elongated — Moderate suspicion"
    else:
        return "Lobular — Moderate suspicion"


def _suspicion_level(circularity, solidity, mean_intensity):
    score = 0
    if circularity < 0.5:  score += 2
    if solidity < 0.7:     score += 2
    if mean_intensity > 200: score += 1
    if score >= 4:   return "High"
    elif score >= 2: return "Moderate"
    else:            return "Low"


def compute_texture_features(img_cv2, gradcam_mask=None):
    """
    Statistical texture features for the region of interest
    (or whole image if no mask).
    """
    try:
        gray = (cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                if len(img_cv2.shape) == 3 else img_cv2.copy())

        if gradcam_mask is not None:
            roi_mask = (gradcam_mask > 0.45).astype(np.uint8)
            roi      = gray[roi_mask > 0] if roi_mask.any() else gray.flatten()
        else:
            roi = gray.flatten()

        if len(roi) == 0:
            roi = gray.flatten()

        roi_f = roi.astype(np.float64)
        mean  = float(np.mean(roi_f))
        std   = float(np.std(roi_f))
        skew  = float(np.mean(((roi_f - mean) / std) ** 3)) if std > 0 else 0.0
        kurt  = float(np.mean(((roi_f - mean) / std) ** 4) - 3) if std > 0 else 0.0

        hist   = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist   = hist / hist.sum()
        hist   = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        return {
            'mean_intensity':  round(mean, 2),
            'std_deviation':   round(std, 2),
            'skewness':        round(skew, 4),
            'kurtosis':        round(kurt, 4),
            'entropy':         round(entropy, 4),
            'dynamic_range':   int(int(gray.max()) - int(gray.min())),
        }
    except Exception as e:
        print(f"[Texture] Error: {e}")
        return {}


def compute_risk_score(ai_prediction, ai_confidence, cv_severity,
                       lesion_data, scan_type):
    """
    BI-RADS–inspired risk score (1–5) with reasoning.
    """
    try:
        score     = 1
        reasoning = []

        if ai_prediction == 'Normal' or ai_prediction == 'Healthy':
            score = 1 if ai_confidence >= 0.70 else 2
            reasoning.append("AI predicts no significant abnormality.")
        elif ai_prediction == 'Benign':
            score = 3 if ai_confidence < 0.65 else 2
            reasoning.append("AI detects likely benign finding.")
        elif ai_prediction in ('Malignant', 'Sick'):
            if ai_confidence >= 0.85:
                score = 5
                reasoning.append("AI is highly confident of malignant pattern.")
            elif ai_confidence >= 0.70:
                score = 4
                reasoning.append("AI detects suspicious abnormality.")
            else:
                score = 3
                reasoning.append("AI detects possible abnormality with moderate confidence.")

        # CV severity adjustment — only promote, never demote
        if cv_severity >= 0.8 and score < 4:
            score += 1
            reasoning.append("CV analysis identified high-intensity suspicious regions.")
        elif cv_severity >= 0.6 and score < 3:
            score += 1
            reasoning.append("CV analysis found moderate imaging abnormality.")

        # Morphology from lesion data
        if lesion_data.get('lesion_count', 0) > 0:
            high_susp = [l for l in lesion_data['lesions']
                         if l.get('suspicion_level') == 'High']
            if high_susp and score < 4:
                score = min(score + 1, 5)
                reasoning.append(
                    f"{len(high_susp)} lesion(s) with high-suspicion morphology detected.")

        score = min(score, 5)

        labels = {
            1: "Negative — No significant finding",
            2: "Benign finding — Routine follow-up",
            3: "Probably benign — Short-interval follow-up (6 months)",
            4: "Suspicious — Tissue sampling recommended",
            5: "Highly suggestive of malignancy — Biopsy required"
        }
        intervals = {
            1: "Routine annual screening",
            2: "Routine annual screening",
            3: "Follow-up imaging in 6 months",
            4: "Biopsy / specialist referral within 2 weeks",
            5: "Urgent biopsy required"
        }

        return {
            'risk_score':            score,
            'risk_category':         f"Category {score}",
            'risk_label':            labels.get(score, ""),
            'recommended_interval':  intervals.get(score, ""),
            'reasoning':             reasoning
        }
    except Exception as e:
        print(f"[RiskScore] Error: {e}")
        return {}


def generate_annotated_image(img_cv2, gradcam_mask, prediction_label):
    """
    Annotated image with lesion contours + bounding boxes (doctor mode).
    """
    try:
        annotated = img_cv2.copy()
        if gradcam_mask is None:
            return None

        binary = (gradcam_mask > 0.45).astype(np.uint8) * 255
        k      = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        colour_map = {
            'Malignant': (0,   0,   255),  # red
            'Sick':      (0,   0,   255),
            'Benign':    (0,  165,  255),  # orange
            'Normal':    (0,  255,    0),  # green
            'Healthy':   (0,  255,    0),
        }
        colour = colour_map.get(prediction_label, (255, 255, 0))

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 60:
                continue
            cv2.drawContours(annotated, [cnt], -1, colour, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 1)
            label_pos = (x, max(15, y - 5))
            cv2.putText(annotated, f"R{i+1}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                        cv2.LINE_AA)

        _, buf = cv2.imencode('.jpg', annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        print(f"[Annotated] Error: {e}")
        return None


# =====================================================================
# --- 11. ROUTES ---
# =====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    # ── Guard: file present? ──
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send a file with key "file".'}), 400

    file      = request.files['file']
    mode      = request.form.get('mode', 'patient').lower()
    scan_type = request.form.get('scan_type', 'mammogram_dmid').strip().lower()

    # ── Guard: valid scan_type? ──
    if scan_type not in VALID_SCAN_TYPES and scan_type != 'auto':
        return jsonify({
            'error': f'Unknown scan_type "{scan_type}". '
                     f'Valid values: {sorted(VALID_SCAN_TYPES | {"auto"})}'
        }), 400

    try:
        # ── Load & decode image ──
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return jsonify({'error': 'Uploaded file is empty.'}), 400

        np_arr   = np.frombuffer(file_bytes, np.uint8)
        img_cv2  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            return jsonify({
                'error': 'Could not decode the image. '
                         'Please upload a valid JPEG, PNG, or BMP file.'
            }), 400

        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # ── Auto-detect scan type if requested ──
        if scan_type == 'auto':
            scan_type = auto_detect_scan_type(img_cv2)
            print(f" * [AUTO-DETECT] Detected scan type: {scan_type}")

        # Resolve generic 'mammogram' to DMID for validation purposes
        # (auto-compare will still run both)
        validation_type = (
            'mammogram_dmid' if scan_type == 'mammogram' else scan_type)

        # ── Medical image validation ──
        is_valid, rejection_reason = validate_medical_image(
            img_cv2, validation_type)
        if not is_valid:
            return jsonify({
                'error':   'Invalid image',
                'message': rejection_reason,
                'action':  'Please retake or re-upload the scan.'
            }), 422

        # ── Preprocessing ──
        if 'mammogram' in scan_type:
            img_cropped   = crop_to_breast_tissue(img_cv2)
            img_processed = preprocess_mammogram(img_cropped)
        else:
            img_cropped   = img_cv2
            img_processed = img_cv2

        img_final     = cv2.resize(img_processed, (224, 224))
        img_pil_final = Image.fromarray(cv2.cvtColor(img_final,
                                                      cv2.COLOR_BGR2RGB))
        input_tensor  = transform(img_pil_final).unsqueeze(0).to('cpu')

        # ─────────────────────────────────────────────────────────────
        # AUTO-COMPARE ENGINE (mammogram generic)
        # ─────────────────────────────────────────────────────────────
        is_auto_compare = (scan_type == 'mammogram')
        winning_model   = scan_type
        cv_details      = {}
        cv_severity     = 0.0

        if is_auto_compare:
            print(" * [AUTO-COMPARE] MIAS model…")
            if not load_pytorch_model('mammogram_mias'):
                return jsonify({'error': 'MIAS model unavailable.'}), 503
            with torch.no_grad():
                out_m   = loaded_model(input_tensor)
                prob_m  = torch.nn.functional.softmax(out_m, dim=1)
                conf_m, idx_m = torch.max(prob_m, 1)
                pred_m  = detected_classes[idx_m.item()]

            print(" * [AUTO-COMPARE] DMID model…")
            if not load_pytorch_model('mammogram_dmid'):
                return jsonify({'error': 'DMID model unavailable.'}), 503
            with torch.no_grad():
                out_d   = loaded_model(input_tensor)
                prob_d  = torch.nn.functional.softmax(out_d, dim=1)
                conf_d, idx_d = torch.max(prob_d, 1)
                pred_d  = detected_classes[idx_d.item()]

            if conf_d.item() >= conf_m.item():
                ai_prediction = pred_d
                ai_score      = conf_d.item()
                winning_model = 'DMID'
                all_probs     = prob_d[0].cpu().numpy()
            else:
                ai_prediction = pred_m
                ai_score      = conf_m.item()
                winning_model = 'MIAS'
                all_probs     = prob_m[0].cpu().numpy()
                load_pytorch_model('mammogram_mias')  # restore winner

        # ─────────────────────────────────────────────────────────────
        # STANDARD ROUTING
        # ─────────────────────────────────────────────────────────────
        else:
            if not load_pytorch_model(scan_type):
                return jsonify({'error': f'Model for "{scan_type}" is unavailable. '
                                         f'Check that the .pth file exists on the server.'}), 503
            with torch.no_grad():
                outputs   = loaded_model(input_tensor)
                probs     = torch.nn.functional.softmax(outputs, dim=1)
                all_probs = probs[0].cpu().numpy()
                confidence, class_idx = torch.max(probs, 1)

            ai_score      = float(confidence.item())
            ai_prediction = detected_classes[class_idx.item()]
            winning_model = scan_type

        # ─────────────────────────────────────────────────────────────
        # CV VALIDATION (mammogram only — tightened override rules)
        # ─────────────────────────────────────────────────────────────
        override_active  = False
        final_prediction = ai_prediction
        final_confidence = float(ai_score)

        probs_sorted      = np.sort(all_probs)[::-1]
        certainty_margin  = (float(probs_sorted[0] - probs_sorted[1])
                             if len(all_probs) > 1 else 1.0)

        if 'mammogram' in scan_type:
            cv_severity, cv_confidence, cv_details = (
                comprehensive_abnormality_analysis(img_cropped))
            cv_prediction = cv_details.get('cv_prediction', 'Normal')

            # ── TIGHTENED overrides (prevents spurious Malignant) ──
            # Only override if AI is very unsure AND CV is very sure
            if (ai_prediction == 'Normal'
                    and cv_severity > 0.75          # raised from 0.6
                    and certainty_margin < 0.20     # raised threshold for "unsure"
                    and cv_confidence > 0.75):
                override_active  = True
                final_prediction = cv_prediction
                final_confidence = cv_confidence
                print(f"   [CV-OVERRIDE] Normal→{cv_prediction} "
                      f"(margin={certainty_margin:.2f}, sev={cv_severity:.2f})")

            elif (ai_prediction == 'Benign'
                  and cv_severity > 0.90            # raised from 0.8
                  and certainty_margin < 0.15
                  and cv_confidence > 0.80):
                override_active  = True
                final_prediction = 'Malignant'
                final_confidence = cv_confidence
                print(f"   [CV-OVERRIDE] Benign→Malignant "
                      f"(margin={certainty_margin:.2f}, sev={cv_severity:.2f})")

        # ─────────────────────────────────────────────────────────────
        # RECOMMENDATION
        # ─────────────────────────────────────────────────────────────
        if final_prediction in ('Malignant', 'Sick'):
            recommendation = ("⚠️ HIGH RISK: Immediate biopsy and "
                              "specialist consultation required.")
        elif final_prediction == 'Benign':
            recommendation = ("⚡ MEDIUM RISK: Close monitoring with "
                              "follow-up imaging in 3–6 months.")
        else:
            recommendation = "✓ LOW RISK: Continue routine annual screening."

        # ─────────────────────────────────────────────────────────────
        # GRAD-CAM + DOCTOR EXTRAS
        # ─────────────────────────────────────────────────────────────
        heatmap_base64      = None
        annotated_base64    = None
        lesion_data         = {}
        texture_data        = {}
        risk_score_data     = {}
        cam_mask            = None

        target_layer = get_target_layer(loaded_model, scan_type)
        if target_layer is not None:
            try:
                final_idx = (detected_classes.index(final_prediction)
                             if final_prediction in detected_classes else 0)
                grad_cam  = GradCAM(loaded_model, target_layer)
                cam_mask  = grad_cam(input_tensor, class_idx=final_idx)
                grad_cam.remove_hooks()

                if cam_mask is not None:
                    heatmap_base64 = generate_heatmap_overlay(
                        img_final, cam_mask, final_prediction, scan_type)
            except Exception as e:
                print(f"[GradCAM-Run] {e}")

        if mode == 'doctor':
            # Lesion boundaries
            lesion_data = detect_lesion_boundaries(
                img_final, cam_mask, scan_type)

            # Texture features on ROI
            texture_data = compute_texture_features(img_final, cam_mask)

            # Risk / BI-RADS score
            risk_score_data = compute_risk_score(
                ai_prediction, float(ai_score),
                cv_severity, lesion_data, scan_type)

            # Annotated boundary image
            if cam_mask is not None:
                annotated_base64 = generate_annotated_image(
                    img_final, cam_mask, final_prediction)

        # ─────────────────────────────────────────────────────────────
        # BUILD RESPONSE
        # ─────────────────────────────────────────────────────────────
        response = {
            'result':         final_prediction,
            'confidence':     f"{final_confidence:.2f}",
            'heatmap':        heatmap_base64,
            'recommendation': recommendation,
            'analysis_details': {
                'scan_type_detected':  scan_type,
                'winning_model':       winning_model,
                'ai_prediction':       ai_prediction,
                'ai_confidence':       f"{ai_score:.2f}",
                'all_class_probs':     {
                    detected_classes[i]: round(float(all_probs[i]), 4)
                    for i in range(len(detected_classes))
                },
                'cv_severity':         f"{cv_severity:.2f}",
                'cv_details':          cv_details,
                'override_active':     override_active,
                'certainty_margin':    f"{certainty_margin:.2f}",
            }
        }

        if mode == 'doctor':
            response['doctor_report'] = {
                'annotated_image':  annotated_base64,
                'lesion_analysis':  lesion_data,
                'texture_features': texture_data,
                'risk_score':       risk_score_data,
                'clinical_notes': _generate_clinical_notes(
                    final_prediction, float(final_confidence),
                    lesion_data, texture_data, scan_type),
            }

        return jsonify(response)

    except Exception as e:
        print(f"[PREDICT] Unhandled error: {e}")
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': 'Internal server error. Please try again.'}), 500


def _generate_clinical_notes(prediction, confidence, lesion_data,
                              texture_data, scan_type):
    """Generate structured plain-text clinical notes for doctors."""
    notes = []

    notes.append(f"Scan type: {scan_type.replace('_',' ').title()}")
    notes.append(f"Primary finding: {prediction} (confidence {confidence:.0%})")

    if confidence < 0.60:
        notes.append("⚠ Low AI confidence — independent radiologist review strongly advised.")

    lc = lesion_data.get('lesion_count', 0)
    if lc > 0:
        notes.append(f"{lc} region(s) of interest identified.")
        for l in lesion_data.get('lesions', []):
            notes.append(
                f"  • Region {l['lesion_id']}: {l['morphology']} | "
                f"Area {l['area_percentage']:.1f}% | "
                f"Suspicion: {l['suspicion_level']}")
    else:
        notes.append("No distinct lesion boundaries detected in attention map.")

    if texture_data:
        notes.append(
            f"Texture — entropy: {texture_data.get('entropy','N/A')}, "
            f"std: {texture_data.get('std_deviation','N/A')}, "
            f"skewness: {texture_data.get('skewness','N/A')}")

    notes.append("Note: AI output is decision-support only. "
                 "Clinical judgement and histopathological confirmation are required.")

    return notes


# ─────────────────────────────────────────────────────────────────────
# --- 12. HEALTH CHECK ---
# ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status':       'online',
        'active_model': active_model_type,
        'classes':      detected_classes,
        'endpoints': {
            'POST /predict': {
                'params': {
                    'file':      'medical image (JPEG/PNG/BMP, max 15 MB)',
                    'scan_type': ('mammogram | mammogram_mias | mammogram_dmid | '
                                  'ultrasound | histopathology | mri | auto'),
                    'mode':      'patient (default) | doctor'
                }
            }
        }
    }), 200


@app.route('/models', methods=['GET'])
def model_status():
    """List which model files exist on disk."""
    status = {}
    for name, path in MODEL_PATHS.items():
        status[name] = {
            'path':   path,
            'exists': os.path.exists(path),
            'active': (name == active_model_type)
        }
    return jsonify(status), 200


# ─────────────────────────────────────────────────────────────────────
# --- 13. ERROR HANDLERS ---
# ─────────────────────────────────────────────────────────────────────
@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'error': 'File too large. Maximum upload size is 15 MB.'}), 413


@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': str(e)}), 400


@app.errorhandler(500)
def internal_error(e):
    gc.collect()
    return jsonify({'error': 'Internal server error.'}), 500


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
