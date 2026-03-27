import os
import io
import gc
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# --- 1. INITIALIZE APP ---
app = Flask(__name__)
application = app

# --- 2. CONFIGURATION & DATASETS ---
UPLOAD_FOLDER = '/tmp'

MODEL_PATHS = {
    'mammogram_mias': 'models/mammo_mias.pth',
    'mammogram_dmid': 'models/mammo_dmid.pth',
    'ultrasound': 'models/ultrasound_model.pth',
    'histopathology': 'models/hist_model.pth',
    'mri': 'models/mri_model.pth'
}

# Dynamic mapping based on your 5 separate training configurations
DATASET_CONFIGS = {
    'histopathology':   {'num_classes': 2, 'labels': ['Benign', 'Malignant']},
    'mri':              {'num_classes': 2, 'labels': ['Healthy', 'Sick']},
    'mammogram_dmid':   {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']},
    'mammogram_mias':   {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']},
    'ultrasound':       {'num_classes': 3, 'labels': ['Benign', 'Malignant', 'Normal']}
}

active_model_type = None
loaded_model = None
detected_classes = []

# =====================================================================
# --- 3. YOUR CUSTOM ARCHITECTURE (RAM-NET) ---
# =====================================================================
class RAMNet(nn.Module):
    def __init__(self, num_classes):
        super(RAMNet, self).__init__()
        
        # ⚠️ IMPORTANT: Paste your EXACT training __init__ logic here!
        # This is a skeleton based on your description. The layer names 
        # must match your saved .pth files perfectly.
        
        # 1. Spatial Branch
        effnet = models.efficientnet_v2_s(weights=None)
        self.spatial_encoder = effnet.features
        
        # 2. Frequency Branch (DCT-based)
        # self.freq_encoder = ... 
        
        # 3. Attention & Downsampling
        # self.downsample = ...
        # self.attention = nn.MultiheadAttention(...)
        
        # 4. Mixture of Experts (MoE)
        # self.moe_gate = ...
        # self.experts = ...
        
        # Final Classifier
        # self.classifier = nn.Linear(..., num_classes)
        
        # --- TEMP FALLBACK FOR TESTING ---
        # If you want to test the server before pasting your full code, 
        # this will just run standard EfficientNetV2-S so it doesn't crash.
        self.temp_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # ⚠️ IMPORTANT: Paste your EXACT training forward() logic here!
        
        # --- TEMP FALLBACK FOR TESTING ---
        s_feat = self.spatial_encoder(x)
        return self.temp_classifier(s_feat)

# =====================================================================
# --- 4. DYNAMIC MODEL LOADER ---
# =====================================================================
def load_pytorch_model(scan_type):
    global loaded_model, active_model_type, detected_classes
    
    if active_model_type == scan_type and loaded_model is not None:
        return True

    if loaded_model is not None:
        print(f" * [GATEKEEPER] Unloading {active_model_type} to free RAM...")
        del loaded_model
        gc.collect()

    path = MODEL_PATHS.get(scan_type)
    if not path or not os.path.exists(path):
        print(f" ! [ERROR] Model not found: {path}")
        return False

    config = DATASET_CONFIGS.get(scan_type)
    detected_classes = config['labels']
    
    print(f" * [GATEKEEPER] Loading {scan_type} ({config['num_classes']} classes) from {path}...")
    device = torch.device('cpu')

    try:
        # Instantiate your custom architecture dynamically
        model = RAMNet(num_classes=config['num_classes'])
        
        # Load weights
        state_dict = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False) 
        
        loaded_model = model
        active_model_type = scan_type
        loaded_model.eval()
        return True
    except Exception as e:
        print(f" ! [FATAL] Error loading RAM-Net model: {e}")
        return False

# Pre-load DMID
load_pytorch_model('mammogram_dmid')


# --- 5. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================================
# --- 6. IMAGE PROCESSING & CV ANOMALY DETECTION (Kept Exact) ---
# =====================================================================
def preprocess_mammogram(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    except Exception: return img_cv2

def crop_to_breast_tissue(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = max(5, min(w, h) // 20)
            h_img, w_img = img_cv2.shape[:2]
            return img_cv2[max(0, y-pad):min(h_img, y+h+pad), max(0, x-pad):min(w_img, x+w+pad)]
        return img_cv2
    except Exception: return img_cv2

def get_tissue_mask_advanced(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.erode(thresh, np.ones((7, 7), np.uint8), iterations=1)
    except Exception: return np.ones(img_cv2.shape[:2], dtype=np.uint8) * 255

def comprehensive_abnormality_analysis(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, bright_mask = cv2.threshold(enhanced, 210, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspicious_regions, total_suspicious_area = [], 0
        for cnt in bright_contours:
            area = cv2.contourArea(cnt)
            if area > 30:
                x, y, w, h = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(enhanced, mask=mask)[0]

                region_score = (3 if area > 100 else 2 if area > 50 else 1)
                region_score += (2 if mean_intensity > 230 else 1 if mean_intensity > 220 else 0)
                region_score += (1 if circularity > 0.6 else 0)

                if region_score >= 3:
                    suspicious_regions.append({'area': area, 'circularity': circularity, 'intensity': mean_intensity, 'score': region_score})
                    total_suspicious_area += area

        num_suspicious = len(suspicious_regions)
        if num_suspicious == 0:
            severity, confidence, category = 0.0, 0.9, "Normal"
        elif num_suspicious == 1:
            region = suspicious_regions[0]
            if region['area'] > 500 or region['intensity'] > 235:
                severity, category, confidence = 0.8, "Malignant", 0.7
            else:
                severity, category, confidence = 0.5, "Benign", 0.7
        else:
            if any(r['area'] > 300 for r in suspicious_regions):
                severity, category, confidence = 0.9, "Malignant", 0.8
            else:
                severity, category, confidence = 0.6, "Benign", 0.8

        return severity, confidence, {
            'suspicious_regions': num_suspicious,
            'total_area': int(total_suspicious_area),
            'max_intensity': max([r['intensity'] for r in suspicious_regions]) if suspicious_regions else 0,
            'cv_prediction': category
        }
    except Exception: return 0.0, 0.0, {}

# =====================================================================
# --- 7. IMPROVED GRAD-CAM (Updated for Custom Architecture) ---
# =====================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
        ]

    def save_activation(self, module, input, output): self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0].detach()
    def remove_hooks(self): [h.remove() for h in self.hooks]

    def __call__(self, x, class_idx=None):
        try:
            self.model.eval()
            output = self.model(x)
            if class_idx is None: class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[:, class_idx].backward(retain_graph=True)
            if self.gradients is None or self.activations is None: return None
            
            gradients = self.gradients.cpu().numpy()[0]
            activations = self.activations.cpu().numpy()[0]
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights): cam += w * activations[i]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            if cam.max() > 0: cam = cam / cam.max()
            return cam
        except Exception: return None

def get_target_layer(model):
    # Dynamic hook finder for custom models (RAM-Net)
    # This automatically finds the last convolutional layer regardless of what you name it
    try:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d): 
                return module
    except Exception: return None

def refine_gradcam_with_intensity(gradcam_mask, original_img):
    try:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
        intensity_map = cv2.resize(np.power(gray.astype(np.float32) / 255.0, 0.7), (224, 224))
        refined = (gradcam_mask * 0.7) + (intensity_map * 0.3)
        return refined / refined.max() if refined.max() > 0 else refined
    except Exception: return gradcam_mask

def generate_heatmap_overlay(original_cv2, heatmap, prediction_label):
    try:
        img = original_cv2.copy()
        heatmap = refine_gradcam_with_intensity(heatmap, img)
        tissue_mask = get_tissue_mask_advanced(img)
        tissue_bool = tissue_mask > 0
        masked_heatmap = heatmap * tissue_bool
        
        threshold = 0.25 if prediction_label == 'Malignant' else 0.30 if prediction_label == 'Benign' else 0.35
        masked_heatmap[masked_heatmap < threshold] = 0
        if np.max(masked_heatmap) > 0: masked_heatmap = masked_heatmap / np.max(masked_heatmap)
        masked_heatmap = cv2.GaussianBlur(masked_heatmap, (15, 15), 0)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * masked_heatmap), cv2.COLORMAP_JET)
        overlay = img.copy()
        heat_mask = (masked_heatmap > 0.01) & tissue_bool

        if np.any(heat_mask):
            overlay[heat_mask] = cv2.addWeighted(heatmap_colored[heat_mask], 0.65, img[heat_mask], 0.35, 0)

        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception: return None

# =====================================================================
# --- 8. ROUTES (With Auto-Compare Engine) ---
# =====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'patient')
    scan_type = request.form.get('scan_type', 'mammogram_dmid').lower() 

    # For general "mammogram" requests, default to the Auto-Compare logic
    is_auto_compare = False
    if scan_type == 'mammogram':
        is_auto_compare = True
        
    try:
        # 1. Process Image
        img_pil = Image.open(file).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if 'mammogram' in scan_type:
            img_cropped = crop_to_breast_tissue(img_cv2)
            img_processed = preprocess_mammogram(img_cropped)
        else:
            img_cropped = img_cv2
            img_processed = img_cv2
            
        img_final = cv2.resize(img_processed, (224, 224))
        img_pil_final = Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil_final).unsqueeze(0).to('cpu')

        # ---------------------------------------------------------
        # THE AUTO-COMPARE ENGINE (MAMMOGRAMS ONLY)
        # ---------------------------------------------------------
        if is_auto_compare:
            print(" * [AUTO-COMPARE] Running MIAS Model...")
            if not load_pytorch_model('mammogram_mias'):
                return jsonify({'error': 'MIAS model missing.'}), 503
            with torch.no_grad():
                out_mias = loaded_model(input_tensor)
                prob_mias = torch.nn.functional.softmax(out_mias, dim=1)
                conf_mias, idx_mias = torch.max(prob_mias, 1)
                pred_mias = detected_classes[idx_mias.item()]

            print(" * [AUTO-COMPARE] Running DMID Model...")
            if not load_pytorch_model('mammogram_dmid'):
                return jsonify({'error': 'DMID model missing.'}), 503
            with torch.no_grad():
                out_dmid = loaded_model(input_tensor)
                prob_dmid = torch.nn.functional.softmax(out_dmid, dim=1)
                conf_dmid, idx_dmid = torch.max(prob_dmid, 1)
                pred_dmid = detected_classes[idx_dmid.item()]

            if conf_dmid.item() >= conf_mias.item():
                ai_prediction = pred_dmid
                ai_score = conf_dmid.item()
                winning_model = 'DMID'
                all_probs = prob_dmid[0].cpu().numpy()
            else:
                ai_prediction = pred_mias
                ai_score = conf_mias.item()
                winning_model = 'MIAS'
                all_probs = prob_mias[0].cpu().numpy()
                load_pytorch_model('mammogram_mias')

        # ---------------------------------------------------------
        # STANDARD ROUTING (ULTRASOUND / HISTO / SPECIFIC MAMMO)
        # ---------------------------------------------------------
        else:
            if not load_pytorch_model(scan_type):
                return jsonify({'error': f'Model {scan_type} missing.'}), 503
            
            with torch.no_grad():
                outputs = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs = probs[0].cpu().numpy()
                confidence, class_idx = torch.max(probs, 1)

            ai_score = confidence.item()
            ai_prediction = detected_classes[class_idx.item()]
            winning_model = scan_type

        # ---------------------------------------------------------
        # CV VALIDATION & GRAD-CAM
        # ---------------------------------------------------------
        override_active = False
        final_prediction = ai_prediction
        final_confidence = ai_score
        certainty_margin = np.sort(all_probs)[::-1][0] - np.sort(all_probs)[::-1][1] if len(all_probs) > 1 else 1.0
        
        cv_details = {}
        cv_severity = 0.0

        if 'mammogram' in scan_type:
            cv_severity, cv_confidence, cv_details = comprehensive_abnormality_analysis(img_cropped)
            cv_prediction = cv_details.get('cv_prediction', 'Normal')

            if ai_prediction == 'Normal' and cv_severity > 0.6 and certainty_margin < 0.3:
                override_active = True
                final_prediction, final_confidence = cv_prediction, cv_confidence
            elif ai_prediction == 'Benign' and cv_severity > 0.8:
                override_active = True
                final_prediction, final_confidence = 'Malignant', cv_confidence

        if final_prediction == 'Malignant' or final_prediction == 'Sick':
            recommendation = "⚠️ HIGH RISK: Immediate biopsy and specialist consultation required."
        elif final_prediction == 'Benign':
            recommendation = "⚡ MEDIUM RISK: Close monitoring with follow-up imaging in 3-6 months."
        else:
            recommendation = "✓ LOW RISK: Continue routine annual screening."

        heatmap_base64 = None
        if mode == 'doctor':
            target_layer = get_target_layer(loaded_model)
            if target_layer:
                final_idx = detected_classes.index(final_prediction) if final_prediction in detected_classes else 0
                grad_cam = GradCAM(loaded_model, target_layer)
                cam_mask = grad_cam(input_tensor, class_idx=final_idx)
                if cam_mask is not None:
                    heatmap_base64 = generate_heatmap_overlay(img_final, cam_mask, final_prediction)
                grad_cam.remove_hooks()

        return jsonify({
            'result': final_prediction,
            'confidence': f"{final_confidence:.2f}",
            'heatmap': heatmap_base64,
            'recommendation': recommendation,
            'analysis_details': {
                'scan_type_used': scan_type,
                'winning_model': winning_model,
                'ai_prediction': ai_prediction,
                'ai_confidence': f"{ai_score:.2f}",
                'cv_severity': f"{cv_severity:.2f}",
                'override_active': override_active,
                'certainty_margin': f"{certainty_margin:.2f}"
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'active_model': active_model_type,
        'classes': detected_classes
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
