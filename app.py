import os
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
from flask import Flask, request, jsonify

# --- 1. INITIALIZE APP ---
app = Flask(__name__)
application = app

# --- 2. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
MODEL_PATH = 'models/mammo_model.pth'
loaded_model = None
detected_classes = ['Benign', 'Malignant', 'Normal']

# --- 3. MODEL LOADER ---
def load_pytorch_model():
    global loaded_model, detected_classes
    print(f" * [STARTUP] Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print(" ! [ERROR] Model file not found.")
        return

    device = torch.device('cpu')

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        combinations = [
            ("ResNet50 (3 Class)", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50 (2 Class)", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18 (3 Class)", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18 (2 Class)", models.resnet18, 2, ['Benign', 'Malignant']),
        ]

        for name, model_func, num_classes, class_names in combinations:
            try:
                temp_model = model_func(weights=None)
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)
                temp_model.load_state_dict(state_dict, strict=False)
                print(f"SUCCESS: Loaded {name}")
                loaded_model = temp_model
                detected_classes = class_names
                loaded_model.eval()
                return
            except Exception as e:
                continue
    except Exception as e:
        print(f" ! [FATAL] Error loading model: {e}")

load_pytorch_model()

# --- 4. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. IMAGE PROCESSING ---
def preprocess_mammogram(img_cv2):
    try:
        if len(img_cv2.shape) == 3:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv2.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        if len(img_cv2.shape) == 3:
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        else:
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return processed
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return img_cv2

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
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            return img_cv2[y1:y2, x1:x2]

        return img_cv2
    except Exception as e:
        print(f"Crop error: {e}")
        return img_cv2

def get_tissue_mask_advanced(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        erode_kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.erode(thresh, erode_kernel, iterations=1)
        
        return thresh
    except Exception as e:
        print(f"Mask error: {e}")
        h, w = img_cv2.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

# --- 6. ADVANCED ANOMALY DETECTION (For validation only) ---
def comprehensive_abnormality_analysis(img_cv2):
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect bright regions (masses/calcifications)
        _, bright_mask = cv2.threshold(enhanced, 210, 255, cv2.THRESH_BINARY)
        bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        suspicious_regions = []
        total_suspicious_area = 0
        
        for cnt in bright_contours:
            area = cv2.contourArea(cnt)
            if area > 30:
                x, y, w, h = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(enhanced, mask=mask)[0]
                
                region_score = 0
                if area > 100:
                    region_score += 3
                elif area > 50:
                    region_score += 2
                else:
                    region_score += 1
                
                if mean_intensity > 230:
                    region_score += 2
                elif mean_intensity > 220:
                    region_score += 1
                
                if circularity > 0.6:
                    region_score += 1
                
                if region_score >= 3:
                    suspicious_regions.append({
                        'area': area,
                        'circularity': circularity,
                        'intensity': mean_intensity,
                        'score': region_score,
                        'bbox': (x, y, w, h)
                    })
                    total_suspicious_area += area
        
        num_suspicious = len(suspicious_regions)
        
        if num_suspicious == 0:
            severity = 0.0
            confidence = 0.9
            category = "Normal"
        elif num_suspicious == 1:
            region = suspicious_regions[0]
            if region['area'] > 500 or region['intensity'] > 235:
                severity = 0.8
                category = "Malignant"
            else:
                severity = 0.5
                category = "Benign"
            confidence = 0.7
        else:
            if any(r['area'] > 300 for r in suspicious_regions):
                severity = 0.9
                category = "Malignant"
            else:
                severity = 0.6
                category = "Benign"
            confidence = 0.8
        
        details = {
            'suspicious_regions': num_suspicious,
            'total_area': int(total_suspicious_area),
            'largest_region': max([r['area'] for r in suspicious_regions]) if suspicious_regions else 0,
            'max_intensity': max([r['intensity'] for r in suspicious_regions]) if suspicious_regions else 0,
            'cv_prediction': category
        }
        
        return severity, confidence, details
        
    except Exception as e:
        print(f"Abnormality analysis error: {e}")
        return 0.0, 0.0, {}

# --- 7. IMPROVED GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

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
            
            gradients = self.gradients.cpu().numpy()[0]
            activations = self.activations.cpu().numpy()[0]
            weights = np.mean(gradients, axis=(1, 2))
            
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam
        except Exception as e:
            print(f"GradCAM calculation error: {e}")
            return None

def get_target_layer(model):
    try:
        if hasattr(model, 'layer4'):
            last_block = model.layer4[-1]
            if hasattr(last_block, 'conv3'):
                return last_block.conv3
            if hasattr(last_block, 'conv2'):
                return last_block.conv2
    except Exception as e:
        print(f"Error getting layer4: {e}")
    
    try:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    except Exception as e:
        print(f"Error finding conv layer: {e}")
    
    return None

def refine_gradcam_with_intensity(gradcam_mask, original_img):
    """
    Refines Grad-CAM by combining it with actual tissue intensity
    This makes it focus more precisely on bright masses
    """
    try:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
        
        # Get intensity map (normalized)
        intensity_map = gray.astype(np.float32) / 255.0
        
        # Enhance bright regions
        intensity_map = np.power(intensity_map, 0.7)  # Gamma correction
        
        # Resize to match Grad-CAM
        intensity_map = cv2.resize(intensity_map, (224, 224))
        
        # Combine: 70% Grad-CAM + 30% Intensity
        # This guides Grad-CAM to focus more on bright areas
        refined = (gradcam_mask * 0.7) + (intensity_map * 0.3)
        
        # Normalize
        if refined.max() > 0:
            refined = refined / refined.max()
        
        return refined
    except Exception as e:
        print(f"Refinement error: {e}")
        return gradcam_mask

def generate_heatmap_overlay(original_cv2, heatmap, prediction_label):
    try:
        img = original_cv2.copy()
        
        # Refine heatmap with intensity information
        heatmap = refine_gradcam_with_intensity(heatmap, img)
        
        tissue_mask = get_tissue_mask_advanced(img)
        tissue_bool = tissue_mask > 0
        masked_heatmap = heatmap * tissue_bool
        
        # Adaptive threshold based on prediction
        if prediction_label == 'Malignant':
            threshold = 0.25
        elif prediction_label == 'Benign':
            threshold = 0.30
        else:
            threshold = 0.35
        
        masked_heatmap[masked_heatmap < threshold] = 0
        
        if np.max(masked_heatmap) > 0:
            masked_heatmap = masked_heatmap / np.max(masked_heatmap)
        
        # Smoother blur
        masked_heatmap = cv2.GaussianBlur(masked_heatmap, (15, 15), 0)
        
        heatmap_uint8 = np.uint8(255 * masked_heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        overlay = img.copy()
        heat_mask = (masked_heatmap > 0.01) & tissue_bool
        
        if np.any(heat_mask):
            alpha = 0.65
            beta = 0.35
            overlay[heat_mask] = cv2.addWeighted(
                heatmap_colored[heat_mask], alpha,
                img[heat_mask], beta,
                0
            )
        
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Heatmap overlay error: {e}")
        return None

# --- 8. ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'patient')

    if not loaded_model:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        img_pil = Image.open(file).convert('RGB')
        img_cv2 = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        img_cropped = crop_to_breast_tissue(img_cv2)
        img_processed = preprocess_mammogram(img_cropped)
        img_final = cv2.resize(img_processed, (224, 224))

        # AI Model Prediction
        img_pil_final = Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil_final).unsqueeze(0).to('cpu')

        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get all class probabilities
            all_probs = probs[0].cpu().numpy()
            confidence, class_idx = torch.max(probs, 1)

        ai_score = confidence.item()
        ai_prediction = detected_classes[class_idx.item()]
        
        # Computer Vision Analysis (for validation)
        cv_severity, cv_confidence, cv_details = comprehensive_abnormality_analysis(img_cropped)
        cv_prediction = cv_details.get('cv_prediction', 'Normal')
        
        # SMART OVERRIDE: Only if there's strong evidence
        override_active = False
        final_prediction = ai_prediction
        final_confidence = ai_score
        
        # Calculate certainty margin
        sorted_probs = np.sort(all_probs)[::-1]
        certainty_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        
        # Override only if:
        # 1. Model is uncertain (low margin) AND CV strongly disagrees
        # 2. CV detects very severe abnormality that model missed
        
        if ai_prediction == 'Normal' and cv_severity > 0.6 and certainty_margin < 0.3:
            print(f"⚠️ OVERRIDE: AI uncertain ({certainty_margin:.2f}), CV severity {cv_severity:.2f}")
            override_active = True
            final_prediction = cv_prediction
            final_confidence = cv_confidence
        
        elif ai_prediction == 'Benign' and cv_severity > 0.8:
            print(f"⚠️ OVERRIDE: CV detected very high severity {cv_severity:.2f}")
            override_active = True
            final_prediction = 'Malignant'
            final_confidence = cv_confidence
        
        # Generate recommendation
        if final_prediction == 'Malignant':
            recommendation = "⚠️ HIGH RISK: Immediate biopsy and specialist consultation required."
        elif final_prediction == 'Benign':
            recommendation = "⚡ MEDIUM RISK: Close monitoring with follow-up imaging in 3-6 months."
        else:
            if cv_severity > 0.3:
                recommendation = "✓ LOW-MEDIUM RISK: Routine screening recommended with follow-up."
            else:
                recommendation = "✓ LOW RISK: Continue routine annual screening."

        heatmap_base64 = None

        # Generate Grad-CAM
        if mode == 'doctor':
            try:
                target_layer = get_target_layer(loaded_model)
                if target_layer:
                    grad_cam = GradCAM(loaded_model, target_layer)
                    cam_mask = grad_cam(input_tensor, class_idx=class_idx.item())
                    
                    if cam_mask is not None:
                        heatmap_base64 = generate_heatmap_overlay(
                            img_final, 
                            cam_mask, 
                            final_prediction
                        )
                    
                    grad_cam.remove_hooks()
                    
            except Exception as e:
                print(f"Heatmap generation error: {e}")

        return jsonify({
            'result': final_prediction,
            'confidence': f"{final_confidence:.2f}",
            'heatmap': heatmap_base64,
            'recommendation': recommendation,
            'analysis_details': {
                'ai_prediction': ai_prediction,
                'ai_confidence': f"{ai_score:.2f}",
                'cv_prediction': cv_prediction,
                'cv_severity': f"{cv_severity:.2f}",
                'suspicious_regions': cv_details.get('suspicious_regions', 0),
                'override_active': override_active,
                'certainty_margin': f"{certainty_margin:.2f}",
                'total_suspicious_area': cv_details.get('total_area', 0),
                'max_intensity': cv_details.get('max_intensity', 0)
            }
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    model_status = "loaded" if loaded_model else "not loaded"
    return jsonify({
        'status': 'online',
        'model': model_status,
        'classes': detected_classes
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
