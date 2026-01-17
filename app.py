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
from werkzeug.utils import secure_filename

# ==========================================
#        BREASTSCAN AI SERVER - CORE
# ==========================================

# --- 1. INITIALIZE APP ---
app = Flask(__name__)

# --- 2. CRASH-PROOF FIX (CRITICAL) ---
# AWS Gunicorn looks for 'application' by default.
# This line ensures it works whether it asks for 'app' OR 'application'.
application = app 

# --- 3. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/mammo_model.pth'

# --- 4. SECURITY & QUALITY THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.60  # Reject if confidence < 60%
MIN_VARIANCE = 50            # Reject blank/solid images

# --- 5. GLOBAL VARIABLES ---
loaded_model = None
detected_classes = ['Benign', 'Malignant'] 

# ==========================================
#      SECTION A: IMAGE VALIDATION
# ==========================================

def is_valid_medical_image(image_pil):
    """
    Checks if image looks valid (not blank, not too dark/bright, has detail).
    Returns (True, None) or (False, "Error Reason").
    """
    try:
        gray = image_pil.convert('L')
        stat = ImageStat.Stat(gray)
        
        # Check Brightness
        mean_brightness = stat.mean[0]
        if mean_brightness < 5: return False, "Image is too dark (black screen)."
        if mean_brightness > 250: return False, "Image is too bright (white screen)."

        # Check Variance (Detail)
        variance = stat.var[0]
        if variance < MIN_VARIANCE: return False, "Image is blank or has no detail."

        return True, None
    except Exception as e:
        print(f" [WARNING] Validation check failed: {e}")
        return True, None 

# ==========================================
#      SECTION B: UNIVERSAL MODEL LOADER
# ==========================================

def load_pytorch_model():
    global loaded_model, detected_classes
    print("\n" + "="*40)
    print(" * [STARTUP] Initializing BreastScan AI Engine...")
    print("="*40)

    if not os.path.exists(MODEL_PATH):
        print(f" ! [ERROR] Model not found at {MODEL_PATH}")
        return

    device = torch.device('cpu') 

    try:
        # 1. Load the File
        print(f" * [INFO] Reading model file: {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # 2. Extract State Dict
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            loaded_model = checkpoint
            loaded_model.eval()
            print(" * [SUCCESS] Model loaded directly!")
            return
        
        if state_dict is None:
            print(" ! [ERROR] Could not extract weights.")
            return

        # 3. Try Architectures
        combinations = [
            ("ResNet50", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18", models.resnet18, 2, ['Benign', 'Malignant']),
            ("ResNet34", models.resnet34, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet34", models.resnet34, 2, ['Benign', 'Malignant']),
        ]

        for name, model_func, num_classes, class_names in combinations:
            try:
                print(f"   - Testing: {name} ({num_classes} classes)...", end=" ")
                temp_model = model_func(weights=None)
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)
                
                missing, unexpected = temp_model.load_state_dict(state_dict, strict=False)
                if len(unexpected) > 20: 
                    print("❌")
                    continue
                
                print("✅ MATCHED!")
                loaded_model = temp_model
                loaded_model.eval()
                detected_classes = class_names
                print(f" * [SUCCESS] Loaded as {name} ({num_classes} classes).")
                return
            except Exception: continue

        print(" ! [CRITICAL] No matching architecture found.")
    except Exception as e:
        print(f" ! [FATAL] {e}")

load_pytorch_model()

# ==========================================
#      SECTION C: PREPROCESSING & AI
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GRAD-CAM ENGINE ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __call__(self, x):
        self.gradients, self.activations = None, None
        output = self.model(x)
        self.model.zero_grad()
        target_class = output.argmax(dim=1).item()
        output[:, target_class].backward()
        if self.gradients is None: return None
        grads = self.gradients.data.numpy()[0]
        acts = self.activations.data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * acts[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
        return cam

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    return base64.b64encode(cv2.imencode('.jpg', superimposed)[1]).decode('utf-8')

# ==========================================
#      SECTION D: SERVER ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def home():
    status = "Active" if loaded_model else "Error"
    return f"BreastScan AI Server is {status}. Detected {len(detected_classes)} classes.", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_mode = request.form.get('mode', 'patient')

    if loaded_model is None: return jsonify({'error': 'Model failed to load'}), 503

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 1. Load & Validate
            image_pil = Image.open(filepath).convert('RGB')
            is_valid, error_msg = is_valid_medical_image(image_pil)
            if not is_valid:
                if os.path.exists(filepath): os.remove(filepath)
                return jsonify({'result': 'Error', 'confidence': '0%', 'error': error_msg})

            # 2. Predict
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu')
            with torch.no_grad():
                output = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

            idx = predicted_idx.item()
            result_text = detected_classes[idx] if idx < len(detected_classes) else "Unknown"
            conf_score = confidence.item()

            # 3. Confidence Check
            if conf_score < CONFIDENCE_THRESHOLD:
                if os.path.exists(filepath): os.remove(filepath)
                return jsonify({'result': 'Uncertain', 'confidence': f"{conf_score*100:.1f}%", 'error': 'Image unclear'})

            response = {'result': result_text, 'confidence': f"{conf_score * 100:.1f}%", 'heatmap': None}

            # 4. Doctor Mode
            if user_mode == 'doctor':
                try:
                    target_layer = None
                    if hasattr(loaded_model, 'layer4'): target_layer = loaded_model.layer4[-1].conv2
                    elif hasattr(loaded_model, 'features'): target_layer = loaded_model.features[-1]
                    if target_layer:
                        cam = GradCAM(loaded_model, target_layer)(input_tensor)
                        if cam is not None: response['heatmap'] = overlay_heatmap(filepath, cam)
                except Exception: pass

            if os.path.exists(filepath): os.remove(filepath)
            return jsonify(response)
        except Exception as e: return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
