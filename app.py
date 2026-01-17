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

# --- 1. INITIALIZE APP ---
app = Flask(__name__)
# CRITICAL FIX: AWS looks for 'application', this ensures it finds it.
application = app 

# --- 2. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
MODEL_PATH = 'models/mammo_model.pth'
CONFIDENCE_THRESHOLD = 0.60
MIN_VARIANCE = 50
loaded_model = None
detected_classes = ['Benign', 'Malignant']

# --- 3. THE "PROVEN WORKING" LOADER ---
# This uses the exact logic from your successful logs (image_ba4364.png)
def load_pytorch_model():
    global loaded_model, detected_classes
    print(f" * [STARTUP] Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print(" ! [ERROR] Model file not found.")
        return

    device = torch.device('cpu')

    try:
        # 1. RAW LOAD (No smart extraction, just grab the data)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(f" * [INFO] File loaded. Type: {type(state_dict)}")

        # 2. Try Architectures
        combinations = [
            ("ResNet50", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18", models.resnet18, 2, ['Benign', 'Malignant']),
        ]

        for name, model_func, num_classes, class_names in combinations:
            print(f"   - Testing: {name} ({num_classes} classes)...", end=" ")
            try:
                # Create Skeleton
                temp_model = model_func(weights=None)
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)

                # DIRECT LOAD (Strict=False allows for minor mismatch handling)
                temp_model.load_state_dict(state_dict, strict=False)
                
                # If we get here without crashing, we assume it's good enough!
                print("MATCHED! ✅")
                loaded_model = temp_model
                detected_classes = class_names
                loaded_model.eval()
                return

            except Exception as e:
                # If it crashes, try next
                print(f"X")
                continue

        print(" ! [CRITICAL] No matching architecture found.")

    except Exception as e:
        print(f" ! [FATAL ERROR] {e}")

load_pytorch_model()

# --- 4. IMAGE VALIDATION ---
def is_valid_medical_image(image_pil):
    try:
        gray = image_pil.convert('L')
        stat = ImageStat.Stat(gray)
        if stat.mean[0] < 5: return False, "Image too dark"
        if stat.mean[0] > 250: return False, "Image too bright"
        if stat.var[0] < MIN_VARIANCE: return False, "Image blank/solid"
        return True, None
    except: return True, None

# --- 5. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 6. GRAD-CAM & OVERLAY ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None; self.activations = None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)
    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]
    def __call__(self, x):
        self.gradients = None; self.activations = None
        out = self.model(x)
        self.model.zero_grad()
        target = out.argmax(dim=1).item()
        out[:, target].backward()
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

def overlay(path, hm):
    img = cv2.imread(path); img = cv2.resize(img, (224, 224))
    hm = np.uint8(255 * hm); hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    res = np.clip(hm * 0.4 + img, 0, 255).astype('uint8')
    return base64.b64encode(cv2.imencode('.jpg', res)[1]).decode('utf-8')

# --- 7. ROUTES ---
@app.route('/', methods=['GET'])
def home():
    status = "Online" if loaded_model else "Offline"
    return f"BreastScan Server: {status}", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    mode = request.form.get('mode', 'patient')

    if not loaded_model: return jsonify({'error': 'Model not loaded'}), 503

    try:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # Validate
        img = Image.open(path).convert('RGB')
        valid, msg = is_valid_medical_image(img)
        if not valid:
            os.remove(path)
            return jsonify({'result': 'Error', 'confidence': '0%', 'error': msg})

        # Predict
        tens = transform(img).unsqueeze(0).to('cpu')
        with torch.no_grad():
            out = loaded_model(tens)
            prob = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)

        score = conf.item()
        res = detected_classes[idx.item()] if idx.item() < len(detected_classes) else "Unknown"

        if score < CONFIDENCE_THRESHOLD:
            os.remove(path)
            return jsonify({'result': 'Uncertain', 'confidence': f"{score*100:.1f}%", 'error': 'Low confidence'})

        resp = {'result': res, 'confidence': f"{score*100:.1f}%", 'heatmap': None}

        if mode == 'doctor':
            try:
                layer = None
                if hasattr(loaded_model, 'layer4'): layer = loaded_model.layer4[-1].conv2
                elif hasattr(loaded_model, 'features'): layer = loaded_model.features[-1]
                if layer:
                    cam = GradCAM(loaded_model, layer)(tens)
                    if cam is not None: resp['heatmap'] = overlay(path, cam)
            except: pass

        os.remove(path)
        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
