import os
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- MODEL SETTINGS ---
MODEL_PATH = 'models/mammo_model.pth'
loaded_model = None

# *** IMPORTANT: CLASS MAPPING ***
# PyTorch ImageFolder usually sorts classes Alphabetically.
# 0 = Benign, 1 = Malignant, 2 = Normal
CLASS_NAMES = ['Benign', 'Malignant', 'Normal'] 

def load_pytorch_model():
    global loaded_model
    print(" * Loading PyTorch Model...")

    if not os.path.exists(MODEL_PATH):
        print(f" ! ERROR: Model not found at {MODEL_PATH}")
        return

    device = torch.device('cpu') # Cloud runs on CPU

    # --- STRATEGY 1: Try loading as a Dictionary (Most Common) ---
    try:
        print(" * [TRYING] Strategy 1: Loading as State Dictionary...")
        # Initialize Skeleton (ResNet50)
        loaded_model = models.resnet50(weights=None) 
        
        # --- CRITICAL FIX: Set to 3 Classes (Benign, Malignant, Normal) ---
        num_ftrs = loaded_model.fc.in_features
        loaded_model.fc = nn.Linear(num_ftrs, 3) 

        # Load Weights (With Security Fix)
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        loaded_model.load_state_dict(state_dict)
        print(" * [SUCCESS] Loaded state_dict into ResNet50 (3 Classes).")

    except Exception as e_state:
        print(f" * Strategy 1 failed ({e_state}). Trying Strategy 2...")

        # --- STRATEGY 2: Try loading as a Full Model (Backup) ---
        try:
            print(" * [TRYING] Strategy 2: Loading as Full Model...")
            loaded_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            print(" * [SUCCESS] Loaded full model object.")

        except Exception as e_full:
            print(f" ! CRITICAL ERROR: Could not load model. {e_full}")
            loaded_model = None

    if loaded_model:
        loaded_model.eval() # Set to evaluation mode

# Load on startup
load_pytorch_model()

# --- PREPROCESSING ---
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

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.gradients = None
        self.activations = None
        output = self.model(x)
        self.model.zero_grad()
        target_class = output.argmax(dim=1).item()
        score = output[:, target_class]
        score.backward()
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/', methods=['GET'])
def home():
    return "BreastScan PyTorch Server (3-Class) is Running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_mode = request.form.get('mode', 'patient')

    if loaded_model is None:
        return jsonify({'error': 'Model failed to load.'}), 503

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 1. Preprocess
            image_pil = Image.open(filepath).convert('RGB')
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu')

            # 2. Predict (Handles 3 Classes Now)
            with torch.no_grad():
                output = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

            # Get Class Name (Benign, Malignant, or Normal)
            idx = predicted_idx.item()
            result_text = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
            conf_score = confidence.item()

            response_data = {
                'result': result_text,
                'confidence': f"{conf_score * 100:.1f}%",
                'heatmap': None
            }

            # 3. Doctor Mode Logic (Adds Heatmap)
            if user_mode == 'doctor':
                try:
                    target_layer = None
                    if hasattr(loaded_model, 'layer4'): # ResNet
                        target_layer = loaded_model.layer4[-1].conv2
                    elif hasattr(loaded_model, 'features'): # VGG
                        target_layer = loaded_model.features[-1]
                    
                    if target_layer:
                        grad_cam = GradCAM(loaded_model, target_layer)
                        heatmap = grad_cam(input_tensor)
                        heatmap_b64 = overlay_heatmap(filepath, heatmap)
                        response_data['heatmap'] = heatmap_b64
                except Exception as e:
                    print(f"GradCAM Error: {e}")

            if os.path.exists(filepath): os.remove(filepath)
            return jsonify(response_data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
