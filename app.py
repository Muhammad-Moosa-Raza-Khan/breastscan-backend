import os
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat, ImageOps
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- 1. INITIALIZE APP ---
app = Flask(__name__)
application = app  # Critical for AWS Elastic Beanstalk

# --- 2. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
MODEL_PATH = 'models/mammo_model.pth'
loaded_model = None
detected_classes = ['Benign', 'Malignant', 'Normal']

# --- 3. SMART RESIZING (THE FIX) ---
def smart_resize_pad(image_pil, target_size=(224, 224)):
    """
    Resizes image to target_size WITHOUT squashing.
    Adds black padding to keep the aspect ratio.
    """
    # 1. Calculate aspect ratio
    w, h = image_pil.size
    ratio = min(target_size[0]/w, target_size[1]/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # 2. Resize with high-quality filter
    img_resized = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Create black background
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    
    # 4. Paste resized image in center
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img

# --- 4. MODEL LOADER ---
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
                
                print(f"SUCCESS: Loaded {name} ✅")
                loaded_model = temp_model
                detected_classes = class_names
                loaded_model.eval()
                return
            except:
                continue
    except Exception as e:
        print(f" ! [FATAL] Error loading model: {e}")

load_pytorch_model()

# --- 5. PREPROCESSING ---
# Removed Resize here because we do it manually in smart_resize_pad
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 6. GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)

    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]

    def __call__(self, x):
        self.gradients = None; self.activations = None
        out = self.model(x)
        self.model.zero_grad()
        target_index = out.argmax(dim=1).item()
        out[:, target_index].backward()
        
        if self.gradients is None or self.activations is None: return None
            
        grads = self.gradients.data.numpy()[0]
        acts = self.activations.data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * acts[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
        return cam

def get_target_layer(model):
    try:
        if hasattr(model, 'layer4'):
            last_block = model.layer4[-1]
            if hasattr(last_block, 'conv3'): return last_block.conv3
            if hasattr(last_block, 'conv2'): return last_block.conv2
    except: pass
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d): return module
    return None

def generate_heatmap_overlay(original_pil, heatmap):
    # 1. Convert PIL to OpenCV (BGR format)
    img = np.array(original_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Gaussian Blur (The "Smoother")
    # This blends the pixelated dots into a smooth coherent blob
    # (15, 15) is the kernel size - larger means smoother
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # 3. Thresholding (The "Noise Gate")
    # Increase slightly to 0.45 to hide weak signals
    heatmap[heatmap < 0.45] = 0 
    
    # 4. Normalize
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    # 5. Create Background Mask (The "Edge Fix")
    # Identify black padding pixels (where value < 10)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, tissue_mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    
    # Erosion: Shrink the mask slightly to move away from the sharp edges
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.erode(tissue_mask, kernel, iterations=2)

    # 6. Apply Heatmap Color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 7. Final Combine
    # Only show heat where:
    # A. Heatmap exists AND
    # B. We are on actual tissue (not padding)
    final_heat_mask = cv2.bitwise_and(heatmap_uint8, tissue_mask)
    
    overlay = img.copy()
    valid_indices = final_heat_mask > 0
    
    if np.any(valid_indices):
         overlay[valid_indices] = cv2.addWeighted(
             heatmap_colored[valid_indices], 0.6, 
             img[valid_indices], 0.4, 
             0
         )

    _, buffer = cv2.imencode('.jpg', overlay)
    return base64.b64encode(buffer).decode('utf-8')

# --- 7. ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    mode = request.form.get('mode', 'patient')

    if not loaded_model: return jsonify({'error': 'Model not loaded'}), 503

    try:
        # 1. Load Image
        img_pil = Image.open(file).convert('RGB')
        
        # 2. SMART RESIZE (Fixes Distortion)
        img_processed = smart_resize_pad(img_pil)

        # 3. PREDICT
        input_tensor = transform(img_processed).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probs, 1)

        score = confidence.item()
        prediction_label = detected_classes[class_idx.item()]

        # 4. HEATMAP GENERATION
        heatmap_base64 = None
        recommendation = "No action."
        
        if prediction_label == 'Malignant': recommendation = "High Risk: Biopsy recommended."
        elif prediction_label == 'Benign': recommendation = "Medium Risk: Monitor regularly."
        elif prediction_label == 'Normal': recommendation = "Low Risk: Routine screening."

        if mode == 'doctor':
            try:
                target_layer = get_target_layer(loaded_model)
                if target_layer:
                    cam_tool = GradCAM(loaded_model, target_layer)
                    heatmap_mask = cam_tool(input_tensor)
                    if heatmap_mask is not None:
                        # Pass the PROCESSED (Padded) image to the overlay function
                        heatmap_base64 = generate_heatmap_overlay(img_processed, heatmap_mask)
            except Exception as e:
                print(f"GradCAM Error: {e}")

        return jsonify({
            'result': prediction_label,
            'confidence': f"{score:.2f}",
            'heatmap': heatmap_base64,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Breast Cancer AI Backend Online", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
