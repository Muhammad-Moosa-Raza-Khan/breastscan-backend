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

# --- 4. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. SMART CROP & MASK (THE FIX) ---
def crop_to_breast_tissue(img_cv2):
    """
    Finds the breast tissue and CROPS the image to remove black background.
    This 'Zooms In' on the actual content.
    """
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest object (the breast)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Add a small padding (10px) so we don't cut the skin edge
        pad = 10
        h_img, w_img = img_cv2.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        return img_cv2[y1:y2, x1:x2] # Return cropped image
    
    return img_cv2 # Fallback if no contour found

def get_tissue_mask(img_cv2):
    """
    Creates a strict mask of the breast tissue and ERODES the edge
    to prevent the heatmap from sticking to the skin line.
    """
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erode the mask (Shrink it by ~10 pixels)
    # This forces the heatmap to stay INSIDE the breast, ignoring the skin edge.
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    
    return thresh

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
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
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

def generate_heatmap_overlay(original_cv2, heatmap):
    # original_cv2 is already the CROPPED, RESIZED image (224x224)
    img = original_cv2
    
    # 1. Get STRICT Tissue Mask
    tissue_mask = get_tissue_mask(img)
    tissue_bool = tissue_mask > 0
    
    # 2. Apply Mask (Kill Edge Artifacts)
    heatmap = heatmap * tissue_bool
    
    # 3. Filter Weak Heat
    heatmap[heatmap < 0.35] = 0 
    
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 4. Blend
    overlay = img.copy()
    final_mask = (heatmap > 0) & tissue_bool
    
    if np.any(final_mask):
         overlay[final_mask] = cv2.addWeighted(
             heatmap_colored[final_mask], 0.6, 
             img[final_mask], 0.4, 
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
        img_cv2 = np.array(img_pil) 
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        
        # 2. AUTO-CROP (Zoom to Breast)
        # This removes the black space that was confusing the AI
        img_cropped = crop_to_breast_tissue(img_cv2)
        
        # 3. Resize the CROP to 224x224
        img_final = cv2.resize(img_cropped, (224, 224))
        
        # Convert back to PIL for Tensor
        img_pil_final = Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        
        # 4. Predict
        input_tensor = transform(img_pil_final).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probs, 1)

        score = confidence.item()
        prediction_label = detected_classes[class_idx.item()]

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
                        # Pass the CROPPED/RESIZED image to heatmap gen
                        heatmap_base64 = generate_heatmap_overlay(img_final, heatmap_mask)
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
    # Running on Port 80 (Sudo required)
    app.run(host='0.0.0.0', port=80)
