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
application = app  # Critical for AWS Elastic Beanstalk

# --- 2. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
MODEL_PATH = 'models/mammo_model.pth'
CONFIDENCE_THRESHOLD = 0.50  # Lowered slightly to allow "Normal" to pass
MIN_VARIANCE = 50
loaded_model = None

# Default to 3 classes (Common for Medical Models: Benign, Malignant, Normal)
detected_classes = ['Benign', 'Malignant', 'Normal']

# --- 3. MODEL LOADER (Robust) ---
def load_pytorch_model():
    global loaded_model, detected_classes
    print(f" * [STARTUP] Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH):
        print(" ! [ERROR] Model file not found.")
        return

    device = torch.device('cpu')

    try:
        # Load the dictionary
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Architecture Search Strategy
        combinations = [
            ("ResNet50 (3 Class)", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50 (2 Class)", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18 (3 Class)", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18 (2 Class)", models.resnet18, 2, ['Benign', 'Malignant']),
        ]

        for name, model_func, num_classes, class_names in combinations:
            print(f"   - Testing Architecture: {name}...", end=" ")
            try:
                temp_model = model_func(weights=None)
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)
                
                # Strict=False allows partial matching (ignores missing keys)
                temp_model.load_state_dict(state_dict, strict=False)
                
                print("SUCCESS ✅")
                loaded_model = temp_model
                detected_classes = class_names
                loaded_model.eval()
                return

            except Exception:
                print("Failed ❌")
                continue

        print(" ! [CRITICAL] Could not match model architecture.")

    except Exception as e:
        print(f" ! [FATAL] Error loading model: {e}")

load_pytorch_model()

# --- 4. IMAGE VALIDATION ( The "Room Photo" Fix ) ---
def is_valid_medical_image(image_pil):
    """
    rejects images that are too dark, too bright, or COLORFUL (Room photos).
    """
    try:
        # 1. Convert to Numpy
        img_np = np.array(image_pil)
        
        # 2. Check for Color (Medical scans are Grayscale)
        # If R, G, and B channels are very different, it's a color photo.
        if len(img_np.shape) == 3: # RGB Image
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            # Calculate variance between channels
            color_variance = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b))
            
            # Threshold: If color variance > 15, it's likely a photo of a room/person
            if color_variance > 15:
                return False, "Non-Medical Image Detected (Color)"

        # 3. Check Brightness/Content
        gray = image_pil.convert('L')
        stat = ImageStat.Stat(gray)
        
        if stat.mean[0] < 10: return False, "Image too dark/blank"
        if stat.var[0] < MIN_VARIANCE: return False, "Image has no content"

        return True, None
    except Exception as e:
        print(f"Validation Error: {e}")
        return True, None # Default to true if check fails

# --- 5. PREPROCESSING ---
# Updated to ensure consistency. 
# We Resize to 224x224 and normalize using standard ImageNet stats.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Standard ImageNet normalization (Matches most ResNet training)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 6. ADVANCED GRAD-CAM ---
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
        self.gradients = None
        self.activations = None
        
        # Forward pass
        out = self.model(x)
        
        # Zero grads
        self.model.zero_grad()
        
        # Target the highest scoring class
        target_index = out.argmax(dim=1).item()
        score = out[:, target_index]
        
        # Backward pass to get gradients
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return None
            
        # Global Average Pooling of Gradients
        grads = self.gradients.data.numpy()[0]
        acts = self.activations.data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        # ReLU (Discard negative activations)
        cam = np.maximum(cam, 0)
        
        # Resize to input size (224x224)
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize 0-1
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
        return cam

def generate_heatmap_overlay(image_path, heatmap):
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Process Heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose (0.4 intensity)
    overlay = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
    
    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', overlay)
    return base64.b64encode(buffer).decode('utf-8')

# --- 7. ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    mode = request.form.get('mode', 'patient') # 'doctor' or 'patient'

    if not loaded_model:
        return jsonify({'error': 'Model not loaded on server'}), 503

    try:
        # Save temp file
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # 1. VALIDATION GATE (Blocks Room/Cat Photos)
        img_pil = Image.open(path).convert('RGB')
        is_valid, error_msg = is_valid_medical_image(img_pil)
        
        if not is_valid:
            os.remove(path)
            # Return specific error so Flutter shows Red Dialog
            return jsonify({
                'result': 'Invalid',
                'confidence': '0.0%',
                'error': error_msg,
                'is_medical': False
            })

        # 2. PREDICTION
        input_tensor = transform(img_pil).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, 1)

        score = confidence.item()
        idx = class_idx.item()
        
        # Safety check for index
        prediction_label = detected_classes[idx] if idx < len(detected_classes) else "Unknown"

        # 3. DOCTOR OUTPUT ENHANCEMENT
        heatmap_base64 = None
        recommendation = "No specific action required."
        
        if prediction_label == 'Malignant':
            recommendation = "High Risk: Immediate biopsy and specialist consultation recommended."
        elif prediction_label == 'Benign':
            recommendation = "Medium Risk: Monitor regularly. Schedule follow-up in 6 months."
        elif prediction_label == 'Normal':
            recommendation = "Low Risk: No anomalies detected. Routine screening suggested."

        # Generate GradCAM only for Doctor Mode OR High Risk
        if mode == 'doctor':
            try:
                # Target the last convolutional layer (Standard for ResNet)
                target_layer = list(loaded_model.modules())[-2][-1].conv2 if 'resnet' in str(type(loaded_model)).lower() else None
                
                # Fallback search for layer
                if target_layer is None:
                    # Generic fallback to last layer with parameters
                    for layer in reversed(list(loaded_model.modules())):
                        if isinstance(layer, torch.nn.Conv2d):
                            target_layer = layer
                            break

                if target_layer:
                    cam_tool = GradCAM(loaded_model, target_layer)
                    heatmap_mask = cam_tool(input_tensor)
                    if heatmap_mask is not None:
                        heatmap_base64 = generate_heatmap_overlay(path, heatmap_mask)
            except Exception as e:
                print(f"GradCAM Failed: {e}")
                heatmap_base64 = None # Fail silently, don't crash app

        # Cleanup
        if os.path.exists(path):
            os.remove(path)

        return jsonify({
            'result': prediction_label,
            'confidence': f"{score:.2f}", # Send as float-string "0.95"
            'heatmap': heatmap_base64,
            'recommendation': recommendation,
            'is_medical': True
        })

    except Exception as e:
        if os.path.exists(path): os.remove(path)
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Breast Cancer AI Backend Online", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
