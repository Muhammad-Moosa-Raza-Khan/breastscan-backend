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

# --- MODEL ROUTER ---
# Make sure your .pth file is named 'mammo_model.pth' inside the models folder
MODEL_PATH = 'models/mammo_model.pth'

loaded_model = None

# =========================================================================
# SECTION A: PASTE YOUR MODEL CLASS HERE (If you used a custom model)
# =========================================================================
# If you trained a custom model (not standard ResNet/VGG),
# paste the "class YourModelName(nn.Module): ..." code right here.
# If you used standard ResNet, you can ignore this.
# =========================================================================


def load_pytorch_model():
    global loaded_model
    print(" * Loading PyTorch Model...")

    if not os.path.exists(MODEL_PATH):
        print(f" ! ERROR: Model not found at {MODEL_PATH}")
        return

    device = torch.device('cpu') # Cloud runs on CPU

    try:
        # STRATEGY 1: Try loading as a full model (Simplest)
        loaded_model = torch.load(MODEL_PATH, map_location=device)
        print(" * [SUCCESS] Loaded full model object.")

    except Exception as e_full:
        print(f" * Full load failed ({e_full}), trying state_dict...")

        try:
            # STRATEGY 2: Load weights into a Skeleton (Standard way)
            # CHANGE THIS if you used a different model (e.g., models.densenet121())
            loaded_model = models.resnet50(pretrained=False)

            # If your output was 2 classes (Benign/Malignant), reset the final layer
            num_ftrs = loaded_model.fc.in_features
            loaded_model.fc = nn.Linear(num_ftrs, 2)

            # Load weights
            loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(" * [SUCCESS] Loaded state_dict into ResNet18.")

        except Exception as e_state:
            print(f" ! CRITICAL ERROR: Could not load model. {e_state}")
            print(" ! You might need to paste your class definition in app.py")
            loaded_model = None

    if loaded_model:
        loaded_model.eval() # Set to evaluation mode

# Load on startup
load_pytorch_model()

# --- PREPROCESSING (Standard for PyTorch) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GRAD-CAM ENGINE (PyTorch Version) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.gradients = None
        self.activations = None

        # Forward pass
        output = self.model(x)

        # Zero grads
        self.model.zero_grad()

        # Target the highest score class
        target_class = output.argmax(dim=1).item()
        score = output[:, target_class]

        # Backward pass
        score.backward()

        # Generate heatmap
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
    return "BreastScan PyTorch Server is Running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    user_mode = request.form.get('mode', 'patient')

    if loaded_model is None:
        return jsonify({'error': 'Model failed to load. Check server logs.'}), 503

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 1. Preprocess
            image_pil = Image.open(filepath).convert('RGB')
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu')

            # 2. Predict
            with torch.no_grad():
                output = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

            conf_score = confidence.item()
            is_malignant = predicted_class.item() == 1 # Assuming 0=Benign, 1=Malignant

            result_text = "Malignant" if is_malignant else "Benign"

            response_data = {
                'result': result_text,
                'confidence': f"{conf_score * 100:.1f}%",
                'heatmap': None
            }

            # 3. Grad-CAM (Only for Doctors)
            if user_mode == 'doctor':
                try:
                    # Try to find the last convolutional layer automatically
                    target_layer = None

                    # Common layers for ResNet/VGG
                    if hasattr(loaded_model, 'layer4'): # ResNet
                        target_layer = loaded_model.layer4[-1].conv2
                    elif hasattr(loaded_model, 'features'): # VGG/DenseNet
                        target_layer = loaded_model.features[-1]

                    if target_layer:
                        grad_cam = GradCAM(loaded_model, target_layer)
                        # We need gradients, so we run a forward/backward pass with grad enabled
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
