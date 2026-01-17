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
MODEL_PATH = 'models/mammo_model.pth'

# Global Model Variable
loaded_model = None
# We will auto-detect these
detected_classes = ['Benign', 'Malignant'] 

def load_pytorch_model():
    global loaded_model, detected_classes
    print(" * [STARTUP] Loading PyTorch Model...")

    if not os.path.exists(MODEL_PATH):
        print(f" ! ERROR: Model not found at {MODEL_PATH}")
        return

    device = torch.device('cpu') 

    try:
        # 1. Load the File (The "Key")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(" * [INFO] File loaded successfully. Now finding the right lock...")

        # Check if it's a state_dict or a full checkpoint
        if isinstance(checkpoint, dict):
            # It might have keys like 'model_state_dict', 'state_dict', or be the raw state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume it's the raw state_dict
                state_dict = checkpoint
        else:
            # If it's a model object directly (unlikely but possible)
            loaded_model = checkpoint
            loaded_model.eval()
            print(" * [SUCCESS] Model loaded directly as saved model object.")
            return

        # 2. Define the Combinations to Try
        # (Architecture Name, Function, Number of Classes, Class List)
        combinations = [
            ("ResNet50", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18", models.resnet18, 2, ['Benign', 'Malignant'])
        ]

        # 3. Try Each Combination
        for name, model_func, num_classes, class_names in combinations:
            try:
                print(f"   - Trying {name} with {num_classes} classes...", end="")
                temp_model = model_func(weights=None)
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)
                
                # Try to insert the key
                temp_model.load_state_dict(state_dict, strict=False)
                
                # IF WE REACH HERE, IT WORKED!
                print(" MATCHED! ✅")
                loaded_model = temp_model
                loaded_model.eval()  # NOW it's safe to call .eval()
                detected_classes = class_names
                print(f" * [SUCCESS] Model loaded as {name} ({num_classes} classes).")
                return # Exit function, we are done
                
            except Exception as e:
                print(f" Failed. ❌ ({str(e)[:50]})")
                # Continue to next loop

        print(" ! CRITICAL ERROR: None of the standard architectures matched.")
        print(" ! Please check if you used VGG, DenseNet, or a custom model.")
        
    except Exception as e:
        print(f" ! FATAL ERROR loading file: {e}")
        import traceback
        traceback.print_exc()

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
    if loaded_model:
        return f"BreastScan Server Running! Detected: {len(detected_classes)} Classes", 200
    else:
        return "Server Running (NO MODEL LOADED - Check Logs)", 500

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

            image_pil = Image.open(filepath).convert('RGB')
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu')

            with torch.no_grad():
                output = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

            # Dynamic Class Mapping
            idx = predicted_idx.item()
            result_text = detected_classes[idx] if idx < len(detected_classes) else "Unknown"
            conf_score = confidence.item()

            response_data = {
                'result': result_text,
                'confidence': f"{conf_score * 100:.1f}%",
                'heatmap': None
            }

            if user_mode == 'doctor':
                try:
                    target_layer = None
                    if hasattr(loaded_model, 'layer4'): 
                        target_layer = loaded_model.layer4[-1].conv2
                    elif hasattr(loaded_model, 'features'):
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
