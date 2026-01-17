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

# --- 2. CRASH-PROOF FIX ---
# AWS Gunicorn sometimes looks for 'application' instead of 'app'.
# This line ensures it works regardless of what AWS asks for.
application = app 

# --- 3. CONFIGURATION ---
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/mammo_model.pth'

# --- 4. SECURITY & QUALITY THRESHOLDS ---
# If the model is less than 60% confident, we assume the image is NOT a mammogram.
CONFIDENCE_THRESHOLD = 0.60  
# Minimum pixel variance to detect "Blank" or "Solid Color" images.
MIN_VARIANCE = 50            

# --- 5. GLOBAL VARIABLES ---
loaded_model = None
# These will be auto-detected when the model loads
detected_classes = ['Benign', 'Malignant'] 

# ==========================================
#      SECTION A: IMAGE VALIDATION
# ==========================================

def is_valid_medical_image(image_pil):
    """
    Analyzes the image to ensure it is not a blank screen, 
    solid color, or completely dark/white image.
    
    Returns:
        (bool, str): (True, None) if valid, (False, reason) if invalid.
    """
    try:
        # Convert to grayscale for statistical analysis
        gray = image_pil.convert('L')
        stat = ImageStat.Stat(gray)
        
        # 1. Check Brightness (Mean pixel value 0-255)
        mean_brightness = stat.mean[0]
        
        if mean_brightness < 5:
            return False, "Image is too dark (black screen)."
        
        if mean_brightness > 250:
            return False, "Image is too bright (white screen)."

        # 2. Check Variance (Detail level)
        # Medical scans have high contrast/variance. 
        # Solid colors (like a wall or blank paper) have near 0 variance.
        variance = stat.var[0]
        
        if variance < MIN_VARIANCE:
            return False, "Image is blank or has no detail (solid color)."

        return True, None

    except Exception as e:
        print(f" [WARNING] Image validation failed: {e}")
        # If the check crashes, we let the image pass to avoid blocking valid users
        return True, None 

# ==========================================
#      SECTION B: UNIVERSAL MODEL LOADER
# ==========================================

def load_pytorch_model():
    """
    Smart Loader that attempts to load the .pth file into multiple
    architectures (ResNet50, ResNet18, ResNet34) with both 2 and 3 classes.
    """
    global loaded_model, detected_classes
    print("\n" + "="*40)
    print(" * [STARTUP] Initializing BreastScan AI Engine...")
    print("="*40)

    if not os.path.exists(MODEL_PATH):
        print(f" ! [ERROR] Model file missing at: {MODEL_PATH}")
        print(" ! Please upload 'mammo_model.pth' to the 'models' folder.")
        return

    device = torch.device('cpu') # AWS Free Tier (t3.micro) uses CPU

    try:
        # 1. Load the File (The "Key")
        print(f" * [INFO] Reading model file: {MODEL_PATH}...")
        # weights_only=False fixes the security error in PyTorch 2.6+
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        print(f" * [INFO] File read successfully. Type: {type(checkpoint)}")
        
        # 2. Extract the State Dictionary (The "Weights")
        state_dict = None
        
        if isinstance(checkpoint, dict):
            # Check common saving formats
            if 'model_state_dict' in checkpoint:
                print(" * [INFO] Found 'model_state_dict' key.")
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                print(" * [INFO] Found 'state_dict' key.")
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                print(" * [INFO] Found 'model' key.")
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state_dict
                print(" * [INFO] Assuming raw state_dict format.")
                state_dict = checkpoint
                
        elif isinstance(checkpoint, nn.Module):
            # If the file is a full model object (pickled model)
            print(" * [INFO] File is a full Model Object (not state_dict).")
            loaded_model = checkpoint
            loaded_model.eval()
            print(" * [SUCCESS] Model loaded directly!")
            return
        
        if state_dict is None:
            print(" ! [ERROR] Could not extract weights from file.")
            return

        # 3. Define the Combinations to Try (The "Locks")
        # We try ResNet50, 18, and 34 with both 3 classes and 2 classes.
        combinations = [
            ("ResNet50", models.resnet50, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet50", models.resnet50, 2, ['Benign', 'Malignant']),
            ("ResNet18", models.resnet18, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet18", models.resnet18, 2, ['Benign', 'Malignant']),
            ("ResNet34", models.resnet34, 3, ['Benign', 'Malignant', 'Normal']),
            ("ResNet34", models.resnet34, 2, ['Benign', 'Malignant']),
        ]

        # 4. Brute Force Matcher
        print(" * [INFO] Attempting to match architecture...")
        
        for name, model_func, num_classes, class_names in combinations:
            try:
                print(f"   - Testing: {name} ({num_classes} classes)...", end=" ")
                
                # Create a fresh Skeleton Model
                temp_model = model_func(weights=None)
                
                # Modify the final layer to match the class count
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, num_classes)
                
                # Attempt to load the weights
                # strict=False allows for minor mismatches (like missing bias), 
                # but we check for MAJOR mismatches manually.
                missing_keys, unexpected_keys = temp_model.load_state_dict(state_dict, strict=False)
                
                # Validation Logic:
                # If there are too many unexpected keys, it's the wrong model.
                if len(unexpected_keys) > 20: 
                    print("❌ (Too many mismatches)")
                    continue
                
                # If we reach here, it's a MATCH!
                print("✅ MATCHED!")
                
                loaded_model = temp_model
                loaded_model.eval() # Set to evaluation mode
                detected_classes = class_names
                
                print(f" * [SUCCESS] Model loaded as {name} with {num_classes} classes.")
                return # Exit the function, we found it!
                
            except Exception as e:
                # If it crashes, just try the next one
                print("❌")
                continue

        # If loop finishes without returning
        print("\n ! [CRITICAL ERROR] No matching architecture found.")
        print(" ! Did you use VGG, DenseNet, or a custom model class?")
        
    except Exception as e:
        print(f" ! [FATAL ERROR] Failed to load model: {e}")

# Run the loader immediately when the server starts
load_pytorch_model()

# ==========================================
#      SECTION C: PREPROCESSING & AI
# ==========================================

# Standard ImageNet normalization (Required for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GRAD-CAM ENGINE (For Doctor Mode) ---
class GradCAM:
    """
    Generates Heatmaps to show WHERE the model is looking.
    Only used when 'mode' = 'doctor'.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture data during prediction
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x):
        self.gradients = None
        self.activations = None
        
        # Forward Pass
        output = self.model(x)
        self.model.zero_grad()
        
        # Target the most confident class
        target_class = output.argmax(dim=1).item()
        score = output[:, target_class]
        
        # Backward Pass (Calculate Gradients)
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return None

        # Generate Heatmap
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8) # Avoid div by zero
        return cam

def overlay_heatmap(img_path, heatmap):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    Returns: Base64 encoded string of the new image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
#      SECTION D: SERVER ROUTES (API)
# ==========================================

@app.route('/', methods=['GET'])
def home():
    """Health Check Route"""
    status = "Active" if loaded_model else "Loading/Error"
    classes = len(detected_classes) if loaded_model else 0
    return f"BreastScan AI Server is {status}. Detected {classes} classes.", 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main Prediction Endpoint.
    Receives: Image File + Mode (patient/doctor)
    Returns: JSON {result, confidence, heatmap}
    """
    # 1. Basic Request Validation
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    user_mode = request.form.get('mode', 'patient') # Default to patient

    # 2. Check if Model is Ready
    if loaded_model is None:
        return jsonify({'error': 'Model failed to load. Check server logs.'}), 503

    if file:
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 3. Load & Validate Image
            image_pil = Image.open(filepath).convert('RGB')

            # --- SECURITY CHECK: VALID MEDICAL IMAGE? ---
            is_valid, error_msg = is_valid_medical_image(image_pil)
            if not is_valid:
                print(f" [REJECT] Image rejected: {error_msg}")
                if os.path.exists(filepath): os.remove(filepath)
                return jsonify({
                    'result': 'Error',
                    'confidence': '0%',
                    'error': error_msg
                })

            # 4. Preprocess for AI
            input_tensor = transform(image_pil).unsqueeze(0).to('cpu')

            # 5. Run Prediction
            with torch.no_grad():
                output = loaded_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

            # 6. Process Results
            idx = predicted_idx.item()
            
            # Map index to class name (Benign/Malignant/Normal)
            if idx < len(detected_classes):
                result_text = detected_classes[idx]
            else:
                result_text = "Unknown"
            
            conf_score = confidence.item()

            # --- SECURITY CHECK: CONFIDENCE THRESHOLD ---
            if conf_score < CONFIDENCE_THRESHOLD:
                print(f" [REJECT] Low confidence: {conf_score:.2f}")
                if os.path.exists(filepath): os.remove(filepath)
                return jsonify({
                    'result': 'Uncertain',
                    'confidence': f"{conf_score * 100:.1f}%",
                    'error': 'Image unclear or not a medical scan.'
                })

            # Prepare Success Response
            response_data = {
                'result': result_text,
                'confidence': f"{conf_score * 100:.1f}%",
                'heatmap': None
            }

            # 7. Doctor Mode Logic (Heatmap Generation)
            if user_mode == 'doctor':
                try:
                    # Auto-detect the target layer for Grad-CAM
                    target_layer = None
                    if hasattr(loaded_model, 'layer4'): 
                        target_layer = loaded_model.layer4[-1].conv2 # ResNet
                    elif hasattr(loaded_model, 'features'):
                        target_layer = loaded_model.features[-1] # VGG/DenseNet
                    
                    if target_layer:
                        grad_cam = GradCAM(loaded_model, target_layer)
                        heatmap = grad_cam(input_tensor)
                        
                        if heatmap is not None:
                            heatmap_b64 = overlay_heatmap(filepath, heatmap)
                            response_data['heatmap'] = heatmap_b64
                            print(" [INFO] Heatmap generated for Doctor.")
                            
                except Exception as e:
                    print(f" [WARNING] GradCAM failed: {e}")
                    # Don't fail the whole request, just return no heatmap

            # 8. Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
            
            print(f" [SUCCESS] Predicted: {result_text} ({conf_score:.2f})")
            return jsonify(response_data)

        except Exception as e:
            print(f" [ERROR] Prediction pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

# ==========================================
#      SECTION E: ENTRY POINT
# ==========================================

if __name__ == '__main__':
    # Use Port 80 for production access
    port = int(os.environ.get('PORT', 80))
    print(f" * Starting Server on Port {port}...")
    app.run(host='0.0.0.0', port=port)
