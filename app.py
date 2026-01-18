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
                print(f"SUCCESS: Loaded {name} ✅")
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

# --- 5. IMPROVED PREPROCESSING ---
def preprocess_mammogram(img_cv2):
    """
    Advanced preprocessing for mammograms:
    - CLAHE for contrast enhancement
    - Noise reduction
    - Proper breast tissue isolation
    """
    try:
        # Convert to grayscale for processing
        if len(img_cv2.shape) == 3:
            gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv2.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convert back to BGR for consistency
        if len(img_cv2.shape) == 3:
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        else:
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return processed
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return img_cv2

def crop_to_breast_tissue(img_cv2):
    """
    Intelligently crops to breast tissue with better boundary detection
    """
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        
        # Use adaptive thresholding for better edge detection
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (breast tissue)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Dynamic padding based on image size
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
    """
    Creates precise tissue mask that preserves internal structures
    while removing background and excessive edge regions
    """
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        
        # Use Otsu's thresholding for automatic threshold selection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Fill holes in the mask to ensure continuous tissue region
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Slightly erode to remove skin edge but preserve internal features
        erode_kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.erode(thresh, erode_kernel, iterations=1)
        
        return thresh
    except Exception as e:
        print(f"Mask error: {e}")
        # Return a basic mask as fallback
        h, w = img_cv2.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

# --- 6. IMPROVED GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
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
            
            # Forward pass
            output = self.model(x)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            output[:, class_idx].backward(retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return None
            
            # Calculate weights using global average pooling
            gradients = self.gradients.cpu().numpy()[0]
            activations = self.activations.cpu().numpy()[0]
            
            # Weights are the average of gradients across spatial dimensions
            weights = np.mean(gradients, axis=(1, 2))
            
            # Weighted combination of activation maps
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]
            
            # Apply ReLU to focus on features that positively influence the prediction
            cam = np.maximum(cam, 0)
            
            # Resize to input size
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam
        except Exception as e:
            print(f"GradCAM calculation error: {e}")
            return None

def get_target_layer(model):
    """Get the last convolutional layer for Grad-CAM"""
    try:
        if hasattr(model, 'layer4'):
            last_block = model.layer4[-1]
            if hasattr(last_block, 'conv3'):
                return last_block.conv3
            if hasattr(last_block, 'conv2'):
                return last_block.conv2
    except Exception as e:
        print(f"Error getting layer4: {e}")
    
    # Fallback: find last conv layer
    try:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    except Exception as e:
        print(f"Error finding conv layer: {e}")
    
    return None

def generate_heatmap_overlay(original_cv2, heatmap, prediction_label):
    """
    Generate high-quality heatmap overlay with intelligent masking
    """
    try:
        img = original_cv2.copy()
        
        # Get tissue mask
        tissue_mask = get_tissue_mask_advanced(img)
        tissue_bool = tissue_mask > 0
        
        # Apply tissue mask to heatmap
        masked_heatmap = heatmap * tissue_bool
        
        # Adaptive thresholding based on prediction
        if prediction_label == 'Malignant':
            threshold = 0.25
        elif prediction_label == 'Benign':
            threshold = 0.30
        else:
            threshold = 0.40
        
        # Apply threshold
        masked_heatmap[masked_heatmap < threshold] = 0
        
        # Normalize after thresholding
        if np.max(masked_heatmap) > 0:
            masked_heatmap = masked_heatmap / np.max(masked_heatmap)
        
        # Apply Gaussian blur to smooth the heatmap
        masked_heatmap = cv2.GaussianBlur(masked_heatmap, (11, 11), 0)
        
        # Convert to uint8
        heatmap_uint8 = np.uint8(255 * masked_heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = img.copy()
        
        # Blend where heatmap is present and within tissue
        heat_mask = (masked_heatmap > 0.01) & tissue_bool
        
        if np.any(heat_mask):
            alpha = 0.65
            beta = 0.35
            
            overlay[heat_mask] = cv2.addWeighted(
                heatmap_colored[heat_mask], alpha,
                img[heat_mask], beta,
                0
            )
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Heatmap overlay error: {e}")
        return None

def detect_abnormalities(img_cv2):
    """
    Traditional computer vision approach to detect bright masses/calcifications
    """
    try:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY) if len(img_cv2.shape) == 3 else img_cv2
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect bright regions
        _, bright_thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze detected regions
        significant_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                significant_regions.append(area)
        
        # Calculate abnormality score
        if significant_regions:
            total_abnormal_area = sum(significant_regions)
            num_regions = len(significant_regions)
            score = min(1.0, (total_abnormal_area / 10000.0) + (num_regions * 0.1))
            return score, len(significant_regions)
        
        return 0.0, 0
    except Exception as e:
        print(f"Abnormality detection error: {e}")
        return 0.0, 0

# --- 7. ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'patient')

    if not loaded_model:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        # 1. Load Image
        img_pil = Image.open(file).convert('RGB')
        img_cv2 = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        # 2. Crop to breast tissue
        img_cropped = crop_to_breast_tissue(img_cv2)
        
        # 3. Apply preprocessing
        img_processed = preprocess_mammogram(img_cropped)
        
        # 4. Resize to model input size
        img_final = cv2.resize(img_processed, (224, 224))

        # 5. Convert to PIL for tensor transformation
        img_pil_final = Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))

        # 6. Prepare input tensor
        input_tensor = transform(img_pil_final).unsqueeze(0).to('cpu')

        # 7. Model prediction
        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probs, 1)

        score = confidence.item()
        prediction_label = detected_classes[class_idx.item()]
        
        # 8. Abnormality detection
        abnormality_score, num_abnormalities = detect_abnormalities(img_cropped)
        
        # 9. Generate recommendation
        recommendation = "No action."
        if prediction_label == 'Malignant':
            recommendation = "⚠️ High Risk: Immediate biopsy and consultation recommended."
        elif prediction_label == 'Benign':
            recommendation = "⚡ Medium Risk: Close monitoring and follow-up imaging in 6 months."
        elif prediction_label == 'Normal':
            if abnormality_score > 0.3:
                recommendation = "✓ Low Risk: Routine screening, but notable tissue density detected - consider follow-up."
            else:
                recommendation = "✓ Low Risk: Continue routine annual screening."

        heatmap_base64 = None

        # 10. Generate Grad-CAM for doctor mode
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
                            prediction_label
                        )
                    
                    grad_cam.remove_hooks()
                    
            except Exception as e:
                print(f"GradCAM generation error: {e}")

        return jsonify({
            'result': prediction_label,
            'confidence': f"{score:.2f}",
            'heatmap': heatmap_base64,
            'recommendation': recommendation,
            'abnormality_detection': {
                'score': f"{abnormality_score:.2f}",
                'regions_detected': num_abnormalities
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
