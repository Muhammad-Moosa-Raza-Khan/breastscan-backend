import os
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import timm
from scipy.fftpack import dct
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- 1. INITIALIZE APP ---
app = Flask(__name__)
application = app

# --- 2. THE MODEL REGISTRY ---
MODEL_REGISTRY = {
    'histopathology': {
        'path': 'models/breakhis_model.pth',
        'classes': ['Benign', 'Malignant']
    },
    'mri': {
        'path': 'models/mri_kaggle_model.pth',
        'classes': ['Healthy', 'Sick']
    },
    'mammogram': {
        'path': 'models/mias_dmid_model.pth', 
        'classes': ['Normal', 'Benign', 'Malignant'] # Order matches your config
    },
    'ultrasound': {
        'path': 'models/ultrasound_model.pth',
        'classes': ['Normal', 'Benign', 'Malignant']
    }
}

# --- 3. FREQUENCY EXTRACTOR (Required for RAM-Net) ---
class FastFrequencyAnalysis:
    @staticmethod
    def dct_transform_fast(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            freq = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            freq = np.stack([freq, freq, freq], axis=2)
        else:
            freq = dct(dct(img.T, norm='ortho').T, norm='ortho')
            freq = np.stack([freq, freq, freq], axis=2)
        return freq

    @staticmethod
    def extract_low_freq_fast(img):
        dct_feat = FastFrequencyAnalysis.dct_transform_fast(img)
        h, w = dct_feat.shape[:2]
        cutoff_h, cutoff_w = int(h*0.3), int(w*0.3)
        low_freq = dct_feat.copy()
        low_freq[cutoff_h:, :] = 0
        low_freq[:, cutoff_w:] = 0
        return low_freq

# --- 4. RAM-NET (MedFormer-ULTRA) ARCHITECTURE ---
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ExpertBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MoE(nn.Module):
    def __init__(self, channels, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([ExpertBlock(channels) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        weights = self.gate(x)
        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)
        weights = weights.view(-1, self.num_experts, 1, 1, 1)
        output = (stacked * weights).sum(dim=1)
        return output, weights.squeeze()

class RAMNet(nn.Module):
    def __init__(self, num_classes=3, feature_dim=256, num_experts=3):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False, features_only=True)
        
        # Determine feature dims by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
        self.feat_dims = [feats[-2].shape[1], feats[-1].shape[1]]
        
        common_dim = feature_dim
        self.proj1 = nn.Sequential(nn.Conv2d(self.feat_dims[0], common_dim, 1), nn.BatchNorm2d(common_dim), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(self.feat_dims[1], common_dim, 1), nn.BatchNorm2d(common_dim), nn.ReLU(inplace=True))
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.moe = MoE(common_dim, num_experts)
        self.attention = EfficientAttention(common_dim, num_heads=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fusion = nn.Sequential(
            nn.Linear(common_dim + 64, 512), nn.LayerNorm(512), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        self.aux_classifier = nn.Linear(common_dim, num_classes)
        
    def forward(self, x, freq, return_all=False):
        feats = self.backbone(x)
        f1, f2 = self.proj1(feats[-2]), self.proj2(feats[-1])
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        
        fused = f1 + f2
        feat, gate_weights = self.moe(fused)
        
        B, C, H, W = feat.shape
        feat_down = F.adaptive_avg_pool2d(feat, (H//2, W//2))
        feat_flat = feat_down.flatten(2).transpose(1, 2)
        feat_att = self.attention(feat_flat)
        feat_spatial = feat_att.transpose(1, 2).reshape(B, C, H//2, W//2)
        feat_spatial = F.interpolate(feat_spatial, size=(H, W), mode='bilinear', align_corners=False)
        feat = feat + feat_spatial
        
        pooled = self.global_pool(feat).flatten(1)
        freq_feat = self.freq_encoder(freq).flatten(1)
        combined = torch.cat([pooled, freq_feat], dim=1)
        fused_feat = self.fusion(combined)
        
        logits = self.classifier(fused_feat)
        aux_logits = self.aux_classifier(pooled)
        
        if return_all: return logits, aux_logits, gate_weights
        return logits, aux_logits

# --- 5. THE MATHEMATICAL GATEKEEPER ---
def detect_image_modality(img_cv2):
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    if saturation > 40: return 'histopathology'

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 15)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    if (black_pixels / total_pixels) > 0.40: return 'mammogram'
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 500: return 'ultrasound'
    return 'mri'

# --- 6. PREPROCESSING & DUAL-INPUT GRAD-CAM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_tissue_mask(img_cv2):
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.erode(thresh, np.ones((10, 10), np.uint8), iterations=2)

class DualInputGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)

    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]

    def __call__(self, x, freq):
        self.gradients = None; self.activations = None
        # RAM-Net returns logits and aux_logits. We only want logits [0]
        out = self.model(x, freq)[0] 
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
        return (cam - np.min(cam)) / (np.max(cam) + 1e-8)

def generate_heatmap_overlay(original_cv2, heatmap):
    img = cv2.resize(original_cv2, (224, 224))
    tissue_mask = get_tissue_mask(img)
    tissue_bool = tissue_mask > 0
    heatmap = heatmap * tissue_bool
    heatmap[heatmap < 0.35] = 0 
    if np.max(heatmap) > 0: heatmap = heatmap / np.max(heatmap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = img.copy()
    final_mask = (heatmap > 0) & tissue_bool
    if np.any(final_mask):
         overlay[final_mask] = cv2.addWeighted(heatmap_colored[final_mask], 0.6, img[final_mask], 0.4, 0)
    _, buffer = cv2.imencode('.jpg', overlay)
    return base64.b64encode(buffer).decode('utf-8')

# --- 7. ROUTING ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    mode = request.form.get('mode', 'patient')

    try:
        # 1. Load Image
        img_pil = Image.open(file).convert('RGB')
        img_cv2 = np.array(img_pil) 
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        
        # 2. THE GATEKEEPER
        modality_key = detect_image_modality(img_cv2)
        model_config = MODEL_REGISTRY[modality_key]
        
        # 3. GENERATE FREQUENCY DATA (For RAM-Net)
        img_for_freq = cv2.resize(img_cv2, (224, 224)).astype(np.float32) / 255.0
        low_freq = FastFrequencyAnalysis.extract_low_freq_fast(img_for_freq)
        freq_tensor = torch.from_numpy(low_freq.transpose(2, 0, 1)).float().unsqueeze(0).to('cpu')
        
        # 4. LOAD MODEL DYNAMICALLY
        device = torch.device('cpu')
        loaded_model = RAMNet(num_classes=len(model_config['classes']))
        
        state_dict = torch.load(model_config['path'], map_location=device, weights_only=False)
        # If your state dict has a 'model_state_dict' key (as seen in your code), extract it:
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        loaded_model.load_state_dict(state_dict, strict=False)
        loaded_model.eval()

        # 5. PREDICT
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs, _ = loaded_model(input_tensor, freq_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probs, 1)

        score = confidence.item()
        prediction_label = model_config['classes'][class_idx.item()]

        # 6. GRAD-CAM
        heatmap_base64 = None
        if mode == 'doctor':
            try:
                # Target the MoE layer for the best spatial-frequency blended heatmap
                target_layer = loaded_model.proj2[-1] 
                cam_tool = DualInputGradCAM(loaded_model, target_layer)
                heatmap_mask = cam_tool(input_tensor, freq_tensor)
                if heatmap_mask is not None:
                    heatmap_base64 = generate_heatmap_overlay(img_cv2, heatmap_mask)
            except Exception as e:
                print(f"GradCAM Error: {e}")

        # 7. CLEAR MEMORY
        del loaded_model
        gc.collect()

        return jsonify({
            'modality_detected': modality_key.capitalize(),
            'result': prediction_label,
            'confidence': f"{score:.2f}",
            'heatmap': heatmap_base64,
            'recommendation': "Consult specialist for detailed diagnosis."
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "BreastScan Multi-Model RAM-Net Backend Online", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
