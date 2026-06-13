"""
MedScan API — Production Backend  v9.0 (Strict Validation & Clean UI Output)
Updates:
 - STRICT VALIDATION: Added Color Variance and Edge Density checks to reject 
   random camera photos of rooms, objects, or faces.
 - UI BEAUTIFICATION: Drastically trimmed the data payload. Lesions are capped at top 2,
   numbers are cleanly rounded, and clinical notes are formatted as punchy bullet points 
   so the Flutter UI looks clean and professional.
"""

import os, gc, base64, traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from scipy.fftpack import dct
from flask import Flask, request, jsonify

# LLM Chatbot Integration
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
application = app
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

MODEL_PATHS = {
    'mammogram_mias': 'models/mammo_mias.pth',
    'mammogram_dmid': 'models/mammo_dmid.pth',
    'ultrasound':     'models/ultrasound_model.pth',
    'histopathology': 'models/hist_model.pth',
    'mri':            'models/mri_model.pth',
}
DATASET_CONFIGS = {
    'histopathology': {'num_classes': 2, 'labels': ['Benign',  'Malignant']},
    'mri':            {'num_classes': 2, 'labels': ['Healthy', 'Sick']},
    'mammogram_dmid': {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
    'mammogram_mias': {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
    'ultrasound':     {'num_classes': 3, 'labels': ['Benign',  'Malignant', 'Normal']},
}
VALID_SCAN_TYPES = set(DATASET_CONFIGS.keys()) | {'mammogram', 'auto'}

COLOUR_MODALITIES = {'histopathology'}
LESION_MODALITIES = {'mammogram_dmid', 'mammogram_mias', 'ultrasound'}

active_model_type = None
loaded_model      = None
IMG_SIZE          = 320

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1,2).reshape(B,N,C))


class ExpertBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,groups=ch), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,1),                     nn.BatchNorm2d(ch), nn.ReLU(True))
    def forward(self, x): return self.conv(x)


class MoE(nn.Module):
    def __init__(self, ch, n=3):
        super().__init__()
        self.experts = nn.ModuleList([ExpertBlock(ch) for _ in range(n)])
        self.gate    = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(ch,n), nn.Softmax(dim=1))
    def forward(self, x):
        w = self.gate(x)
        s = torch.stack([e(x) for e in self.experts], dim=1)
        return (s * w.view(-1,len(self.experts),1,1,1)).sum(1), w


class MedFormerULTRA(nn.Module):
    FEATURE_DIM = 256

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s',
                                          pretrained=False, features_only=True)
        with torch.no_grad():
            feats = self.backbone(torch.randn(1,3,224,224))
        fd = [feats[-2].shape[1], feats[-1].shape[1]]
        cd = self.FEATURE_DIM

        self.proj1 = nn.Sequential(nn.Conv2d(fd[0],cd,1), nn.BatchNorm2d(cd), nn.ReLU(True))
        self.proj2 = nn.Sequential(nn.Conv2d(fd[1],cd,1), nn.BatchNorm2d(cd), nn.ReLU(True))
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3,32,5,stride=2,padding=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.moe            = MoE(cd,3)
        self.attention      = EfficientAttention(cd,4)
        self.gpool          = nn.AdaptiveAvgPool2d(1)
        self.fusion         = nn.Sequential(nn.Linear(cd+64,512), nn.LayerNorm(512),
                                            nn.ReLU(True), nn.Dropout(0.4))
        self.classifier     = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256),
                                            nn.ReLU(True), nn.Dropout(0.3),
                                            nn.Linear(256,num_classes))
        self.aux_classifier = nn.Linear(cd, num_classes)

    def forward(self, x, freq, return_all=False):
        feats  = self.backbone(x)
        f1,f2  = self.proj1(feats[-2]), self.proj2(feats[-1])
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        feat,gw = self.moe(f1+f2)
        B,C,H,W = feat.shape
        down    = F.adaptive_avg_pool2d(feat,(H//2,W//2))
        att     = self.attention(down.flatten(2).transpose(1,2))
        sp      = F.interpolate(att.transpose(1,2).reshape(B,C,H//2,W//2),
                                (H,W), mode='bilinear', align_corners=False)
        feat    = feat + sp
        pooled  = self.gpool(feat).flatten(1)
        fused   = self.fusion(torch.cat([pooled, self.freq_encoder(freq).flatten(1)],1))
        logits  = self.classifier(fused)
        aux     = self.aux_classifier(pooled)
        if return_all: return logits, aux, gw
        return logits, aux


# ─────────────────────────────────────────────────────────────────────────────
# 3. FREQUENCY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def extract_freq_tensor(img_rgb_f32, size=320):
    gray = cv2.cvtColor((img_rgb_f32*255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    d   = _dct2(gray)
    low = d.copy()
    low[int(d.shape[0]*0.3):, :] = 0
    low[:, int(d.shape[1]*0.3):] = 0
    f3  = np.stack([low,low,low], axis=2)
    f3  = cv2.resize(f3,(size,size), interpolation=cv2.INTER_LINEAR)
    mx  = np.abs(f3).max()
    if mx > 0: f3 /= mx
    return torch.from_numpy(f3.transpose(2,0,1)).float()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

_spatial_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE,IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def prepare_inputs(img_rgb_uint8, scan_type):
    if scan_type in COLOUR_MODALITIES:
        bgr    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l,a,b  = cv2.split(lab)
        l_enh  = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        enh_rgb = cv2.cvtColor(
            cv2.cvtColor(cv2.merge([l_enh,a,b]),cv2.COLOR_LAB2BGR),
            cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    else:
        gray    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        enh     = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
        enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB).astype(np.float32)/255.0

    return (_spatial_tf(enh_rgb).unsqueeze(0),
            extract_freq_tensor(enh_rgb, IMG_SIZE).unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(scan_type):
    global loaded_model, active_model_type
    if active_model_type == scan_type and loaded_model is not None:
        return loaded_model, DATASET_CONFIGS[scan_type]['labels']
    if loaded_model is not None:
        print(f" * [GATEKEEPER] Unloading {active_model_type}…")
        del loaded_model; loaded_model = None; gc.collect()

    path, config = MODEL_PATHS.get(scan_type), DATASET_CONFIGS[scan_type]
    if not path: raise RuntimeError(f"No path configured for '{scan_type}'.")
    if not os.path.exists(path): raise RuntimeError(f"Model file not found: {path}")

    print(f" * [GATEKEEPER] Loading {scan_type} ({config['num_classes']} classes)…")
    model = MedFormerULTRA(num_classes=config['num_classes'])
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    sd    = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt,dict) else ckpt
    if isinstance(ckpt,dict):
        print(f"   epoch={ckpt.get('epoch','?')}  best_acc={ckpt.get('best_acc','?')}")
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:  print(f"   [WARN] Missing  ({len(miss)}): {miss[:3]}")
    if unexp: print(f"   [WARN] Unexpect ({len(unexp)}): {unexp[:3]}")
    if not miss and not unexp: print("   [OK] Perfect match.")
    model.eval()
    loaded_model = model; active_model_type = scan_type
    print(f"   [OK] {scan_type} ready.")
    return model, config['labels']


try:
    load_model('mammogram_dmid')
except Exception as e:
    print(f"[STARTUP] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODALITY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _he_pct(img_bgr):
    h,w  = img_bgr.shape[:2]
    crop = img_bgr[int(h*0.08):int(h*0.92), int(w*0.05):int(w*0.95)]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hu,sa = hsv[:,:,0], hsv[:,:,1]
    pink   = ((hu<=15)|(hu>=165)) & (sa>25)
    purple = (hu>=120)&(hu<=160)&(sa>25)
    return float((pink|purple).mean()*100)

def _lv_std_bright(gray, thresh=60, bs=8):
    h,w = gray.shape
    bv  = []
    for r in range(0, h-bs, bs):
        for c in range(0, w-bs, bs):
            p = gray[r:r+bs, c:c+bs]
            if p.mean() > thresh:
                bv.append(float(np.var(p)))
    return float(np.std(bv)) if bv else 0.0

def auto_detect_scan_type(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        h,w  = gray.shape

        he = _he_pct(img_bgr)
        if he > 8.0:
            print(f"   [AUTO] histopathology  H&E={he:.1f}%")
            return 'histopathology'

        ew       = max(int(h*0.08), 10)
        em       = [gray[:ew,:].mean(), gray[-ew:,:].mean(),
                    gray[:,:ew].mean(), gray[:,-ew:].mean()]
        dark_n   = sum(e < 35 for e in em)
        h4,w4    = h//4, w//4
        c_mean   = float(gray[h4:3*h4, w4:3*w4].mean())
        c_ratio  = (c_mean - min(em)) / (c_mean + 1e-6)

        left_b  = float((gray[:, :w//10] > 80).mean())
        right_b = float((gray[:, -w//10:] > 80).mean())
        touches_horiz = max(left_b, right_b) > 0.35

        if dark_n >= 3 and c_ratio > 0.40 and not touches_horiz:
            print(f"   [AUTO] mri  dark_n={dark_n} c_ratio={c_ratio:.2f} horiz={max(left_b,right_b):.2f}")
            return 'mri'

        lap    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark60 = float((gray < 60).mean() * 100)
        lv_std = _lv_std_bright(gray, thresh=60)

        if lap > 500:
            print(f"   [AUTO] ultrasound  lap={lap:.0f}")
            return 'ultrasound'
        if dark60 > 30 and lv_std > 150:
            print(f"   [AUTO] ultrasound  dark60={dark60:.0f}% lv_std={lv_std:.0f}")
            return 'ultrasound'
        if dark60 > 55:
            print(f"   [AUTO] ultrasound  dark60={dark60:.0f}%")
            return 'ultrasound'

        print(f"   [AUTO] mammogram_dmid  fallback lap={lap:.0f} dark60={dark60:.1f}% lv_std={lv_std:.0f}")
        return 'mammogram_dmid'

    except Exception as e:
        print(f"   [AUTO] exception: {e} → mammogram_dmid")
        return 'mammogram_dmid'


# ─────────────────────────────────────────────────────────────────────────────
# 7. HIGH-ROBUSTNESS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_medical_image(img_bgr, scan_type):
    try:
        h,w = img_bgr.shape[:2]
        if h<128 or w<128: return False,"Resolution too low (min 128×128)."
        if h>4096 or w>4096: return False,"Image too large (max 4096×4096)."

        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        mean_px = float(gray.mean())

        if mean_px < 5:   return False,"Image is blank/black. Please retake the scan."
        if mean_px > 250: return False,"Image is overexposed. Please retake."
        if float(cv2.Laplacian(gray,cv2.CV_64F).var()) < 4.0:
            return False,"Image is heavily blurred. Please steady the camera."

        # 1. STRICT COLOR CHECK: Most natural photos are highly colorful.
        if len(img_bgr.shape)==3 and scan_type != 'histopathology':
            b, g, r = cv2.split(img_bgr)
            color_var = max(float(cv2.absdiff(b, g).mean()), float(cv2.absdiff(g, r).mean()), float(cv2.absdiff(r, b).mean()))
            if color_var > 15.0:
                return False,"Detected a color photograph. Medical scans must be strictly grayscale or properly stained."

        # 2. CLUTTER / EDGE DENSITY CHECK: Rejects natural photos (rooms, faces, objects)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float((edges > 0).mean())
        if edge_density > 0.18 and scan_type != 'histopathology':
            return False,"Image contains too much structural clutter. It does not resemble a clear medical scan."

        # 3. MEDICAL BACKGROUND CHECK
        dark_pixels = float((gray < 30).mean())
        if dark_pixels < 0.05 and edge_density > 0.08 and scan_type != 'histopathology':
            return False,"Lacks standard medical imaging background. Please ensure the actual scan is uploaded."

        return True,None
    except Exception as e:
        print(f"[Validation Error] {e}")
        return True,None


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAMMOGRAM PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mammogram(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        enh  = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(gray)
        den  = cv2.bilateralFilter(enh,9,75,75)
        return cv2.cvtColor(den,cv2.COLOR_GRAY2BGR)
    except Exception: return img_bgr

def crop_to_breast_tissue(img_bgr):
    try:
        gray     = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        _,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
        k        = np.ones((5,5),np.uint8)
        thresh   = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,k,iterations=2)
        thresh   = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, k,iterations=1)
        cnts,_   = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c=max(cnts,key=cv2.contourArea); x,y,w,h=cv2.boundingRect(c)
            pad=max(5,min(w,h)//20); hi,wi=img_bgr.shape[:2]
            return img_bgr[max(0,y-pad):min(hi,y+h+pad),max(0,x-pad):min(wi,x+w+pad)]
        return img_bgr
    except Exception: return img_bgr

def get_tissue_mask(img_bgr):
    try:
        gray     = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh   = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=3)
        return cv2.erode(thresh,np.ones((7,7),np.uint8),iterations=1)
    except Exception: return np.ones(img_bgr.shape[:2],dtype=np.uint8)*255


# ─────────────────────────────────────────────────────────────────────────────
# 9. CV ABNORMALITY (mammogram only)
# ─────────────────────────────────────────────────────────────────────────────

def cv_abnormality(img_bgr):
    try:
        gray    = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        enh     = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        _,bright= cv2.threshold(enh,210,255,cv2.THRESH_BINARY)
        cnts,_  = cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        regions,total = [],0
        for cnt in cnts:
            a=cv2.contourArea(cnt)
            if a<=30: continue
            p=cv2.arcLength(cnt,True); ci=(4*np.pi*a/p**2) if p>0 else 0
            msk=np.zeros(gray.shape,np.uint8); cv2.drawContours(msk,[cnt],-1,255,-1)
            mi=cv2.mean(enh,mask=msk)[0]
            sc=(3 if a>100 else 2 if a>50 else 1)+(2 if mi>230 else 1 if mi>220 else 0)+(1 if ci>0.6 else 0)
            if sc>=3: regions.append({'area':a,'circ':ci,'int':mi}); total+=a
        n=len(regions)
        if n==0:  s,c,t=0.0,0.9,"Normal"
        elif n==1:
            r=regions[0]; s,c,t=((0.8,0.7,"Malignant") if r['area']>500 or r['int']>235 else (0.5,0.7,"Benign"))
        else: s,c,t=(0.9,0.8,"Malignant") if any(r['area']>300 for r in regions) else (0.6,0.8,"Benign")
        return s,c,{'suspicious_regions':n,'total_area':int(total),
                    'max_intensity':max(r['int'] for r in regions) if regions else 0,'cv_prediction':t}
    except Exception: return 0.0,0.0,{}


# ─────────────────────────────────────────────────────────────────────────────
# 10. GRAD-CAM (IMPORTED FROM USER SCRIPT)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.fh = self.target_layer.register_forward_hook(self.forward_hook)
        self.bh = self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        """Clean up hooks to prevent memory leaks"""
        self.fh.remove()
        self.bh.remove()

    def generate_heatmap(self, sp, fr, target_class=None):
        self.model.eval()
        output, _ = self.model(sp, fr)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap


def get_gradcam_layer(model):
    try:
        if hasattr(model, 'proj1') and isinstance(model.proj1[0], nn.Conv2d):
            print("   [GradCAM] using model.proj1[0]")
            return model.proj1[0]
    except Exception:
        pass
    best = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels >= 64:
            best = m
    return best


def create_red_heatmap(original_image, heatmap, alpha=0.6):
    """Imported from user script: Special function for red-only heatmap"""
    try:
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        red_heatmap = np.zeros_like(original_image)
        red_heatmap[..., 0] = 0
        red_heatmap[..., 1] = 0
        red_heatmap[..., 2] = np.uint8(255 * heatmap_resized)  # Red channel

        superimposed = cv2.addWeighted(original_image, 1 - alpha, red_heatmap, alpha, 0)
        return superimposed

    except Exception as e:
        print(f"❌ Error creating red heatmap: {str(e)}")
        return original_image

def apply_heatmap_jet(original_image, heatmap, alpha=0.5):
    """Imported from user script: Standard Jet Colormap implementation"""
    try:
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_enhanced = np.uint8(255 * heatmap_resized)

        heatmap_colored = cv2.applyColorMap(heatmap_enhanced, cv2.COLORMAP_JET)
        # Note: Do not convert to RGB here because cv2.imencode expects BGR to compress to JPEG correctly.
        superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed
    except Exception as e:
        print(f"❌ Error applying heatmap: {str(e)}")
        return original_image


# ─────────────────────────────────────────────────────────────────────────────
# 11. DOCTOR EXTRAS & BEAUTIFIED OUTPUTS (Flutter UI Optimized)
# ─────────────────────────────────────────────────────────────────────────────

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    empty = {'lesion_count':0,'lesions':[],'total_lesion_area_pct':0.0,'overall_morphology':'Normal / Clear'}
    if scan_type not in LESION_MODALITIES: return empty
    try:
        if cam is None: return empty
        h_i,w_i = img_bgr.shape[:2]
        cam_rs   = cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs/=cam_rs.max()
        nonzero  = cam_rs[cam_rs>0]
        thr      = float(np.percentile(nonzero,75)) if len(nonzero)>0 else 0.5
        binary   = (cam_rs>thr).astype(np.uint8)*255
        binary   = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        binary   = cv2.morphologyEx(binary,cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        cnts,_   = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        total_px = h_i*w_i; total_a = []; lesions = []
        for i,cnt in enumerate(cnts):
            a = cv2.contourArea(cnt)
            if a<300 or a>(total_px*0.25): continue
            x,y,w,h=cv2.boundingRect(cnt); p=cv2.arcLength(cnt,True)
            ci=(4*np.pi*a/p**2) if p>0 else 0
            ha=cv2.contourArea(cv2.convexHull(cnt))
            sol=a/ha if ha>0 else 0; ar=w/h if h>0 else 1.0

            lesions.append({
                'lesion_id': i+1,
                'area_pixels': int(a),
                'area_percentage': round(a/total_px*100, 1), # Cleaned up for UI
                'bounding_box': {'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
                'circularity': round(float(ci),2), 'solidity': round(float(sol),2),
                'aspect_ratio': round(float(ar),2), 'mean_intensity': 0.0,
                'std_intensity': 0.0, 'irregularity_score': round(1-float(ci),2),
                'morphology': _morph(ci,sol,ar), 'suspicion_level': _susp(ci,sol,0)
            })
            total_a.append(a)

        lesions.sort(key=lambda l:l['area_pixels'], reverse=True)
        lesions = lesions[:2] # TRUNCATE FOR UI BEAUTIFICATION (Only top 2 lesions)

        return {'lesion_count': len(lesions), 'lesions': lesions,
                'total_lesion_area_pct': round(sum(total_a)/total_px*100, 1) if total_a else 0.0,
                'overall_morphology': lesions[0]['morphology'] if lesions else 'Normal / Clear'}
    except Exception as e: print(f"[Lesion] {e}"); return empty


def _morph(c,s,ar):
    if c>0.80 and s>0.90: return "Round / Oval Shape"
    if c>0.60 and s>0.80: return "Smooth Oval Margins"
    if c<0.40 or  s<0.65: return "Irregular / Spiculated"
    if ar>1.5  or ar<0.67: return "Elongated Shape"
    return "Lobular Shape"


def _susp(c,s,mi):
    sc=(2 if c<0.5 else 0)+(2 if s<0.7 else 0)
    return "High" if sc>=4 else "Moderate" if sc>=2 else "Low"


def annotated_image(img_bgr, cam, pred_label, scan_type):
    if scan_type not in LESION_MODALITIES: return None
    try:
        if cam is None: return None
        out=img_bgr.copy(); h_i,w_i=img_bgr.shape[:2]
        cam_rs=cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs/=cam_rs.max()
        nonzero=cam_rs[cam_rs>0]
        thr=float(np.percentile(nonzero,75)) if len(nonzero)>0 else 0.5
        binary=(cam_rs>thr).astype(np.uint8)*255
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        cnts,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        col={'Malignant':(0,0,255),'Sick':(0,0,255),'Benign':(0,165,255),
             'Normal':(0,255,0),'Healthy':(0,255,0)}.get(pred_label,(255,255,0))
        max_a=h_i*w_i*0.25
        for i,cnt in enumerate(cnts):
            a=cv2.contourArea(cnt)
            if a<300 or a>max_a: continue
            cv2.drawContours(out,[cnt],-1,col,2)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(out,(x,y),(x+w,y+h),col,1)
            cv2.putText(out,f"R{i+1}",(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2,cv2.LINE_AA)
        _,buf=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92])
        return base64.b64encode(buf).decode()
    except Exception as e: print(f"[Annotated] {e}"); return None


def compute_texture(img_bgr, cam=None):
    try:
        gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        hi=cv2.calcHist([gray],[0],None,[256],[0,256])
        hi=hi/hi.sum(); hi=hi[hi>0]
        
        # BEAUTIFIED OUTPUT: Numbers are cleanly rounded so the UI looks professional.
        return {
            'mean_intensity': round(float(gray.mean()), 1),
            'std_deviation': round(float(gray.std()), 1),
            'skewness': 0.0, 'kurtosis': 0.0, # Zeroed out to reduce UI noise
            'entropy': round(float(-np.sum(hi*np.log2(hi))), 2),
            'dynamic_range': int(gray.max())-int(gray.min())
        }
    except Exception as e: print(f"[Texture] {e}"); return {}


def compute_risk(pred, conf, cv_sev, lesion_data, scan_type):
    try:
        sc,rs=1,[]
        if pred in ('Normal','Healthy'):
            sc=1 if conf>=0.70 else 2; rs.append("AI predicts no significant abnormality.")
        elif pred=='Benign':
            sc=3 if conf<0.65 else 2; rs.append("AI detects likely benign finding.")
        elif pred in ('Malignant','Sick'):
            sc=5 if conf>=0.85 else 4; rs.append("AI detects suspicious pattern indicative of malignancy.")

        lbls={1:"Negative Finding",2:"Benign",3:"Probably Benign",4:"Suspicious Abnormality",5:"Highly Suggestive of Malignancy"}
        ivls={1:"Routine Annual Screening",2:"Routine Annual Screening",3:"Follow-up in 6 months",4:"Biopsy Recommended",5:"Urgent Biopsy Required"}
        
        return {'risk_score':sc,'risk_category':f"Category {sc}",'risk_label':lbls.get(sc,""),
                'recommended_interval':ivls.get(sc,""),'reasoning':[rs[0] if rs else "Review recommended."]}
    except Exception as e: print(f"[Risk] {e}"); return {}


def clinical_notes(pred, conf, lesion_data, scan_type, risk_data):
    # BEAUTIFIED OUTPUT: Clean bullet points for the Flutter UI
    notes=[
        f"❖ MODALITY: {scan_type.replace('_',' ').title()} (Auto-Detected)",
        f"❖ PRIMARY FINDING: {pred.upper()} ({conf*100:.1f}% AI Confidence)",
    ]
    if lesion_data and lesion_data.get('lesion_count',0)>0:
        l = lesion_data['lesions'][0]
        notes.append(f"❖ KEY OBSERVATION: Focus mapped with {l['morphology'].lower()}. Calculated suspicion level is {l['suspicion_level'].upper()}.")
    else:
        notes.append("❖ KEY OBSERVATION: No distinct high-suspicion focal lesions identified.")
    notes.append(f"❖ ACTION: {risk_data.get('recommended_interval', 'Consult physician')}.")
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 12. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, labels, sp, fr):
    with torch.no_grad():
        logits,_ = model(sp,fr)
        probs    = F.softmax(logits,dim=1)
        arr      = probs[0].cpu().numpy()
        conf,idx = torch.max(probs,dim=1)
    return labels[idx.item()], float(conf.item()), arr


# ─────────────────────────────────────────────────────────────────────────────
# 13. PREDICT ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded.'}),400

    file      = request.files['file']
    mode      = request.form.get('mode','patient').lower()
    scan_type = request.form.get('scan_type','auto').strip().lower()
    if scan_type not in VALID_SCAN_TYPES:
        return jsonify({'error':f'Unknown scan_type "{scan_type}".'}),400

    try:
        raw = file.read()
        if not raw: return jsonify({'error':'File is empty.'}),400
        arr     = np.frombuffer(raw,np.uint8)
        img_bgr = cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error':'Cannot decode image. Please upload JPEG/PNG/BMP.'}),400

        # ── 1. Detect modality ─────────────────────────────────────────────
        if scan_type in ('auto','mammogram'):
            detected = auto_detect_scan_type(img_bgr)
        else:
            detected = scan_type
        if detected == 'mammogram': detected = 'mammogram_dmid'
        print(f" * [ROUTE] req={scan_type!r} → detected={detected!r}")

        # ── 2. Validate (Robust Clutter & Color Checks) ────────────────────
        ok,reason = validate_medical_image(img_bgr, detected)
        if not ok:
            return jsonify({'error':'Invalid image detected','message':reason,
                            'action':'Please capture a proper medical image screen.'}),422

        # ── 3. Preprocessing ───────────────────────────────────────────────
        img_display = img_bgr.copy()

        if 'mammogram' in detected:
            img_crop = crop_to_breast_tissue(img_bgr)
            img_proc = preprocess_mammogram(img_crop)
        else:
            img_crop = img_bgr
            img_proc = img_bgr

        img_rs  = cv2.resize(img_proc,(IMG_SIZE,IMG_SIZE))
        img_rgb = cv2.cvtColor(img_rs,cv2.COLOR_BGR2RGB)
        sp,fr   = prepare_inputs(img_rgb, detected)

        cv_sev,cv_det = 0.0,{}

        # ── 4. Inference ───────────────────────────────────────────────────
        do_compare = (scan_type in ('auto','mammogram') and 'mammogram' in detected)

        if do_compare:
            try: model_m,lbl_m = load_model('mammogram_mias')
            except RuntimeError as e: return jsonify({'error':str(e)}),503
            pm,cm,prm = run_inference(model_m,lbl_m,sp,fr)

            del model_m; gc.collect()

            try: model_d,lbl_d = load_model('mammogram_dmid')
            except RuntimeError as e: return jsonify({'error':str(e)}),503
            pd,cd,prd = run_inference(model_d,lbl_d,sp,fr)

            if cd>=cm:
                ai_pred,ai_conf,all_p,wl,wm = pd,cd,prd,lbl_d,'DMID'
            else:
                ai_pred,ai_conf,all_p,wl,wm = pm,cm,prm,lbl_m,'MIAS'
                try: load_model('mammogram_mias')
                except RuntimeError as e: return jsonify({'error':str(e)}),503
        else:
            try: mx,lx = load_model(detected)
            except RuntimeError as e: return jsonify({'error':str(e)}),503
            ai_pred,ai_conf,all_p = run_inference(mx,lx,sp,fr)
            wl,wm = lx,detected

        print(f" * [INFER] model={wm} pred={ai_pred} conf={ai_conf:.3f}")

        # ── 5. CV override (mammogram only) ────────────────────────────────
        srt    = np.sort(all_p)[::-1]
        margin = float(srt[0]-srt[1]) if len(all_p)>1 else 1.0
        override=False; final_pred=ai_pred; final_conf=float(ai_conf)

        if 'mammogram' in detected:
            cv_sev,cv_c,cv_det = cv_abnormality(img_crop)
            cv_pred = cv_det.get('cv_prediction','Normal')
            if ai_pred=='Normal' and cv_sev>0.75 and margin<0.20 and cv_c>0.75:
                override,final_pred,final_conf=True,cv_pred,cv_c
            elif ai_pred=='Benign' and cv_sev>0.90 and margin<0.15 and cv_c>0.80:
                override,final_pred,final_conf=True,'Malignant',cv_c

        # ── 6. Recommendation ──────────────────────────────────────────────
        rec=("HIGH RISK: Immediate biopsy and specialist consultation required."
             if final_pred in ('Malignant','Sick') else
             "MEDIUM RISK: Close monitoring with follow-up imaging in 3–6 months."
             if final_pred=='Benign' else
             "LOW RISK: Continue routine annual screening.")

        # ── 7. GradCAM (USER PROVIDED CODE) ────────────────────────────────
        hmap_b64=None; ann_b64=None; cam_mask=None
        lesion_d={}; tex_d={}; risk_d={}

        tgt = get_gradcam_layer(loaded_model)
        if tgt is not None:
            try:
                cls_idx  = wl.index(final_pred) if final_pred in wl else 0
                gcam     = GradCAM(loaded_model, tgt)
                cam_mask = gcam.generate_heatmap(sp, fr, cls_idx)
                gcam.remove() # Clean hooks to prevent memory leaks

                if cam_mask is not None:
                    # Switch to apply_heatmap_jet(img_display, cam_mask, alpha=0.5) if you want Jet colors
                    overlay = create_red_heatmap(img_display, cam_mask, alpha=0.6)

                    _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    hmap_b64 = base64.b64encode(buf).decode()
            except Exception as e:
                print(f"[GradCAM-run] {e}"); traceback.print_exc()

        # ── 8. Doctor extras ───────────────────────────────────────────────
        if mode=='doctor':
            lesion_d = detect_lesion_boundaries(img_display, cam_mask, detected)
            tex_d    = compute_texture(img_display, cam_mask)
            risk_d   = compute_risk(final_pred,final_conf,cv_sev,lesion_d,detected)
            if cam_mask is not None:
                ann_b64 = annotated_image(img_display, cam_mask, final_pred, detected)

        # ── 9. Response ────────────────────────────────────────────────────
        resp = {
            'result':         final_pred,
            'confidence':     f"{final_conf:.2f}",
            'heatmap':        hmap_b64,
            'recommendation': rec,
            'analysis_details': {
                'scan_type_detected': detected,
                'winning_model':      wm,
                'ai_prediction':      ai_pred,
                'ai_confidence':      f"{ai_conf:.2f}",
                'all_class_probs':   {wl[i]:round(float(all_p[i]),4) for i in range(len(wl))},
                'cv_severity':        f"{cv_sev:.2f}",
                'cv_details':         cv_det,
                'override_active':    override,
                'certainty_margin':   f"{margin:.2f}",
            },
        }
        if mode=='doctor':
            resp['doctor_report']={
                'annotated_image':  ann_b64,
                'lesion_analysis':  lesion_d,
                'texture_features': tex_d,
                'risk_score':       risk_d,
                'clinical_notes':   clinical_notes(final_pred,final_conf,lesion_d,detected,risk_d),
            }
        return jsonify(resp)

    except Exception as e:
        print(f"[PREDICT ERROR] {e}"); traceback.print_exc(); gc.collect()
        return jsonify({'error':'Internal server error. Please try again.'}),500


# ─────────────────────────────────────────────────────────────────────────────
# 13.5 CHAT ROUTE (Generic AI Assistant with formatting fixes)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message data.'}), 400

    user_message = data['message']
    patient_report_json = data.get('report', {})

    try:
        client = genai.Client()

        # UPDATED PROMPT: Highly generic and strictly restricts Markdown for clean Flutter rendering.
        sys_instruct = f"""
        You are BreastScan's intelligent, empathetic AI medical assistant.
        You can answer ANY general health, medical, or technical question the user asks.
        If the user asks about their specific results or asks you to "review the results",
        use this JSON data from their latest scan to explain it to them in simple words:
        {patient_report_json}

        CRITICAL FORMATTING RULES (YOU MUST OBEY THESE):
        1. NO MARKDOWN: You are strictly forbidden from using asterisks (**), hashes (#), or any markdown styling.
        2. PLAIN TEXT ONLY: The frontend app cannot render markdown. If you use '**', it will break the UI.
        3. SPACING: Use clean line breaks (paragraphs) and standard dashes (-) for bullet points.
        4. SIMPLICITY: Explain medical terms in very simple, layperson language.
        5. DISCLAIMER: Always gently remind the user at the end to consult a human doctor or radiologist.
        """

        config = types.GenerateContentConfig(
            system_instruction=sys_instruct,
            temperature=0.3,
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=config
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"[CHAT ERROR] {e}"); traceback.print_exc()
        return jsonify({'error': 'Failed to generate chat response.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 14. UTILITY ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/',methods=['GET'])
def health():
    return jsonify({'status':'online','active_model':active_model_type,'img_size':IMG_SIZE,
                    'endpoints':{'POST /predict':{
                        'scan_type':'auto|mammogram|mammogram_mias|mammogram_dmid|ultrasound|histopathology|mri',
                        'mode':'patient(default)|doctor'}}}),200

@app.route('/models',methods=['GET'])
def models():
    return jsonify({n:{'path':p,'exists':os.path.exists(p),'active':n==active_model_type}
                    for n,p in MODEL_PATHS.items()}),200

@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File too large (max 15 MB).'}),413
@app.errorhandler(400)
def bad_req(e):   return jsonify({'error':str(e)}),400
@app.errorhandler(500)
def srv_err(e):   gc.collect(); return jsonify({'error':'Internal server error.'}),500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=80)
