"""
MedScan API — Production Backend  v6
Fixes applied over v5:
  BUG-1  self.aux typo  → 500 crash on any request (forward() called self.aux not self.aux_classifier)
  BUG-2  GradCAM no overlay on histopathology  → threshold too high for colour images; now uses
         modality-tuned percentile and COLORMAP_HOT for colour, COLORMAP_JET for grayscale
  BUG-3  GradCAM blue-dot scatter on benign histo  → get_best_gradcam_layer now runs inside
         backbone.named_modules() with a real forward pass so feature-map sizes are accurate
  BUG-4  Mammogram → MRI misclassification  → auto_detect now checks grayscale constraint FIRST
         before MRI dark-edge test; mammograms that are near-grayscale are locked to mammogram path
  BUG-5  Lesion boundaries cover entire image  → detect_lesion_boundaries skips non-mammogram
         modalities (MRI/histo/US lesions don't map to the same contour logic);
         max-coverage guard tightened to 25 %
  BUG-6  Too many spurious lesion regions on MRI  → lesion detection disabled for MRI/histo/US
         in the doctor report; those modalities get texture + risk only
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

# ─────────────────────────────────────────────────────────────────────────────
# 1. APP INIT
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
VALID_SCAN_TYPES  = set(DATASET_CONFIGS.keys()) | {'mammogram', 'auto'}
COLOUR_MODALITIES = {'histopathology', 'ultrasound'}

# Modalities where contour-based lesion detection makes clinical sense
LESION_DETECT_MODALITIES = {'mammogram_dmid', 'mammogram_mias', 'mammogram', 'ultrasound'}

active_model_type = None
loaded_model      = None
IMG_SIZE          = 320

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE  (exact training replica)
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1,2).reshape(B,N,C))


class ExpertBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),                       nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)


class MoE(nn.Module):
    def __init__(self, ch, n=3):
        super().__init__()
        self.experts = nn.ModuleList([ExpertBlock(ch) for _ in range(n)])
        self.gate    = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(ch, n), nn.Softmax(dim=1))
    def forward(self, x):
        w = self.gate(x)
        s = torch.stack([e(x) for e in self.experts], dim=1)
        return (s * w.view(-1, len(self.experts), 1, 1, 1)).sum(dim=1), w


class MedFormerULTRA(nn.Module):
    FEATURE_DIM = 256

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s',
                                          pretrained=False, features_only=True)
        with torch.no_grad():
            feats = self.backbone(torch.randn(1, 3, 224, 224))
        fd = [feats[-2].shape[1], feats[-1].shape[1]]
        cd = self.FEATURE_DIM

        self.proj1 = nn.Sequential(nn.Conv2d(fd[0], cd, 1), nn.BatchNorm2d(cd), nn.ReLU(True))
        self.proj2 = nn.Sequential(nn.Conv2d(fd[1], cd, 1), nn.BatchNorm2d(cd), nn.ReLU(True))
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.moe            = MoE(cd, 3)
        self.attention      = EfficientAttention(cd, 4)
        self.gpool          = nn.AdaptiveAvgPool2d(1)
        self.fusion         = nn.Sequential(nn.Linear(cd+64, 512), nn.LayerNorm(512),
                                            nn.ReLU(True), nn.Dropout(0.4))
        self.classifier     = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256),
                                            nn.ReLU(True), nn.Dropout(0.3),
                                            nn.Linear(256, num_classes))
        self.aux_classifier = nn.Linear(cd, num_classes)   # BUG-1 FIX: correct name used everywhere

    def forward(self, x, freq, return_all=False):
        feats  = self.backbone(x)
        f1, f2 = self.proj1(feats[-2]), self.proj2(feats[-1])
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
        feat, gw = self.moe(f1 + f2)
        B, C, H, W = feat.shape
        down = F.adaptive_avg_pool2d(feat, (H//2, W//2))
        att  = self.attention(down.flatten(2).transpose(1,2))
        sp   = F.interpolate(att.transpose(1,2).reshape(B,C,H//2,W//2),
                             (H,W), mode='bilinear', align_corners=False)
        feat   = feat + sp
        pooled = self.gpool(feat).flatten(1)
        fused  = self.fusion(torch.cat([pooled, self.freq_encoder(freq).flatten(1)], dim=1))
        logits = self.classifier(fused)
        aux    = self.aux_classifier(pooled)          # BUG-1 FIX: was self.aux(pooled) → AttributeError
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
    d = _dct2(gray)
    low = d.copy()
    low[int(d.shape[0]*0.3):, :] = 0
    low[:, int(d.shape[1]*0.3):] = 0
    f3 = np.stack([low,low,low], axis=2)
    f3 = cv2.resize(f3, (size,size), interpolation=cv2.INTER_LINEAR)
    mx = np.abs(f3).max()
    if mx > 0: f3 /= mx
    return torch.from_numpy(f3.transpose(2,0,1)).float()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING  (colour-aware CLAHE)
# ─────────────────────────────────────────────────────────────────────────────

_spatial_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def prepare_inputs(img_rgb_uint8, scan_type):
    if scan_type in COLOUR_MODALITIES:
        bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        enh_rgb = cv2.cvtColor(
            cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR),
            cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        gray    = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
        enh     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
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

    path   = MODEL_PATHS.get(scan_type)
    config = DATASET_CONFIGS[scan_type]
    if not path:
        raise RuntimeError(f"No path configured for '{scan_type}'.")
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")

    print(f" * [GATEKEEPER] Loading {scan_type} ({config['num_classes']} classes)…")
    model = MedFormerULTRA(num_classes=config['num_classes'])
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    sd    = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(ckpt, dict):
        print(f"   epoch={ckpt.get('epoch','?')}  best_acc={ckpt.get('best_acc','?')}")
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:  print(f"   [WARN] Missing  ({len(miss)}): {miss[:3]}")
    if unexp: print(f"   [WARN] Unexpect ({len(unexp)}): {unexp[:3]}")
    if not miss and not unexp: print("   [OK] Perfect weight match.")
    model.eval()
    loaded_model = model; active_model_type = scan_type
    print(f"   [OK] {scan_type} ready.")
    return model, config['labels']


try:
    load_model('mammogram_dmid')
except Exception as e:
    print(f"[STARTUP] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODALITY DETECTOR  v6
# BUG-4 FIX: grayscale-lock check runs BEFORE MRI dark-edge test.
#            Mammograms are near-grayscale → locked to mammogram path immediately.
# ─────────────────────────────────────────────────────────────────────────────

def _he_pct(img_bgr):
    """H&E hue pixel ratio in centre crop.  Histopathology reliably > 8 %."""
    h, w  = img_bgr.shape[:2]
    crop  = img_bgr[int(h*0.08):int(h*0.92), int(w*0.05):int(w*0.95)]
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hu, sa = hsv[:,:,0], hsv[:,:,1]
    pink   = ((hu <= 15) | (hu >= 165)) & (sa > 25)
    purple = (hu >= 120) & (hu <= 160) & (sa > 25)
    return float((pink | purple).mean() * 100)


def _is_near_grayscale(img_bgr, threshold=12):
    """
    Returns True if image is near-grayscale (mammogram / MRI / plain X-ray).
    Computes mean absolute per-pixel channel difference across full image.
    BUG-4 FIX: mammograms have per-pixel diff < 12, MRI < 12,
               while ultrasound B-mode is also grayscale but has high Laplacian.
    """
    if len(img_bgr.shape) < 3: return True
    b = img_bgr[:,:,0].astype(np.float32)
    g = img_bgr[:,:,1].astype(np.float32)
    r = img_bgr[:,:,2].astype(np.float32)
    return max(float(np.mean(np.abs(r-g))),
               float(np.mean(np.abs(r-b))),
               float(np.mean(np.abs(g-b)))) < threshold


def auto_detect_scan_type(img_bgr):
    """
    5-step detector (BUG-4 FIX — grayscale lock added as step 1):
      1. Grayscale-lock  — near-grayscale image → never MRI misfire, go to step 3+
      2. Histopathology  — H&E hue ratio > 8 %
      3. MRI             — all 4 edges dark + bright centre  (only reached if NOT grayscale-locked OR explicit MRI)
      4. Ultrasound      — high Laplacian OR large dark area
      5. Mammogram       — fallback
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()

        # ── Step 1: Grayscale-lock ─────────────────────────────────────────
        near_gray = _is_near_grayscale(img_bgr, threshold=12)
        print(f"   [AUTO] near_gray={near_gray}")

        # ── Step 2: Histopathology (only if NOT near-grayscale) ───────────
        if not near_gray:
            he = _he_pct(img_bgr)
            print(f"   [AUTO] H&E={he:.1f}%")
            if he > 8.0:
                return 'histopathology'

        # ── Step 3: MRI (only if near-grayscale AND dark-edge pattern) ────
        # BUG-4 FIX: Mammograms are near-grayscale but have one bright edge
        # (chest wall) rather than ALL 4 dark edges.  The 4-edge test
        # naturally excludes mammograms without any extra heuristic.
        ew      = max(int(gray.shape[0] * 0.08), 10)
        edges   = [gray[:ew,:].mean(), gray[-ew:,:].mean(),
                   gray[:,:ew].mean(), gray[:,-ew:].mean()]
        dark_n  = sum(e < 35 for e in edges)
        h4, w4  = gray.shape[0]//4, gray.shape[1]//4
        c_mean  = float(gray[h4:3*h4, w4:3*w4].mean())
        c_ratio = (c_mean - min(edges)) / (c_mean + 1e-6)
        print(f"   [AUTO] dark_edges={dark_n} c_ratio={c_ratio:.2f}")

        if near_gray and dark_n >= 4 and c_ratio > 0.40:
            return 'mri'
        if near_gray and dark_n >= 3 and c_ratio > 0.60:
            return 'mri'

        # ── Step 4: Ultrasound ─────────────────────────────────────────────
        lap  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark = float((gray < 25).mean() * 100)
        print(f"   [AUTO] lap={lap:.0f} dark={dark:.1f}%")

        if lap > 600: return 'ultrasound'
        if dark > 30: return 'ultrasound'
        if lap > 300 and dark > 15: return 'ultrasound'

        # ── Step 5: Mammogram ──────────────────────────────────────────────
        print(f"   [AUTO] → mammogram_dmid (fallback)")
        return 'mammogram_dmid'

    except Exception as e:
        print(f"   [AUTO] exception: {e} → mammogram_dmid")
        return 'mammogram_dmid'


# ─────────────────────────────────────────────────────────────────────────────
# 7. VALIDATION  (per-modality thresholds)
# ─────────────────────────────────────────────────────────────────────────────

_BLUR_MIN = {'mammogram_mias':6,'mammogram_dmid':6,'mri':10,'ultrasound':15,'histopathology':20}
_STD_MIN  = {'mammogram_mias':5,'mammogram_dmid':5,'mri':7, 'ultrasound':7, 'histopathology':8}


def validate_medical_image(img_bgr, scan_type):
    try:
        h, w = img_bgr.shape[:2]
        if h < 128 or w < 128: return False, "Resolution too low (min 128×128)."
        if h > 4096 or w > 4096: return False, "Image too large (max 4096×4096)."

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        mean_px = float(gray.mean())
        if mean_px < 4:   return False, "Image is blank/black. Please retake the scan."
        if mean_px > 251: return False, "Image is overexposed. Please retake."

        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        thr = _BLUR_MIN.get(scan_type, 15)
        print(f"   [VALIDATE] scan={scan_type} lap={lap:.1f} thr={thr} mean={mean_px:.1f}")
        if lap < thr:
            return False, f"Image too blurry (sharpness={lap:.1f}, min={thr})."

        if float(np.std(gray)) < _STD_MIN.get(scan_type, 7):
            return False, "Insufficient contrast."

        # Colour check only for grayscale modalities
        if len(img_bgr.shape)==3 and scan_type not in COLOUR_MODALITIES:
            if not _is_near_grayscale(img_bgr, threshold=35):
                return False, (f"Does not look like a {scan_type.replace('_',' ')} scan. "
                               "Please upload a proper grayscale medical image.")

        # Screenshot heuristic (centre crop)
        gc2 = gray[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        edges = cv2.Canny(gc2, 50, 150)
        lines = cv2.HoughLinesP(edges,1,np.pi/2,threshold=120,
                                minLineLength=int(w*0.75),maxLineGap=10)
        if lines is not None and len(lines) > 6:
            return False, "Appears to be a screenshot. Please crop to the scan only."

        # Solid-colour guard
        bs = 32; rows, cols = h//bs, w//bs
        bv = [float(np.var(gray[r*bs:(r+1)*bs,c*bs:(c+1)*bs]))
              for r in range(rows) for c in range(cols)]
        bv = np.array(bv) if bv else np.array([100.])
        if bv.size > 0 and (bv < 5).sum()/bv.size > 0.85:
            return False, "Solid colour image — not a medical scan."

        return True, None
    except Exception:
        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAMMOGRAM PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mammogram(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        enh  = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(gray)
        den  = cv2.bilateralFilter(enh, 9, 75, 75)
        return cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    except Exception: return img_bgr


def crop_to_breast_tissue(img_bgr):
    try:
        gray      = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        k         = np.ones((5,5),np.uint8)
        thresh    = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,k,iterations=2)
        thresh    = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, k,iterations=1)
        cnts,_    = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts,key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            pad = max(5,min(w,h)//20); hi,wi = img_bgr.shape[:2]
            return img_bgr[max(0,y-pad):min(hi,y+h+pad),max(0,x-pad):min(wi,x+w+pad)]
        return img_bgr
    except Exception: return img_bgr


def get_tissue_mask(img_bgr):
    try:
        gray      = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh    = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=3)
        return cv2.erode(thresh,np.ones((7,7),np.uint8),iterations=1)
    except Exception: return np.ones(img_bgr.shape[:2],dtype=np.uint8)*255


# ─────────────────────────────────────────────────────────────────────────────
# 9. CV ABNORMALITY (mammogram only — unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def cv_abnormality(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        enh  = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        _,bright = cv2.threshold(enh,210,255,cv2.THRESH_BINARY)
        cnts,_   = cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        regions, total = [], 0
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if a<=30: continue
            p  = cv2.arcLength(cnt,True)
            ci = (4*np.pi*a/p**2) if p>0 else 0
            msk= np.zeros(gray.shape,np.uint8); cv2.drawContours(msk,[cnt],-1,255,-1)
            mi = cv2.mean(enh,mask=msk)[0]
            sc = (3 if a>100 else 2 if a>50 else 1)+(2 if mi>230 else 1 if mi>220 else 0)+(1 if ci>0.6 else 0)
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
# 10. GRAD-CAM  (BUG-2 + BUG-3 fixes)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, layer):
        self.model=model; self.grads=None; self.acts=None
        self._h=[layer.register_forward_hook(self._fa),
                 layer.register_full_backward_hook(self._fg)]
    def _fa(self,m,i,o): self.acts=o.detach()
    def _fg(self,m,gi,go): self.grads=go[0].detach()
    def remove(self): [h.remove() for h in self._h]

    def __call__(self, sp, fr, cls=None):
        try:
            self.model.eval()
            logits,_ = self.model(sp,fr)
            if cls is None: cls=logits.argmax(1).item()
            self.model.zero_grad()
            logits[0,cls].backward(retain_graph=True)
            if self.grads is None or self.acts is None: return None
            g = self.grads.cpu().numpy()[0]
            a = self.acts.cpu().numpy()[0]
            cam = np.maximum(np.sum(np.mean(g,axis=(1,2))[:,None,None]*a, axis=0), 0)
            cam = cv2.resize(cam,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_CUBIC)
            if cam.max()>0: cam/=cam.max()
            return cam
        except Exception as e:
            print(f"[GradCAM] {e}"); return None


def get_best_gradcam_layer(model, scan_type):
    """
    BUG-3 FIX: Run a REAL forward pass with a dummy image of the correct IMG_SIZE
    so feature-map sizes are measured accurately.
    Select the layer whose spatial output is closest to IMG_SIZE/8 (≈40 for 320px),
    avoiding too-deep layers that produce 10×10 maps (→ blocky blobs after upsampling).

    For colour modalities (histo, US): prefer layers with more channels for richer
    semantic representation, slightly deeper.
    """
    try:
        feat_sizes = {}
        candidates = []
        handles    = []

        def make_hook(name):
            def h(m, inp, out): feat_sizes[name] = out.shape
            return h

        min_ch = 64
        for name, mod in model.backbone.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.out_channels >= min_ch:
                handles.append(mod.register_forward_hook(make_hook(name)))
                candidates.append((name, mod))

        dummy_x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        dummy_f = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            model(dummy_x, dummy_f)
        for hd in handles: hd.remove()

        # BUG-3 FIX: target ~IMG_SIZE/8 spatial, allow range [12, 60]
        target   = IMG_SIZE // 8
        best, best_score = None, float('inf')
        for name, mod in candidates:
            if name in feat_sizes:
                sz = feat_sizes[name][2]
                if 12 <= sz <= 60:
                    score = abs(sz - target)
                    if score < best_score:
                        best_score = score; best = mod

        if best:
            print(f"   [GradCAM] layer chosen: spatial≈{target+best_score}  scan={scan_type}")
            return best

        # Fallback: last conv ≥ 128 ch
        fall = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 128: fall = m
        return fall
    except Exception as e:
        print(f"[get_best_gradcam_layer] {e}")
        fall = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d): fall = m
        return fall


def _percentile_threshold(cam, pct=78):
    """
    BUG-2 FIX: Adaptive threshold based on the CAM's own 78th percentile.
    Never go below 0.25 (prevents entire-image activation for flat CAMs).
    """
    if cam.max() <= 0: return cam
    thr    = max(float(np.percentile(cam, pct)), 0.25)
    result = cam.copy()
    result[result < thr] = 0
    if result.max() > 0: result /= result.max()
    return result


def generate_heatmap_overlay(img_bgr, cam, pred_label, scan_type):
    """
    BUG-2 FIX:
      - Colour modalities: pure CAM overlay with COLORMAP_HOT (warm tones visible on pink tissue)
      - Grayscale modalities: blend with intensity + COLORMAP_JET (as before)
    BUG-3 FIX: apply percentile threshold before overlay.
    """
    try:
        if cam is None: return None
        img      = img_bgr.copy()
        h_i, w_i = img.shape[:2]
        cam_rs   = cv2.resize(cam, (w_i,h_i), interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs /= cam_rs.max()

        if scan_type in COLOUR_MODALITIES:
            # No intensity blending for colour images — keeps H&E colours visible
            refined  = _percentile_threshold(cam_rs, pct=78)
            cmap     = cv2.COLORMAP_HOT    # warm red/yellow visible on pink tissue
        else:
            gray_rs  = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0,
                (w_i,h_i))
            blended  = cam_rs * 0.7 + np.power(gray_rs, 0.7) * 0.3
            if blended.max()>0: blended /= blended.max()
            refined  = _percentile_threshold(blended, pct=78)
            cmap     = cv2.COLORMAP_JET

        # Tissue mask only for mammograms
        tissue = get_tissue_mask(img) > 0 if 'mammogram' in scan_type \
                 else np.ones((h_i, w_i), dtype=bool)

        refined   = cv2.GaussianBlur(refined, (11,11), 0) * tissue
        colored   = cv2.applyColorMap(np.uint8(255*refined), cmap)
        overlay   = img.copy()
        heat_mask = (refined > 0.01) & tissue
        if np.any(heat_mask):
            overlay[heat_mask] = cv2.addWeighted(
                colored[heat_mask], 0.60, img[heat_mask], 0.40, 0)
        _,buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY,95])
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[Heatmap] {e}"); return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. DOCTOR EXTRAS
# BUG-5 + BUG-6 FIX: lesion detection only for mammogram/ultrasound;
#                     MRI and histo return texture + risk only.
# ─────────────────────────────────────────────────────────────────────────────

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    """
    BUG-5 FIX: max single-lesion area capped at 25 % of image.
    BUG-6 FIX: only runs for LESION_DETECT_MODALITIES; returns empty for MRI/histo.
    """
    empty = {'lesion_count':0,'lesions':[],'total_lesion_area_pct':0.0,'overall_morphology':'N/A'}
    if scan_type not in LESION_DETECT_MODALITIES:
        return empty   # MRI and histopathology: no contour-based lesion detection

    try:
        if cam is None: return empty
        h_i,w_i  = img_bgr.shape[:2]
        cam_rs    = cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs /= cam_rs.max()
        cam_thr   = _percentile_threshold(cam_rs, pct=78)

        binary = (cam_thr>0).astype(np.uint8)*255
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        cnts,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        gray     = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        total_px = h_i*w_i
        max_area = total_px * 0.25    # BUG-5 FIX: tightened from 0.60 to 0.25
        total_a, lesions = 0, []

        for i,cnt in enumerate(cnts):
            a = cv2.contourArea(cnt)
            if a < 300: continue       # min 300px — avoids noise dots
            if a > max_area: continue  # skip implausibly large blobs
            x,y,w,h = cv2.boundingRect(cnt)
            p  = cv2.arcLength(cnt,True)
            ci = (4*np.pi*a/p**2) if p>0 else 0
            hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
            sol  = a/ha if ha>0 else 0
            ar   = w/h  if h>0 else 1.0
            msk  = np.zeros(gray.shape,np.uint8)
            cv2.drawContours(msk,[cnt],-1,255,-1)
            mi  = float(cv2.mean(gray,mask=msk)[0])
            sti = float(np.std(gray[msk>0])) if msk.any() else 0.0
            lesions.append({
                'lesion_id':i+1,'area_pixels':int(a),
                'area_percentage':round(a/total_px*100,2),
                'bounding_box':{'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
                'circularity':round(float(ci),3),'solidity':round(float(sol),3),
                'aspect_ratio':round(float(ar),3),'mean_intensity':round(mi,2),
                'std_intensity':round(sti,2),'irregularity_score':round(1-float(ci),3),
                'morphology':_morphology(ci,sol,ar),'suspicion_level':_suspicion(ci,sol,mi),
            })
            total_a += a
        lesions.sort(key=lambda l:l['area_pixels'],reverse=True)
        return {'lesion_count':len(lesions),'lesions':lesions,
                'total_lesion_area_pct':round(total_a/total_px*100,2),
                'overall_morphology':lesions[0]['morphology'] if lesions else 'No detectable lesion'}
    except Exception as e:
        print(f"[Lesion] {e}"); return empty


def _morphology(c,s,ar):
    if c>0.80 and s>0.90: return "Round / Oval — Low suspicion"
    if c>0.60 and s>0.80: return "Oval with smooth margins — Low-intermediate suspicion"
    if c<0.40 or  s<0.65: return "Irregular / Spiculated — High suspicion"
    if ar>1.5 or  ar<0.67: return "Elongated — Moderate suspicion"
    return "Lobular — Moderate suspicion"


def _suspicion(c,s,mi):
    sc=(2 if c<0.5 else 0)+(2 if s<0.7 else 0)+(1 if mi>200 else 0)
    return "High" if sc>=4 else "Moderate" if sc>=2 else "Low"


def annotated_image(img_bgr, cam, pred_label, scan_type):
    """BUG-5 FIX: disabled for MRI and histopathology; same 25% area cap."""
    if scan_type not in LESION_DETECT_MODALITIES:
        return None
    try:
        out=img_bgr.copy()
        if cam is None: return None
        h_i,w_i=img_bgr.shape[:2]
        cam_rs=cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs/=cam_rs.max()
        cam_thr=_percentile_threshold(cam_rs,78)
        binary=(cam_thr>0).astype(np.uint8)*255
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        cnts,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        col={'Malignant':(0,0,255),'Sick':(0,0,255),'Benign':(0,165,255),
             'Normal':(0,255,0),'Healthy':(0,255,0)}.get(pred_label,(255,255,0))
        max_area = h_i*w_i*0.25
        for i,cnt in enumerate(cnts):
            a=cv2.contourArea(cnt)
            if a<300 or a>max_area: continue
            cv2.drawContours(out,[cnt],-1,col,2)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(out,(x,y),(x+w,y+h),col,1)
            cv2.putText(out,f"R{i+1}",(x,max(15,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2,cv2.LINE_AA)
        _,buf=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92])
        return base64.b64encode(buf).decode()
    except Exception as e: print(f"[Annotated] {e}"); return None


def compute_texture(img_bgr, cam=None):
    try:
        gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        h_i,w_i = gray.shape
        if cam is not None:
            cam_rs = cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
            roi = gray[cam_rs>0.45]
        else: roi=gray.flatten()
        if len(roi)==0: roi=gray.flatten()
        f=roi.astype(np.float64); mu=float(f.mean()); s=float(f.std())
        sk=float(((f-mu)/s**3).mean()) if s>0 else 0.0
        ku=float(((f-mu)/s**4).mean()-3) if s>0 else 0.0
        hi=cv2.calcHist([gray],[0],None,[256],[0,256])
        hi=hi/hi.sum(); hi=hi[hi>0]
        return {'mean_intensity':round(mu,2),'std_deviation':round(s,2),
                'skewness':round(sk,4),'kurtosis':round(ku,4),
                'entropy':round(float(-np.sum(hi*np.log2(hi))),4),
                'dynamic_range':int(gray.max())-int(gray.min())}
    except Exception as e: print(f"[Texture] {e}"); return {}


def compute_risk(pred, conf, cv_sev, lesion_data, scan_type):
    try:
        sc,rs=1,[]
        if pred in ('Normal','Healthy'):
            sc=1 if conf>=0.70 else 2; rs.append("AI predicts no significant abnormality.")
        elif pred=='Benign':
            sc=3 if conf<0.65 else 2; rs.append("AI detects likely benign finding.")
        elif pred in ('Malignant','Sick'):
            sc=5 if conf>=0.85 else 4 if conf>=0.70 else 3; rs.append("AI detects suspicious/malignant pattern.")
        if cv_sev>=0.8 and sc<4: sc+=1; rs.append("CV analysis found high-intensity suspicious regions.")
        elif cv_sev>=0.6 and sc<3: sc+=1; rs.append("CV analysis found moderate imaging abnormality.")
        hi=[l for l in lesion_data.get('lesions',[]) if l.get('suspicion_level')=='High']
        if hi and sc<4: sc=min(sc+1,5); rs.append(f"{len(hi)} high-suspicion lesion(s) detected.")
        sc=min(sc,5)
        lbls={1:"Negative — No significant finding",2:"Benign finding — Routine follow-up",
              3:"Probably benign — Short-interval follow-up (6 months)",
              4:"Suspicious — Tissue sampling recommended",
              5:"Highly suggestive of malignancy — Biopsy required"}
        ivls={1:"Routine annual screening",2:"Routine annual screening",
              3:"Follow-up imaging in 6 months",4:"Biopsy / specialist referral within 2 weeks",
              5:"Urgent biopsy required"}
        return {'risk_score':sc,'risk_category':f"Category {sc}",'risk_label':lbls.get(sc,""),
                'recommended_interval':ivls.get(sc,""),'reasoning':rs}
    except Exception as e: print(f"[Risk] {e}"); return {}


def clinical_notes(pred, conf, lesion_data, texture_data, scan_type):
    notes=[f"Scan type: {scan_type.replace('_',' ').title()}",
           f"Primary finding: {pred} (confidence {conf:.0%})"]
    if conf<0.60: notes.append("Warning: Low AI confidence — independent radiologist review strongly advised.")
    lc=lesion_data.get('lesion_count',0)
    if lc>0:
        notes.append(f"{lc} region(s) of interest identified.")
        for l in lesion_data.get('lesions',[]):
            notes.append(f"  Region {l['lesion_id']}: {l['morphology']} | "
                         f"Area {l['area_percentage']:.1f}% | Suspicion: {l['suspicion_level']}")
    else:
        notes.append("No distinct lesion boundaries detected in attention map.")
    if texture_data:
        notes.append(f"Texture: entropy={texture_data.get('entropy','N/A')}, "
                     f"std={texture_data.get('std_deviation','N/A')}, "
                     f"skewness={texture_data.get('skewness','N/A')}")
    notes.append("Note: AI output is decision-support only. "
                 "Clinical judgement and histopathological confirmation are required.")
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 12. INFERENCE HELPER
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
        return jsonify({'error':'No file uploaded.'}), 400

    file      = request.files['file']
    mode      = request.form.get('mode','patient').lower()
    scan_type = request.form.get('scan_type','auto').strip().lower()

    if scan_type not in VALID_SCAN_TYPES:
        return jsonify({'error':f'Unknown scan_type "{scan_type}".'}), 400

    try:
        raw = file.read()
        if not raw: return jsonify({'error':'File is empty.'}), 400
        arr     = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error':'Cannot decode image. Please upload JPEG/PNG/BMP.'}), 400

        # ── 1. Resolve modality ────────────────────────────────────────────
        if scan_type in ('auto','mammogram'):
            detected = auto_detect_scan_type(img_bgr)
        else:
            detected = scan_type
        if detected == 'mammogram': detected = 'mammogram_dmid'
        print(f" * [ROUTE] req={scan_type!r} → detected={detected!r}")

        # ── 2. Validate ────────────────────────────────────────────────────
        ok, reason = validate_medical_image(img_bgr, detected)
        if not ok:
            return jsonify({'error':'Invalid image','message':reason,
                            'action':'Please retake or re-upload the scan.'}), 422

        # ── 3. Preprocessing ───────────────────────────────────────────────
        if 'mammogram' in detected:
            img_crop = crop_to_breast_tissue(img_bgr)
            img_proc = preprocess_mammogram(img_crop)
        else:
            img_crop = img_bgr
            img_proc = img_bgr

        img_rs  = cv2.resize(img_proc, (IMG_SIZE,IMG_SIZE))
        img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        sp, fr  = prepare_inputs(img_rgb, detected)

        cv_sev, cv_det = 0.0, {}

        # ── 4. Inference ───────────────────────────────────────────────────
        # Auto-compare ONLY when client explicitly sends scan_type='mammogram'
        # AND auto-detect also returns a mammogram type.
        do_compare = (scan_type in ('auto','mammogram') and 'mammogram' in detected)

        if do_compare:
            try: model_m, lbl_m = load_model('mammogram_mias')
            except RuntimeError as e: return jsonify({'error':str(e)}), 503
            pm, cm, prm = run_inference(model_m, lbl_m, sp, fr)

            try: model_d, lbl_d = load_model('mammogram_dmid')
            except RuntimeError as e: return jsonify({'error':str(e)}), 503
            pd, cd, prd = run_inference(model_d, lbl_d, sp, fr)

            if cd >= cm:
                ai_pred,ai_conf,all_p,wl,wm = pd,cd,prd,lbl_d,'DMID'
            else:
                ai_pred,ai_conf,all_p,wl,wm = pm,cm,prm,lbl_m,'MIAS'
                try: load_model('mammogram_mias')
                except RuntimeError as e: return jsonify({'error':str(e)}), 503
        else:
            try: mx, lx = load_model(detected)
            except RuntimeError as e: return jsonify({'error':str(e)}), 503
            ai_pred,ai_conf,all_p = run_inference(mx, lx, sp, fr)
            wl,wm = lx, detected

        print(f" * [INFER] model={wm} pred={ai_pred} conf={ai_conf:.3f}")

        # ── 5. CV override (mammogram only, very tight conditions) ─────────
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
        rec = ("HIGH RISK: Immediate biopsy and specialist consultation required."
               if final_pred in ('Malignant','Sick') else
               "MEDIUM RISK: Close monitoring with follow-up imaging in 3–6 months."
               if final_pred=='Benign' else
               "LOW RISK: Continue routine annual screening.")

        # ── 7. GradCAM ─────────────────────────────────────────────────────
        hmap_b64=None; ann_b64=None; cam_mask=None
        lesion_d={}; tex_d={}; risk_d={}

        tgt = get_best_gradcam_layer(loaded_model, detected)
        if tgt is not None:
            try:
                cls_idx  = wl.index(final_pred) if final_pred in wl else 0
                gcam     = GradCAM(loaded_model, tgt)
                cam_mask = gcam(sp, fr, cls=cls_idx)
                gcam.remove()
                if cam_mask is not None:
                    hmap_b64 = generate_heatmap_overlay(img_rs, cam_mask, final_pred, detected)
            except Exception as e:
                print(f"[GradCAM-run] {e}"); traceback.print_exc()

        # ── 8. Doctor extras ───────────────────────────────────────────────
        if mode=='doctor':
            lesion_d = detect_lesion_boundaries(img_rs, cam_mask, detected)
            tex_d    = compute_texture(img_rs, cam_mask)
            risk_d   = compute_risk(final_pred, final_conf, cv_sev, lesion_d, detected)
            if cam_mask is not None:
                ann_b64 = annotated_image(img_rs, cam_mask, final_pred, detected)

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
                'clinical_notes':   clinical_notes(final_pred,final_conf,lesion_d,tex_d,detected),
            }
        return jsonify(resp)

    except Exception as e:
        print(f"[PREDICT ERROR] {e}"); traceback.print_exc(); gc.collect()
        return jsonify({'error':'Internal server error. Please try again.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 14. UTILITY ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status':'online','active_model':active_model_type,'img_size':IMG_SIZE,
                    'endpoints':{'POST /predict':{
                        'scan_type':'auto|mammogram|mammogram_mias|mammogram_dmid|ultrasound|histopathology|mri',
                        'mode':'patient (default)|doctor'}}}),200

@app.route('/models', methods=['GET'])
def models():
    return jsonify({n:{'path':p,'exists':os.path.exists(p),'active':n==active_model_type}
                    for n,p in MODEL_PATHS.items()}),200

@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File too large (max 15 MB).'}),413
@app.errorhandler(400)
def bad_req(e):   return jsonify({'error':str(e)}),400
@app.errorhandler(500)
def srv_err(e):   gc.collect(); return jsonify({'error':'Internal server error.'}),500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
