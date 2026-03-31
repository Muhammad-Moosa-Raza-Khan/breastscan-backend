"""
MedScan API — Production Backend  v7
All issues from v6 fixed:

  FIX-1  GradCAM invisible on histopathology
         → New per-pixel alpha blending: overlay always visible regardless of confidence.
         → Hook moved to model.proj1[0] (reliable 20x20 spatial, not deep backbone internals).
         → GaussianBlur sigma auto-scaled to image size; no hard percentile threshold.

  FIX-2  Ultrasound → mammogram misdetection (lap=20 cyst image fell through all thresholds)
         → Added local-variance-std speckle detector: dark60>30% AND lv_std>150 → ultrasound.
         → Ultrasound removed from COLOUR_MODALITIES (B-mode is genuinely grayscale).

  FIX-3  Mammogram → MRI misdetection (3 dark edges + high c_ratio)
         → Added horizontal-edge brightness guard: if bright breast tissue touches
           left OR right edge (horiz_bright>0.35), cannot be MRI.
         → MRI brain tissue is always SURROUNDED by dark skull, never touching a horizontal edge.

  FIX-4  self.aux typo kept from v5 → self.aux_classifier used consistently.

  FIX-5  Display heatmap on original image at native resolution, not on 320×320 resized crop.

  FIX-6  BI-RADS score: Malignant prediction always maps to Category ≥ 4 (not Category 3).

  FIX-7  GradCAM scattered dots on benign histo → per-pixel alpha removes dot artifacts.
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

# FIX-2: ultrasound is grayscale B-mode — remove from colour set
COLOUR_MODALITIES = {'histopathology'}   # only H&E slides are truly colour

# Modalities where contour lesion detection makes sense
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
            nn.Conv2d(ch,ch,1),                      nn.BatchNorm2d(ch), nn.ReLU(True))
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
        self.aux_classifier = nn.Linear(cd, num_classes)  # FIX-4: correct name

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
        aux     = self.aux_classifier(pooled)   # FIX-4: was self.aux → AttributeError
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
# 4. PREPROCESSING (colour-aware CLAHE)
# ─────────────────────────────────────────────────────────────────────────────

_spatial_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE,IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def prepare_inputs(img_rgb_uint8, scan_type):
    """
    Histopathology: LAB CLAHE preserves H&E colour.
    All other modalities: grayscale CLAHE (correct for mammogram, MRI, ultrasound B-mode).
    FIX-2: ultrasound no longer uses LAB CLAHE.
    """
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
# 6. MODALITY DETECTOR  v7
#
# FIX-2: Ultrasound cyst (lap very low) now detected via dark60+lv_std speckle test
# FIX-3: Mammogram not confused with MRI via horizontal-edge brightness guard
# ─────────────────────────────────────────────────────────────────────────────

def _he_pct(img_bgr):
    """H&E hue pixel ratio in centre crop. Histopathology reliably > 8%."""
    h,w  = img_bgr.shape[:2]
    crop = img_bgr[int(h*0.08):int(h*0.92), int(w*0.05):int(w*0.95)]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hu,sa = hsv[:,:,0], hsv[:,:,1]
    pink   = ((hu<=15)|(hu>=165)) & (sa>25)
    purple = (hu>=120)&(hu<=160)&(sa>25)
    return float((pink|purple).mean()*100)


def _lv_std_bright(gray, thresh=60, bs=8):
    """Std of block-wise variances in non-dark regions. High = speckle (ultrasound)."""
    h,w = gray.shape
    bv  = []
    for r in range(0, h-bs, bs):
        for c in range(0, w-bs, bs):
            p = gray[r:r+bs, c:c+bs]
            if p.mean() > thresh:
                bv.append(float(np.var(p)))
    return float(np.std(bv)) if bv else 0.0


def auto_detect_scan_type(img_bgr):
    """
    Priority chain:
      1. Histopathology — H&E hue pixel ratio > 8%
      2. MRI            — 3+ dark edges + bright centre + NOT touching horizontal edges
      3. Ultrasound     — high Laplacian OR (dark shadow + speckle heterogeneity)
      4. Mammogram      — fallback
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        h,w  = gray.shape

        # ── 1. Histopathology ──────────────────────────────────────────────
        he = _he_pct(img_bgr)
        if he > 8.0:
            print(f"   [AUTO] histopathology  H&E={he:.1f}%")
            return 'histopathology'

        # ── 2. MRI ─────────────────────────────────────────────────────────
        ew       = max(int(h*0.08), 10)
        em       = [gray[:ew,:].mean(), gray[-ew:,:].mean(),
                    gray[:,:ew].mean(), gray[:,-ew:].mean()]
        dark_n   = sum(e < 35 for e in em)
        h4,w4    = h//4, w//4
        c_mean   = float(gray[h4:3*h4, w4:3*w4].mean())
        c_ratio  = (c_mean - min(em)) / (c_mean + 1e-6)

        # FIX-3: mammogram bright tissue always touches a horizontal (left/right) edge
        left_b  = float((gray[:, :w//10] > 80).mean())
        right_b = float((gray[:, -w//10:] > 80).mean())
        touches_horiz = max(left_b, right_b) > 0.35  # breast tissue on one side

        if dark_n >= 3 and c_ratio > 0.40 and not touches_horiz:
            print(f"   [AUTO] mri  dark_n={dark_n} c_ratio={c_ratio:.2f} horiz={max(left_b,right_b):.2f}")
            return 'mri'

        # ── 3. Ultrasound ──────────────────────────────────────────────────
        lap    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark60 = float((gray < 60).mean() * 100)
        lv_std = _lv_std_bright(gray, thresh=60)

        if lap > 500:
            print(f"   [AUTO] ultrasound  lap={lap:.0f}")
            return 'ultrasound'
        if dark60 > 30 and lv_std > 150:   # cyst + speckle (FIX-2)
            print(f"   [AUTO] ultrasound  dark60={dark60:.0f}% lv_std={lv_std:.0f}")
            return 'ultrasound'
        if dark60 > 55:                    # very large dark area (anechoic cyst)
            print(f"   [AUTO] ultrasound  dark60={dark60:.0f}%")
            return 'ultrasound'

        # ── 4. Mammogram ───────────────────────────────────────────────────
        print(f"   [AUTO] mammogram_dmid  fallback lap={lap:.0f} dark60={dark60:.1f}% lv_std={lv_std:.0f}")
        return 'mammogram_dmid'

    except Exception as e:
        print(f"   [AUTO] exception: {e} → mammogram_dmid")
        return 'mammogram_dmid'


# ─────────────────────────────────────────────────────────────────────────────
# 7. VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

_BLUR_MIN = {'mammogram_mias':5,'mammogram_dmid':5,'mri':8,'ultrasound':5,'histopathology':15}
_STD_MIN  = {'mammogram_mias':4,'mammogram_dmid':4,'mri':6,'ultrasound':4,'histopathology':8}


def validate_medical_image(img_bgr, scan_type):
    try:
        h,w = img_bgr.shape[:2]
        if h<128 or w<128: return False,"Resolution too low (min 128×128)."
        if h>4096 or w>4096: return False,"Image too large (max 4096×4096)."
        gray    = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr.copy()
        mean_px = float(gray.mean())
        if mean_px<4:   return False,"Image is blank/black. Please retake the scan."
        if mean_px>251: return False,"Image is overexposed. Please retake."
        lap = float(cv2.Laplacian(gray,cv2.CV_64F).var())
        if lap<_BLUR_MIN.get(scan_type,5):
            return False,f"Image too blurry (sharpness={lap:.1f}, min={_BLUR_MIN.get(scan_type,5)})."
        if float(np.std(gray))<_STD_MIN.get(scan_type,4):
            return False,"Insufficient contrast."
        # Colour check only for grayscale modalities
        if len(img_bgr.shape)==3 and scan_type not in COLOUR_MODALITIES:
            b2,g2,r2 = [img_bgr[:,:,i].astype(float) for i in range(3)]
            ch_diff = max(float(np.mean(np.abs(r2-g2))),
                          float(np.mean(np.abs(r2-b2))),
                          float(np.mean(np.abs(g2-b2))))
            if ch_diff > 40:
                return False,(f"Does not look like a {scan_type.replace('_',' ')} scan. "
                              "Please upload a proper grayscale medical image.")
        # Screenshot guard (centre crop)
        gc2   = gray[int(h*0.05):int(h*0.95),int(w*0.05):int(w*0.95)]
        edges = cv2.Canny(gc2,50,150)
        lines = cv2.HoughLinesP(edges,1,np.pi/2,threshold=120,
                                minLineLength=int(w*0.75),maxLineGap=10)
        if lines is not None and len(lines)>6:
            return False,"Appears to be a screenshot. Please crop to the scan only."
        # Solid-colour guard
        bs=32; bv=np.array([float(np.var(gray[r*bs:(r+1)*bs,c*bs:(c+1)*bs]))
                             for r in range(h//bs) for c in range(w//bs)] or [100.])
        if bv.size>0 and (bv<5).sum()/bv.size>0.85:
            return False,"Solid colour image — not a medical scan."
        return True,None
    except Exception: return True,None


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
# 10. GRAD-CAM  (complete rewrite — FIX-1, FIX-5)
#
# Design:
#   • Hook into model.proj1[0] — a guaranteed 1×1 Conv2d that processes
#     feats[-2] (the second-to-last backbone stage, ~20×20 for 320px input).
#     This gives good spatial resolution and stable gradients.
#   • Per-pixel alpha blending: overlay intensity ∝ cam activation.
#     No threshold needed → always produces visible results.
#   • Heatmap drawn on ORIGINAL image at native resolution (FIX-5).
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM hooked into a single specified layer."""
    def __init__(self, model, layer):
        self.model = model; self.acts = None; self.grads = None
        self._h = [
            layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach())),
            layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach())),
        ]

    def remove(self): [h.remove() for h in self._h]

    def __call__(self, sp, fr, cls_idx=0):
        try:
            self.model.eval()
            logits, _ = self.model(sp, fr)
            self.model.zero_grad()
            logits[0, cls_idx].backward(retain_graph=True)
            if self.acts is None or self.grads is None: return None
            weights = self.grads.cpu().numpy()[0].mean(axis=(1,2))   # (C,)
            acts    = self.acts.cpu().numpy()[0]                      # (C,H,W)
            cam     = np.maximum((weights[:,None,None] * acts).sum(0), 0)
            if cam.max() > 0: cam /= cam.max()
            return cam   # raw spatial map, will be resized in overlay fn
        except Exception as e:
            print(f"[GradCAM.__call__] {e}"); traceback.print_exc(); return None


def get_gradcam_layer(model):
    """
    FIX-1: Hook into model.proj1[0] — the Conv2d that projects feats[-2].
    feats[-2] is the second-to-last EfficientNetV2-S stage: ~20×20 for 320px input.
    This gives clean spatial maps without the blocky artefacts of deeper 10×10 layers.
    Falls back to last Conv2d in backbone if proj1 not accessible.
    """
    try:
        # proj1 = Sequential(Conv2d, BN, ReLU) — index 0 is the Conv2d
        if hasattr(model, 'proj1') and isinstance(model.proj1[0], nn.Conv2d):
            print("   [GradCAM] using model.proj1[0]")
            return model.proj1[0]
    except Exception:
        pass
    # Fallback: last backbone conv with ≥ 64 channels
    best = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels >= 64:
            best = m
    return best


def generate_heatmap_overlay(img_bgr_display, cam, scan_type):
    """
    FIX-1 + FIX-5:
    • img_bgr_display is the ORIGINAL image at its native resolution.
    • cam is resized to match it.
    • Per-pixel alpha blending: alpha = cam_value * 0.80
      → High-activation regions show the heatmap at up to 80% intensity.
      → Zero-activation regions show the original image unchanged.
    • Always produces a visible result — no threshold needed.
    • COLORMAP_JET used universally (visible on both greyscale and pink tissue).
    """
    try:
        if cam is None: return None
        img      = img_bgr_display.copy()
        h_d,w_d  = img.shape[:2]

        # Resize CAM to display image size
        cam_rs = cv2.resize(cam, (w_d,h_d), interpolation=cv2.INTER_CUBIC)
        cam_rs = np.clip(cam_rs, 0, None)
        if cam_rs.max() > 0: cam_rs /= cam_rs.max()

        # Smooth
        ksz = max(11, int(min(h_d,w_d)*0.04) | 1)   # kernel = ~4% of image, always odd
        cam_rs = cv2.GaussianBlur(cam_rs, (ksz,ksz), 0)
        if cam_rs.max() > 0: cam_rs /= cam_rs.max()

        # Tissue mask (mammogram background → zero)
        if 'mammogram' in scan_type:
            tissue = (get_tissue_mask(img) > 0).astype(np.float32)
            cam_rs = cam_rs * tissue

        # Per-pixel alpha blend
        colored = cv2.applyColorMap(np.uint8(255*cam_rs), cv2.COLORMAP_JET)
        alpha   = (cam_rs * 0.80)[:,:,np.newaxis]          # (H,W,1), max 0.80
        overlay = (alpha * colored.astype(np.float32) +
                   (1-alpha) * img.astype(np.float32)).astype(np.uint8)

        _,buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY,95])
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[Heatmap] {e}"); traceback.print_exc(); return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. DOCTOR EXTRAS
# ─────────────────────────────────────────────────────────────────────────────

def detect_lesion_boundaries(img_bgr, cam, scan_type):
    """Only for mammogram and ultrasound. Uses p75 threshold on cam."""
    empty = {'lesion_count':0,'lesions':[],'total_lesion_area_pct':0.0,'overall_morphology':'N/A'}
    if scan_type not in LESION_MODALITIES: return empty
    try:
        if cam is None: return empty
        h_i,w_i = img_bgr.shape[:2]
        cam_rs   = cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
        if cam_rs.max()>0: cam_rs/=cam_rs.max()
        # Threshold at 75th percentile of non-zero values
        nonzero  = cam_rs[cam_rs>0]
        thr      = float(np.percentile(nonzero,75)) if len(nonzero)>0 else 0.5
        binary   = (cam_rs>thr).astype(np.uint8)*255
        binary   = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        binary   = cv2.morphologyEx(binary,cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        cnts,_   = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        gray     = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY) if len(img_bgr.shape)==3 else img_bgr
        total_px = h_i*w_i; max_a=total_px*0.25; total_a=[]; lesions=[]
        for i,cnt in enumerate(cnts):
            a=cv2.contourArea(cnt)
            if a<300 or a>max_a: continue
            x,y,w,h=cv2.boundingRect(cnt); p=cv2.arcLength(cnt,True)
            ci=(4*np.pi*a/p**2) if p>0 else 0
            hull=cv2.convexHull(cnt); ha=cv2.contourArea(hull)
            sol=a/ha if ha>0 else 0; ar=w/h if h>0 else 1.0
            msk=np.zeros(gray.shape,np.uint8); cv2.drawContours(msk,[cnt],-1,255,-1)
            mi=float(cv2.mean(gray,mask=msk)[0])
            sti=float(np.std(gray[msk>0])) if msk.any() else 0.0
            lesions.append({'lesion_id':i+1,'area_pixels':int(a),
                'area_percentage':round(a/total_px*100,2),
                'bounding_box':{'x':int(x),'y':int(y),'width':int(w),'height':int(h)},
                'circularity':round(float(ci),3),'solidity':round(float(sol),3),
                'aspect_ratio':round(float(ar),3),'mean_intensity':round(mi,2),
                'std_intensity':round(sti,2),'irregularity_score':round(1-float(ci),3),
                'morphology':_morph(ci,sol,ar),'suspicion_level':_susp(ci,sol,mi)})
            total_a.append(a)
        lesions.sort(key=lambda l:l['area_pixels'],reverse=True)
        ta=sum(total_a)
        return {'lesion_count':len(lesions),'lesions':lesions,
                'total_lesion_area_pct':round(ta/total_px*100,2),
                'overall_morphology':lesions[0]['morphology'] if lesions else 'No detectable lesion'}
    except Exception as e: print(f"[Lesion] {e}"); return empty


def _morph(c,s,ar):
    if c>0.80 and s>0.90: return "Round / Oval — Low suspicion"
    if c>0.60 and s>0.80: return "Oval with smooth margins — Low-intermediate suspicion"
    if c<0.40 or  s<0.65: return "Irregular / Spiculated — High suspicion"
    if ar>1.5  or ar<0.67: return "Elongated — Moderate suspicion"
    return "Lobular — Moderate suspicion"


def _susp(c,s,mi):
    sc=(2 if c<0.5 else 0)+(2 if s<0.7 else 0)+(1 if mi>200 else 0)
    return "High" if sc>=4 else "Moderate" if sc>=2 else "Low"


def annotated_image(img_bgr, cam, pred_label, scan_type):
    """Lesion boundary overlay — only for mammogram/ultrasound."""
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
        h_i,w_i=gray.shape
        roi=gray.flatten()
        if cam is not None:
            cam_rs=cv2.resize(cam,(w_i,h_i),interpolation=cv2.INTER_CUBIC)
            r2=gray[cam_rs>0.45]
            if len(r2)>100: roi=r2
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
    """FIX-6: Malignant always → Category ≥ 4 regardless of confidence."""
    try:
        sc,rs=1,[]
        if pred in ('Normal','Healthy'):
            sc=1 if conf>=0.70 else 2; rs.append("AI predicts no significant abnormality.")
        elif pred=='Benign':
            sc=3 if conf<0.65 else 2; rs.append("AI detects likely benign finding.")
        elif pred in ('Malignant','Sick'):
            # FIX-6: minimum Category 4 for any malignant call
            sc=5 if conf>=0.85 else 4; rs.append("AI detects suspicious/malignant pattern.")
        if cv_sev>=0.8 and sc<4: sc+=1; rs.append("CV analysis found high-intensity suspicious regions.")
        elif cv_sev>=0.6 and sc<3: sc+=1; rs.append("CV analysis found moderate imaging abnormality.")
        hi=[l for l in lesion_data.get('lesions',[]) if l.get('suspicion_level')=='High']
        if hi and sc<5: sc=min(sc+1,5); rs.append(f"{len(hi)} high-suspicion lesion(s) detected.")
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
    else: notes.append("No distinct lesion boundaries detected in attention map.")
    if texture_data:
        notes.append(f"Texture: entropy={texture_data.get('entropy','N/A')}, "
                     f"std={texture_data.get('std_deviation','N/A')}, "
                     f"skewness={texture_data.get('skewness','N/A')}")
    notes.append("Note: AI output is decision-support only. "
                 "Clinical judgement and histopathological confirmation are required.")
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

        # ── 2. Validate ────────────────────────────────────────────────────
        ok,reason = validate_medical_image(img_bgr, detected)
        if not ok:
            return jsonify({'error':'Invalid image','message':reason,
                            'action':'Please retake or re-upload the scan.'}),422

        # ── 3. Preprocessing ───────────────────────────────────────────────
        # Keep original image for display (FIX-5)
        img_display = img_bgr.copy()

        if 'mammogram' in detected:
            img_crop = crop_to_breast_tissue(img_bgr)
            img_proc = preprocess_mammogram(img_crop)
        else:
            img_crop = img_bgr
            img_proc = img_bgr

        # Resize to IMG_SIZE only for MODEL INPUT
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

        # ── 7. GradCAM (FIX-1, FIX-5) ─────────────────────────────────────
        hmap_b64=None; ann_b64=None; cam_mask=None
        lesion_d={}; tex_d={}; risk_d={}

        tgt = get_gradcam_layer(loaded_model)
        if tgt is not None:
            try:
                cls_idx  = wl.index(final_pred) if final_pred in wl else 0
                gcam     = GradCAM(loaded_model, tgt)
                cam_mask = gcam(sp, fr, cls_idx)
                gcam.remove()
                if cam_mask is not None:
                    # FIX-5: draw on ORIGINAL image at native resolution
                    hmap_b64 = generate_heatmap_overlay(img_display, cam_mask, detected)
            except Exception as e:
                print(f"[GradCAM-run] {e}"); traceback.print_exc()

        # ── 8. Doctor extras ───────────────────────────────────────────────
        if mode=='doctor':
            # Lesion detection on original-size image
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
                'clinical_notes':   clinical_notes(final_pred,final_conf,lesion_d,tex_d,detected),
            }
        return jsonify(resp)

    except Exception as e:
        print(f"[PREDICT ERROR] {e}"); traceback.print_exc(); gc.collect()
        return jsonify({'error':'Internal server error. Please try again.'}),500


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
