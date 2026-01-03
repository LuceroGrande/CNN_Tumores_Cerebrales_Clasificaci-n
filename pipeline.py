import os
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
from skimage.measure import marching_cubes
from tensorflow.keras.models import load_model

SEG_MODEL_PATH = "segmentation_model_nopr.pth"
CLS_MODEL_PATH = "classifier_tumor.h5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Cargando modelo de clasificación...")
cls_model = load_model(CLS_MODEL_PATH)

LABELS_MAP = {
    0: "Normal",
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}

# MODELO DE SEGMENTACIÓN
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(3, 16)
        self.enc2 = self.block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = self.block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = torch.nn.functional.interpolate(
            e2, scale_factor=2, mode="bilinear", align_corners=False
        )
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))


print("Cargando modelo de segmentación...")
seg_model = UNet().to(DEVICE)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
seg_model.eval()

def load_mat(path):
    with h5py.File(path, "r") as f:
        img = np.array(f["cjdata"]["image"])
    return img

def preprocess_seg(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = cv2.resize(img, (256, 256))
    img = np.stack([img]*3, axis=0)  # (3,H,W)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def preprocess_cls(img):
    img = cv2.resize(img, (128, 128))
    img = img / img.max()
    return img.reshape(1, 128, 128, 1)

def run_pipeline(folder):
    files = sorted([
        os.path.join(r, f)
        for r, _, fs in os.walk(folder)
        for f in fs if f.endswith(".mat")
    ])

    if not files:
        raise Exception("No se encontraron archivos .mat")

    volume = []
    masks = []
    preds = []

    for f in files:
        img = load_mat(f)
        volume.append(img)

        # --- SEGMENTACIÓN ---
        with torch.no_grad():
            t = preprocess_seg(img).to(DEVICE)
            mask = seg_model(t)[0, 0].cpu().numpy()
            masks.append(mask)

        # --- CLASIFICACIÓN ---
        p = cls_model.predict(preprocess_cls(img), verbose=0)
        preds.append(np.argmax(p))

    volume = np.stack(volume)
    masks = np.stack(masks)

    # Etiqueta final
    label_idx = max(set(preds), key=preds.count)
    label = LABELS_MAP[label_idx]

    # Confianzas
    confidences = {
        LABELS_MAP[k]: preds.count(k)/len(preds)
        for k in LABELS_MAP
    }

    # 3D
    verts, faces, _, _ = marching_cubes(masks, level=0.5)

    return volume, masks, verts, faces, label, confidences
