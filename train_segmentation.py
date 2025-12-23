# train_segmentation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os

DATA_X = "X.npy"   # (N,256,256,1)
DATA_Y = "Y.npy"   # (N,256,256,1)
MODEL_DIR = "models"

# -----------------------------
# 1. Dataset
# -----------------------------
class SegDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images      # (N,3,H,W)
        self.masks = masks        # (N,1,H,W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# -----------------------------
# 2. Simple U-Net
# -----------------------------
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
        d1 = torch.nn.functional.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))

# -----------------------------
# 3. Training + SAVE MODEL
# -----------------------------
def train_segmentation(train_images, train_masks, epochs=10, batch_size=4, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    dataset = SegDataset(train_images, train_masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(loader):.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(model.state_dict(), f"{MODEL_DIR}/segmentation_model_nopr.pth")
    torch.save(model, f"{MODEL_DIR}/segmentation_model_full_nopr.pth")

    print("Model saved to:", MODEL_DIR)

    return model


# -----------------------------
# MAIN SCRIPT ENTRY
# -----------------------------
if __name__ == "__main__":
    X = np.load(DATA_X)
    Y = np.load(DATA_Y)

    X = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0,3,1,2)

    # expand grayscale to 3 channels
    if X.shape[1] == 1:
        X = X.repeat(1, 3, 1, 1)

    # ---- FIX: crop from 256x512 -> 256x256 ----
    _, _, H, W = X.shape
    start = (W - 256) // 2
    end = start + 256
    X = X[:, :, :, start:end]   # now X is (N,3,256,256)

    print("Loaded:")
    print("X:", X.shape)
    print("Y:", Y.shape)


    # Train model
    train_segmentation(X, Y, epochs=10, batch_size=4, lr=1e-3)
