import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# --- PRO CONFIG ---
IMG_SIZE = 256
BATCH_SIZE = 4  # Smaller batch for better gradient updates
EPOCHS = 50  # Continue training for 50 more epochs
BAD_DIR = "data/bad_256"
CLEAN_DIR = "data/clean_256"
MODEL_PATH = "model_256_pro.pth"

# ==========================================
# IMPROVED DATASET WITH AUGMENTATION
# ==========================================
class BrainDataset(Dataset):
    def __init__(self, bad_dir, clean_dir, augment=True):
        self.bad_files = sorted([f for f in os.listdir(bad_dir) if f.endswith('.png')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
        self.bad_dir = bad_dir
        self.clean_dir = clean_dir
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        
    def __len__(self): 
        return len(self.bad_files)
    
    def __getitem__(self, idx):
        bad = Image.open(os.path.join(self.bad_dir, self.bad_files[idx])).convert("L")
        clean = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert("L")
        
        # Data augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            bad = transforms.functional.hflip(bad)
            clean = transforms.functional.hflip(clean)
        if self.augment and torch.rand(1).item() > 0.5:
            bad = transforms.functional.vflip(bad)
            clean = transforms.functional.vflip(clean)
            
        return self.transform(bad), self.transform(clean)

# ==========================================
# IMPROVED U-NET WITH RESIDUAL CONNECTIONS
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        # Encoder (with BatchNorm for stable training)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            ResBlock(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))

# Keep old UNet for backward compatibility
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1); e2 = self.enc2(p1); p2 = self.pool2(e2); b = self.bottleneck(p2)
        u1 = self.up1(b); d1 = self.dec1(torch.cat((u1, e2), dim=1)); u2 = self.up2(d1); d2 = self.dec2(torch.cat((u2, e1), dim=1))
        return torch.sigmoid(self.final(d2))

# ==========================================
# TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting Improved Training on {device}...")
    
    if not os.path.exists(BAD_DIR):
        print("❌ Error: Run setup_256.py first!")
        exit()

    dataset = BrainDataset(BAD_DIR, CLEAN_DIR, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Use the improved deeper model
    model = ImprovedUNet().to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from existing model if available
    if os.path.exists(MODEL_PATH):
        print(f"📂 Loading existing weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for bad_img, clean_img in loader:
            bad_img, clean_img = bad_img.to(device), clean_img.to(device)
            optimizer.zero_grad()
            
            output = model(bad_img)
            
            # Combined loss: L1 (for sharpness) + MSE (for stability)
            loss_l1 = l1_criterion(output, clean_img)
            loss_mse = mse_criterion(output, clean_img)
            
            # More weight on L1 for sharper results
            total_batch_loss = 0.7 * loss_l1 + 0.3 * loss_mse
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_batch_loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ⭐ BEST"
        else:
            marker = ""
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}{marker}")

    print(f"\n✅ Training Complete! Best Model Saved to {MODEL_PATH} (Loss: {best_loss:.6f})")