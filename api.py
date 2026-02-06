from flask import Flask, request, send_file
import cv2
import numpy as np
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# --- CONFIG ---
IMG_SIZE = 256
MODEL_PATH = "model_256_pro.pth"
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'final_output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# Improved U-Net Model Architecture (must match training)
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
        # Encoder
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

# ==========================================
# Load PyTorch Model at Startup
# ==========================================
device = torch.device("cpu")  # Force CPU for compatibility
model = ImprovedUNet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️ WARNING: Model file '{MODEL_PATH}' not found! Using untrained model.")

# ==========================================
# Post-Processing: CLAHE + Unsharp Masking
# ==========================================
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def apply_unsharp_mask(img, sigma=1.0, strength=1.5):
    """Apply Unsharp Masking to enhance edges"""
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def post_process(img):
    """Full post-processing pipeline for demo sharpness"""
    # Step 1: CLAHE for contrast enhancement
    img = apply_clahe(img, clip_limit=2.5, tile_grid_size=(8, 8))
    # Step 2: Unsharp masking for edge sharpness
    img = apply_unsharp_mask(img, sigma=1.0, strength=1.2)
    return img

# ==========================================
# Flask Route
# ==========================================
@app.route('/restore', methods=['POST'])
def restore():
    print("LOG: Request received...")
    
    if 'image' not in request.files:
        return "No image provided", 400
    file = request.files['image']
    
    # 1. Read Image as Grayscale
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return "Invalid image file", 400
    
    original_size = img.shape[:2]  # (height, width)
    print(f"LOG: Input image size: {original_size}")
    
    # 2. Preprocess: Resize to 256x256 and Normalize to [0, 1]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 3. Convert to PyTorch Tensor: (1, 1, 256, 256) - batch, channel, H, W
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Run Model Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 5. Denormalize: Convert back to [0, 255] uint8
    output_np = output_tensor.squeeze().cpu().numpy()  # Remove batch & channel dims
    output_img = (output_np * 255.0).clip(0, 255).astype(np.uint8)
    
    # 6. Resize back to original size if needed
    if original_size != (IMG_SIZE, IMG_SIZE):
        output_img = cv2.resize(output_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # 7. Apply Post-Processing (CLAHE + Unsharp Masking)
    processed_img = post_process(output_img)
    print("LOG: Post-processing applied (CLAHE + Unsharp Mask)")

    # 8. Save to Output Folder
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    output_filename = f"restored_{timestamp}.png"
    save_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(save_path, processed_img)
    
    print(f"SUCCESS: Saved image to {save_path}")
    
    # 9. Return to n8n
    return send_file(save_path, mimetype='image/png')

if __name__ == '__main__':
    print(f"--- MRI RESTORATION SERVER READY ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Saving images to: {OUTPUT_FOLDER}")
    app.run(host='0.0.0.0', port=5000)