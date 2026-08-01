import numpy as np
import rasterio
import torch

def load_normalization_stats(mean_path, std_path):
    band_mean = np.load(mean_path).reshape(12, 1, 1)
    band_std = np.load(std_path).reshape(12, 1, 1)
    return band_mean, band_std

def preprocess_tif(file_path, band_mean, band_std):
    with rasterio.open(file_path) as src:
        img = src.read().astype(np.float32)  # (12, H, W)

    if img.shape != (12, 128, 128):
        raise ValueError(f"Expected shape (12,128,128), got {img.shape}")

    img = (img - band_mean) / (band_std + 1e-8)
    tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, 12, 128, 128)
    return tensor

def predict_mask(model, tensor, device, threshold=0.5):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255  # (128,128), 0/255
    return mask_np