#!/usr/bin/env python3
"""Compute tLPIPS, warp-L1, FID for Wild V2 outputs (no-reference metrics)."""
import os, sys, json, numpy as np
from PIL import Image
import torch
import cv2

sys.path.insert(0, '/home/user/VSR_Project')
os.chdir('/home/user/VSR_Project')

DEVICE = torch.device("cuda:0")

RESULTS = {
    'bicubic': 'results/wild2_bicubic',
    'basicvsr_pp': 'results/wild2_basicvsr_pp',
    'realesrgan': 'results/wild2_realesrgan',
    'c_hybrid': 'results/part3_wild2_C_hybrid_g0.3',
}

def list_frames(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.png', '.jpg', '.jpeg'))])

def to_float_rgb(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

def compute_tlpips(frames, lpips_model, device):
    """temporal LPIPS between consecutive frames."""
    vals = []
    prev = None
    for f in frames:
        img = to_float_rgb(f).astype(np.float32)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        if prev is not None:
            with torch.no_grad():
                vals.append(float(lpips_model(prev, t).mean().item()))
        prev = t
    return float(np.mean(vals)) if vals else 0.0

def compute_warp_l1(frames):
    """Warp-L1: Farneback flow from t→t+1, warp t+1 to t, L1 error."""
    vals = []
    prev_rgb = None
    prev_gray = None
    for f in frames:
        rgb = to_float_rgb(f)
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # flow[..., 0] = dx, flow[..., 1] = dy
            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = x_coords + flow[..., 0]
            map_y = y_coords + flow[..., 1]
            warped = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            l1 = float(np.mean(np.abs(prev_rgb - warped)))
            vals.append(l1)
        prev_gray = gray
        prev_rgb = rgb
    return float(np.mean(vals)) if vals else 0.0

def compute_fid(act1, act2):
    """FID between two activation matrices (N x D)."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

print("Loading models...")
import lpips
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

from torchvision.models import inception_v3
inception = inception_v3(pretrained=True, transform_input=False).to(DEVICE).eval()

def extract_inception_features(frames, batch_size=16):
    """Extract 2048-d InceptionV3 features."""
    features = []
    for i in range(0, len(frames), batch_size):
        batch_paths = frames[i:i+batch_size]
        batch = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB').resize((299, 299), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) * 2.0
            batch.append(arr)
        t = torch.from_numpy(np.stack(batch, axis=0)).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad():
            feat = inception(t)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

print("Models loaded.")

# Get bicubic features for FID reference
print("Extracting bicubic Inception features...")
bicubic_frames = list_frames(RESULTS['bicubic'])
bicubic_feat = extract_inception_features(bicubic_frames)

results = {}
for name, path in RESULTS.items():
    frames = list_frames(path)
    if not frames:
        print(f"[{name}] No frames found in {path}, skipping")
        continue
    print(f"[{name}] {len(frames)} frames")

    tlpips_val = compute_tlpips(frames, lpips_model, DEVICE)
    print(f"  tLPIPS = {tlpips_val:.4f}")

    warp_val = compute_warp_l1(frames)
    print(f"  warp-L1 = {warp_val:.4f}")

    if name == 'bicubic':
        fid_val = 0.0
    else:
        feat = extract_inception_features(frames)
        fid_val = compute_fid(feat, bicubic_feat)
    print(f"  FID = {fid_val:.2f}")

    results[name] = {
        'tlpips': round(tlpips_val, 4),
        'warp_l1': round(warp_val, 4),
        'fid': round(fid_val, 1),
    }

output_path = 'results/eval_wild2.json'
json.dump(results, open(output_path, 'w'), indent=2)
print(f"\nSaved to {output_path}")

# Print summary for report
print("\n=== Report-ready values ===")
for name, vals in results.items():
    print(f"{name}: tLPIPS={vals['tlpips']:.4f}, warp-L1={vals['warp_l1']:.4f}, FID={vals['fid']:.1f}")
