import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import random
import scipy.ndimage

class LIDCDataset(Dataset):
    def __init__(self, data_dir, patch_size=64, p_positive=0.5):
        self.image_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.patch_size = patch_size
        self.p_positive = p_positive 
        
        # 1. Load all files
        self.filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        
        # --- EMERGENCY FIX ---
        original_count = len(self.filenames)
        self.filenames = [f for f in self.filenames if "0965" not in f]
        
        if len(self.filenames) < original_count:
            print(f"Warning: Removed broken patient LIDC-IDRI-0965. Remaining: {len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load the 3D volumes
        img_path = self.image_dir / self.filenames[idx]
        mask_path = self.mask_dir / self.filenames[idx]
        
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        d, h, w = img.shape
        p = self.patch_size // 2

        # --- 50/50 PROBABILISTIC SAMPLING STRATEGY ---
        has_nodule = np.sum(mask) > 0
        
        if has_nodule and (random.random() < self.p_positive):
            # STRATEGY A: SMART CROP (Find the nodule)
            non_zeros = np.argwhere(mask > 0)
            target_voxel = non_zeros[random.randint(0, len(non_zeros) - 1)]
            cz, cy, cx = target_voxel[0], target_voxel[1], target_voxel[2]
        else:
            # STRATEGY B: RANDOM CROP (Background / Negative)
            if d <= self.patch_size: cz = d // 2
            else: cz = random.randint(p, d - p)
            
            if h <= self.patch_size: cy = h // 2
            else: cy = random.randint(p, h - p)
            
            if w <= self.patch_size: cx = w // 2
            else: cx = random.randint(p, w - p)

        # Ensure coordinates are within bounds (Double Safety)
        cz = max(p, min(cz, d - p))
        cy = max(p, min(cy, h - p))
        cx = max(p, min(cx, w - p))

        # Extract the patch (use .copy() to prevent negative stride issues in PyTorch)
        img_patch = img[cz-p:cz+p, cy-p:cy+p, cx-p:cx+p].copy()
        mask_patch = mask[cz-p:cz+p, cy-p:cy+p, cx-p:cx+p].copy()

        # --- 3D DATA AUGMENTATION ---
        # Apply random spatial transformations to prevent orientation memorization
        if random.random() < 0.5:
            # Random Horizontal, Vertical, or Depth Flip
            axis = random.choice([(0,), (1,), (2,)])
            img_patch = np.flip(img_patch, axis=axis).copy()
            mask_patch = np.flip(mask_patch, axis=axis).copy()
            
        if random.random() < 0.3:
            # Random 3D Rotation (±10 degrees)
            angle = random.uniform(-10, 10)
            img_patch = scipy.ndimage.rotate(img_patch, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
            mask_patch = scipy.ndimage.rotate(mask_patch, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')

        # --- FIX: Force contiguous memory to prevent PyTorch TypeError ---
        img_patch = np.ascontiguousarray(img_patch)
        mask_patch = np.ascontiguousarray(mask_patch)

        # Convert to tensors
        # Using torch.tensor() bypasses the strict memory layout checks of from_numpy()
        img_t = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask_patch, dtype=torch.long)
        
        return img_t, mask_t