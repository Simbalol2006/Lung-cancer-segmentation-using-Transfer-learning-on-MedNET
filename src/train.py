import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import LIDCDataset
import model_lib 

# --- HYPERPARAMETERS (Phase 3 Optimized) --- 
BATCH_SIZE = 64        # Full 4-GPU Capacity
LEARNING_RATE = 5e-4   # Adjusted for Focal Loss stability
EPOCHS = 50           
PATCH_SIZE = 64       
NUM_WORKERS = 4       

# --- FOCAL + DICE COMBO LOSS ---
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, focal_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight

    def forward(self, predict, target):
        if target.dim() == 4: target = target.unsqueeze(1)
        
        # 1. Focal Loss (Combats severe class imbalance)
        bce_loss = F.binary_cross_entropy_with_logits(predict, target.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # 2. Dice Loss (Refines spatial boundaries)
        predict_flat = torch.sigmoid(predict).view(-1)
        target_flat = target.view(-1)
        intersection = (predict_flat * target_flat).sum()
        dice_score = (2. * intersection + 1e-6) / (predict_flat.sum() + target_flat.sum() + 1e-6)
        dice_loss = 1 - dice_score

        # Combine
        return (self.focal_weight * focal_loss) + ((1 - self.focal_weight) * dice_loss)

# --- METRIC HELPER ---
def calculate_dice(predict, target, smooth=1e-6):
    predict = (torch.sigmoid(predict) > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (predict * target).sum()
    return (2. * intersection + smooth) / (predict.sum() + target.sum() + smooth)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- CLUSTER STATUS ---")
    print(f"Total GPUs: {torch.cuda.device_count()}")
    
    # 1. Load Data with 50/50 Sampling
    full_dataset = LIDCDataset(data_dir="../processed_data/", patch_size=PATCH_SIZE, p_positive=0.5)
    total_len = len(full_dataset)
    
    # 80/10/10 Split
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n--- CONFIGURATION ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Loss Function: Focal (0.3) + Dice (0.7)")
    print(f"Sampling Strategy: 50% Nodule Centered / 50% Random")

    # 2. Initialize Model (Make sure shortcut_type='A' is set in model_lib.py)
    model = model_lib.resnet18(sample_input_W=PATCH_SIZE, sample_input_H=PATCH_SIZE, 
                               sample_input_D=PATCH_SIZE, num_seg_classes=1)
    
    # 3A. LOAD MED3D PRE-TRAINED WEIGHTS (The Warm Start Foundation)
    med3d_weights_path = "../pretrain/resnet_18_23dataset.pth"
    if os.path.exists(med3d_weights_path):
        print(f"    [+] Loading Med3D (23-Dataset) Universal Features from: {med3d_weights_path}")
        med3d_state = torch.load(med3d_weights_path, map_location=device)
        # strict=False is mandatory to allow our custom U-Net decoder to randomly initialize
        model.load_state_dict(med3d_state['state_dict'], strict=False)
    else:
        print(f"    [!] WARNING: Med3D weights not found at {med3d_weights_path}. Training from scratch!")

    # 3B. Load Your Custom Checkpoint (If resuming a crash)
    checkpoint_path = "checkpoints/best_model_4gpu.pth"
    if os.path.exists(checkpoint_path):
        print(f"    [+] Resuming custom training from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    # 4. Wrap for 4-GPU Parallelism
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 5. Optimizer & Scheduler setup
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Automatically halves the learning rate if Validation Dice doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    os.makedirs("checkpoints", exist_ok=True)

    # 6. Training Loop
    print("\n--- PHASE 3 TRAINING START ---")
    best_val_dice = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_dice_score = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                val_dice_score += calculate_dice(outputs, masks).item()
        
        avg_val_dice = val_dice_score / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"==> EPOCH {epoch+1} DONE | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        # Step the scheduler based on Validation Dice
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "checkpoints/best_model_4gpu.pth")
            print(f"    [+] Saved Best Model (Dice: {best_val_dice:.4f})")

if __name__ == "__main__":
    train()
