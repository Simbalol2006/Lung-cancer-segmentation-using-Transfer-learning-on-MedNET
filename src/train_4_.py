import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import LIDCDataset
import model_lib 

# --- CONFIGURATION: ALL GPUs ---
# We no longer hide GPU 0. PyTorch will now see 0, 1, 2, and 3.
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16       # 4 patches per GPU x 4 GPUs = 16
LEARNING_RATE = 1e-3  
EPOCHS = 50           
PATCH_SIZE = 64       
VAL_SPLIT = 0.2       
NUM_WORKERS = 8       # Increased workers to keep 4 GPUs fed with data

# --- WEIGHTED DICE LOSS ---
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, fp_weight=1.5):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.fp_weight = fp_weight

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        
        # Ensure target has the channel dimension (B, 1, D, H, W)
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Now that model_lib.py uses F.interpolate, shapes WILL match
        predict_f = predict.reshape(-1)
        target_f = target.reshape(-1)
        
        intersection = (predict_f * target_f).sum()
        fps = (predict_f * (1 - target_f)).sum()
        
        dice_score = (2. * intersection + self.smooth) / (
            predict_f.sum() + target_f.sum() + (self.fp_weight * fps) + self.smooth
        )
        return 1 - dice_score

def calculate_dice(predict, target, smooth=1e-6):
    predict = (torch.sigmoid(predict) > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (predict * target).sum()
    return (2. * intersection + smooth) / (predict.sum() + target.sum() + smooth)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- CLUSTER STATUS ---")
    print(f"Total GPUs Found: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load Data
    full_dataset = LIDCDataset(data_dir="../processed_data/", patch_size=64)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Initialize MedicalNet (ResNet-18)
    model = model_lib.resnet18(sample_input_W=64, sample_input_H=64, 
                               sample_input_D=64, num_seg_classes=1)
    
    # Wrap for 4-GPU Parallelism
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = WeightedDiceLoss(fp_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs("checkpoints", exist_ok=True)

    print("\n--- TRAINING LAUNCHED ---")
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
            
            if batch_idx % 5 == 0: # More frequent updates for faster monitoring
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
        print(f"==> EPOCH {epoch+1} DONE | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "checkpoints/best_model_4gpu.pth")

if __name__ == "__main__":
    train()
