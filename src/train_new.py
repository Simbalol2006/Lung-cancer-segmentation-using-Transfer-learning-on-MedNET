import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import LIDCDataset
import model_lib 

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16       # 4 patches per GPU x 4 GPUs
LEARNING_RATE = 1e-3  
EPOCHS = 50           
PATCH_SIZE = 64       
NUM_WORKERS = 8       # Optimized for 4 GPUs

# --- WEIGHTED DICE LOSS ---
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, fp_weight=1.5):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.fp_weight = fp_weight

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        predict_f = predict.view(-1)
        target_f = target.view(-1)
        
        intersection = (predict_f * target_f).sum()
        fps = (predict_f * (1 - target_f)).sum()
        
        dice_score = (2. * intersection + self.smooth) / (
            predict_f.sum() + target_f.sum() + (self.fp_weight * fps) + self.smooth
        )
        return 1 - dice_score

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
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. Load Data
    full_dataset = LIDCDataset(data_dir="../processed_data/", patch_size=PATCH_SIZE)
    total_len = len(full_dataset)
    
    # 2. Create 80/10/10 Split
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len  # Remaining items
    
    print(f"\n--- DATA SPLIT (80/10/10) ---")
    print(f"Total Patients: {total_len}")
    print(f"Train: {train_len} | Val: {val_len} | Test: {test_len}")

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. Model Setup
    model = model_lib.resnet18(sample_input_W=PATCH_SIZE, sample_input_H=PATCH_SIZE, 
                               sample_input_D=PATCH_SIZE, num_seg_classes=1)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = WeightedDiceLoss(fp_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs("checkpoints", exist_ok=True)

    # 4. Training Loop
    print("\n--- TRAINING START ---")
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
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation Phase
        model.eval()
        val_dice_score = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                val_dice_score += calculate_dice(outputs, masks).item()
        
        avg_val_dice = val_dice_score / len(val_loader)
        print(f"==> EPOCH {epoch+1} DONE | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")

        # Save Best Model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "checkpoints/best_model_4gpu.pth")
            print(f"    [+] Saved Best Model (Dice: {best_val_dice:.4f})")

    # 5. Final Test Phase
    print("\n--- FINAL TEST ON HELD-OUT DATA ---")
    print("Loading best model for testing...")
    model.load_state_dict(torch.load("checkpoints/best_model_4gpu.pth"))
    model.eval()
    
    test_dice_score = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            dice = calculate_dice(outputs, masks).item()
            test_dice_score += dice
            
    avg_test_dice = test_dice_score / len(test_loader)
    print(f"FINAL TEST RESULT (Dice Score): {avg_test_dice:.4f}")
    print("-----------------------------------")

if __name__ == "__main__":
    train()
