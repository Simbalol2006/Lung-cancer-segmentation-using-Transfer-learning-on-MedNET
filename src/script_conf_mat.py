import torch
import numpy as np
import model_lib
from dataset import LIDCDataset
from torch.utils.data import DataLoader

def get_confusion_matrix_elements(predict, target, threshold=0.5):
    """
    Calculates the components of the confusion matrix for 3D volumes.
    """
    # Apply threshold to sigmoid output to get binary prediction
    preds = (torch.sigmoid(predict) > threshold).float()
    target = target.float()

    # Calculate elements
    tp = torch.sum(preds * target).item()              # True Positive: Predicted Nodule, Is Nodule
    fp = torch.sum(preds * (1 - target)).item()        # False Positive: Predicted Nodule, Is Air/Tissue
    fn = torch.sum((1 - preds) * target).item()        # False Negative: Predicted Air, Is Nodule
    tn = torch.sum((1 - preds) * (1 - target)).item()  # True Negative: Predicted Air, Is Air

    return tp, fp, fn, tn

def evaluate_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = model_lib.resnet18(sample_input_W=64, sample_input_H=64, 
                               sample_input_D=64, num_seg_classes=1)
    
    # Load your best weights from Phase 3
    checkpoint_path = "checkpoints/best_model_4gpu.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Clean DataParallel keys if necessary
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    # 2. Load Validation Data
    val_dataset = LIDCDataset(data_dir="../processed_data/", patch_size=64, p_positive=0.5)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    print("Analyzing Validation Set...")
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            tp, fp, fn, tn = get_confusion_matrix_elements(outputs, masks)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    # 3. Calculate Derived Metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6) # Sensitivity
    specificity = total_tn / (total_tn + total_fp + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "="*30)
    print("   3D CONFUSION MATRIX RESULTS   ")
    print("="*30)
    print(f"True Positives (TP):  {int(total_tp):,}")
    print(f"False Positives (FP): {int(total_fp):,}")
    print(f"False Negatives (FN): {int(total_fn):,}")
    print(f"True Negatives (TN):  {int(total_tn):,}")
    print("-" * 30)
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f} (Sensitivity)")
    print(f"Specificity: {specificity:.4f}")
    print(f"Dice (F1):   {f1_score:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_performance()
