
import os
import numpy as np
from tqdm import tqdm               
import random
import matplotlib.pyplot as plt
import torch
from Config import ModelType         
import time
from monai.losses import DiceLoss    
from torch import nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A          
from NUH_Dataset import Dataset      
from collections import defaultdict
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import hausdorff_distance
from scipy.ndimage import binary_erosion
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import cdist


# Get dataset mean and std 
def get_mean_std(npz_dir):
    """
    Compute mean and standard deviation of all images in a dataset (NPZ format).
    Used for normalization.
    """
    sum_pixels = 0.0
    sum_squared = 0.0
    n_pixels = 0

    for file in tqdm(os.listdir(npz_dir)):
        if file.endswith(".npz"):
            data = np.load(os.path.join(npz_dir, file))
            image = data["image"].astype(np.float32)

            # Normalize pixel values to [0,1] 
            if image.max() > 1:
                image = image / 255.0

            # Accumulate sums for mean and variance
            sum_pixels += image.sum()
            sum_squared += (image ** 2).sum()
            n_pixels += image.size

    # Mean and std calculation
    mean = sum_pixels / n_pixels
    std = np.sqrt((sum_squared / n_pixels) - (mean ** 2))

    return [round(mean, 4)], [round(std, 4)]


# Visualize scar and myocardium overlay
def visualize_scar_samples(test_loader, n=5):
    """
    Visualize n random samples from the test set.
    Shows MRI image with myocardium (yellow) and scar (red) overlays.
    """
    images_with_scar = []

    # Collect all samples that contain scar
    for images, masks, myocardium_masks in test_loader:
        for img, scar_mask, myo_mask in zip(images, masks, myocardium_masks):
            if scar_mask.sum() > 0:  # keep only samples with scar
                images_with_scar.append((img, scar_mask, myo_mask))

    if len(images_with_scar) == 0:
        print("No samples with scar found in test set.")
        return

    # Randomly pick up to n samples
    selected_samples = random.sample(images_with_scar, min(n, len(images_with_scar)))

    for i, (image, scar_mask, myo_mask) in enumerate(selected_samples):
        image_np = image.squeeze().cpu().numpy()
        scar_np = scar_mask.squeeze().cpu().numpy()
        myo_np = myo_mask.squeeze().cpu().numpy()

        # Create RGB overlay: yellow = myocardium, red = scar
        overlay = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.float32)
        overlay[myo_np > 0] = [1, 1, 0]  # myocardium
        overlay[scar_np > 0] = [1, 0, 0]  # scar

        plt.figure(figsize=(8, 4))

        # MRI image
        plt.subplot(1, 2, 1)
        plt.imshow(image_np, cmap='gray')
        plt.title("MRI Image")
        plt.axis("off")

        # Scar and myocardium overlay
        plt.subplot(1, 2, 2)
        plt.imshow(image_np, cmap='gray')
        plt.imshow(overlay, alpha=0.5)
        plt.title("Myocardium (yellow) + Scar (red)")
        plt.axis("off")

        plt.suptitle(f"Sample {i+1}", fontsize=12)
        plt.tight_layout()
        plt.show()


# Data Loading and Augmentation 
def load_data(dataset_folder):
    """
    Load train/val/test datasets with augmentations and normalization.
    """
    mean, std = get_mean_std(f"{dataset_folder}/train")

    # Augmentations for training
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0.05, rotate_limit=15,
            interpolation=1, border_mode=0, p=0.3
        ),
        A.GaussianBlur(blur_limit=3, sigma_limit=(0.1, 0.5), p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.RandomGamma(gamma_limit=(90,110), p=0.2),
        A.ElasticTransform(alpha=5, sigma=10, p=0.05),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'scar_mask': 'mask', 'lv_mask': 'mask'})

    # Validation a test data preprocessing 
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'scar_mask': 'mask', 'lv_mask': 'mask'})

    # Dataset splits
    mr_train = Dataset(f"{dataset_folder}/train", transform=train_transform)
    mr_val = Dataset(f"{dataset_folder}/val", transform=test_transform)
    mr_test = Dataset(f"{dataset_folder}/test", transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(mr_train, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(mr_val, batch_size=16, num_workers=4)
    test_loader = DataLoader(mr_test, batch_size=1, num_workers=4)

    return train_loader, val_loader, test_loader


# Reproducibility 
def set_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Metrics
def dice_score(pred, target):
    """
    Compute Dice score between prediction and ground truth mask.
    Thresholded at 0.5 after sigmoid.
    """
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    smooth = 1e-5
    intersect = (pred & target).float().sum((1,2,3))
    union = pred.float().sum((1,2,3)) + target.float().sum((1,2,3))
    return ((2 * intersect + smooth) / (union + smooth)).mean()


# Custom Losses
def myocardium_constrained_loss(scar_pred, lv_mask):
    """
    Penalise scar predictions outside myocardium region.
    """
    scar_pred = torch.sigmoid(scar_pred)
    outside_lv = (1 - lv_mask).float()
    penalty = scar_pred * outside_lv
    return penalty.mean()


def deep_supervision_loss(bce, dice, final_out, ds_out3, target, lv_mask, weights):
    """
    Combined loss = BCE + Dice + anatomical constraint with deep supervision.
    """
    # Final output loss
    loss_final = bce(final_out, target) + dice(final_out, target)

    # Deep supervision loss
    loss_ds3 = bce(ds_out3, target) + dice(ds_out3, target)

    # Constraint loss that penalises scar outside myocardium
    constraint_loss = myocardium_constrained_loss(final_out, lv_mask)

    # Weighted sum
    total_loss = (
        weights[0] * loss_final +
        weights[1] * loss_ds3 +
        weights[2] * constraint_loss
    )
    return total_loss


def get_deep_suv_weights(model_type, epoch):
    """
    Select weights for deep supervision depending on model type and epoch.
    """
    if model_type == ModelType.NDS:
        return (1.0, 0.0, 1.0)
    if model_type == ModelType.NNL:
        return ((1.0, 0.0, 0.0) if epoch < (epoch // 2) else (0.7, 0.3, 0.0))
    else:
        return ((1.0, 0.0, 1.0) if epoch < (epoch // 2) else (0.7, 0.3, 1.0))


# Training 
def train_validate(model, train_loader, val_loader, device, epochs, optimizer, scheduler, chkpt_folder, seed=42):
    """
    Train and validate the model.
    """
    set_seed(seed)

    bce = nn.BCEWithLogitsLoss() 
    dice = DiceLoss(sigmoid=True)

    valid_losses, valid_dices, train_losses = [], [], []
    best_dice = 0
    patience_counter = 0
    start_time = time.time()

    checkpoint = f"{chkpt_folder}/{model.model_type.value}_model.pth"

    # Epoch loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training loop 
        for i, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")):
            imgs, masks, lv_masks = batch
            imgs = imgs.to(device).float()
            masks = masks.to(device).float().permute(0, 3, 1, 2)
            lv_masks = lv_masks.to(device).float().permute(0, 3, 1, 2)

            # Forward pass 
            final_out, ds_out3 = model(imgs)

            # Loss
            loss = deep_supervision_loss(bce, dice, final_out, ds_out3, masks, lv_masks, get_deep_suv_weights(model.model_type, epoch))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            scheduler.step(epoch + i / len(train_loader))  # LR scheduling

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop 
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs, masks, lv_masks = batch
                imgs = imgs.to(device).float()
                masks = masks.to(device).float().permute(0, 3, 1, 2)
                lv_masks = lv_masks.to(device).float().permute(0, 3, 1, 2)

                final_out, ds_out3 = model(imgs)
                loss = deep_supervision_loss(bce, dice, final_out, ds_out3, masks, lv_masks, get_deep_suv_weights(model.model_type, epoch))

                val_loss += loss.item()
                val_dice += dice_score(final_out, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # Log metrics
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Dice={avg_val_dice:.4f}")

        valid_dices.append(avg_val_dice)
        valid_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)

        # Save best checkpoint
        if avg_val_dice > best_dice:  
            best_dice = avg_val_dice
            torch.save(model.state_dict(), checkpoint)


    # Training duration
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration / 60:.2f} minutes.")

    # Plots
    plt.figure(figsize=(8, 6))
    plt.plot(valid_dices, label='Validation Dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice Score'); plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return checkpoint





def dice_score_test(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice



def compute_hd95(pred, target):
    """Compute the 95th percentile Hausdorff Distance (HD95) in pixels."""
    pred_bin = (pred.squeeze().numpy() > 0.5).astype(np.uint8)
    target_bin = (target.squeeze().numpy() > 0.5).astype(np.uint8)

    # Skip if either mask is empty
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return 0.0  # treat as 0 or skip; NaN will inflate averages

    # Get coordinates of boundary points
    pred_surface = pred_bin ^ binary_erosion(pred_bin)
    target_surface = target_bin ^ binary_erosion(target_bin)
    pred_pts = np.argwhere(pred_surface)
    target_pts = np.argwhere(target_surface)

    # Compute pairwise distances
    dists_pred_to_target = cdist(pred_pts, target_pts)
    dists_target_to_pred = cdist(target_pts, pred_pts)

    # Directed Hausdorff distances (percentile)
    hd95 = np.percentile(np.hstack((dists_pred_to_target.min(axis=1),
                                    dists_target_to_pred.min(axis=1))), 95)
    return hd95





# ---------- Test Function ----------

def test(model, test_loader, device, patient_slices):
    """
    Evaluate model on test set using Dice, HD95, and ASD per patient.
    Then compute the mean and std across patients.
    """
    model.eval()
    all_preds, all_masks = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs, masks, _ = batch
            imgs = imgs.to(device).float()
            masks = masks.to(device).float().permute(0, 3, 1, 2)

            preds, _ = model(imgs)
            preds = torch.sigmoid(preds)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Per-patient evaluation
    patient_dice, patient_hd95 = [], []

    for pid, (start_idx, end_idx) in patient_slices.items():
        preds_p = all_preds[start_idx:end_idx + 1]
        masks_p = all_masks[start_idx:end_idx + 1]

        dice_slices, hd95_slices = [], []
        for i in range(len(preds_p)):
            d = dice_score_test(preds_p[i], masks_p[i]).item()
            h = compute_hd95(preds_p[i], masks_p[i])
            if not np.isnan(h): hd95_slices.append(h)
            dice_slices.append(d)

        patient_dice.append(np.nanmean(dice_slices))
        patient_hd95.append(np.nanmean(hd95_slices))

    return (
        np.mean(patient_dice), np.std(patient_dice),
        np.mean(patient_hd95), np.std(patient_hd95)
    )






def visualize_predictions_baseline(model, loader, device, save_dir):
    """
    Visualize predictions: MRI + GT (red) + Pred (cyan).
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    slice_idx = 0

    with torch.no_grad():
        for batch in loader:
            imgs, masks, _ = batch
            imgs = imgs.float().to(device)               # (B, C, H, W)
            masks = masks.float().to(device).permute(0, 3, 1, 2)  # (B, 1, H, W)

            preds, _ = model(imgs)

            # Dice per batch (torch)
            dice_batch = dice_score(preds, masks).item()

            # Threshold predictions for visualization
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            B = imgs.shape[0]
            for i in range(B):
                img = imgs[i,0].cpu().numpy()
                gt  = masks[i,0].cpu().numpy()
                pr  = preds[i,0].cpu().numpy()

                # Make a figure
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # MRI only
                axs[0].imshow(img, cmap="gray")
                axs[0].axis("off")

                # MRI + overlays
                axs[1].imshow(img, cmap="gray")

                h, w = img.shape
                gt_overlay = np.zeros((h, w, 4))
                pr_overlay = np.zeros((h, w, 4))

                # GT = red
                gt_overlay[gt > 0.5] = [1, 0, 0, 0.7]
                # Pred = cyan
                pr_overlay[pr > 0.5] = [0, 1, 1, 0.6]

                axs[1].imshow(gt_overlay)
                axs[1].imshow(pr_overlay)

                axs[1].set_title(f"Dice = {dice_batch:.3f}", fontsize=20, fontweight="bold")
                axs[1].axis("off")

                # Save with Pos_ prefix if prediction has scar
                prefix = "Pos_" if pr.sum() > 0 else ""
                save_path = os.path.join(save_dir, f"{prefix}slice_{slice_idx:03d}.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)

                plt.show()

                slice_idx += 1



 