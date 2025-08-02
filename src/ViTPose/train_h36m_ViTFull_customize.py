import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ViTPoseFull_customize import ViTPoseFull
from H36MPoseDataset import Human36mDataset

#TODO: lower the dim of attention map to test simple ViTPose first

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json

# Load config
with open("config_train_vitpose.json", "r") as f:
    config = json.load(f)
# Extract basic parameters
IMG_SIZE = tuple(config["IMG_SIZE"])
NUM_KEYPOINTS = config["NUM_KEYPOINTS"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
NUM_FRAMES = config["NUM_FRAMES"]
EMBED_DIM = config["EMBED_DIM"]
DEPTH = config["DEPTH"]
MLP_RATIO = config["MLP_RATIO"]
PATCH = config["PATCH"]
HEAD = config["HEAD"]
PATIENCE = config["PATIENCE"]
HEATMAP_SIZE = config["HEATMAP_SIZE"]
VIT_MODEL = config["VIT_MODEL"]
HEATMAP_SIZE = None

# Compute derived parameters
TOKEN = IMG_SIZE[0] * IMG_SIZE[1] // (PATCH * PATCH)
SAVE_PATH = f"vitpose_full_h36m(S1_S9)(head:{HEAD}, token:{TOKEN}, depth:{DEPTH}).pth"
LOG_PATH = f"training_log(head:{HEAD}, token:{TOKEN}, depth:{DEPTH}).txt"


# --- Ground truth Heatmap Generator ---
def generate_heatmaps_from_2d(joints, out_size, sigma=2):
    B, J, _ = joints.shape
    H, W = out_size
    heatmaps = torch.zeros((B, J, H, W), dtype=torch.float32, device=joints.device)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=joints.device),
        torch.arange(W, device=joints.device),
        indexing='ij'
    )
    for b in range(B):
        for j in range(J):
            x, y = joints[b, j]
            if x < 0 or y < 0:
                continue
            x_hm = x * (W / IMG_SIZE[1])
            y_hm = y * (H / IMG_SIZE[0])
            heatmaps[b, j] = torch.exp(-((xx - x_hm)**2 + (yy - y_hm)**2) / (2 * sigma**2))
    return heatmaps

# --- Prediction Decoding from heatmap to coordinate---
def get_max_preds(heatmaps):
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    max_vals, idxs = torch.max(heatmaps_reshaped, dim=2)
    coords = torch.zeros((B, K, 2), dtype=torch.float32, device=heatmaps.device)
    coords[..., 0] = (idxs % W) * (IMG_SIZE[1] / W)
    coords[..., 1] = (idxs // W) * (IMG_SIZE[0] / H)
    return coords

# --- Evaluation ---
def evaluate(model, dataloader, heatmap_size):
    model.eval()
    thresholds = [5.0, 10.0, 20.0]
    pck_totals = {t: 0 for t in thresholds}
    pck_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            poses = batch['pose_13'].to(DEVICE)
            frames = batch['frame'].to(DEVICE)
            preds = model(frames)
            preds = nn.functional.interpolate(preds, size=heatmap_size, mode='bilinear', align_corners=False)
            pred_coords = get_max_preds(preds)
            true_coords = poses

            for pred, true in zip(pred_coords, true_coords):
                dists = torch.norm(pred - true, dim=1)
                for t in thresholds:
                    pck_totals[t] += (dists < t).sum().item()
                pck_count += true.shape[0]

    pck_scores = {t: pck_totals[t] / pck_count for t in thresholds}
    for t in thresholds:
        print(f"PCK@{int(t)}: {pck_scores[t]:.4f}")
    return pck_scores

# --- Main ---
def main():
    global HEATMAP_SIZE

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = Human36mDataset(
        base_directory="/home/skyle/datasets/H36M_FREI", #  # "../data/H36M_FREI"
        split='train',
        num_frames_per_video=NUM_FRAMES,
        transform=transform,
        image_size=IMG_SIZE
    )
    val_dataset = Human36mDataset(
        base_directory="/home/skyle/datasets/H36M_FREI",  # "../data/H36M_FREI",
        split='validation',
        num_frames_per_video=int(NUM_FRAMES/4),
        transform=transform,
        image_size=IMG_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # model = ViTPoseFull(
    #     img_size=IMG_SIZE,
    #     num_keypoints=NUM_KEYPOINTS,
    #     vit_model='vit_base_patch16_224',
    #     pretrained=True
    # ).to(DEVICE)

    # # 1) Build a custom ViT config
    # from transformers import ViTConfig, ViTModel
    # vit_cfg = ViTConfig(
    #     image_size=IMG_SIZE,
    #     patch_size=16,
    #     hidden_size=512,      # smaller total dim
    #     num_hidden_layers=8,  # shallower
    #     num_attention_heads=4,# fewer heads
    #     intermediate_size=2048,
    # )
    # vit_backbone = ViTModel(vit_cfg)"/home/skyle/datasets/H36M_FREI"

    
    # train-from-scratch ViT: 8 heads, 512 embed dim, depth 8, mlp_ratio 2.0
    model = ViTPoseFull(
        img_size=IMG_SIZE,
        patch_size = PATCH, # e.g. 16, the bigger, the less token perimage: N_token = 256*192/16*16
        num_keypoints=NUM_KEYPOINTS, 
        vit_model=VIT_MODEL,
        pretrained=False,     # no ShapeMismatch errors
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=HEAD,
        mlp_ratio=MLP_RATIO
    ).to(DEVICE)
    
    # → torch.Size([2, 13, 32, 24])  
    # assuming 256/16=16→×2×2×2 upsample =32, 
    # similarly 192/16=12→24



    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    best_pck5 = 0.0
    epochs_no_improve = 0

    with open(LOG_PATH, 'w') as log_file:
        log_file.write("Epoch\tLoss\tPCK@5\tPCK@10\tPCK@20\n")

        for epoch in range(1, EPOCHS+1):
            model.train()
            running_loss = 0.0

            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
                poses = batch['pose_13'].to(DEVICE)
                frames = batch['frame'].to(DEVICE)

                with torch.cuda.amp.autocast():
                    preds = model(frames)

                if epoch == 1 and i == 0:
                    HEATMAP_SIZE = preds.shape[2:]
                    print(f"[INFO] Inferred HEATMAP_SIZE from model output: {HEATMAP_SIZE}")

                # Generate ground truth heatmap
                heatmaps = generate_heatmaps_from_2d(poses, out_size=HEATMAP_SIZE)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = criterion(preds, heatmaps)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            # Evaluate and get PCK scores
            pck_scores = evaluate(model, val_loader, heatmap_size=HEATMAP_SIZE)
            pck5 = pck_scores[5.0]
            pck10 = pck_scores[10.0]
            pck20 = pck_scores[20.0]

            # Logging
            log_file.write(f"{epoch}\t{avg_loss:.4f}\t{pck5:.4f}\t{pck10:.4f}\t{pck20:.4f}\n")
            log_file.flush()

            # Check for improvement
            if pck5 > best_pck5:
                best_pck5 = pck5
                epochs_no_improve = 0
                print(f"[INFO] New best PCK@5: {pck5:.4f}, saving model.")
                torch.save(model.state_dict(), SAVE_PATH)
            else:
                epochs_no_improve += 1
                print(f"[INFO] PCK@5 did not improve ({pck5:.4f} <= {best_pck5:.4f}). Patience: {epochs_no_improve}/{PATIENCE}")

            # Early stopping
            if epochs_no_improve >= PATIENCE:
                print(f"[EARLY STOPPING] No improvement for {PATIENCE} epochs.")
                break

if __name__ == '__main__':
    main()
