
import torch 
import json
# --- Ground truth Heatmap Generator ---
with open("config_train_vitpose.json", "r") as f:
    config = json.load(f)
IMG_SIZE = tuple(config["IMG_SIZE"])

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