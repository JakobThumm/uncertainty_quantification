import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def load_scores(pickle_path: str) -> dict:
    """
    Load your scores_dict from disk and convert any JAX arrays to NumPy.
    """
    with open(pickle_path, "rb") as f:
        scores_dict = pickle.load(f)

    for key, val in list(scores_dict.items()):
        if isinstance(val, jnp.ndarray):
            scores_dict[key] = np.array(val)
    return scores_dict

# Example usage
# 
filepath = "/home/skyle/Desktop/uq_benchmark/models/FMNIST/LeNet_h/seed_420/" \
"started_2025-07-01-14-40-38_scores_subsample100_lanczos_seed0_size_HM0of0_LM90of100_sketch_srft_seed0_size1000.pickle"

# smart_lla
filepath = "/home/skyle/Desktop/uq_benchmark/models/FMNIST/LeNet/seed_420/" \
"started_2025-06-29-21-24-19_scores_subsample1000_eig_lanczos_seed0_size_HM0of0_LM90of100_sketch_srft_seed0_size1000.pickle"

# sketched_local_ensemble- LeNet
filepath = "/home/skyle/Desktop/uq_benchmark/models/FMNIST/LeNet/seed_420/" \
"started_2025-06-29-21-24-19_scores_subsample1000_lanczos_seed0_size_HM0of0_LM90of100_sketch_srft_seed0_size1000.pickle"

# low_rank_lla - LeNet
filepath = "/home/skyle/Desktop/uq_benchmark/models/FMNIST/LeNet/seed_420/" \
"started_2025-06-29-21-24-19_scores_subsample1000_eig_lanczos_seed0_size_HM9of10_LM0of0.pickle"

scores = load_scores(filepath)

# 1) Inspect what you got
print("Keys in scores_dict:", scores.keys())

# 2) Identify the “distribution” keys (exclude eigenvals/args_dict and any QF arrays)
dist_keys = [
    k for k in scores
    if k not in ("eigenvals", "args_dict")
       and not k.endswith("_QF")
       and not k.endswith("_QFapprox")
]
for dist in dist_keys:
    print(f"  {dist}: {scores[dist].shape}")

# 3) Build a long-form DataFrame of all scores + labels
df_list = []
for dist in dist_keys:
    arr = scores[dist]
    # label ID→0, OOD→1  (you can adjust if you have multiple OOD classes)
    lbl = 0 if dist == "ID" else 1
    df_list.append(pd.DataFrame({
        "score":      arr.flatten(),
        "dist":       dist,
        "is_ood":     lbl
    }))
df = pd.concat(df_list, ignore_index=True)

# Quick summary stats per split
print(df.groupby("dist")["score"].agg(["mean", "std", "min", "max"]))

# 4) Simple OOD AUC
#    If higher score → more likely OOD, use it directly; otherwise invert: -score.
y_true  = df["is_ood"].values
y_score = df["score"].values  # or -df["score"].values
auc = roc_auc_score(y_true, y_score)
print(f"Overall OOD ROC AUC = {auc:.4f}")
