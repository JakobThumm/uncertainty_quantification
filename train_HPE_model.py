import pickle
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
import argparse
import json
import datetime

from src.datasets import augmented_dataloader_from_string, get_output_dim
from src.models import model_from_string, pretrained_model_from_string
from src.training.trainer import gradient_descent
from src.training.trainer_fancy import gradient_descent_fancy

from src.models import RegressFlowFlax
from src.datasets.h36m import get_h36m
import jax.numpy as jnp
import jax
parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["H36M", "Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "LeNet_h", "GoogleNet", "ConvNeXt", "ConvNeXt_L", "ConvNeXt_XL", "ResNet", "ResNet_NoNorm", "ResNet50", "ResNet50PreAct", "VAN_tiny", "VAN_small", "VAN_base", "VAN_large", "SWIN_tiny", "SWIN_large", "ViT_mnist"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of layers in the MLP.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32) # 128 original
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--decrease_learning_rate", action="store_true", required=False, default=False)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification", "binary_multiclassification"], default="classification")

parser.add_argument("--default_hyperparams", action="store_true", required=False, default=False)
parser.add_argument("--fancy", action="store_true", required=False, default=False)

# extra regularizer
parser.add_argument("--regularizer", type=str, choices=["log_determinant_ggn", "log_determinant_ntk"], default=None)
parser.add_argument("--regularizer_hutch_samples", type=int, default=10)
parser.add_argument("--regularizer_prec_prior", type=float, default=1.)
parser.add_argument("--regularizer_prec_lik", type=float, default=1.)
parser.add_argument("--n_warmup_epochs", type=int, default=0)


# storage
parser.add_argument("--run_name", type=str, default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--run_name_pretrained", type=str, default=None, help="Run name from which to load pretrained parameters. If None parameters are randomly initialized")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=20, help="Frequency of computing validation stats")

# print more stuff
parser.add_argument("--verbose", action="store_true", required=False, default=False)

# -----------------------------
# 3) Train state with batch norm
# -----------------------------
import math, functools, time, pathlib, pickle
from typing import Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import json
import datetime

class TrainState(train_state.TrainState):
    batch_stats: Any

# -----------------------------
# 4) Create model & init params
# -----------------------------
def create_model(rng, num_joints=17, image_size=(256,192), fc_filters=(1024,),
                 accept_nchw=True, batch_size=8):
    model = RegressFlowFlax(
        num_joints=num_joints,
        image_size=image_size,
        fc_filters=fc_filters,
        accept_nchw=accept_nchw
    )
    H, W = image_size
    dummy_x = jnp.zeros((batch_size, 3, H, W), jnp.float32) if accept_nchw \
            else jnp.zeros((batch_size, H, W, 3), jnp.float32)
    variables = model.init(rng, dummy_x, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return model, params, batch_stats
# -----------------------------
# 5) Optimizer / schedule
# -----------------------------
def create_tx(
    base_lr=3e-4, weight_decay=1e-4, warmup_steps=1000, total_steps=100_000, clip_norm=1.0
):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=base_lr * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=sched, weight_decay=weight_decay),
    )
    return tx, sched


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # args.model == "ResNet50":
    args_dict["n_epochs"] = 2
    args_dict["batch_size"] = 128
    args_dict["optimizer"] = "sgd"
    args_dict["learning_rate"] = 1e-5 #0.0001
    args_dict["decrease_learning_rate"] = True
    args_dict["momentum"] = 0.9
    args_dict["weight_decay"] = 1e-4
    args_dict["activation_fun"] = "relu"

    seed = 0
    num_epochs = 5
    steps_per_epoch = 500
    batch_size = 4 
    image_size = (256, 192)
    num_joints = 17
    accept_nchw = True

    base_lr = 3e-4
    weight_decay = 1e-4
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1000, int(0.05 * total_steps))
    rng = jax.random.PRNGKey(seed)


    ###############
    ### dataset ###
    
    # Return the frame image and the pose_13(13 keypoint locations)
    train_loader, valid_loader, _ = get_h36m(
        batch_size = args_dict["batch_size"], 
        shuffle = True,
        seed = seed,
        download = False, 
        #data_path = data_path
    )
    
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")


    #############
    ### model ###
    output_dim = get_output_dim(args.dataset)
    print("output_dim:", output_dim)

    

    model, params, batch_stats = create_model(rng, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size)
    tx, sched = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

    # TODO: check wrapping way
    args_dict["output_dim"] = output_dim
    args_dict["opt_hp"] = {
            "lr": args_dict["learning_rate"],
            "momentum": args_dict["momentum"],
            "weight_decay": args_dict["weight_decay"],
        }
    
    ################
    ### training ###  

    # Skip pretrain code, add if need

    # TODO: 
    best_val = float("inf")
    gstep = 0

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        t0 = time.time()
        losses = []



    params_dict, stats_dict = gradient_descent(
                model, 
                train_loader, 
                valid_loader, 
                args_dict,
                pretrained_params_dict = None if args.run_name_pretrained is None else pretrained_params_dict
            )
    
    model_dict = {"model": args.model, **params_dict}


    ####################################
    ### save params and dictionaries ###
    # first folder is dataset
    save_folder = f"{args.model_save_path}/{args.dataset}"
    if args.n_samples is not None:
        save_folder += f"_samples{args.n_samples}"
    # second folder is model
    if args.model == "MLP":
        save_folder += f"/MLP_depth{args.mlp_num_layers}_hidden{args.mlp_hidden_dim}"
    else:
        save_folder += f"/{args.model}"
    # third folder is seed
    save_folder += f"/seed_{args.seed}"
    os.makedirs(save_folder, exist_ok=True)
    
    if args.run_name is not None:
        save_name = f"{args.run_name}"
    else:
        save_name = f"started_{now_string}"

    print(f"Saving to {save_folder}/{save_name}")
    pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
    pickle.dump(stats_dict, open(f"{save_folder}/{save_name}_stats.pickle", "wb"))
    with open(f"{save_folder}/{save_name}_args.json", "w") as f:
        json.dump(args_dict, f)