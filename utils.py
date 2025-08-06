import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm

def print_metrics(stage, epoch, fused_acc, all_acc, area_accs, area_names, sub_model_accs=None):
    print(f"\n[{stage} Epoch {epoch+1}]")
    print(f"  - Fused Acc: {fused_acc:.2f}%")
    print(f"  - Dsitill Teacher Acc: {all_acc:.2f}%")
    if sub_model_accs:
        print("  - Global_Heter-model Accuracies:")
        for i, acc in enumerate(sub_model_accs, start=1):
            print(f"    • Sub-model {i}: {acc:.2f}%")

    print("  - Local_Per-area Accuracies:")
    if isinstance(area_accs, list):
        for name, acc in zip(area_names, area_accs):
            print(f"    • {name:12s}: {acc:.2f}%")
        
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def count_trainable_params(model):
    # Sum the number of elements in each trainable parameter
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_gaussian_noise(eeg_data, mean=0.0, std=0.01):
    noise = torch.randn_like(eeg_data) * std + mean
    return eeg_data + noise

def print_epoch_summary(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
    try:
        from tabulate import tabulate
        TABULATE_AVAILABLE = True
    except ImportError:
        TABULATE_AVAILABLE = False

    if TABULATE_AVAILABLE:
        table = [
            ["Epoch", f"{epoch}/{num_epochs}"],
            ["Training Loss", f"{train_loss:.4f}"],
            ["Training Accuracy", f"{train_acc:.2f}%"],
            ["Validation Loss", f"{val_loss:.4f}"],
            ["Validation Accuracy", f"{val_acc:.2f}%"],
        ]
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    else:
        # For colored output
        try:
            from colorama import Fore, init
            init(autoreset=True)
            COLORAMA_AVAILABLE = True
        except ImportError:
            COLORAMA_AVAILABLE = False

        # If tabulate is not installed, use a simple print format
        if COLORAMA_AVAILABLE:
            print(
                Fore.GREEN
                + f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%"
            )
            print(
                Fore.BLUE
                + f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )
        else:
            print(
                f"  Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%"
            )
            print(
                f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )    