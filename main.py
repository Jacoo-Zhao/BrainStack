import wandb, time, os, torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from wandb import errors
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from options import parse_args
from utils import count_trainable_params, set_seed, print_metrics
from loss import DynamicEnsembleLossProgress, DynamicEnsembleLoss_No_Dis, DynamicEnsembleLoss_No_Warmup
from assets.eeg_areas import RoI_7
from data.load_data import load_data
from trainer.construct_model import construct_eeg_model
from trainer.train import train
from trainer.validate import validate

try:
    from colorama import Fore, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def main(device, args):
    set_seed(40)
    
    if args.split == 'cross-subject':
        args.subject = ['S05','S07','S08','S09','S10','S11','S12','S13','S14','S15']
    subject_list = args.subject if isinstance(args.subject, list) else [args.subject]


    print("🪇 Subject list:", subject_list)


    wandb_config = {
        "lr": args.lr,
        "batch": args.batch,
        "model": args.model,
        "epoch": args.epoch,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "num_classes": args.num_classes,
        "patience": args.patience,
        "ds": args.ds,
        "eeg_normalization": "mean_std",
        "split": args.split,
        "subject": 'cross-subject' if args.split=='cross-subject' else subject_list[0]
    }
    run = wandb.init(project="NeurIPS-25", config=wandb_config)

    batch_size = args.batch
    learning_rate = args.lr
    num_epochs =args.epoch
    weight_decay = args.weight_decay
    
    best_val_acc = 0.0
    epochs_no_improve = 0

    train_ds, val_ds, test_ds, _, _ = load_data(
        subject_list=subject_list, resample_fs=args.ds, split_method=args.split
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    model = construct_eeg_model(
        option=args.model,
        EEG_AREAS=RoI_7,
        num_classes=args.num_classes,
        dropout_rate=args.dropout,
        input_window_samples=args.ds,
        num_areas=len(RoI_7),
    )

    summary(model, input_size=(batch_size, 122, args.ds), device=device, depth=4)

    model.to(device)

    print("🤖🤖🤖Trainable parameters: {:.2f}M".format(count_trainable_params(model)/1e6))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=3, verbose=True)

    loss_fn = DynamicEnsembleLossProgress(
        criterion=criterion,
        max_alpha=0.5,
        max_beta=1.0,
        warmup_epochs=5,
        loss_global_based=False
    )

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_global_acc, train_local_sub_accs,train_global_sub_accs = train(
            model, train_loader, criterion, loss_fn, batch_size, optimizer, device, epoch, num_epochs,
        )
        val_loss, val_accuracy, val_global_acc, val_local_sub_accs, val_global_sub_accs,f1, kappa = validate(model, val_loader, criterion, loss_fn, device)

        if hasattr(model, 'eeg_areas'):
            print_metrics("Train", epoch, train_accuracy, train_global_acc, train_local_sub_accs, list(model.eeg_areas.keys()), train_global_sub_accs)
            print_metrics("Validation", epoch, val_accuracy, val_global_acc, val_local_sub_accs, list(model.eeg_areas.keys()), val_global_sub_accs)
        else:
            print(f"\n[Train Epoch {epoch+1}] Accuracy: {train_accuracy:.2f}%")
            print(f"[Validation Epoch {epoch+1}] Accuracy: {val_accuracy:.2f}%")
        

        scheduler.step(val_accuracy)

        wandb.log(
            {
                "epoch": epoch + 1,

                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,

                "train_global_acc":train_global_acc,
                "train_global_sub_accs": train_global_sub_accs,
                "train_local_sub_accs": train_local_sub_accs,

                "val_global_acc":val_global_acc,
                "val_global_sub_accs": val_global_sub_accs,
                "val_local_sub_accs": val_local_sub_accs,
                "f1": f1,
                "kappa": kappa,

            }
        )

        if epoch >1:
            if val_accuracy > best_val_acc:
                # Create checkpoints directory in project root
                folder_path = './checkpoints'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                model_save_path = f"{args.model}_{subject_list}_{args.note}_{time_stamp}.pth"
                model_save_path = os.path.join(folder_path, model_save_path)

                torch.save(model.state_dict(), model_save_path)
                if COLORAMA_AVAILABLE:
                    print(Fore.YELLOW + f" Best model saved at {model_save_path}! Validation accuracy improved from {best_val_acc:.2f}% to {val_accuracy:.2f}%")
                else:
                    print(f"  Best model saved at {model_save_path}!")
                    print("Validation accuracy improved from {:.2f}% to {:.2f}%".format(best_val_acc, val_accuracy))
                epochs_no_improve = 0
                best_val_acc = val_accuracy

            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s). Best validation accuracy: {best_val_acc:.2f}%")

            if epochs_no_improve >= args.patience:
                if COLORAMA_AVAILABLE:
                    print(Fore.RED + f"\nEarly stopping triggered after {epoch+1} epochs.")
                else:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

            print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.2f}%")

    
    print("\nEvaluating on Test Set:")
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    test_loss, test_accuracy, test_global_acc, test_local_accs, test_sub_model_accs, test_f1, test_kappa = validate(model, test_loader, criterion, loss_fn, device)

    if COLORAMA_AVAILABLE:
        print(Fore.CYAN + f"\nTest Accuracy: {test_accuracy:.2f}%, All Acc: {test_global_acc:.2f}%, Area Accs: {test_local_accs}, Sub-model Accs: {test_sub_model_accs}") 
    else:
        print(f"\nTest Accuracy: {test_accuracy:.2f}%, All Acc: {test_global_acc:.2f}%, Area Accs: {test_local_accs}, Sub-model Accs: {test_sub_model_accs}")
    
    wandb.summary["Test Loss"] = test_loss
    wandb.summary["Test Accuracy"] = test_accuracy
    wandb.summary["Test All Acc"] = test_global_acc
    wandb.summary["Test Area Accs"] = test_local_accs
    wandb.summary["Test Sub-model Accs"] = test_sub_model_accs
    wandb.summary["F1"] = test_f1
    wandb.summary["Kappa"] = test_kappa
    wandb.summary["Checkpoint"] = model_save_path
    wandb.summary["Marker"]= args.note
    wandb.finish()
    print("Training and evaluation completed successfully!")
    return model_save_path, test_accuracy

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = parse_args()
    print(f"Using model: {args.model}")
    print(f"Using subject: {args.subject}")
    print(f"Using split method: {args.split}")
    print(f"▶️▶️ Running with command line arguments: {args}")

    model_save_path, test_accuracy = main(device, args)
        
    print(f"Test accuracy: {test_accuracy:.2f}%")
