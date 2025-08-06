import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="EEG Model Training")
    parser.add_argument(
        "--model",
        type=str,
        default="brainStack",
        choices=["brainStack"],
        help="Model selection",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="S01",
        choices=["all", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
        help="Subject name to use for training",
    )

    parser.add_argument(
        "--split",
        type=str,
        default='leave-one-session-out',
        choices=['cross-subject','leave-one-session-out'],
        help="Data split method",
    )

    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=7, help="Patience for early stop")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--ds", type=int, default=1000, help="eeg downsampling with origin=2500")
    
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_classes", type=int, default=24, help="Number of classes")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training")
    
    parser.add_argument("--note", type=str, required=True, help="Note for the experiment")
    
    return parser.parse_args()
