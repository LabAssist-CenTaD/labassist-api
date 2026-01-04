import argparse
import pandas as pd
import tempfile
import warnings
from pathlib import Path

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Div255
from torchvision.transforms import Compose, RandomHorizontalFlip
from torchvision.transforms._transforms_video import NormalizeVideo

warnings.filterwarnings('ignore')

seed_everything(0)
torch.set_float32_matmul_precision('medium')

from model import ActionDetectionModel


def create_video_dataset(csv_path, video_dir, transform=None, clip_duration=2):
    """Create a pytorchvideo dataset from CSV and video directory"""
    video_dir = Path(video_dir)
    
    # Read CSV and create manifest
    df = pd.read_csv(csv_path)
    
    # Create label mapping
    unique_labels = sorted(df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Label mapping: {label_map}")
    
    # Create temporary manifest file for pytorchvideo
    manifest_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    valid_count = 0
    for _, row in df.iterrows():
        video_path = video_dir / row['video_name']
        if video_path.exists():
            label_idx = label_map[row['label']]
            manifest_file.write(f"{video_path} {label_idx}\n")
            valid_count += 1
        else:
            warnings.warn(f"Video not found: {video_path}")
    manifest_file.close()
    
    print(f"Loaded {valid_count} videos from manifest")
    
    # Create and return pytorchvideo IterableDataset directly
    dataset = labeled_video_dataset(
        manifest_file.name,
        clip_sampler=make_clip_sampler('random', clip_duration),
        transform=transform,
        decode_audio=False,
    )
    
    return dataset


def create_dataloader(csv_path, video_dir, batch_size=16, num_workers=0, augment=False):
    """Create a dataloader from CSV and video directory"""
    
    if augment:
        video_transform = Compose([
            ApplyTransformToKey(key='video',
            transform=Compose([
                UniformTemporalSubsample(16),
                Div255(),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                RandomHorizontalFlip(p=0.5),
            ])),
        ])
    else:
        video_transform = Compose([
            ApplyTransformToKey(key='video',
            transform=Compose([
                UniformTemporalSubsample(16),
                Div255(),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])),
        ])
    
    dataset = create_video_dataset(csv_path, video_dir, transform=video_transform)
    # For IterableDataset, shuffle must be False in DataLoader
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train action detection model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to validation CSV file (optional)')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to directory containing videos')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_path', type=str, default='trained_model.pth', help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create dataloaders
    print(f"Loading training data from {args.train_csv}")
    train_loader = create_dataloader(
        args.train_csv, 
        args.video_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        augment=True
    )
    
    val_loader = None
    if args.val_csv:
        print(f"Loading validation data from {args.val_csv}")
        val_loader = create_dataloader(
            args.val_csv, 
            args.video_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            augment=False
        )
    
    # Initialize model
    model = ActionDetectionModel(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_worker=args.num_workers
    )
    
    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Detect device and setup trainer
    if torch.cuda.is_available():
        print("Training on GPU")
        accelerator = 'gpu'
        devices = -1
        precision = '16-mixed'
    else:
        print("Training on CPU (No GPU detected)")
        accelerator = 'cpu'
        devices = 'auto'
        precision = '32'
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        accumulate_grad_batches=2,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=[lr_monitor],
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model
    print(f"Saving model to {args.output_path}")
    torch.save(model.state_dict(), args.output_path)
    print("Training complete!")