import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import albumentations as A
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_augmentation_pipeline(seed=None):
    """
    Create an augmentation pipeline using albumentations.
    Returns different augmentation combinations for video diversity.
    """
    # Define multiple augmentation strategies
    augmentation_strategies = [
        # Strategy 1: Brightness and contrast
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        ]),
        
        # Strategy 2: Blur and noise
        A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ]),
        
        # Strategy 3: Geometric transformations
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8),
            A.HorizontalFlip(p=0.5),
        ]),
        
        # Strategy 4: Color adjustments
        A.Compose([
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        ]),
        
        # Strategy 5: Compression and quality
        A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.7),
            A.Downscale(scale_min=0.75, scale_max=0.95, p=0.5),
        ]),
        
        # Strategy 6: Mixed augmentations
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.6),
            A.HorizontalFlip(p=0.5),
        ]),
    ]
    
    return augmentation_strategies


def augment_video(args_tuple):
    """
    Augment a single video and save it with a new name.
    This function is called by multiprocessing.
    
    Args:
        args_tuple: Tuple containing (video_path, output_path, aug_index, total_augmentations)
    
    Returns:
        Tuple of (success, output_filename, error_message)
    """
    video_path, output_path, aug_index, total_augmentations = args_tuple
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (False, None, f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create output video writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Get augmentation pipeline (cycle through strategies)
        strategies = get_augmentation_pipeline()
        transform = strategies[aug_index % len(strategies)]
        
        # Process each frame
        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for albumentations
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = transform(image=frame_rgb)
            frame_aug = augmented['image']
            
            # Convert back to BGR for video writer
            frame_bgr = cv2.cvtColor(frame_aug, cv2.COLOR_RGB2BGR)
            
            out.write(frame_bgr)
            frames_processed += 1
        
        # Release resources
        cap.release()
        out.release()
        
        if frames_processed == 0:
            return (False, None, f"No frames processed for: {video_path}")
        
        return (True, output_path.name, None)
        
    except Exception as e:
        return (False, None, f"Error processing {video_path}: {str(e)}")


def process_video_augmentation(args_tuple):
    """
    Wrapper function for augmenting a single video multiple times.
    
    Args:
        args_tuple: Tuple containing (video_name, label, video_dir, output_dir, num_augmentations)
    
    Returns:
        List of tuples (augmented_video_name, label, success)
    """
    video_name, label, video_dir, output_dir, num_augmentations = args_tuple
    
    video_path = Path(video_dir) / video_name
    if not video_path.exists():
        warnings.warn(f"Video not found: {video_path}")
        return []
    
    results = []
    video_stem = video_path.stem
    
    # Create augmentations for this video
    for i in range(num_augmentations):
        output_filename = f"{video_stem}_aug{i}.mp4"
        output_path = Path(output_dir) / output_filename
        
        success, out_name, error = augment_video((video_path, output_path, i, num_augmentations))
        
        if success:
            results.append((output_filename, label, True))
        else:
            if error:
                warnings.warn(error)
            results.append((None, label, False))
    
    return results


def augment_dataset(csv_path, video_dir, output_dir, output_csv, target_videos_per_label=100, num_workers=None):
    """
    Augment dataset to reach a target number of videos per label using multiprocessing.
    
    Args:
        csv_path: Path to the original CSV file
        video_dir: Directory containing original videos
        output_dir: Directory to save augmented videos
        output_csv: Path to save the new CSV file with augmented video labels
        target_videos_per_label: Target number of total videos per label (original + augmented)
        num_workers: Number of worker processes (default: CPU count)
    """
    # Read the original CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} videos from {csv_path}")
    print(f"Original labels distribution:\n{df['label'].value_counts()}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Calculate how many augmentations needed per label
    label_counts = df['label'].value_counts()
    print(f"\nTarget videos per label: {target_videos_per_label}")
    print("Augmentation plan:")
    
    video_args = []
    for label, count in label_counts.items():
        videos_needed = target_videos_per_label - count
        if videos_needed <= 0:
            print(f"  {label}: {count} videos (already has {count} >= {target_videos_per_label}, no augmentation needed)")
            continue
        
        # Get videos for this label
        label_videos = df[df['label'] == label]
        num_videos = len(label_videos)
        
        # Calculate augmentations per video (round up to ensure we reach target)
        augs_per_video = (videos_needed + num_videos - 1) // num_videos
        
        print(f"  {label}: {count} videos -> need {videos_needed} more (creating {augs_per_video} augmentations per video)")
        
        # Add to video arguments
        for _, row in label_videos.iterrows():
            video_args.append((row['video_name'], row['label'], video_dir, output_dir, augs_per_video))
    
    if not video_args:
        print("\nNo augmentation needed! All labels already have sufficient videos.")
        # Still create output CSV with original data copied to output dir
        print("Copying original videos to output directory...")
        import shutil
        copied_videos = []
        for _, row in df.iterrows():
            src = Path(video_dir) / row['video_name']
            if src.exists():
                dst = output_dir / row['video_name']
                shutil.copy2(src, dst)
                copied_videos.append({'video_name': row['video_name'], 'label': row['label']})
        
        if copied_videos:
            pd.DataFrame(copied_videos).to_csv(output_csv, index=False)
            print(f"Saved CSV to: {output_csv}")
        return len(copied_videos)
    
    # Process videos with multiprocessing
    print(f"\nAugmenting {len(video_args)} videos...")
    all_results = []
    
    with Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        for results in tqdm(pool.imap_unordered(process_video_augmentation, video_args), 
                           total=len(video_args), 
                           desc="Processing videos"):
            all_results.extend(results)
    
    # Filter successful augmentations and create new dataframe
    successful_augmentations = [
        {'video_name': video_name, 'label': label}
        for video_name, label, success in all_results
        if success and video_name is not None
    ]
    
    print(f"\nSuccessfully augmented {len(successful_augmentations)} videos")
    print(f"Failed: {len(all_results) - len(successful_augmentations)}")
    
    # Combine original and augmented data
    print("\nCombining original and augmented data...")
    combined_data = []
    
    # Add all original videos
    for _, row in df.iterrows():
        combined_data.append({'video_name': row['video_name'], 'label': row['label']})
    
    # Add augmented videos
    combined_data.extend(successful_augmentations)
    
    # Create final CSV
    if combined_data:
        final_df = pd.DataFrame(combined_data)
        final_df.to_csv(output_csv, index=False)
        print(f"Saved combined dataset CSV to: {output_csv}")
        print(f"Final labels distribution:\n{final_df['label'].value_counts()}")
    else:
        print("Warning: No data to save!")
    
    return len(combined_data)


def main():
    parser = argparse.ArgumentParser(
        description='Augment video dataset to balance classes using albumentations and multiprocessing'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the original CSV file containing video_name and label columns'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Directory containing the original videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save augmented videos (default: {video_dir}-augmented)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to save the augmented dataset CSV (default: {csv}-augmented.csv)'
    )
    parser.add_argument(
        '--target_videos_per_label',
        type=int,
        required=True,
        help='Target number of total videos per label (original + augmented)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count)'
    )
    
    args = parser.parse_args()
    
    # Set default output paths based on input paths
    if args.output_dir is None:
        args.output_dir = f"{args.video_dir}-augmented"
    
    if args.output_csv is None:
        csv_path = Path(args.csv)
        args.output_csv = str(csv_path.parent / f"{csv_path.stem}-augmented{csv_path.suffix}")
    
    print("=" * 60)
    print("Video Dataset Augmentation (Class Balancing)")
    print("=" * 60)
    print(f"Input CSV: {args.csv}")
    print(f"Input video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Target videos per label: {args.target_videos_per_label}")
    print("=" * 60)
    
    # Run augmentation
    total_videos = augment_dataset(
        csv_path=args.csv,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        target_videos_per_label=args.target_videos_per_label,
        num_workers=args.num_workers
    )
    
    print("=" * 60)
    print(f"Augmentation complete! Total videos in dataset: {total_videos}")
    print("=" * 60)


if __name__ == '__main__':
    main()
