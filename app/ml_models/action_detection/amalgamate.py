import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm


def amalgamate_datasets(csv_path, video_dir, additional_dir, output_dir=None, output_csv=None):
    """
    Combine videos from two directories into one dataset.
    
    Args:
        csv_path: Path to CSV file containing video_name and label for video_dir
        video_dir: Directory containing videos referenced in the CSV
        additional_dir: Directory containing additional videos (all labeled as "Correct")
        output_dir: Output directory for combined videos (default: {video_dir}-{additional_dir}-amalgamation)
        output_csv: Output CSV path (default: {output_dir}/labels.csv)
    
    Returns:
        Total number of videos in the combined dataset
    """
    video_dir = Path(video_dir)
    additional_dir = Path(additional_dir)
    
    # Create output directory name if not provided
    if output_dir is None:
        output_dir = Path(f"{video_dir.name}-{additional_dir.name}-amalgamation")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Set output CSV path
    if output_csv is None:
        output_csv = output_dir / "labels.csv"
    else:
        output_csv = Path(output_csv)
    
    # Read original CSV
    df = pd.read_csv(csv_path)
    print(f"\nOriginal dataset:")
    print(f"  Total videos: {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts()}")
    
    # Prepare combined dataset list
    combined_data = []
    
    # Copy videos from original video_dir
    print(f"\nCopying videos from {video_dir}...")
    copied_count = 0
    not_found_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Original videos"):
        src_path = video_dir / row['video_name']
        if src_path.exists():
            dst_path = output_dir / row['video_name']
            shutil.copy2(src_path, dst_path)
            combined_data.append({'video_name': row['video_name'], 'label': row['label']})
            copied_count += 1
        else:
            print(f"  Warning: Video not found: {src_path}")
            not_found_count += 1
    
    print(f"  Copied: {copied_count} videos")
    if not_found_count > 0:
        print(f"  Not found: {not_found_count} videos")
    
    # Copy videos from additional_dir (all labeled as "Correct")
    print(f"\nCopying videos from {additional_dir} (all labeled 'Correct')...")
    additional_videos = list(additional_dir.glob("*.mp4")) + list(additional_dir.glob("*.avi"))
    additional_count = 0
    
    for src_path in tqdm(additional_videos, desc="Additional videos"):
        # Check if filename already exists from original dataset
        video_name = src_path.name
        dst_path = output_dir / video_name
        
        # If name collision, add suffix
        if dst_path.exists():
            stem = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while dst_path.exists():
                video_name = f"{stem}_add{counter}{suffix}"
                dst_path = output_dir / video_name
                counter += 1
        
        shutil.copy2(src_path, dst_path)
        combined_data.append({'video_name': video_name, 'label': 'Correct'})
        additional_count += 1
    
    print(f"  Copied: {additional_count} videos")
    
    # Create combined CSV
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_csv, index=False)
        print(f"\nCombined dataset saved to: {output_csv}")
        print(f"  Total videos: {len(combined_df)}")
        print(f"  Label distribution:\n{combined_df['label'].value_counts()}")
    else:
        print("\nWarning: No videos were copied!")
    
    return len(combined_data)


def main():
    parser = argparse.ArgumentParser(
        description='Amalgamate videos from two directories into a single dataset'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file containing video_name and label columns for video_dir'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Directory containing videos referenced in the CSV'
    )
    parser.add_argument(
        '--additional_dir',
        type=str,
        required=True,
        help='Directory containing additional videos (all will be labeled as "Correct")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for combined videos (default: {video_dir}-{additional_dir}-amalgamation)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to save combined CSV (default: {output_dir}/labels.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Video Dataset Amalgamation")
    print("=" * 70)
    print(f"Original CSV: {args.csv}")
    print(f"Original video directory: {args.video_dir}")
    print(f"Additional video directory: {args.additional_dir}")
    print("=" * 70)
    
    # Run amalgamation
    total_videos = amalgamate_datasets(
        csv_path=args.csv,
        video_dir=args.video_dir,
        additional_dir=args.additional_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv
    )
    
    print("=" * 70)
    print(f"Amalgamation complete! Total videos: {total_videos}")
    print("=" * 70)


if __name__ == '__main__':
    main()
