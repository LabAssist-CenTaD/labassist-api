from pathlib import Path
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from torch.utils.data import DataLoader, Dataset
import warnings

from pytorchvideo.transforms import (
    ApplyTransformToKey, 
    UniformTemporalSubsample,
    Div255
)

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomGrayscale,
    RandomAutocontrast,
    RandomAdjustSharpness,
)

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

def divide_by_255(video):
    return video / 255.0

def convert_manifest_to_absolute(manifest_path: str, repo_root: Path) -> str:
    """Convert manifest with relative paths to absolute paths"""
    import tempfile
    temp_manifest = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                rel_path = parts[0]
                label = parts[1] if len(parts) > 1 else ''
                # Convert forward slashes to backslashes for Windows and make absolute
                abs_path = (repo_root / rel_path.replace('/', '\\')).resolve()
                temp_manifest.write(f"{abs_path} {label}\n")
    
    temp_manifest.close()
    return temp_manifest.name

class ErrorHandlingDataset(Dataset):
    """Wrapper dataset that skips corrupted videos"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.valid_items = []
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Pre-validate all videos by iterating through the dataset"""
        print("Validating videos...")
        iterator = iter(self.base_dataset)
        idx = 0
        while True:
            try:
                item = next(iterator)
                self.valid_items.append(item)
                idx += 1
            except StopIteration:
                break
            except (RuntimeError, Exception) as e:
                warnings.warn(f"Skipping corrupted video at index {idx}: {str(e)}")
                idx += 1
        print(f"Found {len(self.valid_items)} valid videos")
    
    def __len__(self):
        return len(self.valid_items)
    
    def __getitem__(self, idx):
        return self.valid_items[idx]

class train_dataloader(DataLoader):
    def __init__(self, dataset_df, batch_size, num_workers):
        video_transform = Compose([
            ApplyTransformToKey(key = 'video',
            transform = Compose([
                UniformTemporalSubsample(16),
                Div255(),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                # RandomGrayscale(0.1),
                # RandomAutocontrast(0.5),
                # RandomAdjustSharpness(0.5),
                RandomHorizontalFlip(p=0.5),
            ])),
        ])
        # Get repo root (3 levels up from dataloaders directory)
        repo_root = Path(__file__).resolve().parents[4]
        
        # Convert manifest to absolute paths
        abs_manifest_path = convert_manifest_to_absolute(dataset_df, repo_root)
        
        base_dataset = labeled_video_dataset(
            abs_manifest_path,
            clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform,
            decode_audio=False,
        )
        
        # Wrap with error handling to skip corrupted videos
        dataset = ErrorHandlingDataset(base_dataset)
        
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)