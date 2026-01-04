from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from torch.utils.data import DataLoader

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
        dataset = labeled_video_dataset(
            dataset_df,
            clip_sampler=make_clip_sampler('random', 2),
            transform=video_transform,
            decode_audio=False,
        )
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)