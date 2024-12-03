import torch
import numpy as np

from pytorchvideo.transforms import (
    ApplyTransformToKey, 
    UniformTemporalSubsample,
    Div255
)

from torchvision.transforms import (
    Compose,
)

from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

from app.ml_models.action_detection.model import ActionDetectionModel

classes = {
    0: 'Correct',
    1: 'Incorrect',
    2: 'Stationary',
}

video_transform = Compose([
    UniformTemporalSubsample(16),
    Div255(),
    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

def load_model(model_path):
    model = ActionDetectionModel()
    model.load_state_dict(torch.load(model_path, weights_only=True)) #['state_dict']
    model.eval()
    return model

def predict_action(video, model, transform=video_transform):
    video = transform(video)
    video = video.unsqueeze(0)  
    pred = model(video).detach().cpu().numpy().flatten()
    return classes[int(np.argmax(pred, axis=0))]