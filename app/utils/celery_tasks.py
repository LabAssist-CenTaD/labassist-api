import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from flask import Flask, current_app
from celery import Celery, Task, shared_task

from app.config import Config
from app.ml_models.action_detection.inference import predict_action, load_model
from app.utils.object_detection_utils import predict_on_clips, get_valid_flask, overlay_boxes, square_crop

object_model = YOLO(Config.OBJECT_MODEL_PATH)
action_model = load_model(Config.ACTION_MODEL_PATH)

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    
    
    return celery_app

@shared_task(ignore_result=False)
def example_task(n: int) -> torch.Tensor:
    print(n)
    torch.zeros(1, 1).cuda()
    return torch.tensor(n)

@shared_task(ignore_result=False)
def process_video_clip(clip_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    clip = np.array(cap.read()[1])
    obj_pred = object_model.predict(clip, verbose=False)
    flask_box = get_valid_flask(obj_pred)
    if flask_box is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
        clip = []
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame = square_crop(frame, flask_box, target_size=(224, 224))   
            clip.append(frame)
        clip = np.array(clip)
        clip = torch.tensor(clip, dtype=torch.float32)
        cap.release()

        clip = clip.permute(3, 0, 1, 2)
        pred = predict_action(clip, action_model)
        return int(np.argmax(pred, axis=0))
    else:
        return None
    
    
    