import torch
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask
from celery import Celery, Task, shared_task

from app.config import Config
from app.ml_models.action_detection.inference import predict_action, load_model
from app.utils.object_detection_utils import get_valid_flask, square_crop

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

@shared_task(ignore_result=False, trail=True)
def process_video_clip(clip_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        return {
            "start_seconds": int(start_frame / fps),
            "end_seconds": int(end_frame / fps),
            "swirling": pred
        }
    else:
        return None
    
@shared_task(ignore_result=False, trail=True)
def process_results(results: list) -> list:
    annotations = []
    for result in results:
        if result is not None:
            if result["swirling"] == "Correct":
                type = "info"
                message = "Correct swirling detected"
            elif result["swirling"] == "Incorrect":
                type = "error"
                message = "Conical flask should not be grinded on the white tile"
            elif result["swirling"] == "Stationary":
                type = "error"
                message = "Conical flask should be swirled to ensure proper mixing"
            annotations.append({
                "type": type,
                "message": message,
                "timestamp": f"{result['start_seconds'] // 3600:02}:{result['start_seconds'] % 3600 // 60:02}:{result['start_seconds'] % 60:02}"
            })
    return annotations
    
    
    