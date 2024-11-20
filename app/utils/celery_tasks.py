import torch
import mmap
import numpy as np
from flask import Flask
from celery import Celery, Task, shared_task

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
    # read video clip from memory mapped file
    with open(clip_path, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.seek(start_frame)
        clip = np.frombuffer(mm.read(end_frame - start_frame), dtype=np.uint8)
        print(f'clip shape: {clip.shape}')
        return {'clip': clip_path, 'start_frame': start_frame, 'end_frame': end_frame}