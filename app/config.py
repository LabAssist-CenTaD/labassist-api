import os
from pathlib import Path

class Config:
    DEBUG_MODE = os.environ.get('DEBUG_MODE', False)
    
    UPLOAD_FOLDER = Path(os.environ.get('UPLOAD_FOLDER', 'uploads'))
    CLEANUP_UPLOADS = os.environ.get('CLEANUP_UPLOADS', False)
    
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'db+sqlite:///results.db')
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'amqp://guest@localhost//')
    CELERY_TASK_TRACK_STARTED = os.environ.get('CELERY_TRACK_STARTED', True)
    CELERY_TRACK_STARTED = os.environ.get('CELERY_TRACK_STARTED', True)
    
    SOCKETIO_MESSAGE_QUEUE = os.environ.get('SOCKETIO_MESSAGE_QUEUE', 'db+sqlite:///socketio.db')
    
    VIDEO_JSON_PATH = os.environ.get('VIDEO_JSON_PATH', 'video_json.json')
    
    SECRET_KEY = os.environ.get('SECRET_KEY', 'secret')
    
    ACTION_MODEL_PATH = Path(os.environ.get('ACTION_MODEL_PATH', r'app\ml_models\action_detection\weights\model-v1.pth'))
    OBJECT_MODEL_PATH = Path(os.environ.get('OBJECT_MODEL_PATH', r'app\ml_models\object_detection\weights\obj_detect_best_v5.pt'))
    