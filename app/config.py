from pathlib import Path

class Config:
    DEBUG_MODE = True
    
    UPLOAD_FOLDER = Path('uploads')
    CLEANUP_UPLOADS = False
    
    CELERY_RESULT_BACKEND = 'db+sqlite:///results.db'
    CELERY_BROKER_URL = 'amqp://guest@localhost//'
    CELERY_TASK_TRACK_STARTED = True
    CELERY_TRACK_STARTED = True
    
    SOCKETIO_MESSAGE_QUEUE = 'db+sqlite:///socketio.db'
    
    SQLALCHEMY_DATABASE_URI = 'sqlite:///annotations.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    VIDEO_JSON_PATH = Path('video_json.json')
    
    SECRET_KEY = 'secret'
    
    ACTION_MODEL_PATH = Path(r'app\ml_models\action_detection\weights\model-v1.pth')
    OBJECT_MODEL_PATH = Path(r'app\ml_models\object_detection\weights\obj_detect_best_v5.pt')
    