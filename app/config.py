from pathlib import Path

class Config:
    UPLOAD_FOLDER = Path('uploads')
    CLEANUP_UPLOADS = False
    
    CELERY_RESULT_BACKEND = 'db+sqlite:///results.db'
    CELERY_BROKER_URL = 'amqp://guest@localhost//'
    
    SECRET_KEY = 'secret'
    
    ACTION_MODEL_PATH = Path(r'app\ml_models\action_detection\weights\model-v1.pth')
    OBJECT_MODEL_PATH = Path(r'app\ml_models\object_detection\weights\obj_detect_best_v5.pt')
    