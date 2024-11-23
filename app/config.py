from pathlib import Path

class Config:
    UPLOAD_FOLDER = Path('uploads')
    CLEANUP_UPLOADS = False
    
    CELERY_RESULT_BACKEND = 'db+sqlite:///results.db'
    CELERY_BROKER_URL = 'amqp://guest@localhost//'
    
    SECRET_KEY = 'secret'
    
    ACTION_MODEL_PATH = Path(r'C:\Users\zedon\Documents\GitHub\labassist-api\checkpoints\best\epoch=5-acc=0.8500000238418579.ckpt')
    OBJECT_MODEL_PATH = Path(r'app\ml_models\object_detection\weights\obj_detect_best_v5.pt')
    