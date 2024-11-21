from pathlib import Path

class Config:
    UPLOAD_FOLDER = Path('uploads')
    CLEANUP_UPLOADS = False
    
    CELERY_RESULT_BACKEND = 'db+sqlite:///results.db'
    CELERY_BROKER_URL = 'amqp://guest@localhost//'
    
    SECRET_KEY = 'secret'
    
    