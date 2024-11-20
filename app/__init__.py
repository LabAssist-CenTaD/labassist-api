import subprocess
from flask_cors import CORS
from flask import Flask, session

from app.routes.video_routes import video_routes
from app.utils.celery_tasks import celery_init_app, example_task

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.config.from_mapping(
        CELERY=dict(
            broker_url=app.config['CELERY_BROKER_URL'],
            result_backend=app.config['CELERY_RESULT_BACKEND'],
            task_ignore_result=True,
        ),
    )
    app.register_blueprint(video_routes)
    
    CORS(app)
    
    celery_app = celery_init_app(app)
    # use pickle as serializer as numpy arrays are not serializable with json
    celery_app.conf.update(
        task_serializer='pickle', 
        result_serializer='pickle',
        accept_content=['pickle']
    )
    celery_app
    app.extensions['celery'] = celery_app
    
    # @app.teardown_appcontext
    # def remove_session(exception=None):
    #     # delete all videos in session from the uploads folder
    #     if 'videos' in session:
    #         for video in session['videos']:
    #             video_path = app.config['UPLOAD_FOLDER'] / video
    #             video_path.unlink(missing_ok=True)
    #     session.pop('videos', None)
    #     print('Session removed')
    
    return app