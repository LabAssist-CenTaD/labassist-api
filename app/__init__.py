from flask_cors import CORS
from flask import Flask
from flask_socketio import SocketIO

from app.routes.video_routes import video_routes
from app.utils.celery_tasks import celery_init_app
from app.services.video_json_manager import VideoJSONManager

def create_app() -> tuple[SocketIO, Flask]:
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.config.from_mapping(
        CELERY=dict(
            broker_url=app.config['CELERY_BROKER_URL'],
            result_backend=app.config['CELERY_RESULT_BACKEND'],
            task_track_started = True,
            # task_ignore_result=True,
            broker_connection_retry_on_startup=True,
        ),
    )
        
    vjm = VideoJSONManager(json_path=app.config['VIDEO_JSON_PATH'])
    vjm.load_json()
    vjm.sync_videos(app.config['UPLOAD_FOLDER'])
    app.extensions['vjm'] = vjm
        
    app.register_blueprint(video_routes)
    
    CORS(app)
    
    celery_app = celery_init_app(app)
    # use pickle as serializer as numpy arrays are not serializable with json
    # celery_app.conf.update(
    #     task_serializer='pickle', 
    #     result_serializer='pickle',
    #     accept_content=['pickle']
    # )
    app.extensions['celery','json','application/text'] = celery_app
    
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')
    from app.routes import socket_events
    socket_events.init_socketio(socketio)
    app.extensions['socketio'] = socketio
    
    return socketio, app