from flask import Flask, session

from app.routes.video_routes import video_routes

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    app.register_blueprint(video_routes)
    
    @app.teardown_appcontext
    def remove_session(exception=None):
        # delete all videos in session from the uploads folder
        if 'videos' in session:
            for video in session['videos']:
                video_path = app.config['UPLOAD_FOLDER'] / video
                video_path.unlink(missing_ok=True)
        session.pop('videos', None)
        print('Session removed')
    
    return app