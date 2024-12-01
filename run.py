from app import create_app

socketio, app = create_app()
celery_app = app.extensions['celery']

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)