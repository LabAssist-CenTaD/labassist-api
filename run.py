from app import create_app

socketio, app = create_app()
celery_app = app.extensions['celery']

if __name__ == '__main__':
    # import subprocess
    # subprocess.Popen('celery -A run.celery_app worker --loglevel=info --pool=gevent --concurrency=8'.split(' '))
    socketio.run(app, host='0.0.0.0', port=5500)