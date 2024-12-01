from flask_socketio import emit, join_room, leave_room, SocketIO
from flask import current_app
from werkzeug.utils import secure_filename
import os

socketio = None

def init_socketio(socketio_instance: SocketIO):
    global socketio
    socketio = socketio_instance

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('message', {'data': 'Connected to server!'})
        #socketio.start_background_task(background_task)
        
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        
    @socketio.on('button_click')
    def handle_button_click(data):
        print(f'Button clicked: {data}')
        emit('message', {'data': 'Button clicked!'})
        
    @socketio.on('upload')
    def handle_upload(data):
        print(f'Uploading file: {data}')
        try:
            filename = secure_filename(data['filename'])
            uploads_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(uploads_folder, exist_ok=True)
            # file_path = os.path.join(uploads_folder, filename)
            # with open(file_path, 'wb') as f:
            #     f.write(data['video'])
            emit('upload_response', {'status': 'success'})
        except Exception as e:
            emit('upload_response', {'status': 'error', 'message': str(e)})

    def background_task():
        while True:
            socketio.emit('update', {'data': 'Periodic update'})
            socketio.sleep(1)