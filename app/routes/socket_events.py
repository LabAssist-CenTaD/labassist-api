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
        emit('message', {'message': 'Connected to server!'})
        #socketio.start_background_task(background_task)
        
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        
    @socketio.on('authenticate')
    def handle_authenticate(data):
        print(f'Authenticating: {data}')
        if 'client_id' in data:
            client_id = data['client_id']
            join_room(client_id)
            vjm = current_app.extensions['vjm']
            return "OK", {'message': 'Authenticated!', 'cached_videos': vjm.get_client_videos(client_id)}
        else:
            return "ERROR", {'message': 'Client ID not provided', 'cached_videos': []}
        
    @socketio.on('button_click')
    def handle_button_click(data):
        print(f'Button clicked: {data}')
        emit('message', {'message': 'Button clicked!'})
        
    @socketio.on('patch_backend')
    def handle_apply_patch(data):
        print(f'Applying patch: {data}')
        client_id = data['client_id']
        patch = data['patch']
        vjm = current_app.extensions['vjm']
        result = vjm.apply_patch(client_id, patch)
        emit('update', {'data': result}, room=client_id)

    def background_task():
        while True:
            socketio.emit('update', {'message': 'Periodic update'})
            socketio.sleep(1)