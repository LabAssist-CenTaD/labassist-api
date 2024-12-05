from flask_socketio import emit, join_room, leave_room, SocketIO
from flask import current_app
from werkzeug.utils import secure_filename
from copy import deepcopy
import os

from app.services.video_analysis import get_task_status

socketio = None

def init_socketio(socketio_instance: SocketIO) -> None:
    global socketio
    socketio = socketio_instance

    @socketio.on('connect')
    def handle_connect():
        #print('Device connected')
        emit('message', {'message': 'Connected to server!'})
        
    @socketio.on('disconnect')
    def handle_disconnect():
        #print('Device disconnected')
        pass
        
    @socketio.on('authenticate')
    def handle_authenticate(data):
        #print(f'Authenticating: {data}')
        if 'device_id' in data:
            device_id = data['device_id']
            join_room(device_id)
            vjm = current_app.extensions['vjm']
            socketio.start_background_task(progress_updater, vjm)
            return "OK", {'message': 'Authenticated!', 'cached_videos': vjm.get_device_videos(device_id)}
        else:
            return "ERROR", {'message': 'Device ID not provided', 'cached_videos': []}
        
    @socketio.on('button_click')
    def handle_button_click(data):
        print(f'Button clicked: {data}')
        emit('message', {'message': 'Button clicked!'})
        
    @socketio.on('patch_backend')
    def handle_apply_patch(data):
        print(f'Applying patch: {data}')
        device_id = data['device_id']
        patch = data['patch']
        vjm = current_app.extensions['vjm']
        result = vjm.apply_patch(device_id, patch)
        emit('update', {'data': result}, room=device_id)

    def progress_updater(vjm):
        status_map = {
            'PENDING': 'queued',
            'STARTED': 'predicting',
            'SUCCESS': 'complete',
            'FAILURE': 'warnings-present'
        }
        while True:
            for device_id in vjm.video_json['active_tasks']:
                old_device_videos = deepcopy(vjm.get_device_videos(device_id))
                for video_name, task_id in vjm.video_json['active_tasks'][device_id].copy().items():
                    status, result = get_task_status(task_id)
                    vjm.clear_status(device_id, video_name)
                    vjm.add_status(device_id, video_name, status_map[status])
                    if status in ['SUCCESS', 'FAILURE']:
                        vjm.remove_task(device_id, video_name)
                    if status in ['PENDING', 'STARTED']:
                        vjm.clear_annotations(device_id, video_name)
                    elif status == 'SUCCESS':   
                        for annotation in result:
                            vjm.add_annotation(device_id, video_name, annotation['type'], annotation['message'], annotation['timestamp'])
                new_device_videos = vjm.get_device_videos(device_id)
                if old_device_videos != new_device_videos:
                    patch = vjm.create_patch(old_device_videos, new_device_videos)
                    print(f'Patching frontend: {patch}')
                    socketio.emit('patch_frontend', patch.to_string(), room=device_id)
            socketio.sleep(1)