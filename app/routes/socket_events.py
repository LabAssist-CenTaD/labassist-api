import jsonpatch
from copy import deepcopy
from flask import current_app
from flask_socketio import emit, join_room, SocketIO

from app.services.video_task_manager import get_task_status

socketio = None

def init_socketio(socketio_instance: SocketIO) -> None:
    global socketio
    socketio = socketio_instance

    @socketio.on('connect')
    def handle_connect():
        print('Device connected')
        emit('message', {'message': 'Connected to server!'})
        
    @socketio.on('disconnect')
    def handle_disconnect():
        #print('Device disconnected')
        pass
        
    @socketio.on('authenticate')
    def handle_authenticate(data):
        """Socket event to authenticate a device using a device ID.
        Args:
            data (dict): A dictionary containing the device ID.
        Returns:
            response (str): A string response containing the status of the authentication.
        """
        if 'device_id' in data:
            device_id = data['device_id']
            join_room(device_id)
            vjm = current_app.extensions['vjm']
            socketio.start_background_task(progress_updater, vjm)
            return "OK", {'message': 'Authenticated!', 'cached_videos': vjm.get_device_videos(device_id)}
        else:
            return "ERROR", {'message': 'Device ID not provided', 'cached_videos': []}
        
    @socketio.on('patch_backend')
    def handle_apply_patch(data):
        """Socket event to apply a patch to the backend.
        Args:
            data (dict): A dictionary containing the device ID and the patch to apply.
        """
        device_id = data['device_id']
        patch = jsonpatch.JsonPatch(data['patch'])
        vjm = current_app.extensions['vjm']
        result = vjm.apply_patch(device_id, patch)
        emit('update', {'data': result}, room=device_id)

    def progress_updater(vjm, interval=1):
        """Function to update the progress of video analysis tasks in the background.
        Args:
            vjm (VideoJSONManager): An instance of the VideoJSONManager class.
        """
        status_map = {
            'PENDING': 'predicting', # should be 'queued' instead of 'predicting', but the 'predicting' state is non functional in celery 
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
                            vjm.add_annotation(device_id, video_name, annotation)
                new_device_videos = vjm.get_device_videos(device_id)
                if old_device_videos != new_device_videos:
                    patch = vjm.create_patch(old_device_videos, new_device_videos)
                    print(f'Patching frontend: {patch}')
                    socketio.emit('patch_frontend', patch.to_string(), room=device_id)
            socketio.sleep(interval)