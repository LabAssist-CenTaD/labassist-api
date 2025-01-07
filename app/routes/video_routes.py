import os
import jsonpatch
from copy import deepcopy
from flask import Blueprint, jsonify, request, current_app, Response
from werkzeug.utils import secure_filename

from app.services.video_task_manager import analyze_clip, get_task_status

video_routes = Blueprint('video_routes', __name__)

@video_routes.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API is running'})

@video_routes.route('/upload', methods=['POST'])
def upload_video():
    """Route to upload a video file to the server.
    Args:
        file (FileStorage): The video file to upload.
        device_id (str): The ID of the device that uploaded the video.
    Returns:
        response (json): A JSON response containing the message and filename of the uploaded video
    """
    device_id = request.form.get('device_id')
    file = request.files['video']
    vjm = current_app.extensions['vjm']
    if not (file and file.filename.endswith(('.mp4', '.mov', '.avi', '.MOV'))):
        return jsonify({'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}), 400
    
    filename = secure_filename(file.filename)
    uploads_folder = current_app.config['UPLOAD_FOLDER'] / device_id
    os.makedirs(uploads_folder, exist_ok=True)
    
    file_path = uploads_folder / filename
    file.save(file_path)
    
    patch = vjm.add_video(device_id, filename, str(file_path))
    if isinstance(patch, jsonpatch.JsonPatch):
        current_app.extensions['socketio'].emit('patch_frontend', patch.to_string(), room=device_id)
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename
        }), 201
    else:
        current_app.extensions['socketio'].emit('message', {'data': patch['message']}, room=device_id)
        return jsonify(patch), 400
        

@video_routes.route('/process_video/<clip_name>', methods=['GET'])
def process_video(clip_name):
    """Route to process a video clip.
    Args:
        clip_name (str): The name of the video clip to process.
        device_id (str): The ID of the device that uploaded the video.
    Returns:
        response (json): A JSON response containing the task ID of the video processing task
    """
    device_id = request.args.get('device_id')
    vjm = current_app.extensions['vjm']
    if not device_id:
        return jsonify({'message': 'device ID is required'}), 400
    elif device_id not in vjm.video_json['videos']:
        return jsonify({'message': 'device ID not found'}), 404
    elif device_id not in os.listdir(current_app.config['UPLOAD_FOLDER']) or clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER'] / device_id) or clip_name not in [video['file_name'] for video in vjm.get_device_videos(device_id)]:
        vjm.sync_videos(current_app.config['UPLOAD_FOLDER'])
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': [video['file_name'] for video in vjm.get_device_videos(device_id)]
            }
        ), 404
        
    print(f'Processing video {clip_name} for device {device_id}')
    task_result = analyze_clip(device_id, clip_name, cleanup=current_app.config['CLEANUP_UPLOADS'])
    vjm.add_task(device_id, clip_name, task_result.id)
    old_device_videos = deepcopy(vjm.get_device_videos(device_id))
    vjm.clear_status(device_id, clip_name)
    vjm.add_status(device_id, clip_name, 'queued')
    new_device_videos = vjm.get_device_videos(device_id)
    patch = vjm.create_patch(old_device_videos, new_device_videos)
    if isinstance(patch, jsonpatch.JsonPatch):
        current_app.extensions['socketio'].emit('patch_frontend', patch.to_string(), room=device_id)
        return jsonify({
            'task_id': task_result.id,
            'message': 'Video processing started',
            'filename': clip_name
        }), 202
    else:
        current_app.extensions['socketio'].emit('message', {'data': patch['message']}, room=device_id)
        return jsonify(patch), 400
    

@video_routes.route('/video/<clip_name>', methods=['GET'])
def get_video(clip_name):
    """Route to get a video clip.
    Args:
        clip_name (str): The name of the video clip to get.
        device_id (str): The ID of the device that uploaded the video.
    Returns:
        response (file): The video file to download
    """
    device_id = request.args.get('device_id')
    vjm = current_app.extensions['vjm']
    if not device_id:
        return jsonify({'message': 'device ID is required'}), 400
    elif device_id not in vjm.video_json['videos']:
        return jsonify({'message': 'device ID not found'}), 404
    elif device_id not in os.listdir(current_app.config['UPLOAD_FOLDER']) or clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER'] / device_id) or clip_name not in [video['file_name'] for video in vjm.get_device_videos(device_id)]:
        vjm.sync_videos(current_app.config['UPLOAD_FOLDER'])
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': [video['file_name'] for video in vjm.get_device_videos(device_id)]
            }
        ), 404
    with open(current_app.config['UPLOAD_FOLDER'] / device_id / clip_name, 'rb') as f:
        return Response(
            f.read(), 
            mimetype='video/mp4',
            content_type='video/mp4',
            headers={'Content-Disposition': f'attachment; filename={clip_name}'}
        )
        
@video_routes.route('/delete/<clip_name>', methods=['GET'])
def delete_video(clip_name):
    """Route to delete a video clip.
    Args:
        clip_name (str): The name of the video clip to delete.
        device_id (str): The ID of the device that uploaded the video.
    Returns:
        response (json): A JSON response containing the message of the deletion status
    """
    device_id = request.args.get('device_id')
    vjm = current_app.extensions['vjm']
    if not device_id:
        return jsonify({'message': 'device ID is required'}), 400
    elif device_id not in vjm.video_json['videos']:
        return jsonify({'message': 'device ID not found'}), 404
    elif device_id not in os.listdir(current_app.config['UPLOAD_FOLDER']) or clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER'] / device_id) or clip_name not in [video['file_name'] for video in vjm.get_device_videos(device_id)]:
        vjm.sync_videos(current_app.config['UPLOAD_FOLDER'])
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': [video['file_name'] for video in vjm.get_device_videos(device_id)]
            }
        ), 404
    os.remove(current_app.config['UPLOAD_FOLDER'] / device_id / clip_name)
    try:
        os.remove(current_app.config['UPLOAD_FOLDER'] / device_id / f'{clip_name}.mmap')
    except FileNotFoundError:
        pass
    patch = vjm.remove_video(device_id, clip_name)
    if isinstance(patch, jsonpatch.JsonPatch):
        current_app.extensions['socketio'].emit('patch_frontend', patch.to_string(), room=device_id)
        return jsonify({'message': 'Video deleted successfully'}), 200
    else:
        current_app.extensions['socketio'].emit('message', {'data': patch['message']}, room=device_id)
        return jsonify(patch), 400
    

# This function is currently not in use as polling is done through the socket connection instead
@video_routes.route('/get_task_status/<clip_name>', methods=['GET'])
def get_task_status_route(clip_name):
    """Route to get the status of a video processing task.
    Args:
        clip_name (str): The name of the video clip to process.
        device_id (str): The ID of the device that uploaded the video.
    Returns:
        response (json): A JSON response containing the status of the video processing task
    """
    device_id = request.args.get('device_id')
    vjm = current_app.extensions['vjm']
    if not device_id:
        return jsonify({'message': 'device ID is required'}), 400
    elif device_id not in vjm.video_json['videos']:
        return jsonify({'message': 'device ID not found'}), 404
    elif clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER'] / device_id) or clip_name not in [video['file_name'] for video in vjm.get_device_videos(device_id)]:
        vjm.sync_videos(current_app.config['UPLOAD_FOLDER'])
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': [video['file_name'] for video in vjm.get_device_videos(device_id)]
            }
        ), 404
    task_id = vjm.get_task(device_id, clip_name)
    if task_id is None:
        return jsonify({'message': 'Task not found'}), 404
    return jsonify(get_task_status(task_id)[1])