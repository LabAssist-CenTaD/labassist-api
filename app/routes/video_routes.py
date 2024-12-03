import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from celery.result import GroupResult
from flask_socketio import emit
import jsonpatch

from app.services.video_analysis import analyze_clip, get_task_status

video_routes = Blueprint('video_routes', __name__)

@video_routes.route('/upload', methods=['POST'])
def upload_video():
    device_id = request.form.get('device_id')
    file = request.files['video']
    vjm = current_app.extensions['vjm']
    if file and file.filename.endswith(('.mp4', '.mov', '.avi', '.MOV')):
        filename = secure_filename(file.filename)
        uploads_folder = current_app.config['UPLOAD_FOLDER'] / device_id
        os.makedirs(uploads_folder, exist_ok=True)
        
        file_path = uploads_folder / filename
        file.save(file_path)
        
        patch = vjm.add_video(device_id, filename, str(file_path))
        if isinstance(patch, jsonpatch.JsonPatch):
            current_app.extensions['socketio'].emit('patch_frontend', patch.to_string(), room=device_id)
        else:
            current_app.extensions['socketio'].emit('message', {'data': patch['message']}, room=device_id)
            return jsonify(patch), 400

        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename
        }), 201
    else:
        return jsonify({'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}), 400

@video_routes.route('/process_video/<clip_name>', methods=['GET'])
def process_video(clip_name):
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
    return jsonify({'task_id': task_result.id}), 202

@video_routes.route('/get_task_status/<clip_name>', methods=['GET'])
def get_task_status_route(clip_name):
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