import os
from flask import Blueprint, jsonify, request, session, current_app
from werkzeug.utils import secure_filename
from celery.result import GroupResult
from flask_socketio import emit

from app.services.video_analysis import analyze_clip, get_task_status

video_routes = Blueprint('video_routes', __name__)

@video_routes.route('/upload', methods=['POST'])
def upload_video():
    client_id = request.form.get('id')
    file = request.files['video']
    vjm = current_app.extensions['vjm']
    if file and file.filename.endswith(('.mp4', '.mov', '.avi', '.MOV')):
        filename = secure_filename(file.filename)
        uploads_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_folder, exist_ok=True)
        
        file_path = uploads_folder / filename
        file.save(file_path)
        
        if 'videos' not in session:
            session['videos'] = []
        session['videos'].append(filename)
        
        patch = vjm.add_video(client_id, filename, file_path)
        current_app.extensions['socketio'].emit('create_patch', {'data': patch}, room=client_id)

        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename
        }), 201
    else:
        print('Invalid file format')
        return jsonify({'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}), 400

@video_routes.route('/process_video/<clip_name>', methods=['GET'])
def process_video(clip_name):
    client_id = request.args.get('client_id')
    vjm = current_app.extensions['vjm']
    if not client_id:
        return jsonify({'message': 'Client ID is required'}), 400
    elif client_id not in vjm.video_json:
        return jsonify({'message': 'Client ID not found'}), 404
    elif clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER']) or clip_name not in vjm.get_client_videos(client_id):
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': [video['fileName'] for video in vjm.get_client_videos(client_id)]
            }
        ), 404
        
    print(f'Processing video {clip_name} for client {client_id}')
    task_result = analyze_clip(clip_name, cleanup=current_app.config['CLEANUP_UPLOADS']) #TODO: add progress chord
    #task_dict[client_id][clip_name] = task_result
    return jsonify({'task_id': task_result.id}), 202

# @video_routes.route('/get_task_status/<clip_name>', methods=['GET'])
# def get_task_status_route(clip_name):
#     if 'tasks' not in session or clip_name not in session['tasks']:
#         return jsonify(
#             {
#                 'message': 'No tasks found for this video',
#                 'available_videos': list(session.get('tasks', {}).keys())
#             }
#         ), 404
#     task_status = get_task_status(task_dict[clip_name])
#     return jsonify(task_status), 200