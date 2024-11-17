import os

from flask import Blueprint, jsonify, request, session, current_app
from werkzeug.utils import secure_filename

video_routes = Blueprint('video_routes', __name__)

@video_routes.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file and file.filename.endswith(('.mp4', '.mov', '.avi')):
        filename = secure_filename(file.filename)
        uploads_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_folder, exist_ok=True)
        file_path = uploads_folder / filename
        file.save(file_path)
        
        if 'videos' not in session:
            session['videos'] = []
        session['videos'].append(filename)
        
        return jsonify({'message': 'Video uploaded successfully'}), 201
    else:
        return jsonify({'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}), 400

@video_routes.route('/process_video/<clip_name>', methods=['POST'])
def process_video(clip_name):
    if 'videos' not in session or clip_name not in session['videos']:
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': session.get('videos', [])
            }
        ), 404
    # Process video
    return jsonify({'message': 'Video processed successfully'}), 200