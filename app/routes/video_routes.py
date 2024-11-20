import os
import cv2
import math
import time
import mmap

import numpy as np
from flask import Blueprint, jsonify, request, session, current_app
from werkzeug.utils import secure_filename

from app.utils.celery_tasks import celery_init_app, example_task, process_video_clip

video_routes = Blueprint('video_routes', __name__)

@video_routes.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file and file.filename.endswith(('.mp4', '.mov', '.avi', '.MOV')):
        start_time = time.time()
        filename = secure_filename(file.filename)
        uploads_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_folder, exist_ok=True)
        
        file_path = uploads_folder / filename
        file.save(file_path)
        
        if 'videos' not in session:
            session['videos'] = []
        session['videos'].append(filename)
        
        print('Video uploaded in', time.time() - start_time, 'seconds')
        start_time = time.time()
        mmap_file = current_app.config['UPLOAD_FOLDER'] / f'{filename}.mmap'
        with open(current_app.config['UPLOAD_FOLDER'] / filename, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            with open(mmap_file, 'wb') as mmf:
                mmf.write(mm)
                
        # run object detection on every 4 seconds of the video
        interval = 4
        cap = cv2.VideoCapture(str(mmap_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'frame count: {frame_count}, fps: {fps}')
        for i in range(0, frame_count, int(interval * fps)):
            print(f'extracting clip {i // int(interval * fps) + 1}/{math.ceil(frame_count / int(interval * fps))}', end='\r')
            start_frame = i
            end_frame = min(i + int(interval * fps), frame_count)
            process_video_clip.delay(str(mmap_file), start_frame, end_frame)

        print('Video extracted in', time.time() - start_time, 'seconds')
        return jsonify({'message': 'Video uploaded successfully'}), 201
    else:
        print('Invalid file format')
        return jsonify({'message': 'Invalid file format. Must be .mp4, .mov, or .avi'}), 400

@video_routes.route('/process_video/<clip_name>', methods=['GET'])
def process_video(clip_name):
    # if 'videos' not in session or clip_name not in session['videos']:
    if clip_name not in os.listdir(current_app.config['UPLOAD_FOLDER']):
        return jsonify(
            {
                'message': 'Video not found or session expired. Please upload the video again',
                'available_videos': session.get('videos', [])
            }
        ), 404
    # convert video to memory mapped file
    start_time = time.time()
    mmap_file = current_app.config['UPLOAD_FOLDER'] / f'{clip_name}.mmap'
    with open(current_app.config['UPLOAD_FOLDER'] / clip_name, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        with open(mmap_file, 'wb') as mmf:
            mmf.write(mm)
            
    # run object detection on every 4 seconds of the video
    interval = 4
    cap = cv2.VideoCapture(str(mmap_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'frame count: {frame_count}, fps: {fps}')
    for i in range(0, frame_count, int(interval * fps)):
        print(f'extracting clip {i // int(interval * fps) + 1}/{math.ceil(frame_count / int(interval * fps))}', end='\r')
        start_frame = i
        end_frame = min(i + int(interval * fps), frame_count)
        process_video_clip.delay(str(mmap_file), start_frame, end_frame)

    print('Video extracted in', time.time() - start_time, 'seconds')
    return jsonify({'message': 'Video processed successfully'}), 200

@video_routes.route('/example_task', methods=['GET'])
def run_example_task():
    print('Running example task')
    result = example_task.delay(42)
    print('Task ID:', result.id)
    return jsonify({'task_id': result.id}), 200