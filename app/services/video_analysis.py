import os
import cv2
import math
import mmap
from flask import current_app
from celery import group, chord
from celery.result import AsyncResult, GroupResult
from app.utils.celery_tasks import process_video_clip, process_results
from app.utils.progress_chord import ProgressChord

def analyze_clip(device_id, clip_path, interval=4, cleanup=True) -> GroupResult:
    """Function to start the analysis process of a video clip.

    Args:
    
        clip_path (str): The path to the video clip.
    
        interval (int): The interval in seconds at which to analyze the video clip.
        
        cleanup (bool): A flag to indicate whether to cleanup the uploaded files after analysis.
    
    Returns:
    
        result (GroupResult): The result of the analysis process.
    """
    mmap_file = current_app.config['UPLOAD_FOLDER'] / device_id / f'{clip_path}.mmap'
    with open(current_app.config['UPLOAD_FOLDER'] / device_id / clip_path, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        with open(mmap_file, 'wb') as mmf:
            mmf.write(mm)
            
    cap = cv2.VideoCapture(str(mmap_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    for i in range(0, frame_count, int(interval * fps)):
        start_frame = i
        end_frame = min(i + int(interval * fps), frame_count)
        result = process_video_clip.s(str(mmap_file), start_frame, end_frame)
        results.append(result)
        # break #TODO: remove this line to process the entire video
    # task_group = group(results)
    # result = task_group.apply_async()
    result = chord(results)(process_results.s())
    
    if cleanup:
        # cleanup uploads folder
        os.remove(mmap_file)
        os.remove(current_app.config['UPLOAD_FOLDER'] / device_id / clip_path)
    
    return result
    
def get_task_status(result_id: str) -> tuple[str, dict]:
    """Function to get the status of a list of task IDs.

    Args:
    
        result_id (str): The ID of the task result.
        
    Returns:
    
        tuple[str, dict]: A tuple containing the status of the task and the result of the task.
    """
    result = AsyncResult(result_id)
    if result.state == 'SUCCESS':
        return 'SUCCESS', [item for item in result.get()]
    elif result.state == 'FAILURE':
        return 'FAILURE', {'message': 'An error occurred while processing the video clip'}
    elif result.state == 'STARTED':
        return 'STARTED', {'message': 'The video clip is still being processed'}
    elif result.state == 'PENDING':
        return 'PENDING', {'message': 'The video clip is in the queue'}
    elif result.state == 'RETRY':
        return 'RETRY', {'message': 'The video clip is being retried'}
    else:
        print('unknown state:', result.state)
        return 'UNKNOWN', {'message': 'The video clip status is unknown'}  

