import os
import cv2
import mmap
import gevent
from flask import current_app
from celery import chord
from celery.result import AsyncResult, GroupResult
from app.utils.celery_tasks import process_video_clip, process_results

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
        gevent.sleep(0)
    result = chord(results)(process_results.s())
    
    # if cleanup:
    #     # cleanup uploads folder
    #     os.remove(mmap_file)
    
    return result
    
def get_task_status(result_id: str) -> tuple[str, dict]:
    """Function to get the status of a task ID.
    Args:
        result_id (str): The ID of the task result.
    Returns:
        tuple[str, dict]: A tuple containing the status of the task and the result of the task.
    """
    result = AsyncResult(result_id)
    state = str(result.state)
    if state == 'SUCCESS':
        return 'SUCCESS', [item for item in result.get()]
    elif state == 'FAILURE':
        return 'FAILURE', {'message': 'An error occurred while processing the video clip'}
    elif state == 'STARTED':
        return 'STARTED', {'message': 'The video clip is still being processed'}
    elif state == 'PENDING':
        return 'PENDING', {'message': 'The video clip is in the queue'}
    elif state == 'RETRY':
        return 'RETRY', {'message': 'The video clip is being retried'}
    else:
        print(f'unknown state: {result.state} of type {type(result.state)}')
        return result.state, {'message': 'The video clip status is unknown'}  
        # return 'SUCCESS', [item for item in result.get()] # idk why tf sometimes this would be reached even with a SUCCESS state that is printed

