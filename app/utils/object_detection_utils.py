# %%
import os
import cv2
import json
import torch
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from cv2.typing import MatLike
from ultralytics.models.yolo import YOLO
from ultralytics.engine.results import Results

def pad_and_resize(frame: MatLike, target_size=(640, 640)) -> MatLike:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    diff = abs(height - width)
    if height > width:
        pad = np.zeros((height, diff // 2, 3), dtype=np.uint8)
        frame = np.concatenate((pad, frame, pad), axis=1)
    else:
        pad = np.zeros((diff // 2, width, 3), dtype=np.uint8)
        frame = np.concatenate((pad, frame, pad), axis=0)
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    return frame

def extract_clips(video: str | bytes, interval: int, transform = None) -> tuple[MatLike, float]:
    # handles both file paths and bytes
    if isinstance(video, str):
        if not os.path.exists(video):
            raise FileNotFoundError(f'File {video} not found')
        video_path = video
    elif isinstance(video, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.MOV') as f:
            f.write(video)
            video_path = f.name
    else:
        raise ValueError('video must be a file path or bytes. Got: ', type(video))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_clips = []
    for i in range(0, frame_count, int(interval * fps)):
        frames = []
        for j in range(interval * int(fps)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = pad_and_resize(frame)
            frames.append(frame)
        if len(frames) == interval * int(fps):
            extracted_clips.append(np.array(frames))
    cap.release()
    cv2.destroyAllWindows()
    return extracted_clips, fps

def predict_on_clips(clips: list[MatLike], model) -> list[Results]:
    predictions = []
    for clip in clips:
        # get first frame in clip in the shape of (1, 3, 640, 640)
        clip = torch.tensor(clip).permute(0, 3, 1, 2)[0].unsqueeze(0)
        result = model.predict(clip, verbose=False)
        predictions.append(result)
    return predictions

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x5, y5 = max(x1, x3), max(y1, y3)
    x6, y6 = min(x2, x4), min(y2, y4)
    intersection = max(0, x6 - x5) * max(0, y6 - y5)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection
    return intersection / union

def get_biggest_boxes(boxes: list[dict]) -> dict:
    areas = np.array(box['x1'] - box['x2'] * box['y1'] - box['y2'] for box in boxes)
    return boxes[np.argmax(areas)]

def get_objects(prediction: list, obj_type: str) -> list[dict]:
    return [box['box'] for box in prediction if box['name'] == obj_type]

def get_valid_flask(prediction: Results | str) -> dict:
    prediction = json.loads(prediction.to_json() if isinstance(prediction, Results) else prediction)
    flask_boxes = get_objects(prediction, 'Conical-flask')
    burette_boxes = get_objects(prediction, 'Burette')
    
    valid_flasks = []
    # iterate over all permutations of flask and burette boxes, valid flasks are ones with a burette over them
    for flask_box in flask_boxes:
        for burette_box in burette_boxes:
            # x1, y1, x2, y2
            midpoint_of_burette = (burette_box['x1'] + burette_box['x2']) / 2
            if flask_box['x1'] < midpoint_of_burette < flask_box['x2']:
                valid_flasks.append(flask_box)
    
    if len(valid_flasks) == 0:
        return None
    else:
        return get_biggest_boxes(valid_flasks)
    
def get_valid_tile(prediction: Results | str) -> dict:
    prediction = json.loads(prediction.to_json() if isinstance(prediction, Results) else prediction)
    flask_boxes = get_objects(prediction, 'Conical-flask')
    tile_boxes = get_objects(prediction, 'White-tile')
    
    # iterate over all permutations of flask and tile boxes, valid tiles are ones with a flask over them
    valid_tiles = []
    for flask_box in flask_boxes:
        for tile_box in tile_boxes:
            midpoint_of_flask = (flask_box['x1'] + flask_box['x2']) / 2
            if tile_box['x1'] < midpoint_of_flask < tile_box['x2']:
                valid_tiles.append(tile_box)
    if len(valid_tiles) == 0:
        return None
    else:
        return get_biggest_boxes(valid_tiles)
    
def get_valid_funnel(prediction: Results | str) -> dict:
    prediction = json.loads(prediction.to_json() if isinstance(prediction, Results) else prediction)
    funnel_boxes = get_objects(prediction, 'Funnel')
    burette_boxes = get_objects(prediction, 'Burette')
    
    # iterate over all permutations of funnel and burette boxes, valid funnels are ones on top of a burette
    valid_funnels = []
    for funnel_box in funnel_boxes:
        for burette_box in burette_boxes:
            midpoint_of_burette = (burette_box['x1'] + burette_box['x2']) / 2
            if (funnel_box['x1'] < midpoint_of_burette < funnel_box['x2']) and (funnel_box['y1'] > (burette_box['y1'] * 0.8)):
                valid_funnels.append(funnel_box)
    if len(valid_funnels) == 0:
        return None
    else:
        return get_biggest_boxes(valid_funnels)
    
def get_valid_beaker(prediction: Results | str) -> dict:
    prediction = json.loads(prediction.to_json() if isinstance(prediction, Results) else prediction)
    beaker_boxes = get_objects(prediction, 'Beaker')
    burette_boxes = get_objects(prediction, 'Burette')
    
    # iterate over all permutations of beaker and burette boxes, valid beakers are ones that are within a certain x or y threshold to the top of the burette
    valid_beakers = []
    for burette_box in burette_boxes:
        for beaker_box in beaker_boxes:
            burette_x = (burette_box['x1'] + burette_box['x2']) / 2
            burette_y = burette_box['y1']
            threshold = 2.5 * abs(beaker_box['x2'] - beaker_box['x1']) # threshold is 2.5 times the width of the beaker
            mid_x_of_beaker = (beaker_box['x1'] + beaker_box['x2']) / 2
            mid_y_of_beaker = (beaker_box['y1'] + beaker_box['y2']) / 2
            if abs(mid_x_of_beaker - burette_x) < threshold and abs(mid_y_of_beaker - burette_y) < threshold:
                valid_beakers.append(beaker_box)
    if len(valid_beakers) == 0:
        return None
    else:
        return get_biggest_boxes(valid_beakers)

def overlay_boxes(frame, boxes, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

def square_crop(frame, box, pad_top=0.4, pad_bottom=0.2, target_size=(640, 640)):
    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
    frame_height, frame_width = frame.shape[:2]

    # Adjust y1 and y2 with padding
    y1 = max(0, y1 - int((y2 - y1) * pad_top))
    y2 = min(frame_height, y2 + int((y2 - y1) * pad_bottom))

    # Ensure the cropped area is square
    if y2 - y1 > x2 - x1:
        diff = y2 - y1 - (x2 - x1)
        x1 = max(0, x1 - diff // 2)
        x2 = min(frame_width, x2 + diff // 2)
    else:
        diff = x2 - x1 - (y2 - y1)
        y1 = max(0, y1 - diff // 2)
        y2 = min(frame_height, y2 + diff // 2)

    # Ensure the coordinates are within the frame dimensions
    x1 = max(0, min(x1, frame_width - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    y2 = max(0, min(y2, frame_height - 1))

    # Crop the frame
    cropped_frame = frame[y1:y2, x1:x2]

    # Resize the cropped frame to the target size
    resized_frame = cv2.resize(cropped_frame, target_size, interpolation=cv2.INTER_LINEAR)
    resized_frame = (resized_frame * 255).astype(np.uint8)
    return resized_frame

def save_clips_as_mp4(save_dir: str, clips: list[np.ndarray], base_name: str = 'clip', fps: float = 30) -> bool:
    # Ensure the uploads directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Assuming clips is a list of lists of frames (numpy arrays)
    # Save the clips to the uploads directory as mp4 files
    for i, clip in enumerate(clips):
        if len(clip) == 0:
            continue

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        frame_size = (clip[0].shape[1], clip[0].shape[0])  # Width and height of the frames

        out = cv2.VideoWriter(f'{save_dir}/{base_name}_{i}.mp4', fourcc, fps, frame_size)

        for frame in clip:
            # Ensure the frame is in the correct color format (BGR for OpenCV)
            if frame.shape[2] == 3:  # Check if the frame has 3 color channels
                out.write(frame)

        out.release()
        
    return True

if __name__ == '__main__':
    video_path = r"C:\Users\zedon\Videos\PW2024VIDEOS\IMG_8700.MOV"
    interval = 2
    clips = extract_clips(
        video_path, 
        interval, 
        transform=lambda frame: pad_and_resize(frame, target_size=(640, 640))
    )
    # print(clips[0].shape)

    model = YOLO(r'C:\Users\zedon\Documents\GitHub\labassist-api\app\ml_models\object_detection\weights\obj_detect_best_v5.pt', verbose = False).cpu()
    preds = predict_on_clips(clips, model)

    valid_clips = []
    for i, pred in enumerate(preds):
        flask_box = get_valid_flask(pred)
        if flask_box is not None:
            valid_clips.append([i, clips[i], flask_box])
    print(f'found {len(valid_clips)} valid clips')

    # Display 20 random valid clips as a 4x5 grid
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    for i in range(4):
        for j in range(5):
            idx, clip, flask_box = valid_clips[np.random.randint(len(valid_clips))]
            frame = square_crop(clip[0], flask_box)
            axs[i, j].imshow(frame)
            axs[i, j].axis('off')
    plt.show()
