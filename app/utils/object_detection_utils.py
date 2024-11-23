# %%
import os
import cv2
import math
import torch
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from ultralytics.models.yolo import YOLO

print('Cuda available:', torch.cuda.is_available())

# %%
def pad_and_resize(frame, target_size=(640, 640)):
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

# %%
def extract_clips(video, interval: int, transform = None):
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
    print(f'extracting clips from {video_path}')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'frame count: {frame_count}, fps: {fps}')
    extracted_clips = []
    for i in range(0, frame_count, int(interval * fps)):
        print(f'extracting clip {i // int(interval * fps) + 1}/{math.ceil(frame_count / int(interval * fps))}', end='\r')
        frames = []
        for j in range(interval * int(fps)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = pad_and_resize(frame)
            frames.append(frame)
        if len(frames) == interval * int(fps):
            extracted_clips.append(np.array(frames))
    print('/nDetecting valid clips')
    cap.release()
    cv2.destroyAllWindows()
    return extracted_clips, fps

# %%
def predict_on_clips(clips, model):
    predictions = []
    for clip in clips:
        # get first frame in clip in the shape of (1, 3, 640, 640)
        clip = torch.tensor(clip).permute(0, 3, 1, 2)[0].unsqueeze(0)
        result = model.predict(clip, verbose=False)
        predictions.append(result)
    return predictions

# %%
def get_biggest_boxes(boxes):
    areas = np.array([(box[0] - box[2]) * (box[1] - box[3]) for box in boxes])
    return boxes[np.argmax(areas)]

# %%
def get_valid_flask(prediction):
    all_boxes = np.array(prediction[0].boxes.xyxy.cpu().numpy())
    all_classes = np.array(prediction[0].boxes.cls.cpu().numpy())
    
    flask_boxes = all_boxes[all_classes == 2]
    burette_boxes = all_boxes[all_classes == 1]
    
    valid_flasks = []
    # iterate over all permutations of flask and burette boxes, valid flasks are ones with a burette over them
    for flask_box in flask_boxes:
        for burette_box in burette_boxes:
            # x1, y1, x2, y2
            midpoint_of_burette = (burette_box[0] + burette_box[2]) / 2
            if flask_box[0] < midpoint_of_burette < flask_box[2]:
                valid_flasks.append(flask_box)
    
    if len(valid_flasks) == 0:
        return None
    else:
        return get_biggest_boxes(valid_flasks)

# %%
def overlay_boxes(frame, boxes, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

# %%
def square_crop(frame, box, pad_top=0.4, pad_bottom=0.2, target_size=(640, 640)):
    x1, y1, x2, y2 = box[:4].astype(int)
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
    video_path = 'video-splitter/sample_vid.MOV'
    interval = 2
    clips = extract_clips(
        video_path, 
        interval, 
        transform=lambda frame: pad_and_resize(frame, target_size=(640, 640))
    )
    print(clips[0].shape)

    model = YOLO('video-splitter/models/obj_detect_best_v5.pt', verbose = False).cpu()
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
