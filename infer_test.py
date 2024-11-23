import time
import cv2
import mmap
import torch
import numpy as np
import matplotlib.pyplot as plt

from app.ml_models.action_detection.inference import load_model, predict_action

model = load_model(r'C:\Users\zedon\Documents\GitHub\labassist-api\checkpoints\best\epoch=5-acc=0.8500000238418579.ckpt')
mmap_file = r'uploads\correct-0.mp4.mmap'

interval = 2
cap = cv2.VideoCapture(str(mmap_file))
fps = round(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'fps: {fps}, frame_count: {frame_count}')

num_frames = interval * fps
frame_size = 224 * 224 * 3
total_size = num_frames * frame_size

print(f'num_frames: {num_frames}, frame_size: {frame_size}, total_size: {total_size}')

for i in range(0, frame_count, int(interval * fps)):
    start_time = time.time()
    start_frame = i
    end_frame = min(i + int(interval * fps), frame_count)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    clip = []
    for j in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        clip.append(frame)
    clip = np.array(clip)
    clip = torch.tensor(clip, dtype=torch.float32)
    cap.release()

    clip = clip.permute(3, 0, 1, 2)

    pred = predict_action(clip, model)
    print(np.argmax(pred, axis=0))
    break