
import cv2
import math
import torch
import numpy as np

# interval = 4

# cap = cv2.VideoCapture(r"C:\Users\zedon\Videos\PW2024VIDEOS\IMG_8700.MOV")
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f'frame count: {frame_count}, fps: {fps}')
# extracted_clips = []
# for i in range(0, frame_count, int(interval * fps)):
#     print(f'extracting clip {i // int(interval * fps) + 1}/{math.ceil(frame_count / int(interval * fps))}', end='\r')
#     frames = []
#     for j in range(interval * int(fps)):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     if len(frames) == interval * int(fps):
#         extracted_clips.append(np.array(frames))
# print('Detecting valid clips')

