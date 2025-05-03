import os
from pathlib import Path

import cv2

def convert_avi(video_path, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(video_path)

    for file_path in video_path.rglob("*.avi"):
        print(f"reading file: {file_path}")
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {file_path}")

        frame_idx = 0

        filename = file_path.stem
        output_path = os.path.join(str(output_dir), filename)
        os.makedirs(output_path, exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            # filename: {video name}_0000.bmp
            filepath = os.path.join(output_path, f"{filename}_{frame_idx:04d}.bmp")
            print(filepath)
            cv2.imwrite(filepath, frame)  # save as BMP
            frame_idx += 1

        cap.release()
        print(f"Extracted {frame_idx} frames to '{output_dir}/'")


video_path = "../data_collection/videos"
output_dir = "../data_collection/video_frame"

convert_avi(video_path, output_dir)
