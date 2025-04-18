import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend that supports file output
import multiprocessing
from multiprocessing import Manager, Pool
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypupilext as pp

# from avitobmp import convert_avi


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def Pupil_outline(img, ax=None):

    pupilClass = pp.Pupil()
    assert pupilClass.confidence == -1

    pure = pp.PuRe()
    pure.maxPupilDiameterMM = 7

    im_reized = img
    pupil = pure.runWithConfidence(im_reized)
    data = pd.DataFrame(
        [{"Outline Conf": pupil.outline_confidence, "PupilDiameter": pupil.diameter()}]
    )
    print(data)

    # Check if pupil detection was successful
    if pupil is None or pupil.diameter() <= 0:
        print("Warning: Pupil detection failed or diameter is non-positive.")
        return None

    axes = (int(pupil.minorAxis() / 2), int(pupil.majorAxis() / 2))
    if axes[0] <= 0 or axes[1] <= 0:
        print("Warning: Invalid ellipse axes.")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_plot = cv2.ellipse(
        img,
        (int(pupil.center[0]), int(pupil.center[1])),
        (int(pupil.minorAxis() / 2), int(pupil.majorAxis() / 2)),
        pupil.angle,
        0,
        360,
        (0, 0, 255),
        1,
    )

    resize = ResizeWithAspectRatio(img_plot, width=800)

    return resize


def process_frame(img_path, return_dict):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        print(f"Warning: {img_path.name} could not be read. Skipping...")
        return None
    processed_img = Pupil_outline(img)
    return_dict[img_path.stem] = processed_img


def convert_frames(frame_data):
    return cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

    # frame = ax.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))


PATH = "../data_collection/video_frame/Jessie_37_1513_443"

dir_path = Path(PATH)
# Plot definition
fig, ax = plt.subplots(figsize=(20, 8))

manager = Manager()
return_dict = manager.dict()

files = list(dir_path.rglob("*.bmp"))  # filters for bmps
# Pupil_outline(img)

cpu_count = multiprocessing.cpu_count()
with Pool(processes=cpu_count) as pool:
    pool.starmap(process_frame, [(file, return_dict) for file in files])

frame_data = sorted(return_dict.items(), key=lambda x: x[0])
frame_data = [frame for _, frame in frame_data if frame is not None]

with Pool(processes=cpu_count) as pool:
    frame_results = pool.map(convert_frames, frame_data)

frames = []
for frame in frame_results:
    frame = ax.imshow(frame)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

# fig.canvas.draw()  # Initialize the canvas drawing
print("Exporting Video")
ani.save(os.path.join(PATH, "sample.mp4"))
print("success!")
# plt.show()
