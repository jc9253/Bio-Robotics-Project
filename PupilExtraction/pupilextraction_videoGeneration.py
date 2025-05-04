"""
Parallel pupil-outline video generation for all BMP frames in a single folder.
This refactor moves the worker to module scope so it can be pickled correctly.
"""

from itertools import islice
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pypupilext as pp

# where to log tabular data (unused for video-only)
# PATH_DATA = "./output.txt"


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize `image` preserving aspect ratio, given one of width/height."""
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


def Pupil_outline(img):
    """
    Run PuRe pupil detection on a grayscale image.
    Returns a frame with ellipse overlay or None on failure.
    """
    pure = pp.PuRe()
    pure.maxPupilDiameterMM = 7
    pupil = pure.runWithConfidence(img)
    if pupil is None or pupil.diameter() <= 0:
        return None

    axes = (int(pupil.minorAxis() / 2), int(pupil.majorAxis() / 2))
    if axes[0] <= 0 or axes[1] <= 0:
        return None

    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(
        canvas,
        (int(pupil.center[0]), int(pupil.center[1])),
        axes,
        pupil.angle,
        0,
        360,
        (0, 0, 255),
        1,
    )
    return ResizeWithAspectRatio(canvas, width=800)


def worker(img_path_str, frames_dict):
    """
    Module-level worker to be pickled: processes one image path.
    Adds the processed frame to the shared dict if successful.
    """
    img_path = Path(img_path_str)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    frame = Pupil_outline(img)
    if frame is not None:
        frames_dict[img_path.stem] = frame


def collect_pupil_frames(dir_path: Path):
    """
    Parallel process all BMPs in dir_path, return ordered list of frames.
    """
    manager = Manager()
    frames_dict = manager.dict()
    # parameter testing
    # bmp_files = [str(p) for p in islice(dir_path.rglob("*.bmp"), 10)]
    bmp_files = [str(p) for p in dir_path.rglob("*.bmp")]
    with Pool(processes=cpu_count()) as pool:
        # starmap to pass shared dict
        pool.starmap(worker, [(fp, frames_dict) for fp in bmp_files])

    # sort frames by filename stem
    ordered = sorted(frames_dict.items(), key=lambda x: x[0])
    return [f for _, f in ordered]


def build_animation_matplot(frames, out_path: Path, interval=50, repeat_delay=1000):
    """
    Given a list of BGR frames, create and save an MP4 to `out_path`.
    """
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    fig, ax = plt.subplots(figsize=(20, 8))
    artist_frames = [[ax.imshow(im)] for im in rgb]
    ani = animation.ArtistAnimation(
        fig, artist_frames, interval=interval, blit=True, repeat_delay=repeat_delay
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path))
    plt.close(fig)


def build_animation_cv2(frames, out_path: Path, fps=20):
    """
    Given a list of BGR frames (all the same size), write them to MP4 via cv2.
    VideoWriter.

    Parameters
    ----------
    frames : List[numpy.ndarray]
        List of images in BGR color (HxWx3), all must have identical dimensions.
    out_path : Path
        Path (including filename) to save the .mp4 file.
    fps : int
        Frames per second for the output video.
    """
    if not frames:
        raise ValueError("No frames to write.")

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Grab frame size from first frame
    height, width = frames[0].shape[:2]

    # Define the codec and create VideoWriter object.
    # 'mp4v' is a common codec for .mp4 files; you can also try 'H264' if available.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    # Write out all frames
    for idx, frame in enumerate(frames):
        # Make sure frame size matches; if needed, resize here
        h, w = frame.shape[:2]
        if (h, w) != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print(f"Saved video: {out_path}")


def process_directory(dir_path: Path):
    """
    Orchestrate frame collection and video building for one folder.
    """
    print(f"- Processing folder: {dir_path}")
    frames = collect_pupil_frames(dir_path)
    if not frames:
        print(f"No valid frames in {dir_path}, skipping.")
        return
    out_video = dir_path / "sample.mp4"
    build_animation_cv2(frames, out_video, fps=20)
    # build_animation_matplot(frames, out_video)
    print(f"-- Saved video: {out_video}")


if __name__ == "__main__":
    # INPUT_ROOT = Path("tests/Single_camera_recording_1")
    INPUT_ROOT = Path("output")
    # Directly process the specified folder
    for sub in INPUT_ROOT.iterdir():
        if sub.is_dir():
            process_directory(sub)
