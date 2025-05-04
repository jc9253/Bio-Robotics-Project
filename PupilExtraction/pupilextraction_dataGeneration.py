from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import cv2
import pandas as pd
import pypupilext as pp

INPUT_ROOT = Path("output")
ALL_METRICS_PATH = Path("videos_metrics.tsv")


def pupil_outline(img):
    """Run PuRe and return (frame_img, metrics_dict) or (None, None)."""
    pure = pp.PuRe()
    pure.maxPupilDiameterMM = 4
    pupil = pure.runWithConfidence(img)
    if not pupil or pupil.diameter() <= 0:
        return None, None

    metrics = {
        "video": None,  # filled later
        "frame": None,  # filled later
        "Outline_conf": pupil.outline_confidence,
        "PupilDiameter": pupil.diameter(),
        "RectPoints": pupil.rectPoints(),
        "Size": pupil.size,
        "Major Axis": pupil.majorAxis(),
        "Minor Axis": pupil.minorAxis(),
        "Width": pupil.width(),
        "Height": pupil.height(),
    }

    # draw ellipse (optional if you only want metrics)
    return img, metrics


def worker(args):
    img_path_str, video_name, metrics_list = args
    img_path = Path(img_path_str)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    _, data = pupil_outline(img)
    if data:
        data["video"] = video_name
        data["frame"] = img_path.stem
        metrics_list.append(data)


def collect_all_metrics():
    """
    Walk every subfolder, spawn workers to extract per-frame metrics,
    and return one big list of dicts.
    """
    manager = Manager()
    all_metrics = manager.list()

    # build (img_path, video_name) tuples
    tasks = []
    for sub in INPUT_ROOT.iterdir():
        if not sub.is_dir():
            continue
        video_name = sub.name
        for bmp in sub.glob("*.bmp"):
            tasks.append((str(bmp), video_name, all_metrics))

    # parallel extraction
    with Pool(cpu_count()) as pool:
        pool.map(worker, tasks)

    return list(all_metrics)


if __name__ == "__main__":
    # Collect all per-frame metrics
    metrics = collect_all_metrics()

    # Build DataFrame and save
    df = pd.DataFrame(metrics)
    df.sort_values(["video", "frame"], inplace=True)
    df.to_csv(ALL_METRICS_PATH, sep="\t", index=False)
    print(f"Saved concatenated per-frame metrics to {ALL_METRICS_PATH}")
