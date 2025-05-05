import time
from pathlib import Path

import cv2
import pandas as pd
from insightface.app import FaceAnalysis

from data_collection.data_collection_script import Video
from Preprocessing.insightface_test import get_face_boarder
from PupilExtraction.avitobmp import convert_avi

# from location_model.?
# from zoom_model import zoom_model

# from zoom_ctrl import zoom_interface

app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)  # 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

if __name__ == "__main__":
    # init objects:
    AVI_PATH = Path("videos/")
    BMAP_FOLDER = Path("bmap/")
    source = "source"
    camera = Video(path=str(AVI_PATH) + source, do_thread=False)
    print(str(AVI_PATH) + source)
    """ 
    ####testing zoom control###
    feature_extrac = ?()

    loc_model = ?()
    zoom_model = ?()

    """  ####testing zoom control###
    # os_zoom = zoom_interface()

    ## MAIN LOOP ##
    try:
        while True:

            # Take input data
            recording = camera.start("")  # unlabeled since inference
            time.sleep(2)
            camera.stop()

            convert_avi(
                str(AVI_PATH) + source, str(BMAP_FOLDER)
            )  # convert avi to bit map

            for bmp_path in BMAP_FOLDER.rglob("*.bmp"):
                rel = bmp_path.relative_to(BMAP_FOLDER)
                out_dir = BMAP_FOLDER / rel.parent / "bmap_extracted"
                out_file = out_dir / f"{rel.stem}_face.bmp"
                out_dir.mkdir(parents=True, exist_ok=True)
                face_roi = get_face_boarder(str(bmp_path))
                cv2.imwrite(str(out_file), face_roi)

            ## testing zoom control
            # Run through preprocessing
            # Returns features which is a time series of:
            # - time
            # - pupil size
            # -

            # features = feature_extrac.?(recording)
            # testing zoom control###

            # features = pd.read_csv("PupilExtraction/output.csv")
            # print(features.iloc[0])

            """ ####testing zoom control###
            # Send features to zoom and location models
            loc = loc_model.infer(features)
            """  ####testing zoom control###
            # mag = zoom_model.infer(features)

            loc = [0.5, 0.5]
            mag = 0

            # Send location and zooom to zoom_interface
            # os_zoom.zoom(loc=loc, mag=mag)

    except KeyboardInterrupt:
        print("Complete")
