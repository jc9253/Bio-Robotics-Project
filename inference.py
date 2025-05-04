import time
from pathlib import Path

import pandas as pd

from data_collection.data_collection_script import Video
from Preprocessing.insightface_test import get_face_embedding
from PupilExtraction.avitobmp import convert_avi

# from location_model.?
# from zoom_model import zoom_model

# from zoom_ctrl import zoom_interface


if __name__ == "__main__":
    # init objects:
    avi_path = "video"
    bmap_folder = "bmap/"
    camera = Video(path=avi_path, do_thread=False)
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

            convert_avi(f"{avi_path}.avi", bmap_folder)  # convert avi to bit map

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
