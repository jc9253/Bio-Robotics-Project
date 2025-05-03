import time
import pandas as pd
from data_collection.data_collection_script import Video
from PupilExtraction.avitobmp import convert_avi

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from torch.utils.data import TensorDataset
from torch.utils.data import random_split

# from location_model.?
from zoom_model import zoom_model as zm
import zoom_model
from zoom_ctrl import zoom_interface

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init objects:
    avi_path = "video"
    bmap_folder = "bmap/"
    camera = Video(path=avi_path, do_thread=False)

    num_features = 18
    sequnce_len = 25

    model = zm(num_features=num_features).to(device)
    model.load_state_dict(torch.load("zoom_model.pth"))

    os_zoom = zoom_interface()

    ## MAIN LOOP ##
    try:
        while True:

            # Take input data
            recording = camera.start("")  # unlabeled since inference
            time.sleep(2)
            camera.stop()

            convert_avi("", bmap_folder)  # convert avi to bit map

            """ ####testing zoom control###
            # Run through preprocessing
            # Returns features which is a time series of:
            # - time
            # - pupil size
            # - 

            features = feature_extrac.?(recording)
            """  ####testing zoom control###

            features = pd.read_csv("PupilExtraction/output.csv")
            features = torch.from_numpy(features.values)
            features = features.float()

            size, loc_x, loc_y = zoom_model.infer(model, features, device)

            loc = [loc_x, loc_y]
            mag = size

            # Send location and zooom to zoom_interface
            os_zoom.zoom(loc=loc, mag=mag)

            break

    except KeyboardInterrupt:
        print("Complete")
