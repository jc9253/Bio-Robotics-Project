import time
from zoom_ctrl import zoom_interface
from data_collection.data_collection_script import Video 
# from pupilextraction import ?

os_zoom = zoom_interface()



# init objects:
camera = Video(path="person_0")

''' ####testing zoom control###
pre_processing = ?()

loc_model = ?()
zoom_model = ?()

''' ####testing zoom control###

## MAIN LOOP ##

# Take input data
recording = camera.start("") #unlabeled since inference
time.sleep(2)
camera.stop()

''' ####testing zoom control###
# Run through preprocessing
# Returns features which is a time series of:
# - pupil size
# -
# - 

features = pre_processing.?(recording)

# Send to zoom and location models
loc = loc_model.infer(features)
mag = zoom_model.infer(features)

''' ####testing zoom control###

loc = [0.5, 0.5]
mag = 0

# Send location and zooom to zoom_interface

os_zoom.zoom(loc=loc, mag=mag)
