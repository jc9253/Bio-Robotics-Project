from zoom_ctrl import win_os as zoom_interface  #Select OS here
from ../data_collection_script import video 
from pupilextraction import ?

# init objects:
subject = "person_"
camera = video(path="videos/" + subject)
pre_processing = ?()

loc_model = ?()
zoom_model = ?()

os_zoom = zoom_interface()

# Take input data
recording = camera.start()
#wait
camera.stop()

# Run through preprocessing
features = pre_processing.?(recording)

# Send to zoom and location models
loc = loc_model.infer(features)
mag = zoom_model.infer(features)

loc = [0, 0]
mag = 0

# Send location and zooom to zoom_interface

os_zoom.zoom(loc=loc, mag=mag)
