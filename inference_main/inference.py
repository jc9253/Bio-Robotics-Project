from zoom_ctrl import zoom_interface
# from ../data_collection_script import video 
# from pupilextraction import ?

os_zoom = zoom_interface()

''' ####testing zoom control###

# init objects:
subject = "person_"
camera = video(path="videos/" + subject)
pre_processing = ?()

loc_model = ?()
zoom_model = ?()



# Take input data
recording = camera.start()
#wait
camera.stop()

# Run through preprocessing
features = pre_processing.?(recording)

# Send to zoom and location models
loc = loc_model.infer(features)
mag = zoom_model.infer(features)

'''

loc = [0.5, 0.5]
mag = -5

# Send location and zooom to zoom_interface

os_zoom.zoom(loc=loc, mag=mag)
