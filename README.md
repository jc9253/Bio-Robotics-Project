# Bio-Robotics-Project

different cameras, different people, different background

Collecting labeled data:
while not_bored:
    {text size, text location} = random()
    display = {text size, text location}
    take picture/ video timestamp = [text size, text location] label

preprocessor for training:
  Extract face/ Extract Eye region
    Apply filtering for determining location
    Apply edge filtering for extracting iris
    merge images with labels

Control models:
  Record = location extraction, amount extraction
  determine action by change
  OS agnoistic accessibility API = {}




Data Collection script: (Braley)
  display random text size/location
  save photo/video timestamp with label

Preprocessing: (who ?)
  Eye exctraction
  Irirs edge filtering
  Resave data with label

Model for determing location (who ?)
Model for determing zoom in/out (magnitude or classification?) (who ?)

Control OS assesibility based on threshold

This weekend:
 - Data collection script
WEEK1
 - collect data
WEEK2
 - Create models
WEEK3
- Debug models
WEEK4
 - Have fun with model
  
