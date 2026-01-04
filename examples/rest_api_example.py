import requests

#NOTE: Before running, please ensure that lumaviewpro.py is running locally
#NOTE: You can view documentation for the REST API at http://localhost:8000/docs

#This example shows how to use REST API V1, so here is a URL const that will be the prefix for every URL used
URL = "http://localhost:8000/api/v1"

#---------------MOTION OPERATIONS---------------
MOTION_URL = URL + "/move"

#Get current position
response = requests.get(MOTION_URL + "/position") #GET http://localhost:8000/api/v1/move/position
print(response.json()) #Returns requested information in JSON format

#Move position - absolute
body = { #Request body must be defined for this operation
    "motion_type": "absolute", #Type of positioning, either "absolute" or "relative"
    "axis": "X", #Axis of motion, either "X", "Y", "Z", or "T"
    "um": 100, #Position to move to for absolute motion, Distance to move for relative motion
    "overshoot_enabled": True,
    "ignore_limits": False #Field ignored when using relative motion
}
response = requests.post(MOTION_URL, json=body) #POST http://localhost:8000/api/v1/move
print(response.json()) #Returns motion completed message upon success

#Move position - relative
body = { #Request body must be defined for this operation
    "motion_type": "relative", #Type of positioning, either "absolute" or "relative"
    "axis": "Y", #Axis of motion, either "X", "Y", "Z", or "T"
    "um": 100, #Position to move to for absolute motion, Distance to move for relative motion
    #NOTE: The last two parameters have default values of True and False respectively and can be left out
}
response = requests.post(MOTION_URL, json=body) #POST http://localhost:8000/api/v1/move
print(response.json()) #Returns motion completed message upon success


#---------------CAPTURE OPERATIONS---------------
CAPTURE_URL = URL + "/capture"

#Live capture
body = { #NOTE: All parameters have default values as shown below
    "file_root": "img_", #Prefix for saved image filename
    "append": "ms", #Suffix for saved image filename
    "color": "BF",
    "tail_id_mode": "increment", #Can be "increment" or null, will add an incrementing int at the end of the filename
    "force_to_8bit": True,
    "output_format": "TIFF", #Output format, can be "TIFF" or "OME-TIFF"
    "true_color": "BF",
    "timeout": 0, #Timeout in seconds
    "all_ones_check": False,
    "sum_count": 1,
    "sum_delay_s": 0
}
response = requests.post(CAPTURE_URL + "/live", json=body) #POST http://localhost:8000/api/v1/capture/live
print(response.json()) #Returns capture saved message upon success


#---------------PROTOCOL OPERATIONS---------------
PROTOCOL_URL = URL + "/protocol"

#Get saved protocols
response = requests.get(PROTOCOL_URL) #GET http://localhost:8000/api/v1/protocol
print(response.json()) #Returns a list of all saved protocols (NOTE: protocols must be created and saved through the GUI)

#Run protocol
body = {
    "protocol_name": "test_protocol"
}
response = requests.post(PROTOCOL_URL + "/run", json=body) #POST http://localhost:8000/api/v1/protocol/run
print(response.json()) #Returns protocol started message upon success

#Abort protocol
response = requests.post(PROTOCOL_URL + "/abort") #POST http://localhost:8000/api/v1/protocol/abort
print(response.json())


#---------------STATUS OPERATIONS---------------
STATUS_URL = URL + "/status"

#Get device status
response = requests.get(STATUS_URL) #GET http://localhost:8000/api/v1/status
print(response.json()) #Returns JSON formatted status on device position and protocol