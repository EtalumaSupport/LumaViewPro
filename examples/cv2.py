import cv2, glob

for camera in glob.glob("/dev/video?"):
    print(camera)
