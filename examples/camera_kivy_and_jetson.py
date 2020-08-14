from kivy.app import App
from kivy.uix.button import Button
#from kivy.uix.image import Image
# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Web Cam', frame)
#     k = cv2.waitKey(1)
#     if k == ord('q') or k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

class JetsonApp(App):
    def build(self):
        # img_source  = 'jetson.jpg'
        # kivy_image  = Image(source=img_source)
        # return Image(source=kivy_image)

        # jetsoncam = cv2.VideoCapture(0)
        # ret, frame = jetsoncam.read()

        return Button(text="Jetson App")

JetsonApp().run()
