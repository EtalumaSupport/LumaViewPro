from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image

import cv2
from PIL import Image as PImg
import numpy as np



capture = cv2.VideoCapture(0)
ret, frame = capture.read()
frame = PImg.fromarray(frame)
frame.save('jetson.jpg')
capture.release()
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
        box     = BoxLayout(orientation='vertical')
        image   = Image(source='jetson.jpg')
        button  = Button(text="Update", size_hint_y=.1)

        box.add_widget(image)
        box.add_widget(button)

        return box

JetsonApp().run()
