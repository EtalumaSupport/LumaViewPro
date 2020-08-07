from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import numpy as np
import cv2

# custom class CV_Camera (parent type: Image) to acquire from video using CV2 library
class CV_Camera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(CV_Camera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # display image from the texture
            self.texture = image_texture


class MainDisplay(TabbedPanel):
    pass

class ConfigTab(BoxLayout):
    pass

class ImageTab(BoxLayout):
    #camera_src = CV_Camera(capture=cv2.VideoCapture(0), fps=30)
    #camera_src = camera_src.texture #this is a texture?
    camera_src = '../data/sample.tif'

class MotionTab(BoxLayout):
    pass

class ProtocolTab(BoxLayout):
    pass

class AnalysisTab(BoxLayout):
    pass



class TutorialApp(App):
    def build(self):
        return MainDisplay()

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

TutorialApp().run()
