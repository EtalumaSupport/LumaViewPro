# imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from pypylon import pylon
import numpy as np

class PylonCam(Image):

    def __init__(self,  **kwargs):
        super(PylonCam, self).__init__(**kwargs)

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def play(self):
        # while self.camera.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 500 ms is used.
        img = self.camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
        if img.GrabSucceeded():
            # returns a numpy array in the shape of the image
            img_array = img.GetArray()
             # creates a 1D array of type uint8
            # img_string = img_array.tostring() # equivalent
            img_string = img_array.reshape(-1,)
            img_string = img_array.flatten()
            print(img_string[0])
            # Works up to here...

            # create a texture that has the shape of the image
            texture = Texture.create(size=(img_array.shape[1], img_array.shape[0]), colorfmt='luminance')
            # buffer the 1D array into the texture
            texture.blit_buffer(img_string, colorfmt="luminance", bufferfmt='ubyte')
            self.texture = texture

# BoxLayout containing Camera and Button
class MainBox(BoxLayout):
    pass

# Boxlayout is the App class
class LivePylonApp(App):
   def build(self):
       return MainBox()

   # def on_stop(self):


# Instantiate and run the kivy app
LivePylonApp().run()
