# imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# Open CV 2
import cv2

# Besler Camera Support
import pypylon
from pypylon import pylon
from pypylon import genicam

'''
# Number of images to be grabbed.
countOfImagesToGrab = 100

# The exit code of the sample application.
exitCode = 0

try:
    # Start the grabbing of c_countOfImagesToGrab images.
    # The camera device is parameterized with a default configuration which
    # sets up free-running continuous acquisition.
    camera.StartGrabbingMax(countOfImagesToGrab)

    # Camera.StopGrabbing() is called automatically by the RetrieveResult() method
    # when c_countOfImagesToGrab images have been retrieved.
    while camera.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            print("SizeX: ", grabResult.Width)
            print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            print("Gray value of first pixel: ", img[0, 0])
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    camera.Close()

except genicam.GenericException as e:
    # Error handling.
    print("An exception occurred.")
    print(e.GetDescription())
    exitCode = 1
'''

#sys.exit(exitCode)

class PylonCamera(Camera):
    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            # Print the model name of the camera.
            print("Using device ", self.camera.GetDeviceInfo().GetModelName())

            # demonstrate some feature access
            #new_width = camera.Width.GetValue() - camera.Width.GetInc()
            #if new_width >= camera.Width.GetMin():
            #    camera.Width.SetValue(new_width)

            # The parameter MaxNumBuffer can be used to control the count of buffers
            # allocated for grabbing. The default value of this parameter is 10.
            #camera.MaxNumBuffer = 5

            # Grabing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            #self.converter = pylon.ImageFormatConverter()

            # converting to opencv bgr format
            #self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            #self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())


    def update(self, dt):
        #image_texture = Texture.create(
        #    size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        #self.texture = image_texture
        try:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    # Access the image data
                    #image = self.converter.Convert(grabResult)
                    #img = image.GetArray()
                    image = grabResult.GetArray()
                    #cv2.namedWindow('title', cv2.WINDOW_NORMAL)
                    #cv2.imshow('title', img)
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')
                    # display image from the texture
                    self.texture = image_texture

                grabResult.Release()

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

    def start(self):
        self.fps = 5
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.schedule_interval(self.frame_event)


# Boxlayout is the App class
class BoxLayoutDemo(App):
   def build(self):
       box     = BoxLayout(orientation='vertical')
       #cam     = Image(source='./jetson.jpg')
       camera  = PylonCamera()
       button  = Button(text="button", size_hint_y=0.1)

       box.add_widget(camera)
       box.add_widget(button)
       camera.start()
       #camera.play = True
       return box


# Instantiate and run the kivy app
if __name__ == '__main__':
   BoxLayoutDemo().run()
