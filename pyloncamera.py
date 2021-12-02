import numpy as np
from pypylon import pylon

class PylonCamera:

    def __init__(self, **kwargs):
        self.active = False
        self.array = []
        self.connect()

    def connect(self):
        try:
            # Create an instant active object with the camera device found first.
            self.active = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.active.Open()
            self.active.Width.SetValue(self.active.Width.Max)
            self.active.Height.SetValue(self.active.Height.Max)
            # self.active.PixelFormat.SetValue('PixelFormat_Mono12')
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
            self.active.ReverseX.SetValue(True);
            # Grabbing Continusely (video) with minimal delay
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("LumaViewPro compatible camera is now connected.")

        except:
            self.active = False
            print('Error: Cannot not connect to camera')

    def grab(self):
        if self.active == False:
            self.connect()
        try:
            if self.active.IsGrabbing():
                grabResult = self.active.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    self.array = grabResult.GetArray()
                    # texture = Texture.create(size=(array.shape[1],array.shape[0]), colorfmt='luminance')
                    # texture.blit_buffer(array.flatten(), colorfmt='luminance', bufferfmt='ubyte')
                    # # display image from the texture
                    # self.texture = texture

            grabResult.Release()
            return True

        except:
            print('Error: Cannot grab texture from camera')
            return False

    def frame_size(self, w, h):
        if self.active != False:

            width = int(min(int(w), self.active.Width.Max)/2)*2
            height = int(min(int(h), self.active.Height.Max)/2)*2
            offset_x = int((self.active.Width.Max-width)/4)*2
            offset_y = int((self.active.Height.Max-height)/4)*2

            self.active.StopGrabbing()
            self.active.Width.SetValue(width)
            self.active.Height.SetValue(height)
            self.active.OffsetX.SetValue(offset_x)
            self.active.OffsetY.SetValue(offset_y)
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def gain(self, gain):
        if self.active != False:
            self.active.StopGrabbing()
            self.active.Gain.SetValue(gain)
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def auto_gain(self, state = True):
        if self.active != False:
            self.active.StopGrabbing()
            if state == True:
                self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def exposure_t(self, t):
        if self.active != False:
            self.active.StopGrabbing()
            # (t*1000) in microseconds; therefore t  in milliseconds
            self.active.ExposureTime.SetValue(t*1000)
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def auto_exposure_t(self, state = True):
        if self.active != False:
            self.active.StopGrabbing()
        if state == True:
            self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        else:
            self.active.ExposureAuto.SetValue('Off')
        self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
