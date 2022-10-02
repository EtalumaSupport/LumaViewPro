import numpy as np
from pypylon import pylon

class PylonCamera:

    def __init__(self, **kwargs):
        self.active = False
        self.error_report_count = 0
        self.mssg = 'PylonCamera.__init__()'
        self.array = np.array([])
        self.connect()

    def __delete__(self):
        try:
            self.active.close()
        except:
            print('exception')
    def connect(self):
        try:
            # Create an instant active object with the camera device found first.
            self.active = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.active.Open()
            self.active.Width.SetValue(self.active.Width.Max)
            self.active.Height.SetValue(self.active.Height.Max)
            self.active.BslCenterX.Execute()
            self.active.BslCenterY.Execute()
            # self.active.PixelFormat.SetValue('PixelFormat_Mono12')
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
            self.active.ReverseX.SetValue(True);
            # Grabbing Continuously (video) with minimal delay
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.error_report_count = 0
            self.mssg = 'PylonCamera.connect() succeeded'            

        except:
            self.active = False
            if (self.error_report_count < 6):
                print('Error: Cannot connect to camera')
            self.error_report_count += 1
            self.mssg = 'PylonCamera.connect() failed'            

    def grab(self):
        if self.active == False:
            self.connect()
        try:
            if self.active.IsGrabbing():
                grabResult = self.active.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    self.array = grabResult.GetArray()

            grabResult.Release()
            self.error_report_count = 0
            self.mssg = 'PylonCamera.grab() succeeded'            
            return True

        except:
            if self.error_report_count < 6:
                print('Error: Cannot grab texture from camera')
            self.error_report_count += 1
            self.active = False
            self.mssg = 'PylonCamera.grab() failed'            
            return False

    def frame_size(self, w, h):
        if self.active != False:

            width = int(min(int(w), self.active.Width.Max)/4)*4
            height = int(min(int(h), self.active.Height.Max)/4)*4

            self.active.StopGrabbing()
            self.active.Width.SetValue(width)
            self.active.Height.SetValue(height)
            self.active.BslCenterX.Execute()
            self.active.BslCenterY.Execute()
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.mssg = 'PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; succeeded' 
        else:
            self.mssg = 'PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; inactive' 


    def gain(self, gain):
        if self.active != False:
            self.active.Gain.SetValue(gain)
            self.mssg = 'PylonCamera.gain('+str(gain)+')'+': succeeded' 
        else:
            self.mssg = 'PylonCamera.gain('+str(gain)+')'+': inactive camera' 

    def auto_gain(self, state = True):
        if self.active != False:
            if state == True:
                self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
            self.mssg = 'PylonCamera.auto_gain('+str(state)+')'+': succeeded' 
        else:
            self.mssg = 'PylonCamera.auto_gain('+str(state)+')'+': inactive camera' 
            
    def exposure_t(self, t):
        if self.active != False:
            # (t*1000) in microseconds; therefore t  in milliseconds
            self.active.ExposureTime.SetValue(max(t*1000, self.active.ExposureTime.Min))
            self.mssg = 'PylonCamera.exposure_t('+str(t)+')'+': succeeded' 
        else:
            self.mssg = 'PylonCamera.exposure_t('+str(t)+')'+': inactive camera' 

    def auto_exposure_t(self, state = True):
        if self.active != False:
            if state == True:
                self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.ExposureAuto.SetValue('Off')
            self.mssg = 'PylonCamera.auto_exposure_t('+str(state)+')'+': succeeded' 
        else:
            self.mssg = 'PylonCamera.auto_exposure_t('+str(state)+')'+': inactive camera' 

