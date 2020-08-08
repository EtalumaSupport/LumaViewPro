from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.core.camera import Camera
import numpy as np
import time

class MainDisplay(TabbedPanel):
    pass

class ConfigTab(BoxLayout):
    pass

class ImageTab(BoxLayout):
    def capture(self):
        camera = self.ids['scope']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))

    def get_sliders(self):
        bf_ill = self.ids['bf_ill']
        bf_gain = self.ids['bf_gain']
        bf_exp = self.ids['bf_exp']

        bl_ill = self.ids['bl_ill']
        bl_gain = self.ids['bl_gain']
        bl_exp = self.ids['bl_exp']

        gr_ill = self.ids['gr_ill']
        gr_gain = self.ids['gr_gain']
        gr_exp = self.ids['gr_exp']

        rd_ill = self.ids['rd_ill']
        rd_gain = self.ids['rd_gain']
        rd_exp = self.ids['rd_exp']

        sliders = np.array([[bf_ill.value, bf_gain.value, bf_exp.value],
                            [bl_ill.value, bl_gain.value, bl_exp.value],
                            [gr_ill.value, gr_gain.value, gr_exp.value],
                            [rd_ill.value, rd_gain.value, rd_exp.value]])
        print(sliders)

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
