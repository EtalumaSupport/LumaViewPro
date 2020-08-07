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
