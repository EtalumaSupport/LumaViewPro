from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel

# MainDisplay is organized in lumaviewplus.kv
class MainDisplay(TabbedPanel):
    pass

class LumaViewPlusApp(App):
    def build(self):
        return MainDisplay()

if __name__ == '__main__':
    LumaViewPlusApp().run()
