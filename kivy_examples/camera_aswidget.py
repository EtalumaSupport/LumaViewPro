import kivy
kivy.require('1.7.2')

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.core.window import Window

class CamApp(App):
          # Function to take a screenshot
          def screengrab(self,*largs):
                outname = self.fileprefix+'_%(counter)04d.png'
                Window.screenshot(name=outname)

          def build(self):

                # create a floating layout as base
                camlayout = FloatLayout(size=(600, 600))
                cam = Camera()        #Get the camera
                cam=Camera(resolution=(1024,1024), size=(300,300))
                cam.play=True         #Start the camera
                camlayout.add_widget(cam)

                button=Button(text='Take Picture',size_hint=(0.12,0.12))
                button.bind(on_press=self.screengrab)
                camlayout.add_widget(button)    #Add button to Camera Layout

                self.fileprefix = 'snap'

                return camlayout

if __name__ == '__main__':
    CamApp().run()
