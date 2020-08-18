# imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image

# Boxlayout is the App class
class BoxLayoutDemo(App):
   def build(self):
       box     = BoxLayout(orientation='vertical')
       cam     = Image(source='./jetson.jpg')
       button  = Button(text="button", size_hint_y=0.1)

       box.add_widget(cam)
       box.add_widget(button)
       return box


# Instantiate and run the kivy app
if __name__ == '__main__':
   BoxLayoutDemo().run()
