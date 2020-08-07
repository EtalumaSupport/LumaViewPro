# main.py file of slider

# base Class of your App inherits from the App class.
# app:always refers to the instance of your application
from kivy.app import App

# BoxLayout arranges children in a vertical or horizontal box.
# or help to put the children at the desired location.
from kivy.uix.boxlayout import BoxLayout

# creating the root widget used in .kv file
class SliderWidget(BoxLayout):
	pass

# class in which name .kv file must be named Slider.kv.
# or creating the App class
class Slider(App):
	def build(self):
		# returning the instance of SliderWidget class
		return SliderWidget()

# run the app
if __name__ == '__main__':
	Slider().run()
