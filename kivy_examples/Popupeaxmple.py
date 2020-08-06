# Kivy example for the Popup widget

# Program to Show how to create a switch
# import kivy module
import kivy

# base Class of your App inherits from the App class.
# app:always refers to the instance of your application
from kivy.app import App

# this restrict the kivy version i.e
# below this kivy version you cannot
# use the app or software
kivy.require('1.9.0')

# The Button is a Label with associated actions
# that is triggered when the button
# is pressed (or released after a click/touch).
from kivy.uix.button import Button


# The GridLayout arranges children in a matrix.
# It takes the available space and
# divides it into columns and rows,
# then adds widgets to the resulting “cells”.
from kivy.uix.gridlayout import GridLayout

# Popup widget is used to create popups.
# By default, the popup will cover
# the whole “parent” window.
# When you are creating a popup,
# you must at least set a Popup.title and Popup.content.
from kivy.uix.popup import Popup

# The Label widget is for rendering text.
from kivy.uix.label import Label

# to change the kivy default settings we use this module config
from kivy.config import Config

# 0 being off 1 being on as in true / false
# you can use 0 or 1 && True or False
Config.set('graphics', 'resizable', True)

# Make an app by deriving from the kivy provided app class
class PopupExample(App):
	# override the build method and return the root widget of this App

	def build(self):
		# Define a grid layout for this App
		self.layout = GridLayout(cols = 1, padding = 10)


		# Add a button
		self.button = Button(text ="Click for pop-up")
		self.layout.add_widget(self.button)

		# Attach a callback for the button press event
		self.button.bind(on_press = self.onButtonPress)

		return self.layout

	# On button press - Create a popup dialog with a label and a close button
	def onButtonPress(self, button):

		layout = GridLayout(cols = 1, padding = 10)

		popupLabel = Label(text = "Click for pop-up")
		closeButton = Button(text = "Close the pop-up")

		layout.add_widget(popupLabel)
		layout.add_widget(closeButton)

		# Instantiate the modal popup and display
		popup = Popup(title ='Demo Popup',
					content = layout)
		popup.open()

		# Attach close button press with popup.dismiss action
		closeButton.bind(on_press = popup.dismiss)

# Run the app
if __name__ == '__main__':
	PopupExample().run()
