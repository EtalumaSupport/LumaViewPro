#!/usr/bin/python3
import kivy
kivy.require('2.0.0') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button


class ButtonHolder(Button):
    pass

class SimpleApp(App):
    def build(self):
        return ButtonHolder()
        #return Button(text='Hello world')


if __name__ == '__main__':
    SimpleApp().run()
