#!/usr/bin/env python3
'''
Minimal Kivy program example
'''
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.core.window import Window
Window.size = (200, 300)

kivy.require('2.0.0')

class Main(BoxLayout):
    pass

class ZSettingsApp(App):
    def build(self):
        return Main()

ZSettingsApp().run()
