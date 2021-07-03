#!/usr/bin/env python3
'''
Minimal Kivy program example
'''
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class Main(BoxLayout):
    pass

class IntroKivyApp(App):
    def build(self):
        return Main()

IntroKivyApp().run()
