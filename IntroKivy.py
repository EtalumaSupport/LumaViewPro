#!/usr/bin/python3
'''
Minimal Kivy program example
'''
import kivy
kivy.require('2.0.0')


from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class MainDisplay(BoxLayout):
    pass

class IntroKivyApp(App):
    def build(self):
        return MainDisplay()

IntroKivyApp().run()

'''
class LumaViewProApp(App):
    def build(self):
        self.icon = './data/icon32x.png'
        global lumaview
        lumaview = MainDisplay()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].load_protocol("./data/protocol.json")
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.led_off()
        Window.minimum_width = 800
        Window.minimum_height = 600
        return lumaview

    def on_stop(self):
        global lumaview
        lumaview.led_board.led_off()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].save_protocol("./data/protocol.json")

LumaViewProApp().run()
'''
