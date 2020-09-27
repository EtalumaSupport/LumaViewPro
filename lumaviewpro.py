# General
import sys
import numpy as np
import time
import os
import json
import glob

# Kivy
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock

# User Interface
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.popup import Popup

# Video Related
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.clock import Clock
import cv2
# from kivy.core.camera import Camera

# Pylon Camera Related
from pypylon import pylon
from pypylon import genicam

global lumaview
global protocol
with open('./data/protocol.json', "r") as read_file:
    protocol = json.load(read_file)

# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(FloatLayout):
    def cam_toggle(self):
        if self.ids['viewer_id'].ids['microscope_camera'].play == True:
            self.ids['viewer_id'].ids['microscope_camera'].play = False
            self.ids['live_btn'].text = 'Play Live'
            self.ids['viewer_id'].ids['microscope_camera'].stop()
        else:
            self.ids['viewer_id'].ids['microscope_camera'].play = True
            self.ids['live_btn'].text = 'Freeze'
            self.ids['viewer_id'].ids['microscope_camera'].start()

    def capture(self, dt):
        self.ids['viewer_id'].ids['microscope_camera'].capture()

    def fit_image(self):
        self.ids['viewer_id'].ids['microscope_camera'].keep_ratio = True

    def one2one_image(self):
        self.ids['viewer_id'].ids['microscope_camera'].keep_ratio = False

class PylonCamera(Camera):
    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            # Grabbing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

        self.start()

    def update(self, dt):
        try:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = grabResult.GetArray()
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')                    # display image from the texture
                    self.texture = image_texture

                self.lastGrab = pylon.PylonImage()
                self.lastGrab.AttachGrabResultBuffer(grabResult)
                grabResult.Release()

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

    def start(self):
        self.fps = 10
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def capture(self, save_folder = 'capture/', file_root = 'live_'):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = save_folder + file_root + timestr + '.tiff'
        self.lastGrab.Save(pylon.ImageFileFormat_Tiff, filename)

# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
# -----------------------------------------------------------------------------
fs_header = '''
#ifdef GL_ES
precision highp float;
#endif

/* Outputs from the vertex shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* uniform texture samplers */
uniform sampler2D texture0;

/* fragment attributes
attribute float red_gain;
attribute float green_gain;
attribute float blue_gain; */

/* custom one */
uniform vec2 resolution;
uniform float time;
uniform vec4 black_point;
uniform vec4 white_point;
'''

vs_header = '''
#ifdef GL_ES
precision highp float;
#endif

/* Outputs to the fragment shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* vertex attributes */
attribute vec2     vPosition;
attribute vec2     vTexCoords0;

/* uniform variables */
uniform mat4       modelview_mat;
uniform mat4       projection_mat;
uniform vec4       color;
'''

global illumination_vals
illumination_vals = (0., )*4

global gain_vals
gain_vals = (1., )*4

class ShaderViewer(BoxLayout):
    fs = StringProperty(None)
    vs = StringProperty(None)

    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_shader, 0)

    def update_shader(self, *args):
        global illumination_vals
        global gain_vals
        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = illumination_vals
        c['white_point'] = gain_vals

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)

class ShaderEditor(BoxLayout):
    fs = StringProperty('''
void main (void){
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
	- black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
# }
''')

    viewer = ObjectProperty(None)
    hide_editor = ObjectProperty(None)
    hide_editor = True


    def __init__(self, **kwargs):
        super(ShaderEditor, self).__init__(**kwargs)
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        print('try compile')
        if not self.viewer:
            print('compile fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        print('-->', fs)
        self.viewer.fs = fs
        print('-->', vs)
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        # update protocol
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

class MainSettings(BoxLayout):
    hide_settings = ObjectProperty(None)
    hide_settings = True

    # Hide (and unhide) main settings
    def toggle_settings(self):
        global lumaview
        # update protocol
        if self.hide_settings == False:
            self.hide_settings = True
            self.pos = lumaview.width-15, 0
        else:
            self.hide_settings = False
            self.pos = lumaview.width-300, 0

class MicroscopeSettings(BoxLayout):
    pass

class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    global protocol

    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        protocol[self.layer]['ill'] = self.ids['ill_slider'].value

    def ill_text(self):
        protocol[self.layer]['ill'] = float(self.ids['ill_text'].text)
        self.ids['ill_slider'].value = float(self.ids['ill_text'].text)

    def gain_slider(self):
        protocol[self.layer]['gain'] = self.ids['gain_slider'].value

    def gain_text(self):
        protocol[self.layer]['gain'] = float(self.ids['gain_text'].text)
        self.ids['gain_slider'].value = float(self.ids['gain_text'].text)

    def exp_slider(self):
        protocol[self.layer]['exp'] = self.ids['exp_slider'].value

    def exp_text(self):
        protocol[self.layer]['exp'] = int(self.ids['exp_text'].text)
        self.ids['exp_slider'].value = int(self.ids['exp_text'].text)

    def led_slider(self):
        protocol[self.layer]['led'] = self.ids['led_slider'].value

    def led_text(self):
        protocol[self.layer]['led'] = int(self.ids['led_text'].text)
        self.ids['led_slider'].value = int(self.ids['led_text'].text)

    def choose_folder(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Save Folder", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        protocol[self.layer]['save_folder'] = path
        self.ids['folder_btn'].text = '...'+path[-30:]
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def root_text(self):
        protocol[self.layer]['file_root'] = self.ids['root_text'].text

class TimeLapseSettings(BoxLayout):
    record = ObjectProperty(None)
    record = False

    # load protocol from JSON file
    def load_protocol(self):
        global lumaview
        # determine file to read
        protocol_file = ".\data\protocol.json"

        # load protocol JSON file
        with open(protocol_file, "r") as read_file:
            protocol = json.load(read_file)
            # update GUI values from JSON data
            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = protocol[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = protocol[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = protocol[layer]['exp']
                lumaview.ids['mainsettings_id'].ids[layer].ids['led_slider'].value = protocol[layer]['led']
                lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = '...' + protocol[layer]['save_folder'][-30:]
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = protocol[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = protocol[layer]['acquire']

    # Save protocol to JSON file
    def save_protocol(self):
        global protocol
        # determine file to write
        protocol_file = ".\data\protocol_save.json"
        with open(protocol_file, "w") as write_file:
            json.dump(protocol, write_file)

    # Run the process of capturing one protocol event
    def run_protocol(self):
        global protocol

        # update protocol
        if self.record == False:
            self.record = True
            self.ids['record_btn'].text = 'Stop Recording'

            self.dt = protocol['period']
            self.frame_event = Clock.schedule_interval(self.capture, self.dt)
        else:
            self.record = False
            self.ids['record_btn'].text = 'Record'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    def capture(self, dt):
        global lumaview
        lumaview.ids['viewer_id'].ids['microscope_camera'].capture()

    def movie(self):
        img_array = [] 
        for filename in glob.glob('./capture/*.tiff'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter('./capture/movie.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        global lumaview
        lumaview = MainDisplay()
        return lumaview

LumaViewProApp().run()
