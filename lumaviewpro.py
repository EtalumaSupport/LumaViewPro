# General
import sys
import numpy as np
import time
import os
import json

import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock

# User Interface
from kivy.uix.tabbedpanel import TabbedPanel
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
# from kivy.core.camera import Camera

# Pylon Camera Related
from pypylon import pylon
from pypylon import genicam

# Custom Imports

# -------------------------------------------------------------------------
# MAIN DISPLAY opf LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(TabbedPanel):
    pass

# -------------------------------------------------------------------------
# CONFIGURATION TAB and children
# -------------------------------------------------------------------------
class ConfigTab(BoxLayout):
    pass

# -------------------------------------------------------------------------
# LIVE IMAGE TAB and children
# -------------------------------------------------------------------------
class ImageTab(FloatLayout):
    def cam_toggle(self):
        if self.ids['viewer_id'].ids['microscope_camera'].play == True:
            self.ids['viewer_id'].ids['microscope_camera'].play = False
            self.ids['play_btn'].text = 'Play'
            self.ids['viewer_id'].ids['microscope_camera'].stop()
        else:
            self.ids['viewer_id'].ids['microscope_camera'].play = True
            self.ids['play_btn'].text = 'Pause'
            self.ids['viewer_id'].ids['microscope_camera'].start()
    def capture(self):
        self.ids['viewer_id'].ids['microscope_camera'].capture()

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

                grabResult.Release()

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

    def start(self):
        self.fps = 14
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def capture(self, save_folder = 'capture/', file_root = 'live_'):
        if self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                timestr = time.strftime("%Y%m%d_%H%M%S")
                filename = save_folder + file_root + timestr + '.tiff'
                img = pylon.PylonImage()
                img.AttachGrabResultBuffer(grabResult)
                img.Save(pylon.ImageFileFormat_Tiff, filename)


# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html


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

global exposure_vals
exposure_vals = (150, )*4

class ShaderEditor(BoxLayout):
    fs = StringProperty('''
void main (void){
	gl_FragColor = white_point * frag_color * texture2D(texture0, tex_coord0)
	- black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position = projection_mat * modelview_mat * vec4(vPosition.xy, 0.0, 1.0);
# }
''')

    viewer = ObjectProperty(None)

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

        #print(black_point)
        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = illumination_vals
        c['white_point'] = gain_vals
        c.ask_update()

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)

class ShaderSettings(BoxLayout):
    # get slider values and update global variables where needed
    def get_sliders(self):
        global illumination_vals
        global gain_vals
        global exposure_vals

        bf_ill = self.ids['bf_led_id'].ids['illumination_id'].value
        bf_gain = self.ids['bf_led_id'].ids['gain_id'].value
        bf_exp = self.ids['bf_led_id'].ids['exposure_id'].value

        bl_ill = self.ids['bl_led_id'].ids['illumination_id'].value
        bl_gain = self.ids['bl_led_id'].ids['gain_id'].value
        bl_exp = self.ids['bl_led_id'].ids['exposure_id'].value

        gr_ill = self.ids['gr_led_id'].ids['illumination_id'].value
        gr_gain = self.ids['gr_led_id'].ids['gain_id'].value
        gr_exp = self.ids['gr_led_id'].ids['exposure_id'].value

        rd_ill = self.ids['rd_led_id'].ids['illumination_id'].value
        rd_gain = self.ids['rd_led_id'].ids['gain_id'].value
        rd_exp = self.ids['rd_led_id'].ids['exposure_id'].value

        illumination_vals = (rd_ill, gr_ill, bl_ill, bf_ill)
        gain_vals = (rd_gain, gr_gain, bl_gain, bf_gain)
        exposure_vals = (rd_exp, gr_exp, bl_exp, bf_exp)

class LED_Control(BoxLayout):
    bg_color = ObjectProperty(None)
    ctrl_label = StringProperty(None)

    def __init__(self, **kwargs):
        super(LED_Control, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)
        if self.ctrl_label is None:
            self.ctrl_label = 'Ctrl Label'
# -------------------------------------------------------------------------
# MOTION TAB and children
# -------------------------------------------------------------------------
class MotionTab(BoxLayout):
    pass

# -------------------------------------------------------------------------
# PROTOCOL TAB and children
# -------------------------------------------------------------------------
class ProtocolTab(BoxLayout):
    acquiring = ObjectProperty(None)
    acquiring = False

    # transfer values from live image tab to protocol tab
    def import_vals(self):
        global illumination_vals
        global gain_vals
        global exposure_vals

        self.ids['bf_protocol'].ids['ill_val'].text = str(illumination_vals[3])
        self.ids['bf_protocol'].ids['gain_val'].text = str(gain_vals[3])
        self.ids['bf_protocol'].ids['exp_val'].text = str(exposure_vals[3])

        self.ids['bl_protocol'].ids['ill_val'].text = str(illumination_vals[2])
        self.ids['bl_protocol'].ids['gain_val'].text = str(gain_vals[2])
        self.ids['bl_protocol'].ids['exp_val'].text = str(exposure_vals[2])

        self.ids['gr_protocol'].ids['ill_val'].text = str(illumination_vals[1])
        self.ids['gr_protocol'].ids['gain_val'].text = str(gain_vals[1])
        self.ids['gr_protocol'].ids['exp_val'].text = str(exposure_vals[1])

        self.ids['rd_protocol'].ids['ill_val'].text = str(illumination_vals[0])
        self.ids['rd_protocol'].ids['gain_val'].text = str(gain_vals[0])
        self.ids['rd_protocol'].ids['exp_val'].text = str(exposure_vals[0])

    # load protocol from JSON file
    def load_protocol(self):
        # load protocol JSON file
        with open(".\data\protocol.json", "r") as read_file:
            protocol = json.load(read_file)
            # update GUI values from JSON data
            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            self.ids['bf_protocol'].ids['save_folder_id'].text = str(protocol['BF']['save_folder'])
            self.ids['bf_protocol'].ids['file_root_id'].text = str(protocol['BF']['file_root'])
            self.ids['bf_protocol'].ids['ill_val'].text = str(protocol['BF']['ill'])
            self.ids['bf_protocol'].ids['gain_val'].text = str(protocol['BF']['gain'])
            self.ids['bf_protocol'].ids['exp_val'].text = str(protocol['BF']['exp'])
            self.ids['bf_protocol'].ids['acquire'].active = protocol['BF']['acquire']

            self.ids['bl_protocol'].ids['save_folder_id'].text = str(protocol['Blue']['save_folder'])
            self.ids['bl_protocol'].ids['file_root_id'].text = str(protocol['Blue']['file_root'])
            self.ids['bl_protocol'].ids['ill_val'].text = str(protocol['Blue']['ill'])
            self.ids['bl_protocol'].ids['gain_val'].text = str(protocol['Blue']['gain'])
            self.ids['bl_protocol'].ids['exp_val'].text = str(protocol['Blue']['exp'])
            self.ids['bl_protocol'].ids['acquire'].active = protocol['Blue']['acquire']

            self.ids['gr_protocol'].ids['save_folder_id'].text = str(protocol['Green']['save_folder'])
            self.ids['gr_protocol'].ids['file_root_id'].text = str(protocol['Green']['file_root'])
            self.ids['gr_protocol'].ids['ill_val'].text = str(protocol['Green']['ill'])
            self.ids['gr_protocol'].ids['gain_val'].text = str(protocol['Green']['gain'])
            self.ids['gr_protocol'].ids['exp_val'].text = str(protocol['Green']['exp'])
            self.ids['gr_protocol'].ids['acquire'].active = protocol['Green']['acquire']

            self.ids['rd_protocol'].ids['save_folder_id'].text = str(protocol['Red']['save_folder'])
            self.ids['rd_protocol'].ids['file_root_id'].text = str(protocol['Red']['file_root'])
            self.ids['rd_protocol'].ids['ill_val'].text = str(protocol['Red']['ill'])
            self.ids['rd_protocol'].ids['gain_val'].text = str(protocol['Red']['gain'])
            self.ids['rd_protocol'].ids['exp_val'].text = str(protocol['Red']['exp'])
            self.ids['rd_protocol'].ids['acquire'].active = protocol['Red']['acquire']

            self.ids['composite_protocol'].ids['save_folder_id'].text = str(protocol['Composite']['save_folder'])
            self.ids['composite_protocol'].ids['file_root_id'].text = str(protocol['Composite']['file_root'])
            self.ids['composite_protocol'].ids['ill_val'].text = ''
            self.ids['composite_protocol'].ids['gain_val'].text = ''
            self.ids['composite_protocol'].ids['exp_val'].text = ''
            self.ids['composite_protocol'].ids['acquire'].active = protocol['Composite']['acquire']

    # Save protocol to JSON file
    def save_protocol(self):
        # update protocol
        with open(".\data\protocol.json", "r") as read_file:
            protocol = json.load(read_file)

        protocol['period'] = float(self.ids['capture_period'].text)
        protocol['duration'] = float(self.ids['capture_dur'].text)

        protocol['BF']['save_folder'] = self.ids['bf_protocol'].ids['save_folder_id'].text
        protocol['BF']['file_root'] = self.ids['bf_protocol'].ids['file_root_id'].text
        protocol['BF']['ill'] = float(self.ids['bf_protocol'].ids['ill_val'].text)
        protocol['BF']['gain'] = float(self.ids['bf_protocol'].ids['gain_val'].text)
        protocol['BF']['exp'] = float(self.ids['bf_protocol'].ids['exp_val'].text)
        protocol['BF']['acquire'] = self.ids['bf_protocol'].ids['acquire'].active

        protocol['Blue']['save_folder'] = self.ids['bl_protocol'].ids['save_folder_id'].text
        protocol['Blue']['file_root'] = self.ids['bl_protocol'].ids['file_root_id'].text
        protocol['Blue']['ill'] = float(self.ids['bl_protocol'].ids['ill_val'].text)
        protocol['Blue']['gain'] = float(self.ids['bl_protocol'].ids['gain_val'].text)
        protocol['Blue']['exp'] = float(self.ids['bl_protocol'].ids['exp_val'].text)
        protocol['Blue']['acquire'] = self.ids['bl_protocol'].ids['acquire'].active

        protocol['Green']['save_folder'] = self.ids['gr_protocol'].ids['save_folder_id'].text
        protocol['Green']['file_root'] = self.ids['gr_protocol'].ids['file_root_id'].text
        protocol['Green']['ill'] = float(self.ids['gr_protocol'].ids['ill_val'].text)
        protocol['Green']['gain'] = float(self.ids['gr_protocol'].ids['gain_val'].text)
        protocol['Green']['exp'] = float(self.ids['gr_protocol'].ids['exp_val'].text)
        protocol['Green']['acquire'] = self.ids['gr_protocol'].ids['acquire'].active

        protocol['Red']['save_folder'] = self.ids['rd_protocol'].ids['save_folder_id'].text
        protocol['Red']['file_root'] = self.ids['rd_protocol'].ids['file_root_id'].text
        protocol['Red']['ill'] = float(self.ids['rd_protocol'].ids['ill_val'].text)
        protocol['Red']['gain'] = float(self.ids['rd_protocol'].ids['gain_val'].text)
        protocol['Red']['exp'] = float(self.ids['rd_protocol'].ids['exp_val'].text)
        protocol['Red']['acquire'] = self.ids['rd_protocol'].ids['acquire'].active

        protocol['Composite']['save_folder'] = self.ids['composite_protocol'].ids['save_folder_id'].text
        protocol['Composite']['file_root'] = self.ids['composite_protocol'].ids['file_root_id'].text
        protocol['Composite']['ill'] = ''
        protocol['Composite']['gain'] = ''
        protocol['Composite']['exp'] = ''
        protocol['Composite']['acquire'] = self.ids['composite_protocol'].ids['acquire'].active

        with open("./data/protocol_save.json", "w") as write_file:
            json.dump(protocol, write_file)
        print('Save protocol not yet written')

    # Run the protocol for acquiring image stacks
    def run_protocol(self):
        # update protocol
        if self.acquiring == False:
            self.acquiring = True
            self.ids['run_btn'].text = 'Stop Protocol'


            self.dt = float(self.ids['capture_period'].text)
            self.frame_event = Clock.schedule_interval(self.capture_event, self.dt*60.)
        else:
            self.acquiring = False
            self.ids['run_btn'].text = 'Run Protocol'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    # Run the process of capturing one protocol event
    def capture_event(self, dt):
        print('capture event')


class Protocol_Control(BoxLayout):
    bg_color = ObjectProperty(None)
    protocol_label = StringProperty(None)
    save_folder = StringProperty(None)
    file_root = StringProperty(None)

    def __init__(self, **kwargs):
        super(Protocol_Control, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)
        if self.protocol_label is None:
            self.protocol_label = 'Protocol Label'

    def choose_folder(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Save Folder", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
        # create a folder selection pop-up
        print(self.save_folder)

    def load(self, path):
        self.save_folder = path
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

# -------------------------------------------------------------------------
# ANALYSIS TAB and children
# -------------------------------------------------------------------------
class AnalysisTab(BoxLayout):
    pass

# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        # kwargs = {}
        # if len(sys.argv) > 1:
        #     kwargs['source'] = sys.argv[1]
        return MainDisplay()

# def update_filter_callback():
#     pass

LumaViewProApp().run()
