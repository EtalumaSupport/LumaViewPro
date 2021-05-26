#!/usr/bin/env python3

'''
MIT License

Copyright (c) 2020 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute
Bryan Tiedemann

MODIFIED:
April 30, 2021
'''

# General
import sys
import numpy as np
import time
import os
import json
import serial
import serial.tools.list_ports as list_ports

# Kivy
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.properties import BoundedNumericProperty, ColorProperty, OptionProperty, ListProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.animation import Animation

# User Interface
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button

# From KivyMD
from kivymd.uix.behaviors import HoverBehavior, TouchBehavior

# Video Related
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2

# Pylon Camera Related
from pypylon import pylon

global lumaview
global protocol
with open('./data/protocol.json', "r") as read_file:
    protocol = json.load(read_file)

class PylonCamera(Image):
    record = ObjectProperty(None)
    record = False

    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
        self.camera = False;
        self.play = True;
        self.connect()
        self.start()


    def connect(self):
        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.Width.SetValue(self.camera.Width.Max)
            self.camera.Height.SetValue(self.camera.Height.Max)
            self.camera.GainAuto.SetValue('Off')
            self.camera.ExposureAuto.SetValue('Off')
            # Grabbing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        except:
            if self.camera == False:
                print("It looks like a Lumaview compatible camera or scope is not plugged in")
            self.camera = False

    def start(self):
        self.fps = 10
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def update(self, dt):
        if self.camera == False:
            self.connect()
            if self.camera == False:
                self.source = "./data/camera to USB.png"
                # self.scale = 1
                # self.pos = (0,0)
                return
        try:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = grabResult.GetArray()
                    image = cv2.flip(image, 1)
                    self.array = image
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')                    # display image from the texture
                    self.texture = image_texture

                self.lastGrab = pylon.PylonImage()
                self.lastGrab.AttachGrabResultBuffer(grabResult)

                if self.record == True:
                    self.capture(append = 'ms')

                grabResult.Release()

        except:
            if self.camera == False:
                print("It looks like a Lumaview compatible camera was unplugged")
            self.camera = False

    def capture(self, save_folder = './capture/', file_root = 'live_', append = 'ms'):
        if self.camera == False:
            print("DEBUG: capture() self.camera == False")
            return

        if append == 'time':
            append = time.strftime("%Y%m%d_%H%M%S")
        elif append == 'ms':
            append = str(int(round(time.time() * 1000)))
        else:
            append =''

        filename = save_folder + '/' + file_root + append + '.tiff'
        try:
            self.lastGrab.Save(pylon.ImageFileFormat_Tiff, filename)
        except:
            print("Save folder does not exist")

    def frame_size(self, w, h):
        if self.camera == False:
            print("DEBUG: frame_size() self.camera == False")
            return

        width = int(min(int(w), self.camera.Width.Max)/2)*2
        height = int(min(int(h), self.camera.Height.Max)/2)*2
        offset_x = int((self.camera.Width.Max-width)/4)*2
        offset_y = int((self.camera.Height.Max-height)/4)*2

        self.camera.StopGrabbing()
        self.camera.Width.SetValue(width)
        self.camera.Height.SetValue(height)
        self.camera.OffsetX.SetValue(offset_x)
        self.camera.OffsetY.SetValue(offset_y)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def gain(self, gain):
        if self.camera == False:
            print("DEBUG: gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.Gain.SetValue(gain)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def exposure_t(self, t):
        if self.camera == False:
            print("DEBUG: exposure_t() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.ExposureTime.SetValue(t*1000) # (t*1000) in microseconds; therefore t  in milliseconds
        # # DEBUG:
        # print(camera.ExposureTime.Min)
        # print(camera.ExposureTime.Max)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

class LEDBoard:
    def __init__(self, **kwargs):

        ports = list(list_ports.comports())
        if (len(ports)!=0):
            self.port = ports[0].device
        # self.port="COM5"
        self.baudrate=9600
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=10.0 # seconds
        self.driver = True
        self.connect()

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        try:
            self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
            self.driver.close()
            self.driver.open()
        except:
            if self.driver != False:
                print("It looks like a Lumaview compatible LED driver board is not plugged in")
            self.driver = False

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else:
            return 3

    def led_cal(self, channel):
        command = '{CAL,'+ str(channel) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_on(self, channel, mA):
        command = '{TON,'+ str(channel) + ',H,' + str(mA) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_off(self):
        command = '{TOF}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(FloatLayout):
    led_board = ObjectProperty(None)
    led_board = LEDBoard()

    def choose_folder(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             path=protocol['live_folder'])
        self._popup = Popup(title="Select Save Folder",
                            content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        protocol['live_folder'] = path
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def cam_toggle(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return

        if microscope.play == True:
            microscope.play = False
            self.led_board.led_off()
            microscope.stop()
        else:
            microscope.play = True
            microscope.start()

    def capture(self, dt):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        folder = protocol['live_folder']
        microscope.capture(save_folder = folder)

    def record(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        microscope.record != microscope.record
        # if camera.record == True:
        #     camera.record = False
        # else:
        #     camera.record = True
        #     # self.ids['record_btn'].text = 'Stop Recording'

    def composite(self, dt):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return

        folder = protocol['live_folder']
        img = np.zeros((protocol['frame_height'], protocol['frame_width'], 3))

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            # multicolor image stack

            if protocol[layer]['acquire'] == True:
                # set the gain and expusure
                gain = protocol[layer]['gain']
                microscope.gain(gain)
                exposure = protocol[layer]['exp']
                microscope.exposure_t(exposure)

                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board
                led_board.led_on(led_board.color2ch(layer), illumination)

                microscope.update(0)
                # buffer the images
                if layer == 'Blue':
                    img[:,:,0] = microscope.array
                elif layer == 'Green':
                    img[:,:,1] = microscope.array
                elif layer == 'Red':
                    img[:,:,2] = microscope.array
                else:
                    img[:,:,2] = microscope.array

        led_board.led_off()
        filename = 'composite_' + str(int(round(time.time() * 1000))) + '.png'
        cv2.imwrite(folder+'/'+filename, img)

    def fit_image(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        w = self.width
        h = self.height
        scale_hor = float(microscope.camera.Width.GetValue()) / float(w)
        scale_ver = float(microscope.camera.Height.GetValue()) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (int((w-scale*w)/2),int((h-scale*h)/2))

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

# global illumination_vals
# illumination_vals = (0., )*4

global gain_vals
gain_vals = (1., )*4

class ShaderViewer(Scatter):
    fs = StringProperty('''
void main (void) {
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
}
''')


    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        Clock.schedule_interval(self.update_shader, 0)

    def on_touch_down(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 100:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 0.1:
                    self.scale = self.scale * 0.8
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ShaderViewer, self).on_touch_down(touch)

    def update_shader(self, *args):
        global gain_vals

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (0., )*4
        # adjust for false color
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
}
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
    settings_width = dp(300)
    isOpen = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_draw=self.check_settings)

    # Hide (and unhide) main settings
    def toggle_settings(self):
        global lumaview
        # update protocol
        if self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
            self.isOpen = False
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0
            self.isOpen = True

    def check_settings(self, *args):
        global lumaview
        if not self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0


class MicroscopeSettings(BoxLayout):
#    def port_select(self, port):
#        global protocol
#        self.ids['select_port_btn'].text = port
#        # protocol['objective'] = objective

    def frame_size(self):
        global lumaview
        global protocol

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        # print(w)
        # print(h)
        camera = lumaview.ids['viewer_id'].ids['microscope_camera'].camera

        width = int(min(int(w), camera.Width.Max)/2)*2
        height = int(min(int(h), camera.Height.Max)/2)*2

        protocol['frame_width'] = width
        protocol['frame_height'] = height

        lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(w, h)


# Pass-through class for microscope selection drop-down menu, defined in .kv file
# -------------------------------------------------------------------------------
class MicroscopeDropDown(DropDown):
    pass


# First line of Microscope Settings control panel, to select model of microscope
# ------------------------------------------------------------------------------
class ScopeSelect(BoxLayout):
    # The text displayed in the button
    scope_str = StringProperty(protocol['microscope'])

    def __init__(self, **kwargs):
        super(ScopeSelect, self).__init__(**kwargs)

        # Extract current scope value and put it in scope_str
#        self.scope_str = protocol['microscope']

        # Create label and button here so DropDown menu works properly
        self.mainlabel = Label(text = 'Lumascope Model',
                               size_hint_x = None, width = '150dp', font_size = '14sp')
        self.mainbutton = Button(text = self.scope_str,
                                 size_hint_y = None, height = '40dp')

        # Group widgets together
        self.dropdown = MicroscopeDropDown()
        self.add_widget(self.mainlabel)
        self.add_widget(self.mainbutton)

        # Add actions - mainbutton opens dropdown menu
        self.mainbutton.bind(on_release = self.dropdown.open)

        # Dropdown buttons do stuff based on their text through microscope_selectFN
        self.dropdown.bind(on_select = lambda instance,
                                       scope: setattr(self.mainbutton, 'text', scope))
        self.dropdown.bind(on_select = self.microscope_selectFN)


    def microscope_selectFN(self, instance, scope):
        global protocol
        self.scope_str = scope
        self.parent.ids['image_of_microscope'].source = './data/scopes/'+scope+'.png'
        protocol['microscope'] = scope
        print("Selected microscope: {0}" . format(scope))


# Pass-through class for objective settings drop down menu, defined in .kv file
# ----------------------------------------------------------------------------
class ObjectiveDropDown(DropDown):
    pass


# Second line of Microscope Settings control panel, to select objective
# ---------------------------------------------------------------------
class ObjectiveSelect(BoxLayout):
    # The text displayed in the button
    objective_str = StringProperty('Canon 55mm')

    def __init__(self, **kwargs):
        super(ObjectiveSelect, self).__init__(**kwargs)

        # Create label and button here so DropDown menu works properly
        self.mainlabel = Label(text = 'Objective',
                               size_hint_x = None, width = '150dp', font_size = '14sp')
        self.mainbutton = Button(text = self.objective_str,
                                 size_hint_y = None, height = '40dp')

        # Group widgets together
        self.dropdown = ObjectiveDropDown()
        self.add_widget(self.mainlabel)
        self.add_widget(self.mainbutton)

        # Add actions - mainbutton opens dropdown menu
        self.mainbutton.bind(on_release = self.dropdown.open)

        # Dropdown buttons do stuff based on their text through objective_select
        self.dropdown.bind(on_select = lambda instance,
                           objective: setattr(self.mainbutton, 'text', objective))
        self.dropdown.bind(on_select = self.objective_select)


    def objective_select(self, instance, objective):
        global protocol
        self.objective_str = objective
        protocol['objective'] = objective


class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    global protocol

    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        illumination = self.ids['ill_slider'].value
        protocol[self.layer]['ill'] = illumination
        self.apply_settings()

    def ill_text(self):
        if self.ids['ill_text'].text.isnumeric():
            illumination = float(self.ids['ill_text'].text)
            protocol[self.layer]['ill'] = illumination
            self.ids['ill_slider'].value = illumination
            self.apply_settings()

    def gain_slider(self):
        gain = self.ids['gain_slider'].value
        protocol[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):
        if self.ids['gain_text'].text.isnumeric():
            gain = float(self.ids['gain_text'].text)
            protocol[self.layer]['gain'] = gain
            self.ids['gain_slider'].value = gain
            self.apply_settings()

    def exp_slider(self):
        exposure = self.ids['exp_slider'].value
        protocol[self.layer]['exp'] = exposure
        self.apply_settings()

    def exp_text(self):
        if self.ids['exp_text'].text.isnumeric():
            exposure = float(self.ids['exp_text'].text)
            protocol[self.layer]['exp'] = exposure
            self.ids['exp_slider'].value = exposure
            self.apply_settings()

    def choose_folder(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             path=protocol[self.layer]['save_folder'])
        self._popup = Popup(title="Select Save Folder",
                            content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        protocol[self.layer]['save_folder'] = path
        if len(path) > 30:
            self.ids['folder_btn'].text = '... '+path[-30:]
        else:
            self.ids['folder_btn'].text = path
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def root_text(self):
        protocol[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        protocol[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        protocol[self.layer]['acquire'] = self.ids['acquire'].active

    def apply_settings(self):
        global lumaview
        global gain_vals
        # update false color to currently selected settings
        if self.ids['false_color'].active:
            if(self.layer) == 'Red':
                gain_vals = (1., 0., 0., 1.)
            elif(self.layer) == 'Green':
                gain_vals = (0., 1., 0., 1.)
            elif(self.layer) == 'Blue':
                gain_vals = (0., 0., 1., 1.)
        else:
            gain_vals =  (1., )*4

        if self.layer == 'BF':
            self.ids['false_color_label'].text = '' # Remove 'Colorize' option in brightfield control
            self.ids['false_color'].color = (0., )*4
            #self.ids['false_color'].size = 0, 0

        # update illumination to currently selected settings
        illumination = protocol[self.layer]['ill']

        # update LED illumination to currently selected settings
        led_board = lumaview.led_board
        if self.ids['apply_btn'].state == 'down': # if the button is down
            #  turn the state of all channels to 'normal' and
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
                lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].text = 'OFF'

            # return the state of the current channel to 'down' and text to 'ON'
            self.ids['apply_btn'].state = 'down'
            self.ids['apply_btn'].text = 'ON'

            # turn on the LED
            led_board.led_on(led_board.color2ch(self.layer), illumination)
        else:
            # update the text of the button
            self.ids['apply_btn'].text = 'OFF'
            led_board.led_off() # turn off the LED

        # update gain to currently selected settings
        gain = protocol[self.layer]['gain']
        lumaview.ids['viewer_id'].ids['microscope_camera'].gain(gain)

        # update exposure to currently selected settings
        exposure = protocol[self.layer]['exp']
        lumaview.ids['viewer_id'].ids['microscope_camera'].exposure_t(exposure)


class TimeLapseSettings(BoxLayout):
    record = ObjectProperty(None)
    record = False
    movie_folder = StringProperty(None)
    n_captures = ObjectProperty(None)

    def update_period(self):
        if self.ids['capture_period'].text.isnumeric():
            protocol['period'] = float(self.ids['capture_period'].text)

    def update_duration(self):
        if self.ids['capture_dur'].text.isnumeric():
            protocol['duration'] = float(self.ids['capture_dur'].text)

    # load protocol from JSON file
    def load_protocol(self, file="./data/protocol.json"):
        global lumaview

        # determine file to read

        # load protocol JSON file
        with open(file, "r") as read_file:
            global protocol
            protocol = json.load(read_file)
            # update GUI values from JSON data:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_scope_btn'].scope_str = protocol['microscope']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_width'].text = str(protocol['frame_width'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_height'].text = str(protocol['frame_height'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_obj_btn'].objective_str = str(protocol['objective'])

            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = protocol[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = protocol[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = protocol[layer]['exp']
                if len(protocol[layer]['save_folder']) > 30:
                    lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = '... ' + protocol[layer]['save_folder'][-30:]
                else:
                    lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = protocol[layer]['save_folder'][-30:]
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = protocol[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['false_color'].active = protocol[layer]['false_color']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = protocol[layer]['acquire']

            lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(protocol['frame_width'], protocol['frame_height'])

    # Save protocol to JSON file
    def save_protocol(self, file="./data/protocol.json"):
        global protocol
        with open(file, "w") as write_file:
            json.dump(protocol, write_file)

    # Run the timed process of capture event
    def run_protocol(self):
        global protocol

        # number of capture events remaining
        ## duration is in hours, period is in minutes
        self.n_captures = int(float(protocol['duration'])*60 / float(protocol['period']))

        # update protocol
        if self.record == False:
            self.record = True

            hrs = np.floor(self.n_captures * protocol['period']/60)
            minutes = np.floor((self.n_captures*protocol['period']/60-hrs)*60)
            hrs = '%02d' % hrs
            minutes = '%02d' % minutes
            self.ids['protocol_btn'].text = hrs+':'+minutes+' remaining'

            self.dt = protocol['period']*60 # frame events are measured in seconds
            self.frame_event = Clock.schedule_interval(self.capture, self.dt)
        else:
            self.record = False
            self.ids['protocol_btn'].text = 'Record'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    # One procotol capture event
    def capture(self, dt):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']
        if camera == False:
            return
        try:
            self.n_captures = self.n_captures-1
        except:
            print('single composite mode')

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            if protocol[layer]['acquire'] == True:
                # global lumaview

                # set the gain and exposure
                gain = protocol[layer]['gain']
                camera.gain(gain)
                exposure = protocol[layer]['exp']
                camera.exposure_t(exposure)
                camera.update(0)

                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board
                led_board.led_on(led_board.color2ch(layer), illumination)

                # Wait the delay
                time.sleep(50/1000)

                # capture the image
                save_folder = protocol[layer]['save_folder']
                file_root = protocol[layer]['file_root']
                lumaview.ids['viewer_id'].ids['microscope_camera'].capture(save_folder, file_root)
                # turn off the LED
                led_board.led_off()


    def convert_to_avi(self):

        # self.choose_folder()
        save_location = './capture/movie.avi'

        img_array = []
        for filename in glob.glob('./capture/*.tiff'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def choose_folder(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Image Folder", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        self.movie_folder = path
        print(self.movie_folder)
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = ObjectProperty(None)


# ------------------------------------------------------------------------
# TOOLTIPS
# ------------------------------------------------------------------------
class Tooltip(HoverBehavior, TouchBehavior):
    # Tooltip background color
    tooltip_bg_color = ColorProperty(None)

    # Tooltip text color
    tooltip_text_color = ColorProperty(None)

    # Tooltip text to display
    tooltip_text = StringProperty(None)

    # Radius of rounded rectangle's corner
    tooltip_radius = ListProperty([dp(5),])

    # Tooltip display delay, defaults to 0 and max of 4
    tooltip_display_delay = BoundedNumericProperty(0, min=0, max=4)

    # Y-offset of tooltip text, defaults to 0
    shift_y = NumericProperty(0)

    _tooltip = None


    def delete_clock(self, widget, touch, *args):
        if self.collide_point(touch.x, touch.y) and touch.grab_current:
            try:
                Clock.unschedule(touch.ud["event"])
            except KeyError:
                pass
            self.on_leave()

    # Returns the coordinates of the tooltio that fit in screen borders
    def adjust_tooltip_position(self, x, y):
        # If tooltip position is outside the right border of the screen:
        if x + self._tooltip.width > Window.width:
            x = Window.width - (self._tooltip.width + dp(10))
        elif x < 0:
            # If the tooltip position is outside the left boder of the screen
            x = '10dp'

        # If the tooltip position is below the bottom border:
        if y < 0:
            y = dp(10)
        elif Window.height - self._tooltip.height < y:
            y - Window.height - (self._tooltip.height + dp(10))

        return x, y

    # Display the tooltip using an animated routine after a display delay defined by user
    def display_tooltip(self, interval):
        if not self._tooltip:
            return

        Window.add_widget(self._tooltip)
        pos = self.to_window(self.center_x, self.center_y)
        x = pos[0] - self._tooltip.width / 2

        if not self.shift_y:
            y = pos[1] - self._tooltip.height / 2 - self.height / 2 - dp(20)
        else:
            y = pos[1] - self._tooltip.height / 2 - self.height + self.shift_y

        x, y = self.adjust_tooltip_position(x, y)
        self._tooltip.pos = (x, y)

        Clock.schedule_once(self.animation_tooltip_show, self.tooltip_display_delay)

    # Method that displays tooltip in an animated way
    def animation_tooltip_show(self, interval):
        if not self._tooltip:
            return

        (Animation(_scale_x = 1, _scale_y = 1, d = 0.1)
                + Animation(opacity = 1, d = 0.2)).start(self._tooltip)

    # Makes tooltip disappear
    def remove_tooltip(self, *args):
        Window.remove_widget(self._tooltip)

    def on_long_touch(self, touch, *args):
        return

    def on_enter(self, *args):
        if not self.tooltip_text:
            return

        self._tooltip = TooltipViewClass(
                tooltip_bg_color = self.tooltip_bg_color,
                tooltip_text_color = self.tooltip_text_color,
                tooltip_text = self.tooltip_text,
                tooltip_radius = self.tooltip_radius)
        Clock.schedule_once(self.display_tooltip, -1)

    def on_leave(self):
        if self._tooltip:
            Window.remove_widget(self._tooltip)
            self._tooltip = None


# Holder layout for the tooltip
class TooltipViewClass(BoxLayout):
    tooltip_bg_color = ColorProperty(None)
    tooltip_text_color = ColorProperty(None)
    tooltip_text = StringProperty()
    tooltip_radius = ListProperty()

    _scale_x = NumericProperty(0)
    _scale_y = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = [dp(8), dp(4), dp(8), dp(4)]


# Button-derivative with tooltips
class TooltipButton(Button, Tooltip):
    def __init__(self, **kwargs):
        super(TooltipButton, self).__init__(**kwargs)
        self.tooltip_bg_color = (0.1, 0.1, 0.1, 0.7)
        self.tooltip_display_delay = 0.
        self.shift_y = dp(80)


# Toggle Button-derivative with tooltips
class TooltipToggleButton(ToggleButton, Tooltip):
    def __init__(self, **kwargs):
        super(TooltipToggleButton, self).__init__(**kwargs)
        self.tooltip_bg_color = (0.1, 0.1, 0.1, 0.7)
        self.tooltip_display_delay = 0.5
        self.shift_y = dp(80)


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        Window.minimum_width = 800
        Window.minimum_height = 600
        global lumaview
        lumaview = MainDisplay()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].load_protocol("./data/protocol.json")
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.led_off()
        return lumaview

    def on_stop(self):
        global lumaview
        lumaview.led_board.led_off()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].save_protocol("./data/protocol.json")

LumaViewProApp().run()
