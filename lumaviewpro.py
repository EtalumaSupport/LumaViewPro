# General
import sys
import numpy as np
import time
import os
import json
import glob
import serial

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

class PylonCamera(Camera):
    record = ObjectProperty(None)
    record = False

    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
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

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

        self.start()

    def start(self):
        self.fps = 14
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def update(self, dt):
        try:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = grabResult.GetArray()
                    self.array = image
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')                    # display image from the texture
                    self.texture = image_texture

                self.lastGrab = pylon.PylonImage()
                self.lastGrab.AttachGrabResultBuffer(grabResult)

                if self.record == True:
                    self.capture()

                grabResult.Release()

        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

    def capture(self, save_folder = 'capture/', file_root = 'live_'):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = save_folder + file_root + timestr + '.tiff'
        self.lastGrab.Save(pylon.ImageFileFormat_Tiff, filename)

    def frame_size(self, w, h):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera'].camera

        camera.StopGrabbing()
        camera.Width.SetValue(min(int(w), camera.Width.Max))
        camera.Height.SetValue(min(int(h), camera.Height.Max))
        camera.OffsetX.SetValue(0)
        camera.OffsetY.SetValue(0)
        # camera.CenterX.SetValue(True)
        # camera.CenterY.SetValue(True)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def gain(self, gain):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera'].camera

        camera.StopGrabbing()
        camera.Gain.SetValue(gain)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def exposure_t(self, t):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera'].camera

        camera.StopGrabbing()
        camera.ExposureTime.SetValue(t*1000) # in microseconds
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

class LED:
    # def __init__(self, layer):
    #     self.layer = layer

    def set_layer(self, layer):
        self.layer = layer

    def on(self, current):
        print(self.layer, 'turned ON at i =', current, 'mA')

    def off(self):
        print(self.layer, 'turned OFF')


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

    def record(self):
        print(self.ids['viewer_id'].ids['microscope_camera'].record)
        if self.ids['viewer_id'].ids['microscope_camera'].record == True:
            self.ids['viewer_id'].ids['microscope_camera'].record = False
            self.ids['record_btn'].text = 'Record'
        else:
            self.ids['viewer_id'].ids['microscope_camera'].record = True
            self.ids['record_btn'].text = 'Stop Recording'

    def composite(self, dt):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']

        img = np.zeros((protocol['frame_height'], protocol['frame_width'], 3))

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            # multicolor image stack

            if protocol[layer]['acquire'] == True:
                # set the gain and expusure
                gain = protocol[layer]['gain']
                camera.gain(gain)
                exposure = protocol[layer]['exp']
                camera.exposure_t(exposure)
                camera.update(0)

                # turn on the LED

                # wait for LED time

                # buffer the images
                if layer == 'Blue':
                    img[:,:,0] = camera.array
                elif layer == 'Green':
                    img[:,:,1] = camera.array
                elif layer == 'Red':
                    img[:,:,2] = camera.array

        cv2.imwrite('./capture/composite.png', img)

    def fit_image(self):
        self.ids['viewer_id'].ids['microscope_camera'].keep_ratio = True

    def one2one_image(self):
        self.ids['viewer_id'].ids['microscope_camera'].keep_ratio = False

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

    def accordion_select(self):
        print('ACCORDION SELECT')

    def item_select(self, item):
        print('ITEM SELECT', item)

class MicroscopeSettings(BoxLayout):
    def microscope_select(self, scope):
        global protocol
        self.ids['select_scope_btn'].text = scope
        self.ids['image_of_microscope'].source = './data/'+scope+'.png'
        protocol['microscope'] = scope

    def objective_select(self, objective):
        global protocol
        self.ids['select_obj_btn'].text = 'Objective '+objective
        protocol['objective'] = objective

    def frame_size(self):
        global lumaview
        global protocol

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        protocol['frame_width'] = w
        protocol['frame_height'] = h

        lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(w, h)

class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    global protocol

    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)
        self.led = LED()

    def ill_slider(self):
        protocol[self.layer]['ill'] = self.ids['ill_slider'].value

    def ill_text(self):
        protocol[self.layer]['ill'] = float(self.ids['ill_text'].text)
        self.ids['ill_slider'].value = float(self.ids['ill_text'].text)

    def gain_slider(self):
        gain = self.ids['gain_slider'].value
        protocol[self.layer]['gain'] = gain
        lumaview.ids['viewer_id'].ids['microscope_camera'].gain(gain)

    def gain_text(self):
        gain = float(self.ids['gain_text'].text)
        protocol[self.layer]['gain'] = gain
        self.ids['gain_slider'].value = gain
        lumaview.ids['viewer_id'].ids['microscope_camera'].gain(gain)

    def exp_slider(self):
        exposure = self.ids['exp_slider'].value
        protocol[self.layer]['exp'] = exposure
        lumaview.ids['viewer_id'].ids['microscope_camera'].exposure_t(exposure)

    def exp_text(self):
        exposure = int(self.ids['exp_text'].text)
        protocol[self.layer]['exp'] = exposure
        self.ids['exp_slider'].value = exposure
        lumaview.ids['viewer_id'].ids['microscope_camera'].exposure_t(exposure)

    # def led_slider(self):
    #     protocol[self.layer]['led'] = self.ids['led_slider'].value
    #
    # def led_text(self):
    #     protocol[self.layer]['led'] = int(self.ids['led_text'].text)
    #     self.ids['led_slider'].value = int(self.ids['led_text'].text)
    #
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

    # def led_button(self):
    #     self.led.set_layer(self.layer)
    #     if self.ids['led_btn'].state == 'normal':
    #         self.led.off() # turn LED off
    #         self.ids['led_btn'].text = 'is off'
    #     else:
    #         self.led.on(50) # turn LED on
    #         self.ids['led_btn'].text = 'is on'

    def false_color(self):
        protocol[self.layer]['false_color'] = self.ids['acquire'].active
        # apply false color using Shader Editor

    def update_acquire(self):
        protocol[self.layer]['acquire'] = self.ids['acquire'].active

class TimeLapseSettings(BoxLayout):
    record = ObjectProperty(None)
    record = False
    movie_folder = StringProperty(None)
    n_captures = ObjectProperty(None)

    def update_period(self):
        protocol['period'] = float(self.ids['capture_period'].text)

    def update_duration(self):
        protocol['duration'] = float(self.ids['capture_dur'].text)

    # load protocol from JSON file
    def load_protocol(self, file=".\data\protocol.json"):
        global lumaview

        # determine file to read

        # load protocol JSON file
        with open(file, "r") as read_file:
            global protocol
            protocol = json.load(read_file)
            # update GUI values from JSON data
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_scope_btn'].text = protocol['microscope']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_width'].text = str(protocol['frame_width'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_height'].text = str(protocol['frame_height'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_obj_btn'].text = str(protocol['objective'])

            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = protocol[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = protocol[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = protocol[layer]['exp']
                # lumaview.ids['mainsettings_id'].ids[layer].ids['led_slider'].value = protocol[layer]['led']
                lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = '...' + protocol[layer]['save_folder'][-30:]
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = protocol[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['false_color'].active = protocol[layer]['false_color']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = protocol[layer]['acquire']

            lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(protocol['frame_width'], protocol['frame_height'])

    # Save protocol to JSON file
    def save_protocol(self):
        global protocol
        # determine file to write
        protocol_file = ".\data\protocol_save.json"
        with open(protocol_file, "w") as write_file:
            json.dump(protocol, write_file)

    # Run the timed process of capture event
    def run_protocol(self):
        global protocol

        # number of capture events remaining
        self.n_captures = int(float(protocol['duration'])*60 / float(protocol['period']))

        # update protocol
        if self.record == False:
            self.record = True

            hrs = np.floor(self.n_captures*protocol['period']/60)
            minutes = np.floor((self.n_captures*protocol['period']/60-hrs)*60)
            hrs = '%02d' % hrs
            minutes = '%02d' % minutes
            self.ids['protocol_btn'].text = hrs+':'+minutes+' remaining'

            self.dt = protocol['period']*60
            self.frame_event = Clock.schedule_interval(self.capture, self.dt)
        else:
            self.record = False
            self.ids['protocol_btn'].text = 'Record'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    # One procotol capture event
    def capture(self, dt):
        global lumaview
        try:
            self.n_captures = self.n_captures-1
        except:
            print('single composite mode')

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            if protocol[layer]['acquire'] == True:
                global lumaview
                camera = lumaview.ids['viewer_id'].ids['microscope_camera']

                # set the gain and exposure
                gain = protocol[layer]['gain']
                camera.gain(gain)
                exposure = protocol[layer]['exp']
                camera.exposure_t(exposure)
                camera.update(0)

                # turn on the LED

                # Wait the delay
                # time.sleep(float(protocol[layer]['led'])/1000)

                # capture the image
                save_folder = protocol[layer]['save_folder']
                file_root = protocol[layer]['file_root']
                lumaview.ids['viewer_id'].ids['microscope_camera'].capture(save_folder, file_root)

                # turn off the LED


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

# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        global lumaview
        lumaview = MainDisplay()
        return lumaview

LumaViewProApp().run()
