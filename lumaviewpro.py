#!/usr/bin/python3

'''
MIT License

Copyright (c) 2020 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
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
Bryan Tiedemann, The Earthineering Company

MODIFIED:
May 19, 2022
'''

# General
import os
import numpy as np
import pandas as pd
import time
import json
import glob
import math
import threading
from plyer import filechooser
# from scipy.optimized import curve_fit

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
from kivy.graphics import Line, Color, Rectangle, Ellipse
# from kivy.config import Config
# Config.set('graphics', 'width', '1920')
# Config.set('graphics', 'height', '1080')

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
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button

# Video Related
from kivy.graphics.texture import Texture
import cv2
from scipy import signal

# Additional LumaViewPro files
from trinamic850 import *
from ledboard import *
from pyloncamera import *
from labware import *

from kivy.config import Config
Config.set('input', 'mouse', 'mouse, disable_multitouch')

global lumaview
global settings
home_wd = os.getcwd()

start_str = time.strftime("%Y %m %d %H_%M_%S")
start_str = str(int(round(time.time() * 1000)))

def error_log(mssg):
    if True:
        os.chdir(home_wd)
        file = open('LVP_log '+start_str+'.txt', 'a')
        file.write(mssg + '\n')
        file.close()
        print(mssg)

# -------------------------------------------------------------------------
# SCOPE DISPLAY Image representing the microscope camera
# -------------------------------------------------------------------------
class ScopeDisplay(Image):
    record = BooleanProperty(None)
    record = False
    play = BooleanProperty(None)
    play = True

    def __init__(self, **kwargs):
        super(ScopeDisplay,self).__init__(**kwargs)
        error_log('ScopeDisplay.__init__()')
        self.start()

    def start(self, fps = 10):
        error_log('ScopeDisplay.start()')
        self.fps = fps
        error_log('Clock.schedule_interval(self.update, 1.0 / self.fps)')
        Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        error_log('ScopeDisplay.stop()')
        # self.frame_event.cancel()
        error_log('Clock.unschedule(self.update)')
        Clock.unschedule(self.update)

    def update(self, dt=0):
        global lumaview

        if lumaview.camera.grab():
            array = lumaview.camera.array
            texture = Texture.create(size=(array.shape[1],array.shape[0]), colorfmt='luminance')
            texture.blit_buffer(array.flatten(), colorfmt='luminance', bufferfmt='ubyte')
            # display image from the texture
            self.texture = texture
        else:
            self.source = "./data/icons/camera to USB.png"

        if self.record == True:
            lumaview.live_capture()


# -------------------------------------------------------------------------
# COMPOSITE CAPTURE FloatLayout with shared capture capabilities
# -------------------------------------------------------------------------
class CompositeCapture(FloatLayout):

    def live_capture(self):
        error_log('CompositeCapture.live_capture()')
        global lumaview

        save_folder = settings['live_folder']
        file_root = 'live_'
        append = 'ms'
        color = 'BF'
        if lumaview.ids['mainsettings_id'].currentLayer != 'microscope':
            color = lumaview.ids['mainsettings_id'].currentLayer

        lumaview.camera.grab()
        error_log(lumaview.camera.mssg)
        self.save_image(save_folder, file_root, append, color)

    def custom_capture(self, channel, illumination, gain, exposure):
        error_log('CompositeCapture.custom_capture()')
        global lumaview
        global settings
        
        # Set gain and exposure
        lumaview.camera.gain(gain)
        error_log(lumaview.camera.mssg)
        lumaview.camera.exposure_t(exposure)
        error_log(lumaview.camera.mssg)
 
        # Save Settings
        color = lumaview.led_board.ch2color(channel)
        save_folder =  settings[color]['save_folder']
        file_root = settings[color]['file_root']
        append = 'ms'

        # Illuminate
        lumaview.led_board.led_on(channel, illumination)
        error_log(lumaview.led_board.mssg)

        # Grab image and save
        lumaview.camera.grab()
        error_log(lumaview.camera.mssg)
        self.save_image(save_folder, file_root, append, color)

        # Turn off LEDs
        lumaview.led_board.leds_off()
        error_log(lumaview.led_board.mssg)

    # Save image from camera buffer to specified location
    def save_image(self, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
        error_log('CompositeCapture.save_image()')
        global lumaview

        if lumaview.camera.active == False:
            return

        img = np.zeros((lumaview.camera.array.shape[0], lumaview.camera.array.shape[1], 3))

        if color == 'Blue':
            img[:,:,0] = lumaview.camera.array
        elif color == 'Green':
            img[:,:,1] = lumaview.camera.array
        elif color == 'Red':
            img[:,:,2] = lumaview.camera.array
        else:
            img[:,:,0] = lumaview.camera.array
            img[:,:,1] = lumaview.camera.array
            img[:,:,2] = lumaview.camera.array

        img = np.flip(img, 0)

        # set filename options
        if append == 'ms':
            append = str(int(round(time.time() * 1000)))
        elif append == 'time':
            append = time.strftime("%Y%m%d_%H%M%S")
        else:
            append =''

        # generate filename string
        filename =  file_root + append + '.tiff'

        try:
            cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))
            # cv2.imwrite(filename, img.astype(np.uint8))
        except:
            error_log("Error: Unable to save. Perhaps save folder does not exist?")

    # capture and save a composite image using the current settings
    def composite_capture(self):
        error_log('CompositeCapture.composite_capture()')
        global lumaview

        if self.camera.active == False:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']

        img = np.zeros((settings['frame']['height'], settings['frame']['width'], 3))

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            if settings[layer]['acquire'] == True:

                # Go to focus and wait for arrival
                lumaview.ids['mainsettings_id'].ids[layer].goto_focus()
                while not lumaview.motion.target_status('Z'):
                    time.sleep(.05)

                # set the gain and exposure
                gain = settings[layer]['gain']
                lumaview.camera.gain(gain)
                error_log(lumaview.camera.mssg)
                exposure = settings[layer]['exp']
                lumaview.camera.exposure_t(exposure)
                error_log(lumaview.camera.mssg)

                # update illumination to currently selected settings
                illumination = settings[layer]['ill']

                # Dark field capture
                lumaview.led_board.leds_off()
                error_log(lumaview.led_board.mssg)
                time.sleep(exposure/1000)  # Should be replaced with Clock
                scope_display.update()
                darkfield = lumaview.camera.array

                # Florescent capture
                lumaview.led_board.led_on(lumaview.led_board.color2ch(layer), illumination)
                error_log(lumaview.led_board.mssg)
                lumaview.camera.grab()
                error_log(lumaview.camera.mssg)
                scope_display.update()
                corrected = lumaview.camera.array - np.minimum(lumaview.camera.array,darkfield)
                # buffer the images
                if layer == 'Blue':
                    img[:,:,0] = corrected
                elif layer == 'Green':
                    img[:,:,1] = corrected
                elif layer == 'Red':
                    img[:,:,2] = corrected
                # # if Brightfield is included
                # else:
                #     a = 0.3
                #     img[:,:,0] = img[:,:,0]*a + corrected*(1-a)
                #     img[:,:,1] = img[:,:,1]*a + corrected*(1-a)
                #     img[:,:,2] = img[:,:,2]*a + corrected*(1-a)

            lumaview.led_board.leds_off()
            error_log(lumaview.led_board.mssg)

        lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
        lumaview.ids['composite_btn'].state = 'normal'

        img = np.flip(img, 0)

        save_folder = settings['live_folder']
        file_root = 'composite_'
        append = str(int(round(time.time() * 1000)))
        filename =  file_root + append + '.tiff'
        cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))


# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(CompositeCapture): # i.e. global lumaview
    led_board = ObjectProperty(None)
    led_board = LEDBoard()
    motion = ObjectProperty(None)
    motion = TrinamicBoard()
    camera = ObjectProperty(None)
    camera = PylonCamera()

    def cam_toggle(self):
        error_log('MainDisplay.cam_toggle()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return

        if scope_display.play == True:
            scope_display.play = False
            self.led_board.leds_off()
            error_log(self.led_board.mssg)
            scope_display.stop()
        else:
            scope_display.play = True
            scope_display.start()

    def record(self):
        error_log('MainDisplay.record()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return
        scope_display.record = not scope_display.record

    def fit_image(self):
        error_log('MainDisplay.fit_image()')
        if self.camera.active == False:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        error_log('MainDisplay.one2one_image()')
        if self.camera.active == False:
            return
        w = self.width
        h = self.height
        scale_hor = float(lumaview.camera.active.Width.GetValue()) / float(w)
        scale_ver = float(lumaview.camera.active.Height.GetValue()) / float(h)
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

# global gain_vals
# gain_vals = (1., )*4

class ShaderViewer(Scatter):
    black = ObjectProperty(0.)
    white = ObjectProperty(1.)

    fs = StringProperty('''
void main (void) {
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
    //gl_FragColor = pow(glFragColor.rgb, 1/gamma)
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
        super(ShaderViewer, self).__init__(**kwargs)
        error_log('ShaderViewer.__init__()')
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        self.white = 1.
        self.black = 0.

    def on_touch_down(self, touch):
        error_log('ShaderViewer.on_touch_down()')
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

    def update_shader(self, false_color='BF'):
        # error_log('ShaderViewer.update_shader()')

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (self.black, )*4
        c['gamma'] = 2.2

        if false_color == 'Red':
            c['white_point'] = (self.white, 0., 0., 1.)
        elif false_color == 'Green':
            c['white_point'] = (0., self.white, 0., 1.)
        elif false_color == 'Blue':
            c['white_point'] = (0., 0., self.white, 1.)
        else:
            c['white_point'] = (self.white, )*4

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)


class MotionSettings(BoxLayout):
    settings_width = dp(300)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        error_log('MotionSettings.__init__()')

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        error_log('MotionSettings.toggle_settings()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        self.ids['verticalcontrol_id'].update_gui()
        self.ids['xy_stagecontrol_id'].update_gui()

        # move position of motion control
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

        if scope_display.play == True:
            scope_display.start()

    def check_settings(self, *args):
        error_log('MotionSettings.check_settings()')
        # global lumaview
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

class PostProcessing(BoxLayout):

    def convert_to_avi(self):

        error_log('PostProcessing.convert_to_avi()')

        # # self.choose_folder()
        # save_location = './capture/movie.avi'

        # img_array = []
        # for filename in glob.glob('./capture/*.tiff'):
        #     img = cv2.imread(filename)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)

        # if len(img_array) > 0:
        #     out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

    def stitch(self):
        error_log('PostProcessing.stitch()')



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
        error_log('ShaderEditor.__init__()')
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        error_log('ShaderEditor.compile_shaders()')
        if not self.viewer:
            error_log('ShaderEditor.compile_shaders() Fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        self.viewer.fs = fs
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        error_log('ShaderEditor.toggle_editor()')
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

class MainSettings(BoxLayout):
    settings_width = dp(300)
    notCollapsing = BooleanProperty(True)
    currentLayer = StringProperty('microscope')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        error_log('MainSettings.toggle_editor()')

    # Hide (and unhide) main settings
    def toggle_settings(self):
        error_log('MainSettings.toggle_settings()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()

        # move position of settings and stop histogram if collapsed
        if self.ids['toggle_mainsettings'].state == 'normal':
            self.pos = lumaview.width - 30, 0
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                Clock.unschedule(lumaview.ids['mainsettings_id'].ids[layer].ids['histo_id'].histogram)
        else:
            self.pos = lumaview.width - self.settings_width, 0
 
        if scope_display.play == True:
            scope_display.start()

    def accordion_collapse(self, layer):
        error_log('MainSettings.accordion_collapse()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        lumaview.led_board.leds_off()
        error_log(lumaview.led_board.mssg)
        # turn off all toglles
        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        self.currentLayer = layer
 
        self.notCollapsing = not(self.notCollapsing)
        if scope_display.play == True:
            scope_display.start()

        if self.notCollapsing:
            layers = ['BF', 'Blue', 'Green', 'Red']
            if self.currentLayer in layers:
                error_log('Clock.schedule_interval(...histogram, 0.1)')
                Clock.schedule_interval(lumaview.ids['mainsettings_id'].ids[layer].ids['histo_id'].histogram, 0.1)
 
    def check_settings(self, *args):
        error_log('MainSettings.check_settings()')
        global lumaview
        if self.ids['toggle_mainsettings'].state == 'normal':
            self.pos = lumaview.width - 30, 0
        else:
            self.pos = lumaview.width - self.settings_width, 0

class Histogram(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        error_log('Histogram.__init__()')
        if self.bg_color is None:
            self.bg_color = (1, 1, 1, 1)

        self.hist_range_set = False
        self.edges = [0,255]
        self.stablize = 0.3

    def histogram(self, *args):
        # error_log('Histogram.histogram()')
        global lumaview

        if lumaview.camera != False:
            image = lumaview.camera.array
            hist = np.histogram(image, bins=256,range=(0,256))
            if self.hist_range_set:
                edges = self.edges
            else:
                edges = np.histogram_bin_edges(image, bins=1)
                edges[0] = self.stablize*self.edges[0] + (1 - self.stablize)*edges[0]
                edges[1] = self.stablize*self.edges[1] + (1 - self.stablize)*edges[1]

            # mean = np.mean(hist[1],hist[0])
            lumaview.ids['viewer_id'].black = 0.0 # float(edges[0])/255.
            lumaview.ids['viewer_id'].white = 1.0 # float(edges[1])/255.

            # UPDATE SHADER

            self.canvas.clear()
            r, b, g, a = self.bg_color
            self.hist = hist
            self.edges = edges
            with self.canvas:
                x = self.x
                y = self.y
                w = self.width
                h = self.height
                Color(r, b, g, a/12)
                Rectangle(pos=(x, y), size=(256, h))
                Color(r, b, g, a/4)
                Rectangle(pos=(x + edges[0], y), size=(edges[1]-edges[0], h))
                Color(r, b, g, a/2)
                #self.color = Color(rgba=self.color)
                logHistogram = lumaview.ids['mainsettings_id'].ids[self.layer].ids['logHistogram_id'].active
                if logHistogram:
                    maxheight = np.log(np.max(hist[0])+1)
                else:
                    maxheight = np.max(hist[0])
                if maxheight > 0:
                    scale=h/maxheight
                    for i in range(len(hist[0])):
                        if logHistogram:
                            counts = scale*np.log(hist[0][i] + 1)
                        else:
                            counts = np.ceil(scale*hist[0][i])
                        self.pos = self.pos
                        self.line = Line(points=(x+i, y, x+i, y+counts), width=1)


class VerticalControl(BoxLayout):

    def update_gui(self):
        error_log('VerticalControl.update_gui()')
        set_pos = lumaview.motion.target_pos('Z')  # Get target value
        error_log(lumaview.motion.mssg)

        self.ids['obj_position'].value = max(0, set_pos)
        self.ids['z_position_id'].text = format(max(0, set_pos), '.2f')

    def course_up(self):
        error_log('VerticalControl.course_up()')
        course = settings['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', course)                  # Move UP
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def fine_up(self):
        error_log('VerticalControl.fine_up()')
        fine = settings['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', fine)                    # Move UP
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def fine_down(self):
        error_log('VerticalControl.fine_down()')
        fine = settings['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', -fine)                   # Move DOWN
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def course_down(self):
        error_log('VerticalControl.course_down()')
        course = settings['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', -course)                 # Move DOWN
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def set_position(self, pos):
        error_log('VerticalControl.set_position()')
        lumaview.motion.move_abs_pos('Z', float(pos))
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def set_bookmark(self):
        error_log('VerticalControl.set_bookmark()')
        height = lumaview.motion.current_pos('Z')  # Get current z height in um
        error_log(lumaview.motion.mssg)
        settings['bookmark']['z'] = height

    def set_all_bookmarks(self):
        error_log('VerticalControl.set_all_bookmarks()')
        height = lumaview.motion.current_pos('Z')  # Get current z height in um
        error_log(lumaview.motion.mssg)
        settings['bookmark']['z'] = height
        settings['BF']['focus'] = height
        settings['Blue']['focus'] = height
        settings['Green']['focus'] = height
        settings['Red']['focus'] = height

    def goto_bookmark(self):
        error_log('VerticalControl.goto_bookmark()')
        pos = settings['bookmark']['z']
        lumaview.motion.move_abs_pos('Z', pos)
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def home(self):
        error_log('VerticalControl.home()')
        lumaview.motion.zhome()
        error_log(lumaview.motion.mssg)
        self.update_gui()

    # User selected the autofocus function
    def autofocus(self):
        error_log('VerticalControl.autofocus()')
        
        global lumaview
        if lumaview.camera.active == False:
            error_log('Error: VerticalControl.autofocus()')
            return

        # TODO Needs to be set by the user
        center = settings['bookmark']['z']
        range =  settings['objective']['AF_range']
        fine =   settings['objective']['AF_min']
        course = settings['objective']['AF_max']

        self.z_min = max(0, center-range)
        self.z_max = center+range
        self.z_step = course

        # TODO change this based on focus and exposure time
        dt = 0.5

        self.positions = []
        self.focus_measures = []

        if self.ids['autofocus_id'].state == 'down':
            self.ids['autofocus_id'].text = 'Focusing...'
            lumaview.motion.move_abs_pos('Z', self.z_min) # Go to z_min
            error_log(lumaview.motion.mssg)

            error_log('Clock.schedule_interval(self.focus_iterate, dt)')
            Clock.schedule_interval(self.focus_iterate, dt)

    def focus_iterate(self, dt):
        error_log('VerticalControl.focus_iterate()')
        global lumaview
        image = lumaview.camera.array

        target = lumaview.motion.current_pos('Z') # Get current value
        error_log(lumaview.motion.mssg)

        self.positions.append(target)
        self.focus_measures.append(self.focus_function(image))

        fine =   settings['objective']['AF_min']
        course = settings['objective']['AF_max']
        #closeness = 1/(len(self.positions) + 1
        n = len(self.positions)
        closeness = 1/(n + 0.1)

        step = course*closeness + fine*(1 - closeness)
        error_log('fine: ' + str(fine) + '; course: ' + str(course) + '; step' + str(step))

        lumaview.motion.move_rel_pos('Z', step) # move by z_step
        error_log(lumaview.motion.mssg)

        if self.ids['autofocus_id'].state == 'normal':
            self.ids['autofocus_id'].text = 'Autofocus'
            error_log('Clock.unschedule(self.focus_iterate)')
            Clock.unschedule(self.focus_iterate)

        elif target >= self.z_max:
            self.ids['autofocus_id'].state = 'normal'
            self.ids['autofocus_id'].text = 'Autofocus'
            error_log('Clock.unschedule(self.focus_iterate)')
            Clock.unschedule(self.focus_iterate)

            focus = self.focus_best(self.positions, self.focus_measures)
            lumaview.motion.move_abs_pos('Z', focus) # move to absolute target
            error_log(lumaview.motion.mssg)

        self.update_gui()

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        error_log('VerticalControl.focus_function()')
        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            error_log('difference\t' + str(sum_one - sum_two))
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            error_log('skew\t' + str(skew))
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            error_log('pixel_variation\t' + str(var))
            return var

        elif algorithm == 'convolve2D':
            # Bueno-Ibarra et al. Optical Engineering 44(6), 063601 (June 2005)
            kernel = np.array([ [0, -1, 0],
                                [-1, 4,-1],
                                [0, -1, 0]], dtype='float') / 6
            n = 9
            a = 1
            kernel = np.zeros([n,n])
            for i in range(n):
                for j in range(n):
                    r2 = ((i-(n-1)/2)**2 + (j-(n-1)/2)**2)/a**2
                    kernel[i,j] = 2*(1-r2)*np.exp(-0.5*r2)/np.sqrt(3*a)
            error_log('kernel\t' + str(kernel))
            convolve = signal.convolve2d(image, kernel, mode='valid')
            sum = np.sum(convolve)
            error_log('sum\t' + str(sum))
            return sum

        else:
            return 0

    def focus_best(self, positions, values, algorithm='mov_avg'):
        error_log('VerticalControl.focus_best()')
        if algorithm == 'direct':
            max_value = max(values)
            max_index = values.index(max_value)
            return positions[max_index]

        elif algorithm == 'mov_avg':
            avg_values = np.convolve(values, [.5, 1, 0.5], 'same')
            max_index = avg_values.argmax()
            return positions[max_index]

        else:
            return positions[0]

    # Image Sharpening using Richardson-Lucy deconvolution algorithm
    def Richardson_Lucy(self, image):
        # https://scikit-image.org/docs/dev/auto_examples/filters/
        # plot_deconvolution.html#sphx-glr-download-auto-examples-filters-plot-deconvolution-py
        pass



class XYStageControl(BoxLayout):

    def update_gui(self):
        error_log('XYStageControl.update_gui()')
        global lumaview
        x_target = lumaview.motion.target_pos('X')  # Get target value
        error_log(lumaview.motion.mssg)
        y_target = lumaview.motion.target_pos('Y')  # Get target value
        error_log(lumaview.motion.mssg)

        self.ids['x_pos_id'].text = format(max(0, x_target)/1000, '.2f')
        self.ids['y_pos_id'].text = format(max(0, y_target)/1000, '.2f')

    def fine_left(self):
        error_log('XYStageControl.fine_left()')
        global lumaview
        lumaview.motion.move_rel_pos('X', -10)  # Move LEFT
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def fine_right(self):
        error_log('XYStageControl.fine_right()')
        global lumaview
        lumaview.motion.move_rel_pos('X', 10)  # Move RIGHT
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def course_left(self):
        error_log('XYStageControl.course_left()')
        global lumaview
        lumaview.motion.move_rel_pos('X', -100)  # Move LEFT relative
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def course_right(self):
        error_log('XYStageControl.course_right()')
        global lumaview
        lumaview.motion.move_rel_pos('X', 100)  # Move RIGHT
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def fine_back(self):
        error_log('XYStageControl.fine_back()')
        global lumaview
        lumaview.motion.move_rel_pos('Y', -10)  # Move BACK by 1000
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def fine_fwd(self):
        error_log('XYStageControl.fine_fwd()')
        global lumaview
        lumaview.motion.move_rel_pos('Y', 10)  # Move FORWARD by 1000
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def course_back(self):
        error_log('XYStageControl.course_back()')
        global lumaview
        lumaview.motion.move_rel_pos('Y', -100)  # Move BACK relative by 10000
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def course_fwd(self):
        error_log('XYStageControl.course_fwd()')
        global lumaview
        lumaview.motion.move_rel_pos('Y', 100)  # Move FORWARD by 10000
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def set_xposition(self, pos):
        error_log('XYStageControl.set_xposition()')
        global lumaview
        lumaview.motion.move_abs_pos('X', float(pos)*1000)  # position in text is in mm
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def set_yposition(self, pos):
        error_log('XYStageControl.set_yposition()')
        global lumaview
        lumaview.motion.move_abs_pos('Y', float(pos)*1000)  # position in text is in mm
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def set_xbookmark(self):
        error_log('XYStageControl.set_xbookmark()')
        global lumaview
        x_pos = lumaview.motion.current_pos('X')  # Get current x position in um
        error_log(lumaview.motion.mssg)
        settings['bookmark']['x'] = x_pos

    def set_ybookmark(self):
        error_log('XYStageControl.set_ybookmark()')
        global lumaview
        y_pos = lumaview.motion.current_pos('Y')  # Get current x position in um
        error_log(lumaview.motion.mssg)
        settings['bookmark']['y'] = y_pos

    def goto_xbookmark(self):
        error_log('XYStageControl.goto_xbookmark()')
        global lumaview
        x_pos = settings['bookmark']['x']
        lumaview.motion.move_abs_pos('X', x_pos)  # set current x position in um
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def goto_ybookmark(self):
        error_log('XYStageControl.goto_ybookmark()')
        global lumaview
        y_pos = settings['bookmark']['y']
        lumaview.motion.move_abs_pos('Y', y_pos)  # set current y position in um
        error_log(lumaview.motion.mssg)
        self.update_gui()

    def home(self):
        error_log('XYStageControl.home()')
        global lumaview

        self.go_home = threading.Thread(target=lumaview.motion.xyhome) # threaded to prevent freezing
        self.go_home.start()
        self.ids['x_pos_id'].text = '0.00'
        self.ids['y_pos_id'].text = '0.00'

# Protocol settings tab
class ProtocolSettings(CompositeCapture):
    global settings

    def __init__(self, **kwargs):

        super(ProtocolSettings, self).__init__(**kwargs)
        error_log('ProtocolSettings.__init__()')
 
        # Load all Possible Labware from JSON
        os.chdir(home_wd)
        with open('./data/labware.json', "r") as read_file:
            self.labware = json.load(read_file)
        
        try:
            self.load_protocol(file = settings['protocol']['location'])
        except:
            self.step_names = list()
            self.step_values = []

        self.c_step = 0

    # Update Protocol Period   
    def update_period(self):
        error_log('ProtocolSettings.update_period()')
        try:
            settings['protocol']['period'] = float(self.ids['capture_period'].text)
        except:
            error_log('Update Period is not an acceptable value')

    # Update Protocol Duration   
    def update_duration(self):
        error_log('ProtocolSettings.update_duration()')
        try:
            settings['protocol']['duration'] = float(self.ids['capture_dur'].text)
        except:
            error_log('Update Duration is not an acceptable value')

    # Labware Selection
    def select_labware(self):
        error_log('ProtocolSettings.select_labware()')
        spinner = self.ids['labware_spinner']
        spinner.values = list(self.labware['Wellplate'].keys())
        settings['protocol']['labware'] = spinner.text
    
        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()

 
    # Create New Protocol
    def new_protocol(self):
        error_log('ProtocolSettings.new_protocol()')

        os.chdir(home_wd)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])
        current_labware.set_positions()
        self.step_values = np.array([])
 
         # Iterate through all the positions in the scan
        for pos in current_labware.pos_list:

            # Iterate through all the colors to create the steps
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                if settings[layer]['acquire'] == True:

                    x = pos[0]
                    y = pos[1]
                    z = settings[layer]['focus']
                    ch = lumaview.led_board.color2ch(layer)
                    ill = settings[layer]['ill']
                    gain = settings[layer]['gain']
                    gain_auto = int(settings[layer]['gain_auto'])
                    exp = settings[layer]['exp']

                    step = np.array([[x, y, z, ch, ill, gain, gain_auto, exp]])
 
                    if self.step_values.size == 0:
                        self.step_values = step
                    else:
                        self.step_values = np.append(self.step_values, step, axis=0)

        # Number of Steps

        length =  self.step_values.shape[0]
        # length = 1
              
        # Update text with current step and number of steps in protocol
        self.c_step = -1
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.ids['step_total_input'].text = str(length)

        self.step_names = [self.ids['step_name_input'].text] * length

        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()

    # Load Protocol from File
    def load_protocol(self, file="./data/sample_protocol.csv"):
        error_log('ProtocolSettings.load_protocol()')

        # Open protocol as DataFrame
        protocol_df = pd.read_csv(file)
        # Change columns to List and NPArray
        self.step_names = list(protocol_df['Name'])
        self.step_values = protocol_df[['X', 'Y', 'Z', 'Channel', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure']].to_numpy()

        self.c_step = -1
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.ids['step_name_input'].text = ''

        # Update total number of steps to GUI
        self.ids['step_total_input'].text = str(len(self.step_names))

        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()

    # Save Protocol to File
    def save_protocol(self, file="./data/sample_protocol.csv"):
        error_log('ProtocolSettings.save_protocol()')

        # Create Empty DataFrame
        protocol_df = pd.DataFrame(columns=['Name', 'X', 'Y', 'Z', 'Channel', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure'])
        # Change columns to List and NPArray
        protocol_df['Name'] = self.step_names
        protocol_df[['X', 'Y', 'Z', 'Channel', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure']] = self.step_values

        protocol_df.to_csv(file, index=False)

    # Goto to Previous Step
    def prev_step(self):
        error_log('ProtocolSettings.prev_step()')
        if len(self.step_names) <= 0:
            return
        self.c_step = max(self.c_step - 1, 0)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
 
    # Go to Next Step
    def next_step(self):
        error_log('ProtocolSettings.next_step()')
        if len(self.step_names) <= 0:
            return
        self.c_step = min(self.c_step + 1, len(self.step_names)-1)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
    
    # Go to Input Step
    def go_to_step(self):
        error_log('ProtocolSettings.go_to_step()')

        if len(self.step_names) <= 0:
            self.ids['step_number_input'].text = '0'
            return
        
        # Get the Current Step Number
        self.c_step = int(self.ids['step_number_input'].text)-1

        if self.c_step < 0 or self.c_step > len(self.step_names):
            self.ids['step_number_input'].text = '0'
            return

        # Extract Values from protocol list and array
        name =      str(self.step_names[self.c_step])
        x =         self.step_values[self.c_step, 0]
        y =         self.step_values[self.c_step, 1]
        z =         self.step_values[self.c_step, 2]
        ch =        self.step_values[self.c_step, 3]
        ill =       self.step_values[self.c_step, 4]
        gain =      self.step_values[self.c_step, 5]
        auto_gain = self.step_values[self.c_step, 6]
        exp =       self.step_values[self.c_step, 7]

        self.ids['step_name_input'].text = name
        lumaview.motion.move_abs_pos('X', x*1000)
        error_log(lumaview.motion.mssg)
        lumaview.motion.move_abs_pos('Y', y*1000)
        error_log(lumaview.motion.mssg)
        lumaview.motion.move_abs_pos('Z', z)
        error_log(lumaview.motion.mssg)

        ch = lumaview.led_board.ch2color(ch)
        layer  = lumaview.ids['mainsettings_id'].ids[ch]

        # TODO: open accordian to correct channel or display channel in some way

        # set illumination settings, text, and slider
        error_log('ill:     ' + str(ill))
        settings[ch]['ill'] = ill
        layer.ids['ill_text'].text = str(ill)
        layer.ids['ill_slider'].value = float(ill)

        # set gain settings, text, and slider
        error_log('gain:     ' + str(gain))
        settings[ch]['gain'] = gain
        layer.ids['gain_text'].text = str(gain)
        layer.ids['gain_slider'].value = float(gain)

        # set exposure settings, text, and slider
        error_log('exp:       ' + str(exp))
        settings[ch]['exp'] = exp
        layer.ids['exp_text'].text = str(exp)
        layer.ids['exp_slider'].value = float(exp)

        # set auto-gain checkbox
        error_log('auto_gain: ' + str(auto_gain))
        settings[ch]['gain_auto'] = bool(auto_gain)
        layer.ids['gain_auto'].active = bool(auto_gain)

        for i in range(10):
            Clock.schedule_once(self.ids['stage_widget_id'].draw_labware, i/2)

    # Delete Current Step of Protocol
    def delete_step(self):
        error_log('ProtocolSettings.delete_step()')
        
        self.step_names.pop(self.c_step)
        self.step_values = np.delete(self.step_values, self.c_step, axis = 0)
        self.c_step = self.c_step - 1

        # Update total number of steps to GUI
        self.ids['step_total_input'].text = str(len(self.step_names))
        self.next_step()

    # Modify Current Step of Protocol
    def modify_step(self):
        error_log('ProtocolSettings.modify_step()')

        self.step_names[self.c_step] = self.ids['step_name_input'].text
        self.step_values[self.c_step, 0] = lumaview.motion.current_pos('X')/1000 # x
        error_log(lumaview.motion.mssg)
        self.step_values[self.c_step, 1] = lumaview.motion.current_pos('Y')/1000 # y
        error_log(lumaview.motion.mssg)
        self.step_values[self.c_step, 2] = lumaview.motion.current_pos('Z')      # z
        error_log(lumaview.motion.mssg)

        c_layer = False

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            accordion = layer + '_accordion'
            if lumaview.ids['mainsettings_id'].ids[accordion].collapse == False:
                c_layer = layer

        if c_layer == False:
            mssg = "No layer currently selected"
            return mssg

        ch = lumaview.led_board.color2ch(c_layer)

        layer_id = lumaview.ids['mainsettings_id'].ids[c_layer]
        self.step_values[self.c_step, 3] = ch # ch
        self.step_values[self.c_step, 4] = layer_id.ids['ill_slider'].value # ill
        self.step_values[self.c_step, 5] = layer_id.ids['gain_slider'].value # gain
        self.step_values[self.c_step, 6] = int(layer_id.ids['gain_auto'].active) # auto_gain
        self.step_values[self.c_step, 7] = layer_id.ids['exp_slider'].value # exp

    # Insert Current Step to Protocol at Current Position
    def add_step(self):
        error_log('ProtocolSettings.add_step()')

         # Determine Values
        name = self.ids['step_name_input'].text
        c_layer = False

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            accordion = layer + '_accordion'
            if lumaview.ids['mainsettings_id'].ids[accordion].collapse == False:
                c_layer = layer

        if c_layer == False:
            mssg = "No layer currently selected"
            return mssg

        ch = lumaview.led_board.color2ch(c_layer)
        layer_id = lumaview.ids['mainsettings_id'].ids[c_layer]

        step = [lumaview.motion.current_pos('X')/1000, # x
                lumaview.motion.current_pos('Y')/1000, # y
                lumaview.motion.current_pos('Z'),      # z
                ch, # ch 
                layer_id.ids['ill_slider'].value, # ill
                layer_id.ids['gain_slider'].value, # gain
                int(layer_id.ids['gain_auto'].active), # auto_gain
                layer_id.ids['exp_slider'].value, # exp
        ]

        # Insert into List and Array
        self.step_names.insert(self.c_step, name)
        self.step_values = np.insert(self.step_values, self.c_step, step, axis=0)

        self.ids['step_total_input'].text = str(len(self.step_names))



















        

    def run_autofocus(self):
        error_log('ProtocolSettings.run_autofocus()')

        if len(self.step_names) < 1:
            error_log('Protocol has no steps.')
            self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
            return

        if self.ids['run_autofocus_btn'].state == 'down':
            self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

            self.c_step = 0
            self.ids['step_number_input'].text = str(self.c_step+1)

            x = self.step_values[self.c_step, 0]
            y = self.step_values[self.c_step, 1]
            z =  self.step_values[self.c_step, 2]
 
            lumaview.motion.move_abs_pos('X', x*1000)
            error_log(lumaview.motion.mssg)
            lumaview.motion.move_abs_pos('Y', y*1000)
            error_log(lumaview.motion.mssg)
            lumaview.motion.move_abs_pos('Z', z)
            error_log(lumaview.motion.mssg)

            error_log('Clock.schedule_interval(self.autofocus_iterate, 0.1)')
            Clock.schedule_interval(self.autofocus_iterate, 0.1)

        else:  # self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
            error_log('Clock.unschedule(self.autofocus_iterate)')
            Clock.unschedule(self.autofocus_iterate) # unschedule all copies of scan iterate
        
    def autofocus_iterate(self, dt):
        global lumaview
        global settings

        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()
        self.ids['step_number_input'].text = str(self.c_step+1)

        # Check if at desired position
        x_status = lumaview.motion.target_status('X')
        y_status = lumaview.motion.target_status('Y')
        z_status = lumaview.motion.target_status('Z')

        # If target location has been reached
        if x_status and y_status and z_status:
            error_log('Scan Step:' + str(self.step_names[self.c_step]) )

            # identify image settings
            ch =        self.step_values[self.c_step, 3]
            ill =       self.step_values[self.c_step, 4]
            gain =      self.step_values[self.c_step, 5]
            auto_gain = self.step_values[self.c_step, 6]
            exp =       self.step_values[self.c_step, 7]


            # increment to the next step
            self.c_step += 1

            if self.c_step < len(self.step_names):
                x = self.step_values[self.c_step, 0]
                y = self.step_values[self.c_step, 1]
                z =  self.step_values[self.c_step, 2]

                lumaview.motion.move_abs_pos('X', x*1000)  # move to x
                lumaview.motion.move_abs_pos('Y', y*1000)  # move to y
                lumaview.motion.move_abs_pos('Z', z)       # move to z

            # if all positions have already been reached
            else:
                error_log('Scan Complete')
                self.ids['run_autofocus_btn'].state = 'normal'
                self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
                error_log('Clock.unschedule(self.autofocus_iterate)')
                Clock.unschedule(self.autofocus_iterate) # unschedule all copies of scan iterate
























    # Run one scan of the protocol
    def run_scan(self, protocol = False):
        error_log('ProtocolSettings.run_scan()')
 
        if len(self.step_names) < 1:
            error_log('Protocol has no steps.')
            self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'
            return

        if self.ids['run_scan_btn'].state == 'down' or protocol == True:
            self.ids['run_scan_btn'].text = 'Running Scan'

            self.c_step = 0
            self.ids['step_number_input'].text = str(self.c_step+1)

            x = self.step_values[self.c_step, 0]
            y = self.step_values[self.c_step, 1]
            z =  self.step_values[self.c_step, 2]
 
            lumaview.motion.move_abs_pos('X', x*1000)
            error_log(lumaview.motion.mssg)
            lumaview.motion.move_abs_pos('Y', y*1000)
            error_log(lumaview.motion.mssg)
            lumaview.motion.move_abs_pos('Z', z)
            error_log(lumaview.motion.mssg)

            error_log('Clock.schedule_interval(self.scan_iterate, 0.1)')
            Clock.schedule_interval(self.scan_iterate, 0.1)

        else:  # self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'
            error_log('Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
    
    def scan_iterate(self, dt):
        global lumaview
        global settings

        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()
        self.ids['step_number_input'].text = str(self.c_step+1)

        # Check if at desired position
        x_status = lumaview.motion.target_status('X')
        y_status = lumaview.motion.target_status('Y')
        z_status = lumaview.motion.target_status('Z')

        # If target location has been reached
        if x_status and y_status and z_status:
            error_log('Scan Step:' + str(self.step_names[self.c_step]) )

            # identify image settings
            ch =        self.step_values[self.c_step, 3]
            ill =       self.step_values[self.c_step, 4]
            gain =      self.step_values[self.c_step, 5]
            auto_gain = self.step_values[self.c_step, 6]
            exp =       self.step_values[self.c_step, 7]

            # capture image
            self.custom_capture(ch, ill, gain, exp)

            # increment to the next step
            self.c_step += 1

            if self.c_step < len(self.step_names):
                x = self.step_values[self.c_step, 0]
                y = self.step_values[self.c_step, 1]
                z =  self.step_values[self.c_step, 2]

                lumaview.motion.move_abs_pos('X', x*1000)  # move to x
                lumaview.motion.move_abs_pos('Y', y*1000)  # move to y
                lumaview.motion.move_abs_pos('Z', z)       # move to z

            # if all positions have already been reached
            else:
                error_log('Scan Complete')
                self.ids['run_scan_btn'].state = 'normal'
                self.ids['run_scan_btn'].text = 'Run One Scan'
                error_log('Clock.unschedule(self.scan_iterate)')
                Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate

    # Run protocol without xy movement
    def run_stationary(self):
        error_log('ProtocolSettings.run_stationary()')

        if self.ids['run_stationary_btn'].state == 'down':
            self.ids['run_stationary_btn'].text = 'State == Down'
        else:
            self.ids['run_stationary_btn'].text = 'Run Stationary Protocol' # 'normal'


    # Run the complete protocol 
    def run_protocol(self):
        error_log('ProtocolSettings.run_protocol()')
        self.n_scans = int(float(settings['protocol']['duration'])*60 / float(settings['protocol']['period']))
        self.start_t = time.time() # start of cycle in seconds

        if self.ids['run_protocol_btn'].state == 'down':
            error_log('Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.run_scan(protocol = True)
            error_log('Clock.schedule_interval(self.protocol_iterate, 1)')
            Clock.schedule_interval(self.protocol_iterate, 1)

        else:
            self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'
            error_log('Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            error_log('Clock.unschedule(self.protocol_iterate)')
            Clock.unschedule(self.protocol_iterate) # unschedule all copies of scan iterate
            # self.protocol_event.cancel()
 
    def protocol_iterate(self, dt):
        error_log('ProtocolSettings.protocol_iterate()')

        # Simplified variables
        start_t = self.start_t # start of cycle in seconds
        curr_t = time.time()   # current time in seconds
        n_scans = self.n_scans # current number of scans left
        period = settings['protocol']['period']*60 # length of cycle in seconds

        # compute time remaining
        sec_remaining = n_scans*period - (curr_t - start_t)
        min_remaining = sec_remaining / 60
        hrs_remaining = min_remaining / 60

        hrs = np.floor(hrs_remaining)
        minutes = np.floor((hrs_remaining - hrs)*60)

        hrs = '%d' % hrs
        minutes = '%02d' % minutes

        # Update Button
        # self.ids['run_protocol_btn'].text = hrs+':'+minutes+' remaining'
        self.ids['run_protocol_btn'].text = str(n_scans)

        # Check if reached next Period
        if (time.time()-self.start_t) > period:

            # reset the start time and update number of scane remaining
            self.start_t = time.time()
            self.n_scans = self.n_scans - 1

            if self.n_scans > 0:
                error_log('Scans Remaining: ' + str(self.n_scans))
                self.run_scan(protocol = True)
            else:
               self.ids['run_protocol_btn'].state = 'normal' # 'normal'
               self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'
               error_log('Clock.unschedule(self.protocol_iterate)')
               Clock.unschedule(self.protocol_iterate) # unschedule all copies of scan iterate

# Widget for displaying Microscope Stage area, labware, and current position 
class Stage(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)
        error_log('Stage.__init__()')


    def on_touch_down(self, touch):
        error_log('Stage.on_touch_down()')
        if self.collide_point(*touch.pos):
            error_log('clicked on: ' + str(touch.pos))
        
    def draw_labware(self, *args):
        # error_log('Stage.draw_labware()')
        global lumaview
        global settings

        # Create current labware instance
        os.chdir(home_wd)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])

        self.canvas.clear()
        r, b, g, a = (0.5, 0.5, 0.5, 0.5)

        with self.canvas:
            w = 12 * math.floor(self.width/12)
            h = 12 * math.floor(self.height/12)
            x = self.x + (self.width - w)/2
            y = self.y + (self.height - h)/2
            Color(r, b, g, a)
            Rectangle(pos=(x, y), size=(w, h))


            d = w/current_labware.plate['columns']
            r = math.floor(d/2 - 0.5)

            for i in range(current_labware.plate['columns']):
                for j in range(current_labware.plate['rows']):
                    Line(circle=(x + d*i + r, y + d*j + r, r))

            # Green Circle
            x_target = lumaview.motion.target_pos('X')/1000
            y_target = lumaview.motion.target_pos('Y')/1000
            i, j = current_labware.get_well_index(x_target, y_target)
            Color(0., 1., 0., 1.)
            Line(circle=(x + d*i + r  , y + d*j + r  , r))

            # Red Crosshairs
            x_current = lumaview.motion.current_pos('X')/1000
            y_current= lumaview.motion.current_pos('Y')/1000
            i, j = current_labware.get_screen_position(x_current, y_current)
            Color(1., 0., 0., 1.)
            Line(points=(x + d*i      , y + d*j + d/2, x + d*i + d  , y + d*j + d/2), width = 1)
            Line(points=(x + d*i + d/2, y + d*j      , x + d*i + d/2, y + d*j + d  ), width = 1)


class MicroscopeSettings(BoxLayout):

    def __init__(self, **kwargs):
        super(MicroscopeSettings, self).__init__(**kwargs)
        error_log('MicroscopeSettings.__init__()')

        os.chdir(home_wd)
        with open('./data/scopes.json', "r") as read_file:
            self.scopes = json.load(read_file)

        os.chdir(home_wd)
        with open('./data/objectives.json', "r") as read_file:
            self.objectives = json.load(read_file)

    # load settings from JSON file
    def load_settings(self, file="./data/current.json"):
        error_log('MicroscopeSettings.load_settings()')
        global lumaview

        # load settings JSON file
        os.chdir(home_wd)
        with open(file, "r") as read_file:
            global settings
            settings = json.load(read_file)
            # update GUI values from JSON data:
            self.ids['scope_spinner'].text = settings['microscope']
            self.ids['objective_spinner'].text = settings['objective']['ID']
            self.ids['magnification_id'].text = str(settings['objective']['magnification'])
            self.ids['frame_width'].text = str(settings['frame']['width'])
            self.ids['frame_height'].text = str(settings['frame']['height'])

            protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
            protocol_settings.ids['capture_period'].text = str(settings['protocol']['period'])
            protocol_settings.ids['capture_dur'].text = str(settings['protocol']['duration'])
            protocol_settings.ids['labware_spinner'].text = settings['protocol']['labware']
            #protocol_settings.ids['stage_widget_id'].draw_labware()

            zstack_settings = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['zstack_id']
            zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
            zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])
            zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']


            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = settings[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = settings[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = settings[layer]['exp']
                # lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = float(np.log10(settings[layer]['exp']))
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = settings[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['false_color'].active = settings[layer]['false_color']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = settings[layer]['acquire']

            lumaview.camera.frame_size(settings['frame']['width'], settings['frame']['height'])
            error_log(lumaview.camera.mssg)

    # Save settings to JSON file
    def save_settings(self, file="./data/current.json"):
        error_log('MicroscopeSettings.save_settings()')
        global settings
        os.chdir(home_wd)
        with open(file, "w") as write_file:
            json.dump(settings, write_file, indent = 4)

    def load_scopes(self):
        error_log('MicroscopeSettings.load_scopes()')
        spinner = self.ids['scope_spinner']
        spinner.values = list(self.scopes.keys())

    def select_scope(self):
        error_log('MicroscopeSettings.select_scope()')
        global lumaview
        global settings

        spinner = self.ids['scope_spinner']
        settings['microscope'] = spinner.text

    def load_ojectives(self):
        error_log('MicroscopeSettings.load_ojectives()')
        spinner = self.ids['objective_spinner']
        spinner.values = list(self.objectives.keys())

    def select_objective(self):
        error_log('MicroscopeSettings.select_objective()')
        global lumaview
        global settings

        spinner = self.ids['objective_spinner']
        settings['objective'] = self.objectives[spinner.text]
        settings['objective']['ID'] = spinner.text
        microscope_settings_id = lumaview.ids['mainsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['magnification_id'].text = str(settings['objective']['magnification'])

    def frame_size(self):
        error_log('MicroscopeSettings.frame_size()')
        global lumaview
        global settings

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        width = int(min(int(w), lumaview.camera.active.Width.Max)/2)*2
        height = int(min(int(h), lumaview.camera.active.Height.Max)/2)*2

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame_width'].text = str(width)
        self.ids['frame_height'].text = str(height)

        lumaview.camera.frame_size(width, height)
        error_log(lumaview.camera.mssg)


# Modified Slider Class to enable on_release event
# ---------------------------------------------------------------------
class ModSlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(ModSlider, self).__init__(**kwargs)
        error_log('ModSlider.__init__()')

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super(ModSlider, self).on_touch_up(touch)
        # error_log('ModSlider.on_touch_up()')
        if touch.grab_current == self:
            self.dispatch('on_release')
            return True

# LayerControl Layout class
# ---------------------------------------------------------------------
class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    global settings

    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)
        error_log('LayerControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        error_log('LayerControl.ill_slider()')
        illumination = self.ids['ill_slider'].value
        settings[self.layer]['ill'] = illumination
        self.apply_settings()

    def ill_text(self):
        error_log('LayerControl.ill_text()')
        ill_min = self.ids['ill_slider'].min
        ill_max = self.ids['ill_slider'].max
        ill_val = float(self.ids['ill_text'].text)
        illumination = float(np.clip(ill_val, ill_min, ill_max))

        settings[self.layer]['ill'] = illumination
        self.ids['ill_slider'].value = illumination
        self.ids['ill_text'].text = str(illumination)

        self.apply_settings()

    def gain_auto(self):
        error_log('LayerControl.gain_auto()')
        if self.ids['gain_auto'].state == 'down':
            state = True
        else:
            state = False
        settings[self.layer]['gain_auto'] = state
        self.apply_settings()

    def gain_slider(self):
        error_log('LayerControl.gain_slider()')
        gain = self.ids['gain_slider'].value
        settings[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):
        error_log('LayerControl.gain_text()')
        gain_min = self.ids['gain_slider'].min
        gain_max = self.ids['gain_slider'].max
        gain_val = float(self.ids['gain_text'].text)
        gain = float(np.clip(gain_val, gain_min, gain_max))

        settings[self.layer]['gain'] = gain
        self.ids['gain_slider'].value = gain
        self.ids['gain_text'].text = str(gain)

        self.apply_settings()

    def exp_slider(self):
        error_log('LayerControl.exp_slider()')
        exposure = self.ids['exp_slider'].value
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp'] = exposure        # exposure in ms
        self.apply_settings()

    def exp_text(self):
        error_log('LayerControl.exp_text()')
        exp_min = self.ids['exp_slider'].min
        exp_max = self.ids['exp_slider'].max
        exp_val = float(self.ids['exp_text'].text)
        exposure = float(np.clip(exp_val, exp_min, exp_max))

        settings[self.layer]['exp'] = exposure
        self.ids['exp_slider'].value = exposure
        # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
        self.ids['exp_text'].text = str(exposure)

        self.apply_settings()

    def root_text(self):
        error_log('LayerControl.root_text()')
        settings[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        error_log('LayerControl.false_color()')
        settings[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        error_log('LayerControl.update_acquire()')
        settings[self.layer]['acquire'] = self.ids['acquire'].active

    def save_focus(self):
        error_log('LayerControl.save_focus()')
        global lumaview
        pos = lumaview.motion.current_pos('Z')
        error_log(lumaview.motion.mssg)
        settings[self.layer]['focus'] = pos

    def goto_focus(self):
        error_log('LayerControl.goto_focus()')
        global lumaview
        pos = settings[self.layer]['focus']
        lumaview.motion.move_abs_pos('Z', pos)  # set current z height in usteps
        error_log(lumaview.motion.mssg)
        control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
        control.update_gui()

    def apply_settings(self):
        error_log('LayerControl.apply_settings()')
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        illumination = settings[self.layer]['ill']
        if self.ids['apply_btn'].state == 'down': # if the button is down
            # In active channel,turn on LED
            lumaview.led_board.led_on(lumaview.led_board.color2ch(self.layer), illumination)
            error_log(lumaview.led_board.mssg)
            #  turn the state of remaining channels to 'normal' and text to 'OFF'
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                if(layer != self.layer):
                    lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        else: # if the button is 'normal' meaning not active
            # In active channel, and turn off LED
            lumaview.led_board.leds_off()
            error_log(lumaview.led_board.mssg)

        # update gain to currently selected settings
        # -----------------------------------------------------
        state = settings[self.layer]['gain_auto']
        lumaview.camera.auto_gain(state)
        error_log(lumaview.camera.mssg)


        if not(state):
            gain = settings[self.layer]['gain']
            lumaview.camera.gain(gain)
            error_log(lumaview.camera.mssg)

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = settings[self.layer]['exp']
        lumaview.camera.exposure_t(exposure)
        error_log(lumaview.camera.mssg)

        # choose correct active toggle button image based on color
        # -----------------------------------------------------
        if self.ids['apply_btn'].state == 'down':
            if(self.layer) == 'Red':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRR.png'
            elif(self.layer) == 'Green':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRG.png'
            elif(self.layer) == 'Blue':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRB.png'
        else:
            self.ids['apply_btn'].background_down = './data/icons/ToggleR.png'

        # Remove 'Colorize' option in brightfield control
        # -----------------------------------------------------
        if self.layer == 'BF':
            self.ids['false_color_label'].text = ''
            self.ids['false_color'].color = (0., )*4

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        for i in np.arange(0.1, 2, 0.1):
            Clock.schedule_once(self.update_shader, i)

    def update_shader(self, dt):
        # error_log('LayerControl.update_shader()')
        if self.ids['false_color'].active:
            lumaview.ids['viewer_id'].update_shader(self.layer)
        else:
            lumaview.ids['viewer_id'].update_shader('none')


class ZStack(CompositeCapture):
    def set_steps(self):
        error_log('ZStack.set_steps()')

        step_size = self.ids['zstack_stepsize_id'].text
        step_size = float(step_size)
        settings['zstack']['step_size'] = step_size

        range = self.ids['zstack_range_id'].text
        range = float(range)
        settings['zstack']['range'] = range

        if step_size != 0:
            n_steps = np.floor( range / step_size)
            self.ids['zstack_steps_id'].text = str(int(n_steps))
        else:
            self.ids['zstack_steps_id'].text = '0'

    def set_position(self):
        settings['zstack']['position'] = self.ids['zstack_spinner'].text

    def aquire_zstack(self):
        error_log('ZStack.aquire_zstack()')
        global lumaview

        step_size = self.ids['zstack_stepsize_id'].text
        step_size = float(step_size)

        range = self.ids['zstack_range_id'].text
        range = float(range)

        n_steps = self.ids['zstack_steps_id'].text
        n_steps = int(n_steps)

        if n_steps <= 0:
            return False

        spinner_values = self.ids['zstack_spinner'].values
        spinner_value = self.ids['zstack_spinner'].text

        # Get current position
        current_pos = lumaview.motion.current_pos('Z')
        error_log(lumaview.motion.mssg)

        # Set start position
        if spinner_value == spinner_values[0]:   # 'Current Position at Top'
            start_pos = current_pos - range
        elif spinner_value == spinner_values[1]: # 'Current Position at Center'
            start_pos = current_pos - range / 2
        elif spinner_value == spinner_values[2]: # 'Current Position at Bottom'
            start_pos = current_pos

        # Make array of positions
        self.positions = np.arange(n_steps)*step_size + start_pos

        # begin moving to the first position
        self.n_pos = 0
        lumaview.motion.move_abs_pos('Z', self.positions[self.n_pos])
        error_log(lumaview.motion.mssg)

        if self.ids['ztack_aqr_btn'].state == 'down':
            error_log('Clock.schedule_interval(self.zstack_iterate, 0.01)')
            Clock.schedule_interval(self.zstack_iterate, 0.01)
            self.ids['ztack_aqr_btn'].text = 'Acquiring ZStack'

        else:
            self.ids['ztack_aqr_btn'].text = 'Acquire'
            # self.zstack_event.cancel()
            error_log('Clock.unschedule(self.zstack_iterate)')
            Clock.unschedule(self.zstack_iterate)

    def zstack_iterate(self, dt):
        error_log('ZStack.zstack_iterate()')

        if lumaview.motion.target_status('Z'):
            error_log('Z at target')
            self.live_capture()
            self.n_pos += 1

            if self.n_pos < len(self.positions):
                lumaview.motion.move_abs_pos('Z', self.positions[self.n_pos])
            else:
                self.ids['ztack_aqr_btn'].text = 'Acquire'
                self.ids['ztack_aqr_btn'].state = 'normal'
                # self.zstack_event.cancel()
                error_log('Clock.unschedule(self.zstack_iterate)')
                Clock.unschedule(self.zstack_iterate)

# Button the triggers 'filechooser.open_file()' from plyer
class FileChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        error_log('FileChooseBTN.choose()')
        # Call plyer filechooser API to run a filechooser Activity.
        self.context = context
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        error_log('FileChooseBTN.handle_selection()')
        self.selection = selection

    def on_selection(self, *a, **k):
        error_log('FileChooseBTN.on_selection()')
        global lumaview
        
        if self.selection:
            if self.context == 'load_settings':
                lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings(self.selection[0])

            elif self.context == 'load_protocol':
                lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(file = self.selection[0])
        else:
            return

# Button the triggers 'filechooser.choose_dir()' from plyer
class FolderChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        error_log('FolderChooseBTN.choose()')
        self.context = context
        filechooser.choose_dir(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        error_log('FolderChooseBTN.handle_selection()')
        self.selection = selection

    def on_selection(self, *a, **k):
        error_log('FolderChooseBTN.on_selection()')
        if self.selection:
            path = self.selection[0]
        else:
            return

        if self.context == 'live_folder':
            settings['live_folder'] = path

        elif self.context == 'movie_folder':
            save_location = path + '/movie.avi'

            img_array = []
            for filename in glob.glob(path + '/*.tiff'):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

            out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

        else: # Channel Save Folder selections
            settings[self.context]['save_folder'] = path

# Button the triggers 'filechooser.save_file()' from plyer
class FileSaveBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        error_log('FileSaveBTN.choose()')
        self.context = context
        filechooser.save_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        error_log('FileSaveBTN.handle_selection()')
        self.selection = selection

    def on_selection(self, *a, **k):
        error_log('FileSaveBTN.on_selection()')
        global lumaview
        
        if self.context == 'save_settings':
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].save_settings(self.selection[0])
            error_log('Saving Settings to File:' + self.selection[0])

        elif self.context == 'save_protocol':
            lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].save_protocol(file = self.selection[0])
            error_log('Saving Protocol to File:' + self.selection[0])


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):

    def build(self):
        error_log('-----------------------------------------')
        error_log('Latest Code Change: 9/4/2022')
        error_log('Run Time: ' + time.strftime("%Y %m %d %H:%M:%S"))
        error_log('-----------------------------------------')

        error_log('LumaViewProApp.build()')
        self.icon = './data/icons/icon32x.png'
        Window.maximize()
        Window.bind(on_resize=self._on_resize)

        global lumaview
        lumaview = MainDisplay()
        try:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings("./data/current.json")
        except:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings("./data/settings.json")
            
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.leds_off()
        error_log(lumaview.led_board.mssg)
        lumaview.ids['motionsettings_id'].ids['xy_stagecontrol_id'].home()

        return lumaview

    def _on_resize(self, window, w, h):
        Clock.schedule_once(lumaview.ids['motionsettings_id'].check_settings, 0.1)
        Clock.schedule_once(lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].ids['stage_widget_id'].draw_labware, 0.1)
        Clock.schedule_once(lumaview.ids['mainsettings_id'].check_settings, 0.1)

        # Does not seem to be working ... slowing down even!
        # Clock.schedule_once(lumaview.ids['viewer_id'].ids['scope_display_id'].update, 0.1) # This will fix image size on resize issue

    def on_stop(self):
        error_log('LumaViewProApp.on_stop()')
        global lumaview
        lumaview.led_board.leds_off()
        error_log(lumaview.led_board.mssg)
        lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].save_settings("./data/current.json")

LumaViewProApp().run()
