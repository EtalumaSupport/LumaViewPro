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

global lumaview
global settings
home_wd = os.getcwd()

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
        self.start()

    def start(self, fps = 10):
        self.fps = fps
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        # self.frame_event.cancel()
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
        global lumaview

        save_folder = settings['live_folder']
        file_root = 'live_'
        append = 'ms'
        color = 'BF'
        if lumaview.ids['mainsettings_id'].currentLayer != 'microscope':
            color = lumaview.ids['mainsettings_id'].currentLayer

        lumaview.camera.grab()
        self.save_image(save_folder, file_root, append, color)

    def custom_capture(self, channel, illumination, gain, exposure):
        global lumaview
        global settings
        
        # Set gain and exposure
        lumaview.camera.gain(gain)
        lumaview.camera.exposure_t(exposure)

 
        # Save Settings
        color = lumaview.led_board.ch2color(channel)
        save_folder =  settings[color]['save_folder']
        file_root = settings[color]['file_root']
        append = 'ms'

        # Illuminate
        lumaview.led_board.led_on(channel, illumination)

        # Grab image and save
        lumaview.camera.grab()
        self.save_image(save_folder, file_root, append, color)

        # Turn off LEDs
        lumaview.led_board.leds_off()

    # Save image from camera buffer to specified location
    def save_image(self, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
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
            print("Error: Unable to save. Perhaps save folder does not exist?")

    # capture and save a composite image using the current settings
    def composite_capture(self):
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
                exposure = settings[layer]['exp']
                lumaview.camera.exposure_t(exposure)

                # update illumination to currently selected settings
                illumination = settings[layer]['ill']

                # Dark field capture
                lumaview.led_board.leds_off()
                time.sleep(exposure/1000)  # Should be replaced with Clock
                scope_display.update()
                darkfield = lumaview.camera.array

                # Florescent capture
                lumaview.led_board.led_on(lumaview.led_board.color2ch(layer), illumination)
                lumaview.camera.grab()
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
        lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

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
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return

        if scope_display.play == True:
            scope_display.play = False
            self.led_board.leds_off()
            scope_display.stop()
        else:
            scope_display.play = True
            scope_display.start()

    def record(self):
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return
        scope_display.record = not scope_display.record

    def fit_image(self):
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
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
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        self.white = 1.
        self.black = 0.

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

    def update_shader(self, false_color='BF'):

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

        # print('b:', self.black, 'w:', self.white)

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)


class MotionSettings(BoxLayout):
    settings_width = dp(300)
    isOpen = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_draw=self.check_settings)

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        self.ids['verticalcontrol_id'].update_gui()
        self.ids['xy_stagecontrol_id'].update_gui()

        # move position of motion control
        if self.isOpen:
            self.ids['toggle_motionsettings'].state = 'normal'
            self.pos = -self.settings_width+30, 0
            self.isOpen = False
        else:
            self.ids['toggle_motionsettings'].state = 'down'
            self.pos = 0, 0
            self.isOpen = True

        if scope_display.play == True:
            scope_display.start()

    def check_settings(self, *args):
        # global lumaview
        if not self.isOpen:
            self.ids['toggle_motionsettings'].state = 'normal'
            self.pos = -self.settings_width+30, 0
        else:
            self.ids['toggle_motionsettings'].state = 'down'
            self.pos = 0, 0

class PostProcessing(BoxLayout):

    def convert_to_avi(self):

        # self.choose_folder()
        save_location = './capture/movie.avi'

        img_array = []
        for filename in glob.glob('./capture/*.tiff'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        if len(img_array) > 0:
            out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


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
        print('ShaderEditor.compile_shaders()')
        if not self.viewer:
            print('ShaderEditor.compile_shaders() Fail')
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
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

class MainSettings(BoxLayout):
    settings_width = dp(300)
    isOpen = BooleanProperty(False)
    notCollapsing = BooleanProperty(True)
    currentLayer = StringProperty('microscope')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_draw=self.check_settings)

    # Hide (and unhide) main settings
    def toggle_settings(self):
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()

        # move position of settings
        if self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
            self.isOpen = False
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0
            self.isOpen = True

        if scope_display.play == True:
            scope_display.start()

    def accordion_collapse(self, layer):
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        lumaview.led_board.leds_off()
        self.currentLayer = layer
        self.notCollapsing = not(self.notCollapsing)
        if self.notCollapsing:
            if scope_display.play == True:
                scope_display.start()

    def check_settings(self, *args):
        global lumaview
        if not self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0


class Histogram(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (1, 1, 1, 1)

        self.event = Clock.schedule_interval(self.histogram, 1)

        self.hist_range_set = False
        self.edges = [0,255]
        self.stablize = 0.3

    def histogram(self, *args):
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
        set_pos = lumaview.motion.target_pos('Z')  # Get target value
        get_pos = lumaview.motion.current_pos('Z') # Get current value

        self.ids['obj_position'].value = max(0, set_pos)
        self.ids['z_position_id'].text = format(max(0, set_pos), '.2f')

    def course_up(self):
        course = settings['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', course)                  # Move UP
        self.update_gui()

    def fine_up(self):
        fine = settings['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', fine)                    # Move UP
        self.update_gui()

    def fine_down(self):
        fine = settings['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', -fine)                   # Move DOWN
        self.update_gui()

    def course_down(self):
        course = settings['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', -course)                 # Move DOWN
        self.update_gui()

    def set_position(self, pos):
        lumaview.motion.move_abs_pos('Z', float(pos))
        self.update_gui()

    def set_bookmark(self):
        height = lumaview.motion.current_pos('Z')  # Get current z height in um
        settings['bookmark']['z'] = height

    def goto_bookmark(self):
        pos = settings['bookmark']['z']
        lumaview.motion.move_abs_pos('Z', pos)
        self.update_gui()

    def home(self):
        lumaview.motion.zhome()
        self.update_gui()

    # User selected the autofocus function
    def autofocus(self):
        # camera = lumaview.ids['viewer_id'].ids['scope_display_id']
        global lumaview
        if lumaview.camera.active == False:
            print('Error: VerticalControl.autofocus()')
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
            self.autofocus_event = Clock.schedule_interval(self.focus_iterate, dt)
            print('DEBUG: Vertical Control self.autofocus_event = Clock.schedule_interval(self.focus_iterate, dt)')

    def focus_iterate(self, dt):
        global lumaview
        image = lumaview.camera.array

        target = lumaview.motion.current_pos('Z') # Get current value
        self.positions.append(target)
        self.focus_measures.append(self.focus_function(image))

        fine =   settings['objective']['AF_min']
        course = settings['objective']['AF_max']
        #closeness = 1/(len(self.positions) + 1
        n = len(self.positions)
        closeness = 1/(n + 0.1)
        #print(closeness)
        step = course*closeness + fine*(1 - closeness)
        print("fine: ", fine, end="")
        print(" course: ", course, end="")
        print(" step:", step)

        lumaview.motion.move_rel_pos('Z', step) # move by z_step

        if self.ids['autofocus_id'].state == 'normal':
            self.ids['autofocus_id'].text = 'Autofocus'
            # self.autofocus_event.cancel()
            Clock.unschedule(self.focus_iterate)

        elif target >= self.z_max:
            self.ids['autofocus_id'].state = 'normal'
            self.ids['autofocus_id'].text = 'Autofocus'
            # self.autofocus_event.cancel()
            Clock.unschedule(self.focus_iterate)

            focus = self.focus_best(self.positions, self.focus_measures)
            # print(self.positions, '\t', self.focus_measures)
            print("Focus Position:", -lumaview.motion.z_ustep2um(focus))
            lumaview.motion.move_abs_pos('Z', focus) # move to absolute target

        self.update_gui()

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            print(sum_one - sum_two)
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            print(skew)
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            print('pixel_variation:', var)
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
            print(kernel)
            convolve = signal.convolve2d(image, kernel, mode='valid')
            sum = np.sum(convolve)
            print(sum)
            return sum

        else:
            return 0

    def focus_best(self, positions, values, algorithm='mov_avg'):
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
        global lumaview
        x_target = lumaview.motion.target_pos('X')  # Get target value
        y_target = lumaview.motion.target_pos('Y')  # Get target value

        self.ids['x_pos_id'].text = format(max(0, x_target)/1000, '.2f')
        self.ids['y_pos_id'].text = format(max(0, y_target)/1000, '.2f')

    def fine_left(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', -10)  # Move LEFT
        self.update_gui()

    def fine_right(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', 10)  # Move RIGHT
        self.update_gui()

    def course_left(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', -100)  # Move LEFT relative
        self.update_gui()

    def course_right(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', 100)  # Move RIGHT
        self.update_gui()

    def fine_back(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', -10)  # Move BACK by 1000
        self.update_gui()

    def fine_fwd(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', 10)  # Move FORWARD by 1000
        self.update_gui()

    def course_back(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', -100)  # Move BACK relative by 10000
        self.update_gui()
    def course_fwd(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', 100)  # Move FORWARD by 10000
        self.update_gui()

    def set_xposition(self, pos):
        global lumaview
        lumaview.motion.move_abs_pos('X', float(pos)*1000)  # position in text is in mm
        self.update_gui()

    def set_yposition(self, pos):
        global lumaview
        lumaview.motion.move_abs_pos('Y', float(pos)*1000)  # position in text is in mm
        self.update_gui()

    def set_xbookmark(self):
        global lumaview
        x_pos = lumaview.motion.current_pos('X')  # Get current x position in um
        settings['bookmark']['x'] = x_pos

    def set_ybookmark(self):
        global lumaview
        y_pos = lumaview.motion.current_pos('Y')  # Get current x position in um
        settings['bookmark']['y'] = y_pos

    def goto_xbookmark(self):
        global lumaview
        x_pos = settings['bookmark']['x']
        lumaview.motion.move_abs_pos('X', x_pos)  # set current x position in um
        self.update_gui()

    def goto_ybookmark(self):
        global lumaview
        y_pos = settings['bookmark']['y']
        lumaview.motion.move_abs_pos('Y', y_pos)  # set current y position in um
        self.update_gui()

    def home(self):
        global lumaview
        lumaview.motion.xyhome()
        self.ids['x_pos_id'].text = '0.00'
        self.ids['y_pos_id'].text = '0.00'

# Protocol settings tab
class ProtocolSettings(CompositeCapture):
    global settings

    def __init__(self, **kwargs):

        super(ProtocolSettings, self).__init__(**kwargs)
 
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
        try:
            settings['protocol']['period'] = float(self.ids['capture_period'].text)
        except:
            print('Update Period is not an acceptable value')

    # Update Protocol Duration   
    def update_duration(self):
        try:
            settings['protocol']['duration'] = float(self.ids['capture_dur'].text)
        except:
            print('Update Duration is not an acceptable value')

    # Labware Selection
    def select_labware(self):
        spinner = self.ids['labware_spinner']
        spinner.values = list(self.labware['Wellplate'].keys())
        settings['protocol']['labware'] = spinner.text
    
        # Draw the Labware on Stage
        self.ids['stage_widget_id'].draw_labware()

 
    # Create New Protocol
    def new_protocol(self):
        pass

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

        # # Print Out
        print(self.step_names)
        print(self.step_values)

    # Load Protocol from File
    def load_protocol(self, file="./data/sample_protocol.csv"):

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

        print(protocol_df)
        print(self.step_names)
        print(self.step_values)

    # Save Protocol to File
    def save_protocol(self, file="./data/sample_protocol.csv"):

        # Create Empty DataFrame
        protocol_df = pd.DataFrame(columns=['Name', 'X', 'Y', 'Z', 'Channel', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure'])
        # Change columns to List and NPArray
        protocol_df['Name'] = self.step_names
        protocol_df[['X', 'Y', 'Z', 'Channel', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure']] = self.step_values

        protocol_df.to_csv(file, index=False)

    # Goto to Previous Step
    def prev_step(self):
        self.c_step = max(self.c_step - 1, 0)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
 
    # Go to Next Step
    def next_step(self):
        self.c_step = min(self.c_step + 1, len(self.step_names)-1)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
    
    # Go to Input Step
    def go_to_step(self):
        # Get the Current Step Number
        self.c_step = int(self.ids['step_number_input'].text)-1

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
        print(name)
        lumaview.motion.move_abs_pos('X', x*1000)
        lumaview.motion.move_abs_pos('Y', y*1000)
        lumaview.motion.move_abs_pos('Z', z)

        ch = lumaview.led_board.ch2color(ch)
        layer  = lumaview.ids['mainsettings_id'].ids[ch]

        # TODO: open accordian to correct channel or display channel in some way

        # set illumination settings, text, and slider
        print(ill, type(ill))
        settings[ch]['ill'] = ill
        layer.ids['ill_text'].text = str(ill)
        layer.ids['ill_slider'].value = float(ill)

        # set gain settings, text, and slider
        print(gain, type(gain))
        settings[ch]['gain'] = gain
        layer.ids['gain_text'].text = str(gain)
        layer.ids['gain_slider'].value = float(gain)

        # set exposure settings, text, and slider
        print(exp, type(exp))
        settings[ch]['exp'] = exp
        layer.ids['exp_text'].text = str(exp)
        layer.ids['exp_slider'].value = float(exp)

        # set auto-gain checkbox
        print(auto_gain, type(auto_gain))
        settings[ch]['gain_auto'] = bool(auto_gain)
        layer.ids['gain_auto'].active = bool(auto_gain)

        for i in range(10):
            Clock.schedule_once(self.ids['stage_widget_id'].draw_labware, i/2)

    # Delete Current Step of Protocol
    def delete_step(self):
        
        self.step_names.pop(self.c_step)
        self.step_values = np.delete(self.step_values, self.c_step, axis = 0)
        self.c_step = self.c_step - 1

        # Update total number of steps to GUI
        self.ids['step_total_input'].text = str(len(self.step_names))
        self.next_step()

    # Modify Current Step of Protocol
    def modify_step(self):

        self.step_names[self.c_step] = self.ids['step_name_input'].text
        self.step_values[self.c_step, 0] = lumaview.motion.current_pos('X')/1000 # x
        self.step_values[self.c_step, 1] = lumaview.motion.current_pos('Y')/1000 # y
        self.step_values[self.c_step, 2] = lumaview.motion.current_pos('Z')      # z

        # TODO
        ch = 0
        layer  = lumaview.ids['mainsettings_id'].ids[ch]

        self.step_values[self.c_step, 3] = ch # ch
        self.step_values[self.c_step, 4] = layer.ids['ill_slider'].value # ill
        self.step_values[self.c_step, 5] = layer.ids['gain_slider'].value # gain
        self.step_values[self.c_step, 6] = int(layer.ids['gain_auto'].active) # auto_gain
        self.step_values[self.c_step, 7] = layer.ids['exp_slider'].value # exp

    # Insert Current Step to Protocol at Current Position
    def add_step(self):

         # Determine Values
        name = self.ids['step_name_input'].text
        ch = 0
        layer  = lumaview.ids['mainsettings_id'].ids[ch]

        # TODO
        step = [lumaview.motion.current_pos('X')/1000, # x
                lumaview.motion.current_pos('Y')/1000, # y
                lumaview.motion.current_pos('Z'),      # z
                ch, # ch 
                layer.ids['ill_slider'].value, # ill
                layer.ids['gain_slider'].value, # gain
                int(layer.ids['gain_auto'].active), # auto_gain
                layer.ids['exp_slider'].value, # exp
        ]

        # Insert into List and Array
        self.step_names.insert(self.c_step, name)
        self.step_values = np.insert(self.step_values, self.c_step, step, axis=0)

        self.ids['step_total_input'].text = str(len(self.step_names))



    # Run one scan of the protocol
    def run_scan(self, protocol = False):
 
        if len(self.step_names) < 1:
            print('Protocol has no steps.')
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
            lumaview.motion.move_abs_pos('Y', y*1000)
            lumaview.motion.move_abs_pos('Z', z)

            Clock.schedule_interval(self.scan_iterate, 0.1)

        else:  # self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'
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
            print('Scan Step:', self.step_names[self.c_step])

            # identify image settings
            ch =        self.step_values[self.c_step, 3]
            ill =       self.step_values[self.c_step, 4]
            gain =      self.step_values[self.c_step, 5]
            auto_gain = self.step_values[self.c_step, 6]
            exp =       self.step_values[self.c_step, 7]

            print(ch, ill, gain, auto_gain, exp)

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
                print('Scan Complete')
                self.ids['run_scan_btn'].state = 'normal'
                self.ids['run_scan_btn'].text = 'Run One Scan'
                Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate

    # Run protocol without xy movement
    def run_stationary(self):

        if self.ids['run_stationary_btn'].state == 'down':
            self.ids['run_stationary_btn'].text = 'State == Down'
        else:
            self.ids['run_stationary_btn'].text = 'Run Stationary Protocol' # 'normal'


    # Run the complete protocol 
    def run_protocol(self):
        self.n_scans = int(float(settings['protocol']['duration'])*60 / float(settings['protocol']['period']))
        self.start_t = time.time() # start of cycle in seconds

        if self.ids['run_protocol_btn'].state == 'down':
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.run_scan(protocol = True)
            self.protocol_event = Clock.schedule_interval(self.protocol_iterate, 1)

        else:
            self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            Clock.unschedule(self.protocol_iterate) # unschedule all copies of scan iterate
            # self.protocol_event.cancel()
 
    def protocol_iterate(self, dt):

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
        # print(hrs + ':' + minutes + ' remaining')
        # self.ids['run_protocol_btn'].text = hrs+':'+minutes+' remaining'
        self.ids['run_protocol_btn'].text = str(n_scans)

        # Check if reached next Period
        if (time.time()-self.start_t) > period:

            # reset the start time and update number of scane remaining
            self.start_t = time.time()
            self.n_scans = self.n_scans - 1

            if self.n_scans > 0:
                print('Scans Remaining:', self.n_scans)
                self.run_scan(protocol = True)
            else:
                # self.protocol_event.cancel()
                Clock.unschedule(self.protocol_iterate) # unschedule all copies of scan iterate

# Widget for displaying Microscope Stage area, labware, and current position 
class Stage(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)


    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print('clicked on:', touch.pos)
        
    def draw_labware(self, *args):
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

        os.chdir(home_wd)
        with open('./data/scopes.json', "r") as read_file:
            self.scopes = json.load(read_file)

        os.chdir(home_wd)
        with open('./data/objectives.json', "r") as read_file:
            self.objectives = json.load(read_file)

    # load settings from JSON file
    def load_settings(self, file="./data/current.json"):
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

    # Save settings to JSON file
    def save_settings(self, file="./data/current.json"):
        global settings
        os.chdir(home_wd)
        with open(file, "w") as write_file:
            json.dump(settings, write_file, indent = 4)

    def load_scopes(self):
        spinner = self.ids['scope_spinner']
        spinner.values = list(self.scopes.keys())

    def select_scope(self):
        global lumaview
        global settings

        spinner = self.ids['scope_spinner']
        print(spinner.text)
        settings['microscope'] = spinner.text

    def load_ojectives(self):
        spinner = self.ids['objective_spinner']
        spinner.values = list(self.objectives.keys())

    def select_objective(self):
        global lumaview
        global settings

        spinner = self.ids['objective_spinner']
        print(spinner.text)
        settings['objective'] = self.objectives[spinner.text]
        settings['objective']['ID'] = spinner.text
        microscope_settings_id = lumaview.ids['mainsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['magnification_id'].text = str(settings['objective']['magnification'])

    def frame_size(self):
        global lumaview
        global settings

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        width = int(min(int(w), lumaview.camera.active.Width.Max)/2)*2
        height = int(min(int(h), lumaview.camera.active.Height.Max)/2)*2

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame']['width'].text = str(width)
        self.ids['frame']['height'].text = str(height)

        lumaview.camera.frame_size(width, height)


# Modified Slider Class to enable on_release event
# ---------------------------------------------------------------------
class ModSlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(ModSlider, self).__init__(**kwargs)

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super(ModSlider, self).on_touch_up(touch)
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
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        illumination = self.ids['ill_slider'].value
        settings[self.layer]['ill'] = illumination
        self.apply_settings()

    def ill_text(self):
        ill_min = self.ids['ill_slider'].min
        ill_max = self.ids['ill_slider'].max
        ill_val = float(self.ids['ill_text'].text)
        illumination = float(np.clip(ill_val, ill_min, ill_max))

        settings[self.layer]['ill'] = illumination
        self.ids['ill_slider'].value = illumination
        self.ids['ill_text'].text = str(illumination)

        self.apply_settings()

    def gain_auto(self):
        if self.ids['gain_auto'].state == 'down':
            state = True
        else:
            state = False
        settings[self.layer]['gain_auto'] = state
        self.apply_settings()

    def gain_slider(self):
        gain = self.ids['gain_slider'].value
        settings[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):

        gain_min = self.ids['gain_slider'].min
        gain_max = self.ids['gain_slider'].max
        gain_val = float(self.ids['gain_text'].text)
        gain = float(np.clip(gain_val, gain_min, gain_max))

        settings[self.layer]['gain'] = gain
        self.ids['gain_slider'].value = gain
        self.ids['gain_text'].text = str(gain)

        self.apply_settings()

    def exp_slider(self):
        exposure = self.ids['exp_slider'].value
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp'] = exposure        # exposure in ms
        self.apply_settings()

    def exp_text(self):
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
        settings[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        settings[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        settings[self.layer]['acquire'] = self.ids['acquire'].active

    def save_focus(self):
        global lumaview
        pos = lumaview.motion.current_pos('Z')
        settings[self.layer]['focus'] = pos

    def goto_focus(self):
        global lumaview
        pos = settings[self.layer]['focus']
        lumaview.motion.move_abs_pos('Z', pos)  # set current z height in usteps
        control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
        control.update_gui()

    def apply_settings(self):
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        illumination = settings[self.layer]['ill']
        if self.ids['apply_btn'].state == 'down': # if the button is down
            # In active channel,turn on LED
            lumaview.led_board.led_on(lumaview.led_board.color2ch(self.layer), illumination)
            #  turn the state of remaining channels to 'normal' and text to 'OFF'
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                if(layer != self.layer):
                    lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        else: # if the button is 'normal' meaning not active
            # In active channel, and turn off LED
            lumaview.led_board.leds_off()

        # update gain to currently selected settings
        # -----------------------------------------------------
        state = settings[self.layer]['gain_auto']
        lumaview.camera.auto_gain(state)

        if not(state):
            gain = settings[self.layer]['gain']
            lumaview.camera.gain(gain)

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = settings[self.layer]['exp']
        lumaview.camera.exposure_t(exposure)

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
        if self.ids['false_color'].active:
            lumaview.ids['viewer_id'].update_shader(self.layer)
        else:
            lumaview.ids['viewer_id'].update_shader('none')


class ZStack(CompositeCapture):
    def set_steps(self):

        step_size = self.ids['zstack_stepsize_id'].text
        step_size = float(step_size)

        range = self.ids['zstack_range_id'].text
        range = float(range)

        if step_size != 0:
            n_steps = np.floor( range / step_size)
            self.ids['zstack_steps_id'].text = str(int(n_steps))
        else:
            self.ids['zstack_steps_id'].text = '0'

    def aquire_zstack(self):
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

        if self.ids['ztack_aqr_btn'].state == 'down':
            self.zstack_event = Clock.schedule_interval(self.zstack_iterate, 0.01)
            self.ids['ztack_aqr_btn'].text = 'Acquiring ZStack'
            print('DEBUG: ZStack self.zstack_event = Clock.schedule_interval(self.zstack_iterate, 0.01)')

        else:
            self.ids['ztack_aqr_btn'].text = 'Acquire'
            # self.zstack_event.cancel()
            Clock.unschedule(self.zstack_iterate)
            print('cancel')

    def zstack_iterate(self, dt):
        print('Iterate at:', lumaview.motion.current_pos('Z'), lumaview.motion.target_pos('Z'))


        if lumaview.motion.target_status('Z'):
            print('at target')
            self.live_capture()
            self.n_pos += 1

            if self.n_pos < len(self.positions):
                lumaview.motion.move_abs_pos('Z', self.positions[self.n_pos])
            else:
                self.ids['ztack_aqr_btn'].text = 'Acquire'
                self.ids['ztack_aqr_btn'].state = 'normal'
                # self.zstack_event.cancel()
                Clock.unschedule(self.zstack_iterate)

# Button the triggers 'filechooser.open_file()' from plyer
class FileChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        # Call plyer filechooser API to run a filechooser Activity.
        self.context = context
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        self.selection = selection

    def on_selection(self, *a, **k):
        global lumaview
        
        if self.context == 'load_settings':
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings(self.selection[0])

        elif self.context == 'load_protocol':
            lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(file = self.selection[0])

# Button the triggers 'filechooser.choose_dir()' from plyer
class FolderChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        self.context = context
        filechooser.choose_dir(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        self.selection = selection

    def on_selection(self, *a, **k):
        path = self.selection[0]

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
        self.context = context
        filechooser.save_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        self.selection = selection

    def on_selection(self, *a, **k):
        global lumaview
        
        if self.context == 'save_settings':
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].save_settings(self.selection[0])
            print('Saving Settings to File:', self.selection[0])

        elif self.context == 'save_protocol':
            lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].save_protocol(file = self.selection[0])
            print('Saving Protocol to File:', self.selection[0])


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):

    def build(self):
        self.icon = './data/icons/icon32x.png'
        Window.maximize()
        global lumaview
        lumaview = MainDisplay()

        try:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings("./data/current.json")
        except:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].load_settings("./data/settings.json")
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.leds_off()
        lumaview.motion.xyhome()
        return lumaview

    def on_stop(self):
        global lumaview
        lumaview.led_board.leds_off()
        lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].save_settings("./data/current.json")

LumaViewProApp().run()
