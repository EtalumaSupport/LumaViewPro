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
import sys
import numpy as np
import time
import os
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

# From KivyMD
from kivymd.uix.behaviors import HoverBehavior, TouchBehavior

# Video Related
from kivy.graphics.texture import Texture
import cv2
from scipy import signal

# Additional LumaViewPro files
from trinamic850 import *
from ledboard import *
from pyloncamera import *

global lumaview
global protocol

with open('./data/current.json', "r") as read_file:
    protocol = json.load(read_file)
    print(protocol['labware'])

class ScopeDisplay(Image):
    record = BooleanProperty(None)
    record = False
    play = BooleanProperty(None)
    play = True

    def __init__(self, **kwargs):
        super(ScopeDisplay,self).__init__(**kwargs)
        self.start()

    def start(self, fps = 14):
        self.fps = fps
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def update(self, dt):
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
            lumaview.capture()


# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(FloatLayout): # i.e. global lumaview
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

    def capture(self, dt=0, save_folder = './capture/', file_root = 'live_', append = 'ms', color = 'BF'):
        if self.camera.active == False:
            return

        img = np.zeros((self.camera.array.shape[0], self.camera.array.shape[1], 3))

        if self.ids['mainsettings_id'].currentLayer != 'protocol':
            color = self.ids['mainsettings_id'].currentLayer

        if color == 'Blue':
            img[:,:,0] = self.camera.array
        elif color == 'Green':
            img[:,:,1] = self.camera.array
        elif color == 'Red':
            img[:,:,2] = self.camera.array
        else:
            img[:,:,0] = self.camera.array
            img[:,:,1] = self.camera.array
            img[:,:,2] = self.camera.array

        img = np.flip(img, 0)

        folder = protocol['live_folder']

        # set filename options
        if append == 'time':
            append = time.strftime("%Y%m%d_%H%M%S")
        elif append == 'ms':
            append = str(int(round(time.time() * 1000)))
        else:
            append =''

        # generate filename string
        filename =  file_root + append + '.tiff'

        try:
            cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))
        except:
            print("Error: Unable to save. Perhaps save folder does not exist?")

    def record(self):
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return
        scope_display.record = not scope_display.record

    # TODO recombine
    def composite(self, dt=0):
        global lumaview
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.camera.active == False:
            return

        folder = protocol['live_folder']
        img = np.zeros((protocol['frame_height'], protocol['frame_width'], 3))

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            # multicolor image stack

            if protocol[layer]['acquire'] == True:
                lumaview.ids['mainsettings_id'].ids[layer].goto_focus()

                # Wait for focus to be reached
                set_value = lumaview.motion.target_pos('Z')   # Get target value
                get_value = lumaview.motion.current_pos('Z')  # Get current value

                while set_value != get_value:
                    time.sleep(.01)
                    get_value = lumaview.motion.current_pos('Z') # Get current value

                # set the gain and exposure
                gain = protocol[layer]['gain']
                lumaview.camera.gain(gain)
                exposure = protocol[layer]['exp']
                lumaview.camera.exposure_t(exposure)
                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board

                # Dark field capture
                led_board.leds_off()
                time.sleep(exposure/1000)  # Should be replaced with Clock
                scope_display.update(0)
                darkfield = lumaview.camera.array
                # Florescent capture
                led_board.led_on(led_board.color2ch(layer), illumination) #self.layer??
                time.sleep(exposure/1000)  # Should be replaced with Clock
                scope_display.update(0)
                #corrected = np.max(microscope.array - darkfield, np.zeros(like=darkfield))
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

            led_board.leds_off()
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        img = np.flip(img, 0)

        filename = 'composite_' + str(int(round(time.time() * 1000))) + '.tiff'
        cv2.imwrite(folder+'/'+filename, img.astype(np.uint8))
        # TODO save file in 16 bit TIFF, OMETIFF, and others
        # cv2.imwrite(folder+'/'+filename, img.astype(np.uint16)) # This works

        # # TODO display captured composite
        # microscope.stop()
        # microscope.source = filename
        # time.sleep(5) #Needs to be user selected
        # microscope.start()

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

    def update_shader(self, false_color):

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
        self.ids['verticalcontrol_id'].update_event = Clock.schedule_interval(self.ids['verticalcontrol_id'].update_gui, 0.5)
        self.ids['xy_stagecontrol_id'].update_event = Clock.schedule_interval(self.ids['xy_stagecontrol_id'].update_gui, 0.5)

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
            lumaview.ids['viewer_id'].black = float(edges[0])/255.
            lumaview.ids['viewer_id'].white = float(edges[1])/255.

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
        # else:
        #     print("Can't find image.")

    # def on_touch_down(self, touch):
    #     x = touch.x - self.x
    #     print(x)
    #     if abs(x - self.edges[0]) < 20:
    #         self.edges[0] = x
    #         self.hist_range_set = True
    #     if abs(x - self.edges[1]) < 20:
    #         self.edges[1] = x
    #         self.hist_range_set = True
    #     '''if touch.is_mouse_scrolling:
    #         if touch.button == 'scrolldown':
    #             if self.scale < 100:
    #                 self.scale = self.scale * 1.1
    #         elif touch.button == 'scrollup':
    #             if self.scale > 0.1:
    #                 self.scale = self.scale * 0.8
    #                 '''
    #     # If some other kind of "touch": Fall back on Scatter's behavior
    #     #else:
    #         #super(ShaderViewer, self).on_touch_down(touch)


class VerticalControl(BoxLayout):

    def course_up(self):
        course = protocol['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', course)                  # Move UP
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def fine_up(self):
        fine = protocol['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', fine)                    # Move UP
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def fine_down(self):
        fine = protocol['objective']['step_fine']
        lumaview.motion.move_rel_pos('Z', -fine)                   # Move DOWN
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def course_down(self):
        course = protocol['objective']['step_course']
        lumaview.motion.move_rel_pos('Z', -course)                 # Move DOWN
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_position(self, pos):
        lumaview.motion.move_abs_pos('Z', pos)
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_bookmark(self):
        height = lumaview.motion.current_pos('Z')  # Get current z height in um
        protocol['z_bookmark'] = height

    def goto_bookmark(self):
        pos = protocol['z_bookmark']
        lumaview.motion.move_abs_pos('Z', pos)
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def home(self):
        self.ids['home_id'].text = 'Homing...'
        lumaview.motion.zhome()
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def update_gui(self, dt):
        set_pos = lumaview.motion.target_pos('Z')  # Get target value
        get_pos = lumaview.motion.current_pos('Z') # Get current value

        # z_home = lumaview.motion.limit_status('Z')

        # if z_home:
        #     self.ids['obj_position'].value = 0
        #     self.ids['set_position_id'].text = '0.00'
        # else:
        #     self.ids['obj_position'].value = set_pos
        #     self.ids['set_position_id'].text = format(set_pos, '.2f')

        self.ids['obj_position'].value = max(0, set_pos)
        self.ids['set_position_id'].text = format(max(0, set_pos), '.2f')
        self.ids['get_position_id'].text = format(get_pos, '.2f')

        if set_pos == get_pos:
            Clock.unschedule(self.update_event)
            self.ids['home_id'].text = 'Home'
            self.ids['home_id'].state = 'normal'

    # User selected the autofocus function
    def autofocus(self):
        # camera = lumaview.ids['viewer_id'].ids['scope_display_id']
        global lumaview
        if lumaview.camera.active == False:
            print('Error: VerticalControl.autofocus()')
            return

        # TODO Needs to be set by the user
        center = protocol['z_bookmark']
        range =  protocol['objective']['AF_range']
        fine =   protocol['objective']['AF_min']
        course = protocol['objective']['AF_max']

        self.z_min = center-range
        self.z_max = center+range
        self.z_step = course

        # dt = 0.2 # TODO change this based on focus and exposure time
        layers = ['BF', 'Blue', 'Green', 'Red']
        #for layer in layers:
        #    if lumaview.ids['mainsettings_id'].ids[layer].collapse == False:
        #        dt = protocol[layer]['exp']*2
        dt = 0.5

        self.positions = [0]
        self.focus_measures = [0]

        if self.ids['autofocus_id'].state == 'down':
            self.ids['autofocus_id'].text = 'Focusing...'
            lumaview.motion.move_abs_pos('Z', self.z_min) # Go to z_min
            self.autofocus_event = Clock.schedule_interval(self.focus_iterate, dt)

    def focus_iterate(self, dt):
        global lumaview
        image = lumaview.camera.array

        target = lumaview.motion.current_pos('Z') # Get current value

        self.positions.append(target)
        self.focus_measures.append(self.focus_function(image))

        fine =   protocol['objective']['AF_min']
        course = protocol['objective']['AF_max']
        #closeness = 1/(len(self.positions) + 1
        n = len(self.positions)
        closeness = 1/(n + 0.1)
        #print(closeness)
        step = course*closeness + fine*(1 - closeness)
        print("fine: ",fine, end="")
        print(" course: ",course, end="")
        print(" step: ",step)


        lumaview.motion.move_rel_pos('Z', step) # move by z_step

        if self.ids['autofocus_id'].state == 'normal':
            self.ids['autofocus_id'].text = 'Autofocus'
            Clock.unschedule(self.autofocus_event)
            print("autofocus cancelled")

        elif target >= self.z_max:
            self.ids['autofocus_id'].state = 'normal'
            self.ids['autofocus_id'].text = 'Autofocus'
            Clock.unschedule(self.autofocus_event)

            focus = self.focus_best(self.positions, self.focus_measures)
            # print(self.positions, '\t', self.focus_measures)
            print("Focus Position:", -lumaview.motion.z_ustep2um(focus))
            lumaview.motion.move_abs_pos('Z', focus) # move to absolute target

        self.update_gui(0)

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

    def course_left(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', -100)  # Move LEFT relative
        self.update_gui(0)

    def fine_left(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', -10)  # Move LEFT
        self.update_gui(0)

    def fine_right(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', 10)  # Move RIGHT
        self.update_gui(0)

    def course_right(self):
        global lumaview
        lumaview.motion.move_rel_pos('X', 100)  # Move RIGHT
        self.update_gui(0)

    def course_back(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', -100)  # Move BACK relative by 10000
        self.update_gui(0)

    def fine_back(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', -10)  # Move BACK by 1000
        self.update_gui(0)

    def fine_fwd(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', 10)  # Move FORWARD by 1000
        self.update_gui(0)

    def course_fwd(self):
        global lumaview
        lumaview.motion.move_rel_pos('Y', 100)  # Move FORWARD by 10000
        self.update_gui(0)

    def set_xposition(self, pos):
        global lumaview
        lumaview.motion.move_abs_pos('X', float(pos)*1000)  # position in text is in mm
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_yposition(self, pos):
        global lumaview
        lumaview.motion.move_abs_pos('Y', float(pos)*1000)  # position in text is in mm
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_xbookmark(self):
        global lumaview
        x_pos = lumaview.motion.current_pos('X')  # Get current x position in um
        protocol['x_bookmark'] = x_pos

    def set_ybookmark(self):
        global lumaview
        y_pos = lumaview.motion.current_pos('Y')  # Get current x position in um
        protocol['y_bookmark'] = y_pos

    def goto_xbookmark(self):
        global lumaview
        x_pos = protocol['x_bookmark']
        lumaview.motion.move_abs_pos('X', x_pos)  # set current x position in um
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def goto_ybookmark(self):
        global lumaview
        y_pos = protocol['y_bookmark']
        lumaview.motion.move_abs_pos('Y', y_pos)  # set current y position in um
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def home(self):
        global lumaview
        self.ids['home_id'].text = 'Homing...'
        lumaview.motion.xyhome()
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def update_gui(self, dt):
        global lumaview
        x_target = lumaview.motion.target_pos('X')  # Get target value
        y_target = lumaview.motion.target_pos('Y')  # Get target value

        x_current = lumaview.motion.current_pos('X')  # Get current value
        y_current = lumaview.motion.current_pos('Y')  # Get current value

        # x_home = lumaview.motion.limit_status('X') # Get reference switch status
        # y_home = lumaview.motion.limit_status('Y') # Get reference switch status
        #
        # if x_home or y_home:
        #     self.ids['x_pos_id'].text = '0.00'
        #     self.ids['y_pos_id'].text = '0.00'
        # else:
        #     self.ids['x_pos_id'].text = format(x_target/1000, '.2f')
        #     self.ids['y_pos_id'].text = format(y_target/1000, '.2f')


        self.ids['x_pos_id'].text = format(max(0, x_target)/1000, '.2f')
        self.ids['y_pos_id'].text = format(max(0, y_target)/1000, '.2f')

        if (x_target == x_current) and (y_target == y_current):
            Clock.unschedule(self.update_event)
            self.ids['home_id'].text = 'Home'
            self.ids['home_id'].state = 'normal'

# Labware settings tab
class LabwareSettings(BoxLayout):
    def __init__(self, **kwargs):
        super(LabwareSettings, self).__init__(**kwargs)
        with open('./data/labware.json', "r") as read_file:
            self.labware = json.load(read_file)

    def load_labware(self):
        spinner = self.ids['labware_spinner']
        spinner.values = self.labware['Wellplate']

    def select_labware(self):
        spinner = self.ids['labware_spinner']
        protocol['labware'] = spinner.text
        labware = self.ids['labware_widget_id']
        labware.columns = self.labware['Wellplate'][spinner.text]['columns']
        labware.rows = self.labware['Wellplate'][spinner.text]['rows']
        labware.dimensions = self.labware['Wellplate'][spinner.text]['dimensions']
        labware.spacing = self.labware['Wellplate'][spinner.text]['spacing']
        labware.offset = self.labware['Wellplate'][spinner.text]['offset']
        labware.draw_labware()

    def scan_labware(self):
        labware = self.ids['labware_widget_id']
        labware.scan_labware() # Pass function to the Labware class



class Labware(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Labware, self).__init__(**kwargs)

    def draw_labware(self):
        global lumaview

        self.canvas.clear()
        r, b, g, a = (0.5, 0.5, 0.5, 0.5)
        with self.canvas:
            w = 12 * math.floor(self.width/12)
            h = 12 * math.floor(self.height/12)
            x = self.x + (self.width - w)/2
            y = self.y + (self.height - h)/2
            Color(r, b, g, a)
            Rectangle(pos=(x, y), size=(w, h))

            d = w/self.columns
            r = math.floor(d/2 - 0.5)
            #print(d)
            for i in range(self.columns):
                for j in range(self.rows):
                    Line(circle=(x + d*i + d/2, y + d*j + d/2, r))
                    #print(x + d*i + d/2, y + d*j + d/2)
            x_curr = lumaview.motion.current_pos('X')/1000
            y_curr = lumaview.motion.current_pos('Y')/1000
            i, j = self.get_well_numbers(x_curr, y_curr)
            Color(0., i%2, 0, 1.)
            Line(circle=(x + d*i + d/2, y + d*j + d/2, r))
            #print(x + d*i + d/2, y + d*j + d/2)
            print(i,j)

    def scan_labware(self):
        global lumaview

        for j in range(self.rows):
            for i in range(self.columns):
                if i % 2 == 1:
                    j = self.rows - j
                x, y = self.get_well_position(i, j)
                print(x,y)

                # Go to position
                # On arrival, take image
                # Move to next position
                lumaview.motion.move_abs_pos('X', x*1000)
                lumaview.motion.move_abs_pos('Y', y*1000)
                dt = 2
                self.autoscan_event = Clock.schedule_interval(self.scan_iterate, dt)

    def scan_iterate(self, dt):
        global lumaview

        self.draw_labware()

        x_status = lumaview.motion.target_status('X')
        y_status = lumaview.motion.target_status('Y')

        if x_status and y_status:
            # take the images
            print("Position Reached")
            Clock.unschedule(self.autoscan_event)



























    def get_well_position(self, i, j):
        x = self.offset['x'] + i*self.spacing['x']
        y = self.offset['y'] + j*self.spacing['y']
        return x, y

    def get_well_numbers(self, x, y):
        i = (x - self.offset['x']) / self.spacing['x']
        j = (y - self.offset['y']) / self.spacing['y']
        i = np.clip(i, 0, self.columns-1)
        j = np.clip(j, 0, self.rows-1)
        return i, j

class MicroscopeSettings(BoxLayout):
    def __init__(self, **kwargs):
        super(MicroscopeSettings, self).__init__(**kwargs)

        with open('./data/scopes.json', "r") as read_file:
            self.scopes = json.load(read_file)

        with open('./data/objectives.json', "r") as read_file:
            self.objectives = json.load(read_file)

    def load_scopes(self):
        spinner = self.ids['scope_spinner']
        spinner.values = ['LS850']

    def select_scope(self):
        global lumaview
        global protocol

        spinner = self.ids['scope_spinner']
        print(spinner.text)
        protocol['microscope'] = spinner.text
        microscope_settings_id = lumaview.ids['mainsettings_id'].ids['microscope_settings_id']
        # microscope_settings_id.ids['image_of_microscope'].source = './data/scopes/'+spinner.text+'.png'

    def load_ojectives(self):
        spinner = self.ids['objective_spinner']
        spinner.values = ['4x phase', '4x other', '10x', '20x', '40x', '60x', '100x']

    def select_objective(self):
        global lumaview
        global protocol

        spinner = self.ids['objective_spinner']
        print(spinner.text)
        protocol['objective'] = self.objectives[spinner.text]
        protocol['objective']['ID'] = spinner.text
        microscope_settings_id = lumaview.ids['mainsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['magnification_id'].text = str(protocol['objective']['magnification'])

    def frame_size(self):
        global lumaview
        global protocol

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        width = int(min(int(w), lumaview.camera.active.Width.Max)/2)*2
        height = int(min(int(h), lumaview.camera.active.Height.Max)/2)*2

        protocol['frame_width'] = width
        protocol['frame_height'] = height

        self.ids['frame_width'].text = str(width)
        self.ids['frame_height'].text = str(height)

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
        try:
            illumination = float(self.ids['ill_text'].text)
            protocol[self.layer]['ill'] = illumination
            self.ids['ill_slider'].value = illumination
            self.apply_settings()
        except:
            print('Illumination value is not in acceptable range of 0 to 600 mA.')

    def gain_auto(self):
        if self.ids['gain_auto'].state == 'down':
            state = True
        else:
            state = False
        protocol[self.layer]['gain_auto'] = state
        self.apply_settings()

    def gain_slider(self):
        gain = self.ids['gain_slider'].value
        protocol[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):
        try:
            gain = float(self.ids['gain_text'].text)
            protocol[self.layer]['gain'] = gain
            self.ids['gain_slider'].value = gain
            self.apply_settings()
        except:
            print('Gain value is not in acceptable range of 0 to 24 dB.')

    def exp_slider(self):
        exposure = self.ids['exp_slider'].value
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        protocol[self.layer]['exp'] = exposure        # protocol is ms
        self.apply_settings()

    def exp_text(self):
        try:
            exposure = float(self.ids['exp_text'].text)
            protocol[self.layer]['exp'] = exposure
            self.ids['exp_slider'].value = exposure
            # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
            self.apply_settings()
        except:
            print('Exposure value is not in acceptable range of 0.01 to 1000ms.')

    def root_text(self):
        protocol[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        protocol[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        protocol[self.layer]['acquire'] = self.ids['acquire'].active

    def save_focus(self):
        global lumaview
        pos = lumaview.motion.current_pos('Z')
        protocol[self.layer]['focus'] = pos

    def goto_focus(self):
        global lumaview
        pos = protocol[self.layer]['focus']
        lumaview.motion.move_abs_pos('Z', pos)  # set current z height in usteps
        control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
        control.update_event = Clock.schedule_interval(control.update_gui, 0.5)


    def apply_settings(self):
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        illumination = protocol[self.layer]['ill']
        led_board = lumaview.led_board
        if self.ids['apply_btn'].state == 'down': # if the button is down
            # In active channel,turn on LED
            led_board.led_on(led_board.color2ch(self.layer), illumination)
            #  turn the state of remaining channels to 'normal' and text to 'OFF'
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                if(layer != self.layer):
                    lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        else: # if the button is 'normal' meaning not active
            # In active channel, and turn off LED
            led_board.leds_off()

        # update gain to currently selected settings
        # -----------------------------------------------------
        state = protocol[self.layer]['gain_auto']
        lumaview.camera.auto_gain(state)

        if not(state):
            gain = protocol[self.layer]['gain']
            lumaview.camera.gain(gain)

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = protocol[self.layer]['exp']
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
        for i in np.arange(0.1, 3, 0.1):
            Clock.schedule_once(self.update_shader, i)

    def update_shader(self, dt):
        if self.ids['false_color'].active:
            lumaview.ids['viewer_id'].update_shader(self.layer)
        else:
            lumaview.ids['viewer_id'].update_shader('none')

class CompositeCapture(BoxLayout):
    # One procotol capture event
    def capture(self, dt):
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        if lumaview.camera.active == False:
            return
        try:
            self.n_captures = self.n_captures-1
        except:
            print('Capturing a Single Composite Image')

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            if protocol[layer]['acquire'] == True:
                # global lumaview

                # set the gain and exposure
                gain = protocol[layer]['gain']
                lumaview.camera.gain(gain)
                exposure = protocol[layer]['exp']
                lumaview.camera.exposure_t(exposure)
                scope_display.update(0)

                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board
                led_board.led_on(led_board.color2ch(layer), illumination)

                # capture the image
                save_folder = protocol[layer]['save_folder']
                file_root = protocol[layer]['file_root']
                lumaview.capture(0, save_folder, file_root, color = layer)
                # turn off the LED
                led_board.leds_off()
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'



class TimeLapseSettings(CompositeCapture):
    record = ObjectProperty(None)
    record = False
    movie_folder = StringProperty(None)
    n_captures = ObjectProperty(None)

    def update_period(self):
        # if self.ids['capture_period'].text.isnumeric(): # Did not allow for floating point numbers
        try:
            protocol['period'] = float(self.ids['capture_period'].text)
        except:
            print('Update Period is not an acceptable value')

    def update_duration(self):
        # if self.ids['capture_dur'].text.isnumeric():  # Did not allow for floating point numbers
        try:
            protocol['duration'] = float(self.ids['capture_dur'].text)
        except:
            print('Update Duration is not an acceptable value')

    # load protocol from JSON file
    def load_protocol(self, file="./data/current.json"):
        global lumaview

        # load protocol JSON file
        with open(file, "r") as read_file:
            global protocol
            protocol = json.load(read_file)
            # update GUI values from JSON data:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['scope_spinner'].text = protocol['microscope']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['objective_spinner'].text = protocol['objective']['ID']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['magnification_id'].text = str(protocol['objective']['magnification'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_width'].text = str(protocol['frame_width'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_height'].text = str(protocol['frame_height'])

            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = protocol[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = protocol[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = protocol[layer]['exp']
                # lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = float(np.log10(protocol[layer]['exp']))
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = protocol[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['false_color'].active = protocol[layer]['false_color']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = protocol[layer]['acquire']

            lumaview.camera.frame_size(protocol['frame_width'], protocol['frame_height'])

    # Save protocol to JSON file
    def save_protocol(self, file="./data/current.json"):
        global protocol
        with open(file, "w") as write_file:
            json.dump(protocol, write_file, indent = 4)

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
            self.ids['protocol_btn'].text = 'Run Protocol'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    '''
    # One procotol capture event
    def capture(self, dt):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']
        if camera == False:
            return
        try:
            self.n_captures = self.n_captures-1
        except:
            print('Capturing a Single Composite Image')

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

                # capture the image
                save_folder = protocol[layer]['save_folder']
                file_root = protocol[layer]['file_root']
                lumaview.capture(0, save_folder, file_root, color = layer)
                # turn off the LED
                led_board.leds_off()
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
    '''

    # def convert_to_avi(self):
    #
    #     # self.choose_folder()
    #     save_location = './capture/movie.avi'
    #
    #     img_array = []
    #     for filename in glob.glob('./capture/*.tiff'):
    #         img = cv2.imread(filename)
    #         height, width, layers = img.shape
    #         size = (width,height)
    #         img_array.append(img)
    #
    #     out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    #
    #     for i in range(len(img_array)):
    #         out.write(img_array[i])
    #     out.release()


class ZStack(CompositeCapture):
    def set_range(self):
        n_steps = self.ids['zstack_steps_id'].text
        n_steps = int(n_steps)

        step_size = self.ids['zstack_stepsize_id'].text
        step_size = float(step_size)

        range = n_steps * step_size
        self.ids['zstack_range_id'].text = str(range)

    def aquire_zstack(self):
        global lumaview

        n_steps = self.ids['zstack_steps_id'].text
        n_steps = int(n_steps)

        step_size = self.ids['zstack_stepsize_id'].text
        step_size = float(step_size)

        z_range = n_steps * step_size

        spinner_values = self.ids['zstack_spinner'].values
        spinner_value = self.ids['zstack_spinner'].text

        # Get current position
        current_pos = lumaview.motion.current_pos('Z')

        # Set start position
        if spinner_value == spinner_values[0]:   # 'Current Position at Top'
            start_pos = current_pos - z_range
        elif spinner_value == spinner_values[1]: # 'Current Position at Center'
            start_pos = current_pos - z_range / 2
        elif spinner_value == spinner_values[2]: # 'Current Position at Bottom'
            start_pos = current_pos

        # Make array of positions
        positions = np.arange(n_steps)*step_size + start_pos

        # Acquire z-stack
        for pos in positions:
            # Move to position
            lumaview.motion.move_abs_pos('Z', pos)


            # REPLACE BELOW WITH:
            # Schedule Regular Checks
            # Use target_status
            # If target_status is True capture image and unschedule
            # Include a timeout

            # Wait to arrive
            set_value = lumaview.motion.target_pos('Z')   # Get target value
            get_value = lumaview.motion.current_pos('Z')  # Get current value

            while set_value != get_value:
                time.sleep(.01)
                get_value = lumaview.motion.current_pos('Z') # Get current value

            # Capture image
            # print(pos)


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
        if self.context == 'load_protocol':
            global lumaview
            lumaview.ids['mainsettings_id'].ids['time_lapse_id'].load_protocol(self.selection[0])

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
            protocol['live_folder'] = path

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
            protocol[self.context]['save_folder'] = path

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
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].save_protocol(self.selection[0])
        print('Saving Protocol to File:', self.selection[0])

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
        self.tooltip_display_delay = 0.
        self.shift_y = dp(80)


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        Window.size = (1280, 800)
        self.icon = './data/icons/icon32x.png'
        global lumaview
        lumaview = MainDisplay()

        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].load_protocol("./data/current.json")
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.leds_off()
        # how to keep loading software while this is happening?
        lumaview.motion.xyhome()
        return lumaview

    def on_stop(self):
        global lumaview
        lumaview.led_board.leds_off()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].save_protocol("./data/current.json")

LumaViewProApp().run()
