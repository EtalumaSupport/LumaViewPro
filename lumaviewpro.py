#!/usr/bin/python3

'''
MIT License

Copyright (c) 2023 Etaluma, Inc.

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
Bryan Tiedemann, The Earthineering Company
Gerard Decker, The Earthineering Company

MODIFIED:
June 24, 2023
'''

# General
import os
import numpy as np
import csv
import time
import json
import glob
from lvp_logger import logger
from plyer import filechooser

# Deactivate kivy logging
#os.environ["KIVY_NO_CONSOLELOG"] = "1"

# Kivy configurations
# Configurations must be set befor Kivy is imported
from kivy.config import Config
Config.set('input', 'mouse', 'mouse, disable_multitouch')
Config.set('graphics', 'resizable', True) # this seemed to have no effect so may be unnessesary

# if fixed size at launch
#Config.set('graphics', 'width', '1920')
#Config.set('graphics', 'height', '1080')

# if maximized at launch
Config.set('graphics', 'window_state', 'maximized')

import kivy
kivy.require("2.1.0")

from kivy.app import App
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, ListProperty
#from kivy.properties import BoundedNumericProperty, ColorProperty, OptionProperty, NumericProperty
from kivy.clock import Clock
from kivy.metrics import dp
#from kivy.animation import Animation
from kivy.graphics import Line, Color, Rectangle, Ellipse

# User Interface
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

# Video Related
from kivy.graphics.texture import Texture
import cv2

# Hardware
from labware import WellPlate
import lumascope_api
import post_processing

global lumaview
global settings

abspath = os.path.abspath(__file__)
basename = os.path.basename(__file__)
source_path = abspath[:-len(basename)]
print(source_path)

start_str = time.strftime("%Y %m %d %H_%M_%S")
start_str = str(int(round(time.time() * 1000)))

global focus_round
focus_round = 0


def focus_log(positions, values):
    global focus_round
    if False:
        os.chdir(source_path)
        try:
            file = open('./logs/focus_log.txt', 'a')
        except:
            if not os.path.isdir('./logs'):
                raise FileNotFoundError("Couldn't find 'logs' directory.")
            else:
                raise
        for i, p in enumerate(positions):
            mssg = str(focus_round) + '\t' + str(p) + '\t' + str(values[i]) + '\n'
            file.write(mssg)
        file.close()
        focus_round += 1

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
        logger.info('[LVP Main  ] ScopeDisplay.__init__()')
        self.start()

    def start(self, fps = 10):
        logger.info('[LVP Main  ] ScopeDisplay.start()')
        self.fps = fps
        logger.info('[LVP Main  ] Clock.schedule_interval(self.update, 1.0 / self.fps)')
        Clock.schedule_interval(self.update_scopedisplay, 1.0 / self.fps)

    def stop(self):
        logger.info('[LVP Main  ] ScopeDisplay.stop()')
        logger.info('[LVP Main  ] Clock.unschedule(self.update)')
        Clock.unschedule(self.update_scopedisplay)

    def update_scopedisplay(self, dt=0):
        global lumaview

        if lumaview.scope.camera.active != False:
            array = lumaview.scope.get_image()
            if array is False:
                return
            # Convert to texture for display (using OpenGL)
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

    def __init__(self, **kwargs):
        super(CompositeCapture,self).__init__(**kwargs)

    # Gets the current well label (ex. A1, C2, ...) 
    def get_well_label(self):
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']

        # Get target position
        try:
            x_target = lumaview.scope.get_target_position('X')
            y_target = lumaview.scope.get_target_position('Y')
        except:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            raise

        x_target, y_target = protocol_settings.stage_to_plate(x_target, y_target)

        well_x, well_y = current_labware.get_well_index(x_target, y_target)
        letter = chr(ord('A') + well_y)
        return f'{letter}{well_x + 1}'

    def live_capture(self):
        logger.info('[LVP Main  ] CompositeCapture.live_capture()')
        global lumaview

        save_folder = settings['live_folder']
        file_root = 'live_'
        color = 'BF'
        well_label = self.get_well_label()
        append = f'{well_label}_{color}'

        layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
        for layer in layers:
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                if lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active:
                    color = layer
            
        # lumaview.scope.get_image()
        lumaview.scope.save_live_image(save_folder, file_root, append, color)

    def custom_capture(self, channel, illumination, gain, exposure, false_color = True):
        logger.info('[LVP Main  ] CompositeCapture.custom_capture()')
        global lumaview
        global settings
        
        # Set gain and exposure
        lumaview.scope.set_gain(gain)
        lumaview.scope.set_exposure_time(exposure)
 
        # Save Settings
        color = lumaview.scope.ch2color(channel)
        save_folder =  settings[color]['save_folder']
        file_root = settings[color]['file_root']
        well_label = self.get_well_label()
        append = f'{well_label}_{color}'

        # Illuminate
        if lumaview.scope.led:
            lumaview.scope.led_on(channel, illumination)
            logger.info('[LVP Main  ] lumaview.scope.led_on(channel, illumination)')
        else:
            logger.warning('LED controller not available.')

        # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        # Grab image and save
        time.sleep(2*exposure/1000+0.2) 
        lumaview.scope.get_image()

        if false_color: 
            lumaview.scope.save_live_image(save_folder, file_root, append, color)
        else:
            lumaview.scope.save_live_image(save_folder, file_root, append, 'BF')

        # Turn off LEDs
        if lumaview.scope.led:
            lumaview.scope.leds_off()
            logger.info('[LVP Main  ] lumaview.scope.leds_off()')
        else:
            logger.warning('LED controller not available.')

    # capture and save a composite image using the current settings
    def composite_capture(self):
        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')
        global lumaview

        if lumaview.scope.camera.active == False:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        img = np.zeros((settings['frame']['height'], settings['frame']['width'], 3))

        layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
        for layer in layers:
            if settings[layer]['acquire'] == True:

                # Go to focus and wait for arrival
                lumaview.ids['imagesettings_id'].ids[layer].goto_focus()

                while not lumaview.scope.get_target_status('Z'):
                    time.sleep(.001)

                # set the gain and exposure
                gain = settings[layer]['gain']
                lumaview.scope.set_gain(gain)
                exposure = settings[layer]['exp']
                lumaview.scope.set_exposure_time(exposure)

                # update illumination to currently selected settings
                illumination = settings[layer]['ill']

                # Dark field capture
                if lumaview.scope.led:
                    lumaview.scope.leds_off()
                    logger.info('[LVP Main  ] lumaview.scope.leds_off()')
                else:
                    logger.warning('LED controller not available.')

                # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
                time.sleep(2*exposure/1000+0.2)
                scope_display.update_scopedisplay() # Why?
                darkfield = lumaview.scope.get_image()

                # Florescent capture
                if lumaview.scope.led:
                    lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)
                    logger.info('[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)')
                else:
                    logger.warning('LED controller not available.')

                # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
                time.sleep(2*exposure/1000+0.2)
                exposed = lumaview.scope.get_image()

                scope_display.update_scopedisplay() # Why?
                corrected = exposed - np.minimum(exposed,darkfield)
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

            if lumaview.scope.led:
                lumaview.scope.leds_off()
                logger.info('[LVP Main  ] lumaview.scope.leds_off()')
            else:
                logger.warning('LED controller not available.')

            # turn off all LED toggle buttons and histograms
            lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
            Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        # lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
        lumaview.ids['composite_btn'].state = 'normal'

        img = np.flip(img, 0)

        save_folder = settings['live_folder']
        file_root = 'composite_'

        # append = str(int(round(time.time() * 1000)))
        well_label = self.get_well_label()
        append = f'{well_label}'

        # generate filename and save path string
        initial_id = '_000001'
        filename =  file_root + append + initial_id + '.tiff'
        path = save_folder + '/' + filename

        # Obtain next save path if current directory already exists
        while os.path.exists(path):
            path = lumaview.scope.get_next_save_path(path)

        cv2.imwrite(path, img.astype(np.uint8))

# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(CompositeCapture): # i.e. global lumaview
    
    def __init__(self, **kwargs):
        super(MainDisplay,self).__init__(**kwargs)
        self.scope = lumascope_api.Lumascope()

    def cam_toggle(self):
        logger.info('[LVP Main  ] MainDisplay.cam_toggle()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.scope.camera.active == False:
            return

        if scope_display.play == True:
            scope_display.play = False
            if self.scope.led:
                self.scope.leds_off()
                logger.info('[LVP Main  ] self.scope.leds_off()')
            scope_display.stop()
        else:
            scope_display.play = True
            scope_display.start()

    def record(self):
        logger.info('[LVP Main  ] MainDisplay.record()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.scope.camera.active == False:
            return
        scope_display.record = not scope_display.record

    def fit_image(self):
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if self.scope.camera.active == False:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        logger.info('[LVP Main  ] MainDisplay.one2one_image()')
        if self.scope.camera.active == False:
            return
        w = self.width
        h = self.height
        scale_hor = float(lumaview.scope.get_width()) / float(w)
        scale_ver = float(lumaview.scope.get_height()) / float(h)
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
        logger.info('[LVP Main  ] ShaderViewer.__init__()')
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        self.white = 1.
        self.black = 0.

    def on_touch_down(self, touch):
        logger.info('[LVP Main  ] ShaderViewer.on_touch_down()')
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 100:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 1:
                    self.scale = max(1, self.scale * 0.8)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ShaderViewer, self).on_touch_down(touch)

    def update_shader(self, false_color='BF'):
        # logger.info('[LVP Main  ] ShaderViewer.update_shader()')

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
        logger.info('[LVP Main  ] MotionSettings.__init__()')

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        logger.info('[LVP Main  ] MotionSettings.toggle_settings()')
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
        logger.info('[LVP Main  ] MotionSettings.check_settings()')
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

class PostProcessingAccordion(BoxLayout):

    def __init__(self, **kwargs):
        super(PostProcessingAccordion,self).__init__(**kwargs)
        self.post = post_processing.PostProcessing()

    def convert_to_avi(self):
        self.post.convert_to_avi('')

    def stitch(self):
        logger.debug('[LVP Main  ] PostProcessing.stitch() not yet implemented')
        self.post.stitch('')

    def open_folder(self):
        logger.debug('[LVP Main  ] PostProcessing.open_folder() not yet implemented')

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
        logger.info('[LVP Main  ] ShaderEditor.__init__()')
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        logger.info('[LVP Main  ] ShaderEditor.compile_shaders()')
        if not self.viewer:
            logger.warning('[LVP Main  ] ShaderEditor.compile_shaders() Fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        self.viewer.fs = fs
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        logger.info('[LVP Main  ] ShaderEditor.toggle_editor()')
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

class ImageSettings(BoxLayout):
    settings_width = dp(300)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('[LVP Main  ] ImageSettings.__init__()')

    # Hide (and unhide) main settings
    def toggle_settings(self):
        self.update_transmitted()
        logger.info('[LVP Main  ] ImageSettings.toggle_settings()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()

        # move position of settings and stop histogram if main settings are collapsed
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.pos = lumaview.width - 30, 0

            layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
            for layer in layers:
                Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
                logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')
        else:
            self.pos = lumaview.width - self.settings_width, 0
 
        if scope_display.play == True:
            scope_display.start()

    def update_transmitted(self):
        layers = ['BF', 'PC', 'EP']
        for layer in layers:
            accordion = layer + '_accordion'

            # Remove 'Colorize' option in transmitted channels control
            # -----------------------------------------------------
            self.ids[layer].ids['false_color_label'].text = ''
            self.ids[layer].ids['false_color'].color = (0., )*4

            # Adjust 'Illumination' range
            self.ids[layer].ids['ill_slider'].max = 50

    def accordion_collapse(self):
        logger.info('[LVP Main  ] ImageSettings.accordion_collapse()')
        global lumaview

        # turn off the camera update and all LEDs
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        if lumaview.scope.led:
            lumaview.scope.leds_off()
            logger.info('[LVP Main  ] lumaview.scope.leds_off()')
        else:
            logger.warning('LED controller not available.')

        # turn off all LED toggle buttons and histograms
        layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
        for layer in layers:
            lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'
            Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                if lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active:
                    lumaview.ids['viewer_id'].update_shader(false_color=layer)
                else:
                    lumaview.ids['viewer_id'].update_shader(false_color='BF')

        # Restart camera feed
        if scope_display.play == True:
            scope_display.start()

    def check_settings(self, *args):
        logger.info('[LVP Main  ] ImageSettings.check_settings()')
        global lumaview
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.pos = lumaview.width - 30, 0
        else:
            self.pos = lumaview.width - self.settings_width, 0

class Histogram(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        logger.info('[LVP Main  ] Histogram.__init__()')
        if self.bg_color is None:
            self.bg_color = (1, 1, 1, 1)

        self.hist_range_set = False
        self.edges = [0,255]
        self.stablize = 0.3

    def histogram(self, *args):
        # logger.info('[LVP Main  ] Histogram.histogram()')
        global lumaview

        if lumaview.scope.camera != False:
            image = lumaview.scope.get_image()
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
                logHistogram = lumaview.ids['imagesettings_id'].ids[self.layer].ids['logHistogram_id'].active
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

    def __init__(self, **kwargs):
        super(VerticalControl, self).__init__(**kwargs)
        logger.info('[LVP Main  ] VerticalControl.__init__()')

        # boolean describing whether the scope is currently in the process of autofocus
        self.is_autofocus = False
        self.is_complete = False
        
    def update_gui(self):
        logger.info('[LVP Main  ] VerticalControl.update_gui()')
        try:
            set_pos = lumaview.scope.get_target_position('Z')  # Get target value
        except:
            logger.warning('[LVP Main  ] Error talking to Motor board.')

        self.ids['obj_position'].value = max(0, set_pos)
        self.ids['z_position_id'].text = format(max(0, set_pos), '.2f')

    def coarse_up(self):
        logger.info('[LVP Main  ] VerticalControl.coarse_up()')
        coarse = settings['objective']['z_coarse']
        lumaview.scope.move_relative_position('Z', coarse)                  # Move UP
        self.update_gui()

    def fine_up(self):
        logger.info('[LVP Main  ] VerticalControl.fine_up()')
        fine = settings['objective']['z_fine']
        lumaview.scope.move_relative_position('Z', fine)                    # Move UP
        self.update_gui()

    def fine_down(self):
        logger.info('[LVP Main  ] VerticalControl.fine_down()')
        fine = settings['objective']['z_fine']
        lumaview.scope.move_relative_position('Z', -fine)                   # Move DOWN
        self.update_gui()

    def coarse_down(self):
        logger.info('[LVP Main  ] VerticalControl.coarse_down()')
        coarse = settings['objective']['z_coarse']
        lumaview.scope.move_relative_position('Z', -coarse)                 # Move DOWN
        self.update_gui()

    def set_position(self, pos):
        logger.info('[LVP Main  ] VerticalControl.set_position()')
        lumaview.scope.move_absolute_position('Z', float(pos))
        self.update_gui()

    def set_bookmark(self):
        logger.info('[LVP Main  ] VerticalControl.set_bookmark()')
        height = lumaview.scope.get_current_position('Z')  # Get current z height in um
        settings['bookmark']['z'] = height

    def set_all_bookmarks(self):
        logger.info('[LVP Main  ] VerticalControl.set_all_bookmarks()')
        height = lumaview.scope.get_current_position('Z')  # Get current z height in um
        settings['bookmark']['z'] = height
        settings['BF']['focus'] = height
        settings['PC']['focus'] = height
        settings['EP']['focus'] = height
        settings['Blue']['focus'] = height
        settings['Green']['focus'] = height
        settings['Red']['focus'] = height

    def goto_bookmark(self):
        logger.info('[LVP Main  ] VerticalControl.goto_bookmark()')
        pos = settings['bookmark']['z']
        lumaview.scope.move_absolute_position('Z', pos)
        self.update_gui()

    def home(self):
        logger.info('[LVP Main  ] VerticalControl.home()')
        lumaview.scope.zhome()
        self.update_gui()

    # User selected the autofocus function
    def autofocus(self):
        logger.info('[LVP Main  ] VerticalControl.autofocus()')

        global lumaview
        self.is_autofocus = True
        self.is_complete = False
        
        if lumaview.scope.camera.active == False:
            logger.warning('[LVP Main  ] Error: VerticalControl.autofocus()')

            self.ids['autofocus_id'].state == 'normal'
            self.is_autofocus = False

            return

        center = lumaview.scope.get_current_position('Z')
        range =  settings['objective']['AF_range']

        self.z_min = max(0, center-range)                   # starting minimum z-height for autofocus
        self.z_max = center+range                           # starting maximum z-height for autofocus
        self.resolution = settings['objective']['AF_max']   # starting step size for autofocus
        self.exposure = lumaview.scope.get_exposure_time()  # camera exposure to determine 'wait' time

        self.positions = []       # List of positions to step through
        self.focus_measures = []  # Measure focus score at each position
        self.last_focus = 0       # Last / Previous focus score
        self.last = False         # Are we on the last scan for autofocus?

        # set button text if button is pressed
        if self.ids['autofocus_id'].state == 'down':
            self.ids['autofocus_id'].text = 'Focusing...'
            self.is_autofocus = True

            # Start the autofocus process at z-minimum
            lumaview.scope.move_absolute_position('Z', self.z_min)

            # schedule focus iterate
            logger.info('[LVP Main  ] Clock.schedule_interval(self.focus_iterate, 0.01)')
            Clock.schedule_interval(self.focus_iterate, 0.01)

    def focus_iterate(self, dt):

        logger.info('[LVP Main  ] VerticalControl.focus_iterate()')
        global lumaview

        # If the z-height has reached its target
        if lumaview.scope.get_target_status('Z') and not lumaview.scope.get_overshoot():
        # if lumaview.scope.get_target_status('Z'):

            # Wait two exposure lengths
            time.sleep(2*self.exposure/1000+0.2) # TODO: msec into sec

            # observe the image 
            image = lumaview.scope.get_image()
            rows, cols = image.shape

            # Use center quarter of image for focusing
            image = image[int(rows/4):int(3*rows/4),int(cols/4):int(3*cols/4)]

            # calculate the position and focus measure
            try:
                current = lumaview.scope.get_current_position('Z')
                focus = self.focus_function(image)
                next_target = lumaview.scope.get_target_position('Z') + self.resolution
            except:
                logger.warning('[LVP Main  ] Error talking to motion controller.')
                raise

            # append to positions and focus measures
            self.positions.append(current)
            self.focus_measures.append(focus)

            # if (focus < self.last_focus) or (next_target > self.z_max):
            if next_target > self.z_max:

                # Calculate new step size for resolution
                AF_min = settings['objective']['AF_min']
                prev_resolution = self.resolution
                self.resolution = prev_resolution / 3 # SELECT DESIRED RESOLUTION FRACTION

                if self.resolution < AF_min:
                    self.resolution = AF_min

                # As long as the step size is larger than or equal to the minimum and not the last pass
                if self.resolution >= AF_min and not self.last:

                    # compute best focus
                    focus = self.focus_best(self.positions, self.focus_measures)

                    # assign new z_min, z_max, resolution, and sweep
                    self.z_min = focus-prev_resolution 
                    self.z_max = focus+prev_resolution 

                    # reset positions and focus measures
                    self.positions = []
                    self.focus_measures = []

                    # go to new z_min
                    lumaview.scope.move_absolute_position('Z', self.z_min)

                    if self.resolution == AF_min:
                        self.last = True

                else:
                    # compute best focus
                    focus = self.focus_best(self.positions, self.focus_measures)

                    # go to best focus
                    lumaview.scope.move_absolute_position('Z', focus) # move to absolute target

                    # end autofocus sequence
                    logger.info('[LVP Main  ] Clock.unschedule(self.focus_iterate)')
                    Clock.unschedule(self.focus_iterate)

                    # update button status
                    self.ids['autofocus_id'].state = 'normal'
                    self.ids['autofocus_id'].text = 'Autofocus'
                    self.is_autofocus = False
                    self.is_complete = True

            else:
                # move to next position
                lumaview.scope.move_relative_position('Z', self.resolution)

            # update last focus
            self.last_focus = focus

        # In case user cancels autofocus, end autofocus sequence
        if self.ids['autofocus_id'].state == 'normal':
            self.ids['autofocus_id'].text = 'Autofocus'
            self.is_autofocus = False

            logger.info('[LVP Main  ] Clock.unschedule(self.focus_iterate)')
            Clock.unschedule(self.focus_iterate)

        self.update_gui()

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        logger.info('[LVP Main  ] VerticalControl.focus_function()')
        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            logger.info('[LVP Main  ] Focus Score Vollath: ' + str(sum_one - sum_two))
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            logger.info('[LVP Main  ] Focus Score Skew: ' + str(skew))
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            logger.info('[LVP Main  ] Focus Score Pixel Variation: ' + str(var))
            return var
        
            '''
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
            logger.info('[LVP Main  ] kernel\t' + str(kernel))
            convolve = signal.convolve2d(image, kernel, mode='valid')
            sum = np.sum(convolve)
            logger.info('[LVP Main  ] sum\t' + str(sum))
            return sum
            '''
        else:
            return 0

    def focus_best(self, positions, values, algorithm='direct'):
        logger.info('[LVP Main  ] VerticalControl.focus_best()')
        if algorithm == 'direct':
            max_value = max(values)
            max_index = values.index(max_value)
            focus_log(positions, values)
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

    def turret_left(self):
        lumaview.scope.turret_bias -= 1
        angle = 90*lumaview.scope.turret_id + lumaview.scope.turret_bias
        lumaview.scope.tmove(angle)
        
    def turret_right(self):
        lumaview.scope.turret_bias += 1
        angle = 90*lumaview.scope.turret_id + lumaview.scope.turret_bias
        lumaview.scope.tmove(angle)
    
    def turret_home(self):
        lumaview.scope.turret_bias = 0
        lumaview.scope.thome()
        self.ids['turret_pos_1_btn'].state = 'normal'
        self.ids['turret_pos_2_btn'].state = 'normal'
        self.ids['turret_pos_3_btn'].state = 'normal'
        self.ids['turret_pos_4_btn'].state = 'normal'

    def turret_select(self, position):
        #TODO check if turret has been HOMED turret first
        lumaview.scope.turret_id = int(position) - 1
        angle = 90*lumaview.scope.turret_id #+ lumaview.scope.turret_bias
        lumaview.scope.tmove(angle)
        
        #self.ids['turret_pos_1_btn'].state = 'normal'
        #self.ids['turret_pos_2_btn'].state = 'normal'
        #self.ids['turret_pos_3_btn'].state = 'normal'
        #self.ids['turret_pos_4_btn'].state = 'normal'
        #self.ids[f'turret_pos_{position}_btn'].state = 'down'
                    
        if position == '1':
            self.ids['turret_pos_1_btn'].state = 'down'
            self.ids['turret_pos_2_btn'].state = 'normal'
            self.ids['turret_pos_3_btn'].state = 'normal'
            self.ids['turret_pos_4_btn'].state = 'normal'

        elif position == '2':
            self.ids['turret_pos_1_btn'].state = 'normal'
            self.ids['turret_pos_2_btn'].state = 'down'
            self.ids['turret_pos_3_btn'].state = 'normal'
            self.ids['turret_pos_4_btn'].state = 'normal'

        elif position == '3':
            self.ids['turret_pos_1_btn'].state = 'normal'
            self.ids['turret_pos_2_btn'].state = 'normal'
            self.ids['turret_pos_3_btn'].state = 'down'
            self.ids['turret_pos_4_btn'].state = 'normal'

        elif position == '4':
            self.ids['turret_pos_1_btn'].state = 'normal'
            self.ids['turret_pos_2_btn'].state = 'normal'
            self.ids['turret_pos_3_btn'].state = 'normal'
            self.ids['turret_pos_4_btn'].state = 'down'


class XYStageControl(BoxLayout):

    def update_gui(self, dt=0):
        # logger.info('[LVP Main  ] XYStageControl.update_gui()')
        global lumaview
        try:
            x_target = lumaview.scope.get_target_position('X')  # Get target value in um
            x_target = np.clip(x_target, 0, 120000) # prevents crosshairs from leaving the stage area
            y_target = lumaview.scope.get_target_position('Y')  # Get target value in um
            y_target = np.clip(y_target, 0, 80000) # prevents crosshairs from leaving the stage area
        except:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
        else:
            # Convert from plate position to stage position
            protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
            stage_x, stage_y =  protocol_settings.stage_to_plate(x_target, y_target)
            self.ids['x_pos_id'].text = format(max(0, stage_x), '.2f') # display coordinate in mm
            self.ids['y_pos_id'].text = format(max(0, stage_y), '.2f') # display coordinate in mm
            self.ids['stage_control_id'].draw_labware()

    def fine_left(self):
        logger.info('[LVP Main  ] XYStageControl.fine_left()')
        fine = settings['objective']['xy_fine']
        lumaview.scope.move_relative_position('X', -fine)  # Move LEFT fine step
        self.update_gui()

    def fine_right(self):
        logger.info('[LVP Main  ] XYStageControl.fine_right()')
        fine = settings['objective']['xy_fine']
        lumaview.scope.move_relative_position('X', fine)  # Move RIGHT fine step
        self.update_gui()

    def coarse_left(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_left()')
        coarse = settings['objective']['xy_coarse']
        lumaview.scope.move_relative_position('X', -coarse)  # Move LEFT coarse step
        self.update_gui()

    def coarse_right(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_right()')
        coarse = settings['objective']['xy_coarse']
        lumaview.scope.move_relative_position('X', coarse)  # Move RIGHT
        self.update_gui()

    def fine_back(self):
        logger.info('[LVP Main  ] XYStageControl.fine_back()')
        fine = settings['objective']['xy_fine']
        lumaview.scope.move_relative_position('Y', -fine)  # Move BACK 
        self.update_gui()

    def fine_fwd(self):
        logger.info('[LVP Main  ] XYStageControl.fine_fwd()')
        fine = settings['objective']['xy_fine']
        lumaview.scope.move_relative_position('Y', fine)  # Move FORWARD 
        self.update_gui()

    def coarse_back(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_back()')
        coarse = settings['objective']['xy_coarse']
        lumaview.scope.move_relative_position('Y', -coarse)  # Move BACK
        self.update_gui()

    def coarse_fwd(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_fwd()')
        coarse = settings['objective']['xy_coarse']
        lumaview.scope.move_relative_position('Y', coarse)  # Move FORWARD 
        self.update_gui()

    def set_xposition(self, x_pos):
        logger.info('[LVP Main  ] XYStageControl.set_xposition()')
        global lumaview

        # x_pos is the the plate position in mm
        # Find the coordinates for the stage
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        stage_x, stage_y =  protocol_settings.plate_to_stage(float(x_pos), 0)
        logger.info('[LVP Main  ] X pos', x_pos, 'Stage X', stage_x)

        # Move to x-position
        lumaview.scope.move_absolute_position('X', stage_x)  # position in text is in mm
        self.update_gui()

    def set_yposition(self, y_pos):
        logger.info('[LVP Main  ] XYStageControl.set_yposition()')
        global lumaview

        # y_pos is the the plate position in mm
        # Find the coordinates for the stage
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        stage_x, stage_y =  protocol_settings.plate_to_stage(0, float(y_pos))

        # Move to y-position
        lumaview.scope.move_absolute_position('Y', stage_y)  # position in text is in mm
        self.update_gui()

    def set_xbookmark(self):
        logger.info('[LVP Main  ] XYStageControl.set_xbookmark()')
        global lumaview

        # Get current stage x-position in um     
        x_pos = lumaview.scope.get_current_position('X')
 
        # Save plate x-position to settings
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        plate_x, plate_y =  protocol_settings.stage_to_plate(x_pos, 0)
        settings['bookmark']['x'] = plate_x

    def set_ybookmark(self):
        logger.info('[LVP Main  ] XYStageControl.set_ybookmark()')
        global lumaview

        # Get current stage y-position in um
        y_pos = lumaview.scope.get_current_position('Y')  

        # Save plate y-position to settings
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        plate_x, plate_y =  protocol_settings.stage_to_plate(0, y_pos)
        settings['bookmark']['y'] = plate_y

    def goto_xbookmark(self):
        logger.info('[LVP Main  ] XYStageControl.goto_xbookmark()')
        global lumaview

        # Get bookmark plate x-position in mm
        x_pos = settings['bookmark']['x']

        # Move to x-position
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        stage_x, stage_y =  protocol_settings.plate_to_stage(x_pos, 0)
        lumaview.scope.move_absolute_position('X', stage_x)  # set current x position in um

    def goto_ybookmark(self):
        logger.info('[LVP Main  ] XYStageControl.goto_ybookmark()')
        global lumaview

        # Get bookmark plate y-position in mm
        y_pos = settings['bookmark']['y']

        # Move to y-position
        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        stage_x, stage_y =  protocol_settings.plate_to_stage(0, y_pos)
        lumaview.scope.move_absolute_position('Y', stage_y)  # set current y position in um

    # def calibrate(self):
    #     logger.info('[LVP Main  ] XYStageControl.calibrate()')
    #     global lumaview
    #     x_pos = lumaview.scope.get_current_position('X')  # Get current x position in um
    #     y_pos = lumaview.scope.get_current_position('Y')  # Get current x position in um

    #     current_labware = WellPlate()
    #     current_labware.load_plate(settings['protocol']['labware'])
    #     x_plate_offset = current_labware.plate['offset']['x']*1000
    #     y_plate_offset = current_labware.plate['offset']['y']*1000

    #     settings['stage_offset']['x'] = x_plate_offset-x_pos
    #     settings['stage_offset']['y'] = y_plate_offset-y_pos
    #     self.update_gui()

    def home(self):
        logger.info('[LVP Main  ] XYStageControl.home()')
        global lumaview

        if lumaview.scope.motion.driver: # motor controller is actively connected
            lumaview.scope.xyhome()
            # TODO: update GUI, 
            
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

# Protocol settings tab
class ProtocolSettings(CompositeCapture):
    global settings

    def __init__(self, **kwargs):

        super(ProtocolSettings, self).__init__(**kwargs)
        logger.info('[LVP Main  ] ProtocolSettings.__init__()')

        # Load all Possible Labware from JSON
        os.chdir(source_path)
        try:
            read_file = open('./data/labware.json', "r")
        except:
            logger.exception("[LVP Main  ] Error reading labware definition file 'data/labware.json'")
            if not os.path.isdir('./data'):
                raise FileNotFoundError("Couldn't find 'data' directory.")
            else:
                raise
            #self.labware = False
        else:
            self.labware = json.load(read_file)
            read_file.close()

        self.step_names = list()
        self.step_values = []
        self.c_step = 0

    # Update Protocol Period   
    def update_period(self):
        logger.info('[LVP Main  ] ProtocolSettings.update_period()')
        try:
            settings['protocol']['period'] = float(self.ids['capture_period'].text)
        except:
            logger.exception('[LVP Main  ] Update Period is not an acceptable value')

    # Update Protocol Duration   
    def update_duration(self):
        logger.info('[LVP Main  ] ProtocolSettings.update_duration()')
        try:
            settings['protocol']['duration'] = float(self.ids['capture_dur'].text)
        except:
            logger.warning('[LVP Main  ] Update Duration is not an acceptable value')

    # Labware Selection
    def select_labware(self):
        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        spinner = self.ids['labware_spinner']
        spinner.values = list(self.labware['Wellplate'].keys())
        settings['protocol']['labware'] = spinner.text
    
    def plate_to_stage(self, px, py):
        # plate coordinates in mm from top left
        # stage coordinates in um from bottom right

        # Determine current labware
        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])

        # Get labware dimensions
        x_max = current_labware.plate['dimensions']['x'] # in mm
        y_max = current_labware.plate['dimensions']['y'] # in mm

        # Convert coordinates
        sx = x_max - settings['stage_offset']['x']/1000 - px
        sy = y_max - settings['stage_offset']['y']/1000 - py

        sx = sx*1000
        sy = sy*1000

        # return
        return sx, sy
    
    def stage_to_plate(self, sx, sy):
        # stage coordinates in um from bottom right
        # plate coordinates in mm from top left

        # Determine current labware
        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])

        # Get labware dimensions
        x_max = current_labware.plate['dimensions']['x']
        y_max = current_labware.plate['dimensions']['y']

        # Convert coordinates
        px = x_max - (settings['stage_offset']['x'] + sx)/1000
        py = y_max - (settings['stage_offset']['y'] + sy)/1000
 
        return px, py
    
    def plate_to_pixel(self, px, py, scale_x, scale_y):
        # plate coordinates in mm from top left
        # pixel coordinates in px from bottom left

        # Determine current labware
        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])

        # Get labware dimensions
        x_max = current_labware.plate['dimensions']['x']
        y_max = current_labware.plate['dimensions']['y']

        # Convert coordinates
        pixel_x = px*scale_x
        pixel_y = (y_max-py)*scale_y

        return pixel_x, pixel_y

    def stage_to_pixel(self, sx, sy, scale_x, scale_y):
        # stage coordinates in um from bottom right
        # plate coordinates in mm from top left
        # pixel coordinates in px from bottom left

        px, py = self.stage_to_plate(sx, sy)
        pixel_x, pixel_y = self.plate_to_pixel(px, py, scale_x, scale_y)

        return pixel_x, pixel_y
        


    # Create New Protocol
    def new_protocol(self):
        logger.info('[LVP Main  ] ProtocolSettings.new_protocol()')

        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])
        current_labware.set_positions()
        
        self.step_names = list()
        self.step_values = []

         # Iterate through all the positions in the scan
        for pos in current_labware.pos_list:

            # Iterate through all the colors to create the steps
            layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
            for layer in layers:
                if settings[layer]['acquire'] == True:

                    x = pos[0] # in 'plate' coordinates
                    y = pos[1] # in 'plate' coordinates
                    z = settings[layer]['focus']
                    af = settings[layer]['autofocus']
                    ch = lumaview.scope.color2ch(layer)
                    fc = settings[layer]['false_color']
                    ill = settings[layer]['ill']
                    gain = settings[layer]['gain']
                    auto_gain = int(settings[layer]['auto_gain'])
                    exp = settings[layer]['exp']

                    self.step_values.append([x, y, z, af, ch, fc, ill, gain, auto_gain, exp])

        self.step_values = np.array(self.step_values)

        # Number of Steps
        length =  self.step_values.shape[0]
              
        # Update text with current step and number of steps in protocol
        self.c_step = -1
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.ids['step_total_input'].text = str(length)

        # self.step_names = [self.ids['step_name_input'].text] * length
        self.step_names = [''] * length
        settings['protocol']['filepath'] = ''        
        self.ids['protocol_filename'].text = ''

    # Load Protocol from File
    def load_protocol(self, filepath="./data/example_protocol.tsv"):
        logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

        # Load protocol
        file_pointer = open(filepath, 'r')                      # open the file
        csvreader = csv.reader(file_pointer, delimiter='\t') # access the file using the CSV library
        verify = next(csvreader)
        if not (verify[0] == 'LumaViewPro Protocol'):
            return
        period = next(csvreader)
        period = float(period[1])
        duration = next(csvreader)
        duration = float(duration[1])
        labware = next(csvreader)
        labware = labware[1]
        header = next(csvreader) # skip a line

        self.step_names = list()
        self.step_values = []

        for row in csvreader:
            self.step_names.append(row[0])
            self.step_values.append(row[1:])

        file_pointer.close()
        self.step_values = np.array(self.step_values)
        self.step_values = self.step_values.astype(float)

        settings['protocol']['filepath'] = filepath
        self.ids['protocol_filename'].text = os.path.basename(filepath)

        # Update GUI
        self.c_step = -1
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.ids['step_name_input'].text = ''
        self.ids['step_total_input'].text = str(len(self.step_names))
        self.ids['capture_period'].text = str(period)
        self.ids['capture_dur'].text = str(duration)
       
        # Update Protocol
        settings['protocol']['period'] = period
        settings['protocol']['duration'] = duration
        settings['protocol']['labware'] = labware

        # Update Labware Selection in Spinner
        self.ids['labware_spinner'].text = settings['protocol']['labware']
    
    # Save Protocol to File
    def save_protocol(self, filepath=''):
        logger.info('[LVP Main  ] ProtocolSettings.save_protocol()')

        # Gather information
        period = settings['protocol']['period']
        duration = settings['protocol']['duration']
        labware = settings['protocol']['labware'] 
        self.step_names
        self.step_values

        if len(filepath)==0:
            if len(settings['protocol']['filepath']) == 0:
                return
            filepath = settings['protocol']['filepath']
        else:
            settings['protocol']['filepath'] = filepath

        if filepath[-4:].lower() != '.tsv':
            filepath = filepath+'.tsv'

        self.ids['protocol_filename'].text = os.path.basename(filepath)

        # Write a TSV file
        file_pointer = open(filepath, 'w')                      # open the file
        csvwriter = csv.writer(file_pointer, delimiter='\t', lineterminator='\n') # access the file using the CSV library

        csvwriter.writerow(['LumaViewPro Protocol'])
        csvwriter.writerow(['Period', period])
        csvwriter.writerow(['Duration', duration])
        csvwriter.writerow(['Labware', labware])
        csvwriter.writerow(['Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Channel', 'False_Color', 'Illumination', 'Gain', 'Auto_Gain', 'Exposure'])

        for i in range(len(self.step_names)):
            c_name = self.step_names[i]
            c_values = list(self.step_values[i])
            c_values.insert(0, c_name)

            # write the row
            csvwriter.writerow(c_values)

        # Close the TSV file
        file_pointer.close()

    # Goto to Previous Step
    def prev_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        if len(self.step_names) <= 0:
            return
        self.c_step = max(self.c_step - 1, 0)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
 
    # Go to Next Step
    def next_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        if len(self.step_names) <= 0:
            return
        self.c_step = min(self.c_step + 1, len(self.step_names)-1)
        self.ids['step_number_input'].text = str(self.c_step+1)
        self.go_to_step()
    
    # Go to Input Step
    def go_to_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.go_to_step()')

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
        af =        self.step_values[self.c_step, 3]
        ch =        self.step_values[self.c_step, 4]
        fc =        self.step_values[self.c_step, 5]
        ill =       self.step_values[self.c_step, 6]
        gain =      self.step_values[self.c_step, 7]
        auto_gain = self.step_values[self.c_step, 8]
        exp =       self.step_values[self.c_step, 9]

        self.ids['step_name_input'].text = name

        # Convert plate coordinates to stage coordinates
        sx, sy = self.plate_to_stage(x, y)

        # Move into position
        if lumaview.scope.motion.driver:
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', z)
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

        ch = lumaview.scope.ch2color(ch)
        layer  = lumaview.ids['imagesettings_id'].ids[ch]

        # open ImageSettings
        lumaview.ids['imagesettings_id'].ids['toggle_imagesettings'].state = 'down'
        lumaview.ids['imagesettings_id'].toggle_settings()
        
        # set accordion item to corresponding channel
        id = ch + '_accordion'
        lumaview.ids['imagesettings_id'].ids[id].collapse = False

        # set autofocus checkbox
        logger.info('[LVP Main  ] autofocus:  ' + str(bool(af)))
        settings[ch]['autofocus'] = bool(af)
        layer.ids['autofocus'].active = bool(af)
        
        # set false_color checkbox
        logger.info('[LVP Main  ] false_color:' + str(bool(fc)))
        settings[ch]['false_color'] = bool(fc)
        layer.ids['false_color'].active = bool(fc)

        # set illumination settings, text, and slider
        logger.info('[LVP Main  ] ill:        ' + str(ill))
        settings[ch]['ill'] = ill
        layer.ids['ill_text'].text = str(ill)
        layer.ids['ill_slider'].value = float(ill)

        # set gain settings, text, and slider
        logger.info('[LVP Main  ] gain:       ' + str(gain))
        settings[ch]['gain'] = gain
        layer.ids['gain_text'].text = str(gain)
        layer.ids['gain_slider'].value = float(gain)

        # set auto_gain checkbox
        logger.info('[LVP Main  ] auto_gain:  ' + str(auto_gain))
        settings[ch]['auto_gain'] = bool(auto_gain)
        layer.ids['auto_gain'].active = bool(auto_gain)

        # set exposure settings, text, and slider
        logger.info('[LVP Main  ] exp:        ' + str(exp))
        settings[ch]['exp'] = exp
        layer.ids['exp_text'].text = str(exp)
        layer.ids['exp_slider'].value = float(exp)

        # update position in stage control
        lumaview.ids['motionsettings_id'].ids['xy_stagecontrol_id'].update_gui()


    # Delete Current Step of Protocol
    def delete_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.delete_step()')

        if len(self.step_names) < 1:
            return
        
        self.step_names.pop(self.c_step)
        self.step_values = np.delete(self.step_values, self.c_step, axis = 0)
        self.c_step = self.c_step - 1

        # Update total number of steps to GUI
        self.ids['step_total_input'].text = str(len(self.step_names))
        self.next_step()

    # Modify Current Step of Protocol
    def modify_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if len(self.step_names) < 1:
            return

        self.step_names[self.c_step] = self.ids['step_name_input'].text

        # Determine and update plate position
        if lumaview.scope.motion.driver:
            sx = lumaview.scope.get_current_position('X')
            sy = lumaview.scope.get_current_position('Y')
            px, py = self.stage_to_plate(sx, sy)

            self.step_values[self.c_step, 0] = px # x
            self.step_values[self.c_step, 1] = py # y
            self.step_values[self.c_step, 2] = lumaview.scope.get_current_position('Z')      # z
        else:
            logger.warning('[LVP Main  ] Motion controller not availabble.')

        c_layer = False

        layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
        for layer in layers:
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                c_layer = layer

        if c_layer == False:
            mssg = "No layer currently selected"
            return mssg

        ch = lumaview.scope.color2ch(c_layer)

        layer_id = lumaview.ids['imagesettings_id'].ids[c_layer]
        self.step_values[self.c_step, 3] = int(layer_id.ids['autofocus'].active) # autofocus
        self.step_values[self.c_step, 4] = ch # channel
        self.step_values[self.c_step, 5] = int(layer_id.ids['false_color'].active) # false color
        self.step_values[self.c_step, 6] = layer_id.ids['ill_slider'].value # ill
        self.step_values[self.c_step, 7] = layer_id.ids['gain_slider'].value # gain
        self.step_values[self.c_step, 8] = int(layer_id.ids['auto_gain'].active) # auto_gain
        self.step_values[self.c_step, 9] = layer_id.ids['exp_slider'].value # exp

    # Insert Current Step to Protocol at Current Position
    def add_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.add_step()')

         # Determine Values
        name = self.ids['step_name_input'].text
        c_layer = False

        layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
        for layer in layers:
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                c_layer = layer

        if c_layer == False:
            mssg = "No layer currently selected"
            return mssg

        ch = lumaview.scope.color2ch(c_layer)
        layer_id = lumaview.ids['imagesettings_id'].ids[c_layer]

        # Determine and update plate position
        sx = lumaview.scope.get_current_position('X')
        sy = lumaview.scope.get_current_position('Y')
        px, py = self.stage_to_plate(sx, sy)

        step = [px,                                      # x
                py,                                      # y
                lumaview.scope.get_current_position('Z'),# z
                int(layer_id.ids['autofocus'].active),   # autofocus
                ch,                                      # ch 
                int(layer_id.ids['false_color'].active), # false color
                layer_id.ids['ill_slider'].value,        # ill
                layer_id.ids['gain_slider'].value,       # gain
                int(layer_id.ids['auto_gain'].active),   # auto_gain
                layer_id.ids['exp_slider'].value,        # exp
        ]

        # Insert into List and Array
        self.step_names.insert(self.c_step, name)
        self.step_values = np.insert(self.step_values, self.c_step, step, axis=0)

        self.ids['step_total_input'].text = str(len(self.step_names))

        
    # Run one scan of protocol, autofocus at each step, and update protocol
    def run_autofocus_scan(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_autofocus_scan()')

        # If there are no steps, do not continue
        if len(self.step_names) < 1:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
            return

        # If the toggle button is in the down position: start autofocus scan
        if self.ids['run_autofocus_btn'].state == 'down':
            self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # begin at current step (c_step = 0)
            self.c_step = 0
            self.ids['step_number_input'].text = str(self.c_step+1)

            x = self.step_values[self.c_step, 0]
            y = self.step_values[self.c_step, 1]
            z = self.step_values[self.c_step, 2]
 
            # Convert plate coordinates to stage coordinates
            sx, sy = self.plate_to_stage(x, y)

            # Move into position
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', z)

            logger.info('[LVP Main  ] Clock.schedule_interval(self.autofocus_scan_iterate, 0.1)')
            Clock.schedule_interval(self.autofocus_scan_iterate, 0.1)

        # If the toggle button is in the up position: Stop Running Autofocus Scan
        else:  # self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'

            # toggle all LEDs AND TOGGLE BUTTONS OFF
            if lumaview.scope.led:
                lumaview.scope.leds_off()
                logger.info('[LVP Main  ] lumaview.scope.leds_off()')
            else:
                logger.warning('[LVP Main  ] LED controller not available.')

            layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

            logger.info('[LVP Main  ] Clock.unschedule(self.autofocus_scan_iterate)')
            Clock.unschedule(self.autofocus_scan_iterate) # unschedule all copies of autofocus scan iterate
        
    def autofocus_scan_iterate(self, dt):
        global lumaview
        global settings

        # If the autofocus is currently active, leave the function before continuing step
        is_autofocus = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_autofocus
        if is_autofocus:
            return

        # If the autofocus just completed, go to next steps
        is_complete = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete

        if is_complete:
            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # update protocol to focused z-position
            self.step_values[self.c_step, 2] = lumaview.scope.get_current_position('Z')

            # increment to the next step
            self.c_step += 1

            # determine and go to next positions
            if self.c_step < len(self.step_names):
                # Update Step number text
                self.ids['step_number_input'].text = str(self.c_step+1)
                self.go_to_step()

            # if all positions have already been reached
            else:
                logger.info('[LVP Main  ] Autofocus Scan Complete')
                self.ids['run_autofocus_btn'].state = 'normal'
                self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'


                logger.info('[LVP Main  ] Clock.unschedule(self.autofocus_scan_iterate)')
                Clock.unschedule(self.autofocus_scan_iterate) # unschedule all copies of scan iterate
            
            return

        # Check if at desired position 
        x_status = lumaview.scope.get_target_status('X')
        y_status = lumaview.scope.get_target_status('Y')
        z_status = lumaview.scope.get_target_status('Z')

        # If target location has been reached
        if x_status and y_status and z_status and not lumaview.scope.get_overshoot():
            logger.info('[LVP Main  ] Autofocus Scan Step:' + str(self.step_names[self.c_step]) )

            # identify image settings
            ch =        self.step_values[self.c_step, 4] # LED channel
            fc =        self.step_values[self.c_step, 5] # image false color
            ill =       self.step_values[self.c_step, 6] # LED illumination
            gain =      self.step_values[self.c_step, 7] # camera gain
            auto_gain = self.step_values[self.c_step, 8] # camera autogain
            exp =       self.step_values[self.c_step, 9] # camera exposure
            
            # set camera settings and turn on LED
            lumaview.scope.leds_off()
            lumaview.scope.led_on(ch, ill)
            lumaview.scope.set_gain(gain)
            lumaview.scope.set_auto_gain(bool(auto_gain))
            lumaview.scope.set_exposure_time(exp)

            # Begin autofocus routine
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['autofocus_id'].state = 'down'
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].autofocus()
            return
   
    # Run one scan of the protocol
    def run_scan(self, protocol = False):
        logger.info('[LVP Main  ] ProtocolSettings.run_scan()')
 
        # If there are no steps, do not continue
        if len(self.step_names) < 1:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'
            return

        # If the toggle button is in the down position: Start Running Scan
        if self.ids['run_scan_btn'].state == 'down' or protocol == True:
            self.ids['run_scan_btn'].text = 'Running Scan'

            # TODO: shut off live updates

            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # begin at current step set to 0 (c_step = 0)
            self.c_step = 0
            self.ids['step_number_input'].text = str(self.c_step+1)

            x = self.step_values[self.c_step, 0]
            y = self.step_values[self.c_step, 1]
            z = self.step_values[self.c_step, 2]
 
            # Convert plate coordinates to stage coordinates
            sx, sy = self.plate_to_stage(x, y)

            # Move into position
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', z)

            logger.info('[LVP Main  ] Clock.schedule_interval(self.scan_iterate, 0.1)')
            Clock.schedule_interval(self.scan_iterate, 0.1)

        # If the toggle button is in the up position: Stop Running Scan
        else:  # self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'

            # toggle all LEDs AND TOGGLE BUTTONS OFF
            if lumaview.scope.led:
                lumaview.scope.leds_off()
                logger.info('[LVP Main  ] lumaview.scope.leds_off()')
            else:
                logger.warning('[LVP Main  ] LED controller not available.')

            layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
    
    def scan_iterate(self, dt):
        global lumaview
        global settings

        # If the autofocus is currently active, leave the function before continuing step
        is_autofocus = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_autofocus
        if is_autofocus:
            return

        # Identify if an autofocus cycle completed
        is_complete = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete

        # Check if at desired position
        x_status = lumaview.scope.get_target_status('X')
        y_status = lumaview.scope.get_target_status('Y')
        z_status = lumaview.scope.get_target_status('Z')

        # If target location has been reached
        if x_status and y_status and z_status and not lumaview.scope.get_overshoot():
            logger.info('[LVP Main  ] Scan Step:' + str(self.step_names[self.c_step]))

            # identify image settings
            af =        self.step_values[self.c_step, 3] # autofocus
            ch =        self.step_values[self.c_step, 4] # LED channel
            fc =        self.step_values[self.c_step, 5] # image false color
            ill =       self.step_values[self.c_step, 6] # LED illumination
            gain =      self.step_values[self.c_step, 7] # camera gain
            auto_gain = self.step_values[self.c_step, 8] # camera autogain
            exp =       self.step_values[self.c_step, 9] # camera exposure
            
            # Set camera settings
            lumaview.scope.set_gain(gain)
            lumaview.scope.set_auto_gain(bool(auto_gain))
            lumaview.scope.set_exposure_time(exp)
            
            # If the autofocus is selected, is not currently running and has not completed, begin autofocus
            if af and not is_complete:
                # turn on LED
                lumaview.scope.leds_off()
                lumaview.scope.led_on(ch, ill)

                # Begin autofocus routine
                lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['autofocus_id'].state = 'down'
                lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].autofocus()
                
                return
            
            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # capture image
            self.custom_capture(ch, ill, gain, exp, bool(fc))

            # increment to the next step
            self.c_step += 1

            if self.c_step < len(self.step_names):

                # Update Step number text
                self.ids['step_number_input'].text = str(self.c_step+1)
                self.go_to_step()

            # if all positions have already been reached
            else:
                logger.info('[LVP Main  ] Scan Complete')
                self.ids['run_scan_btn'].state = 'normal'
                self.ids['run_scan_btn'].text = 'Run One Scan'

                logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
                Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate

    # Run protocol without xy movement
    def run_stationary(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_stationary()')

        if self.ids['run_stationary_btn'].state == 'down':
            self.ids['run_stationary_btn'].text = 'State == Down'
        else:
            self.ids['run_stationary_btn'].text = 'Run Stationary Protocol' # 'normal'


    # Run the complete protocol 
    def run_protocol(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_protocol()')
        self.n_scans = int(float(settings['protocol']['duration'])*60 / float(settings['protocol']['period']))
        self.start_t = time.time() # start of cycle in seconds

        if self.ids['run_protocol_btn'].state == 'down':
            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.run_scan(protocol = True)
            logger.info('[LVP Main  ] Clock.schedule_interval(self.protocol_iterate, 1)')
            Clock.schedule_interval(self.protocol_iterate, 1)

        else:
            self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            logger.info('[LVP Main  ] Clock.unschedule(self.protocol_iterate)')
            Clock.unschedule(self.protocol_iterate) # unschedule all copies of protocol iterate
            # self.protocol_event.cancel()
 
    def protocol_iterate(self, dt):
        logger.info('[LVP Main  ] ProtocolSettings.protocol_iterate()')

        # Simplified variables
        start_t = self.start_t # start of cycle in seconds
        curr_t = time.time()   # current time in seconds
        n_scans = self.n_scans # current number of scans left
        period = settings['protocol']['period']*60 # length of cycle in seconds

        # compute time remaining
        sec_remaining = n_scans*period - (curr_t - start_t)
        # compute time remaining until next scan
        # sec_remaining = period - (curr_t - start_t)
        min_remaining = sec_remaining / 60
        hrs_remaining = min_remaining / 60

        hrs = np.floor(hrs_remaining)
        minutes = np.floor((hrs_remaining - hrs)*60)

        hrs = '%d' % hrs
        minutes = '%02d' % minutes

        # Update Button
        self.ids['run_protocol_btn'].text = str(n_scans) + ' scans remaining. Press to ABORT'

        # Check if reached next Period
        if (time.time()-self.start_t) > period:

            # reset the start time and update number of scans remaining
            self.start_t = time.time()
            self.n_scans = self.n_scans - 1

            if self.n_scans > 0:
                logger.info('[LVP Main  ] Scans Remaining: ' + str(self.n_scans))
                self.run_scan(protocol = True)
            else:
               self.ids['run_protocol_btn'].state = 'normal' # 'normal'
               self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'

               logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
               Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
               logger.info('[LVP Main  ] Clock.unschedule(self.protocol_iterate)')
               Clock.unschedule(self.protocol_iterate) # unschedule all copies of protocol iterate

# Widget for displaying Microscope Stage area, labware, and current position 
class Stage(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)
        logger.info('[LVP Main  ] Stage.__init__()')

    def on_touch_down(self, touch):
        logger.info('[LVP Main  ] Stage.on_touch_down()')

        if self.collide_point(*touch.pos) and touch.button == 'left':

            # Get mouse position in pixels
            (mouse_x, mouse_y) = touch.pos

            # Convert to relative mouse position in pixels
            mouse_x = mouse_x-self.x
            mouse_y = mouse_y-self.y
           
            # Create current labware instance
            current_labware = WellPlate()
            current_labware.load_plate(settings['protocol']['labware'])

            # Get labware dimensions
            x_max = current_labware.plate['dimensions']['x']
            y_max = current_labware.plate['dimensions']['y']

            # Scale from pixels to mm (from the bottom left)
            scale_x = x_max / self.width
            scale_y = y_max / self.height

            # Convert to plate position in mm (from the top left)
            plate_x = mouse_x*scale_x
            plate_y = y_max- mouse_y*scale_y

            # Convert from plate position to stage position
            stage_x, stage_y =  lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].plate_to_stage(plate_x, plate_y)

            lumaview.scope.move_absolute_position('X', stage_x)
            lumaview.scope.move_absolute_position('Y', stage_y)
            lumaview.ids['motionsettings_id'].ids['xy_stagecontrol_id'].update_gui()
    

    def draw_labware(self, *args): # View the labware from front and above
        # logger.info('[LVP Main  ] Stage.draw_labware()')
        global lumaview
        global settings

        # Create current labware instance
        
        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])

        self.canvas.clear()

        with self.canvas:
            w = self.width
            h = self.height
            x = self.x
            y = self.y

            # Get labware dimensions
            x_max = current_labware.plate['dimensions']['x']
            y_max = current_labware.plate['dimensions']['y']

            # mm to pixels scale
            scale_x = w/x_max
            scale_y = h/y_max

            # Stage Coordinates (120x80 mm)
            stage_w = 120
            stage_h = 80

            stage_x = settings['stage_offset']['x']/1000
            stage_y = settings['stage_offset']['y']/1000

            # Outline of Stage Area from Above
            # ------------------
            
            Color(.2, .2, .2 , 0.5)                # dark grey
            Rectangle(pos=(x+(x_max-stage_w-stage_x)*scale_x, y+stage_y*scale_y),
                           size=(stage_w*scale_x, stage_h*scale_y))

            # Outline of Plate from Above
            # ------------------
            Color(50/255, 164/255, 206/155, 1.)                # kivy aqua
            Line(points=(x, y, x, y+h-15), width = 1)          # Left
            Line(points=(x+w, y, x+w, y+h), width = 1)         # Right
            Line(points=(x, y, x+w, y), width = 1)             # Bottom
            Line(points=(x+15, y+h, x+w, y+h), width = 1)      # Top
            Line(points=(x, y+h-15, x+15, y+h), width = 1)     # Diagonal

            # Draw all wells
            # ------------------
            cols = current_labware.plate['columns']
            rows = current_labware.plate['rows']

            Color(0.4, 0.4, 0.4, 0.5)
            rx = current_labware.plate['spacing']['x']
            ry = current_labware.plate['spacing']['y']
            for i in range(cols):
                for j in range(rows):
                    #  THIS ONE
                    well_x, well_y = current_labware.get_well_position(i, j)
                    x_center = int(x+well_x*scale_x) # on screen center
                    y_center = int(y+well_y*scale_y) # on screen center
                    Ellipse(pos=(x_center-rx, y_center-ry), size=(rx*2, ry*2))

            # Green Circle
            # ------------------
            protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']

            # Get target position
            try:
                x_target = lumaview.scope.get_target_position('X')
                y_target = lumaview.scope.get_target_position('Y')
            except:
                logger.exception('[LVP Main  ] Error talking to Motor board.')
                raise
                
            x_target, y_target = protocol_settings.stage_to_plate(x_target, y_target)

            i, j = current_labware.get_well_index(x_target, y_target)
            well_x, well_y = current_labware.get_well_position(i, j)

            # Convert plate coordinates to relative pixel coordinates
            sx, sy = protocol_settings.plate_to_pixel(well_x, well_y, scale_x, scale_y)

            Color(0., 1., 0., 1.)
            Line(circle=(x+sx, y+sy, rx))
            
            #  Red Crosshairs
            # ------------------
            x_current = lumaview.scope.get_current_position('X')
            x_current = np.clip(x_current, 0, 120000) # prevents crosshairs from leaving the stage area
            y_current = lumaview.scope.get_current_position('Y')
            y_current = np.clip(y_current, 0, 80000) # prevents crosshairs from leaving the stage area

            # Convert stage coordinates to relative pixel coordinates
            pixel_x, pixel_y = protocol_settings.stage_to_pixel(x_current, y_current, scale_x, scale_y)
            Color(1., 0., 0., 1.)
            Line(points=(x+pixel_x-10, y+pixel_y, x+pixel_x+10, y+pixel_y), width = 1) # horizontal line
            Line(points=(x+pixel_x, y+pixel_y-10, x+pixel_x, y+pixel_y+10), width = 1) # vertical line


class MicroscopeSettings(BoxLayout):

    def __init__(self, **kwargs):
        super(MicroscopeSettings, self).__init__(**kwargs)
        logger.info('[LVP Main  ] MicroscopeSettings.__init__()')

        try:
            os.chdir(source_path)
            with open('./data/scopes.json', "r") as read_file:
                self.scopes = json.load(read_file)
        except:
            logger.exception('[LVP Main  ] Unable to read scopes.json.')
            raise

        try:
            os.chdir(source_path)
            with open('./data/objectives.json', "r") as read_file:
                self.objectives = json.load(read_file)
        except:
            logger.exception('[LVP Main  ] Unable to open objectives.json.')
            raise

    # load settings from JSON file
    def load_settings(self, filename="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.load_settings()')
        global lumaview
        global settings

        # load settings JSON file
        try:
            os.chdir(source_path)
            read_file = open(filename, "r")
        except:
            logger.exception('[LVP Main  ] Unable to open file '+filename)
            raise
            
        else:
            try:
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
                protocol_settings.select_labware()

                zstack_settings = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['zstack_id']
                zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']
                zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
                zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])

                if settings['zstack']['step_size'] != 0:
                    n_steps = np.floor( settings['zstack']['range'] / settings['zstack']['step_size'])
                    zstack_settings.ids['zstack_steps_id'].text = str(int(n_steps))
                else:
                    zstack_settings.ids['zstack_steps_id'].text = '0'

                layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
                for layer in layers:
                    lumaview.ids['imagesettings_id'].ids[layer].ids['ill_slider'].value = settings[layer]['ill']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['gain_slider'].value = settings[layer]['gain']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].value = settings[layer]['exp']
                    # lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].value = float(np.log10(settings[layer]['exp']))
                    lumaview.ids['imagesettings_id'].ids[layer].ids['root_text'].text = settings[layer]['file_root']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active = settings[layer]['false_color']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['acquire'].active = settings[layer]['acquire']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['autofocus'].active = settings[layer]['autofocus']

                lumaview.scope.set_frame_size(settings['frame']['width'], settings['frame']['height'])
            except:
                logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')

    # Save settings to JSON file
    def save_settings(self, file="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.save_settings()')
        global settings
        os.chdir(source_path)
        with open(file, "w") as write_file:
            json.dump(settings, write_file, indent = 4)

    def load_scopes(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_scopes()')
        spinner = self.ids['scope_spinner']
        spinner.values = list(self.scopes.keys())

    def select_scope(self):
        logger.info('[LVP Main  ] MicroscopeSettings.select_scope()')
        global lumaview
        global settings

        spinner = self.ids['scope_spinner']
        settings['microscope'] = spinner.text

    def load_ojectives(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_ojectives()')
        spinner = self.ids['objective_spinner']
        spinner.values = list(self.objectives.keys())

    def select_objective(self):
        logger.info('[LVP Main  ] MicroscopeSettings.select_objective()')
        global lumaview
        global settings

        spinner = self.ids['objective_spinner']
        settings['objective'] = self.objectives[spinner.text]
        settings['objective']['ID'] = spinner.text
        microscope_settings_id = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['magnification_id'].text = str(settings['objective']['magnification'])

    def frame_size(self):
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        global lumaview
        global settings

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        width = int(min(int(w), lumaview.scope.get_max_width())/4)*4
        height = int(min(int(h), lumaview.scope.get_max_height())/4)*4

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame_width'].text = str(width)
        self.ids['frame_height'].text = str(height)

        lumaview.scope.set_frame_size(width, height)


# Modified Slider Class to enable on_release event
# ---------------------------------------------------------------------
class ModSlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(ModSlider, self).__init__(**kwargs)
        logger.info('[LVP Main  ] ModSlider.__init__()')

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super(ModSlider, self).on_touch_up(touch)
        # logger.info('[LVP Main  ] ModSlider.on_touch_up()')
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
        logger.info('[LVP Main  ] LayerControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        logger.info('[LVP Main  ] LayerControl.ill_slider()')
        illumination = self.ids['ill_slider'].value
        settings[self.layer]['ill'] = illumination
        self.apply_settings()

    def ill_text(self):
        logger.info('[LVP Main  ] LayerControl.ill_text()')
        ill_min = self.ids['ill_slider'].min
        ill_max = self.ids['ill_slider'].max
        ill_val = float(self.ids['ill_text'].text)
        illumination = float(np.clip(ill_val, ill_min, ill_max))

        settings[self.layer]['ill'] = illumination
        self.ids['ill_slider'].value = illumination
        self.ids['ill_text'].text = str(illumination)

        self.apply_settings()

    def auto_gain(self):
        logger.info('[LVP Main  ] LayerControl.auto_gain()')
        if self.ids['auto_gain'].state == 'down':
            state = True
        else:
            state = False
        settings[self.layer]['auto_gain'] = state
        self.apply_settings()

    def gain_slider(self):
        logger.info('[LVP Main  ] LayerControl.gain_slider()')
        gain = self.ids['gain_slider'].value
        settings[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):
        logger.info('[LVP Main  ] LayerControl.gain_text()')
        gain_min = self.ids['gain_slider'].min
        gain_max = self.ids['gain_slider'].max
        gain_val = float(self.ids['gain_text'].text)
        gain = float(np.clip(gain_val, gain_min, gain_max))

        settings[self.layer]['gain'] = gain
        self.ids['gain_slider'].value = gain
        self.ids['gain_text'].text = str(gain)

        self.apply_settings()

    def exp_slider(self):
        logger.info('[LVP Main  ] LayerControl.exp_slider()')
        exposure = self.ids['exp_slider'].value
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp'] = exposure        # exposure in ms
        self.apply_settings()

    def exp_text(self):
        logger.info('[LVP Main  ] LayerControl.exp_text()')
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
        logger.info('[LVP Main  ] LayerControl.root_text()')
        settings[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        logger.info('[LVP Main  ] LayerControl.false_color()')
        settings[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        logger.info('[LVP Main  ] LayerControl.update_acquire()')
        settings[self.layer]['acquire'] = self.ids['acquire'].active

    def update_autofocus(self):
        logger.info('[LVP Main  ] LayerControl.update_autofocus()')
        settings[self.layer]['autofocus'] = self.ids['autofocus'].active

    def save_focus(self):
        logger.info('[LVP Main  ] LayerControl.save_focus()')
        global lumaview
        pos = lumaview.scope.get_current_position('Z')
        settings[self.layer]['focus'] = pos

    def goto_focus(self):
        logger.info('[LVP Main  ] LayerControl.goto_focus()')
        global lumaview
        pos = settings[self.layer]['focus']
        lumaview.scope.move_absolute_position('Z', pos)  # set current z height in usteps
        control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
        control.update_gui()

    def apply_settings(self):
        logger.info('[LVP Main  ] LayerControl.apply_settings()')
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        illumination = settings[self.layer]['ill']
        if self.ids['apply_btn'].state == 'down': # if the button is down
            # In active channel,turn on LED
            if lumaview.scope.led:
                lumaview.scope.led_on(lumaview.scope.color2ch(self.layer), illumination)
                logger.info('[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch(self.layer), illumination)')
            else:
                logger.warning('[LVP Main  ] LED controller not available.')
            

            
            #  turn the state of remaining channels to 'normal' and text to 'OFF'
            layers = ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']
            for layer in layers:
                if layer == self.layer:
#                    Clock.schedule_interval(lumaview.ids['imagesettings_id'].ids[self.layer].ids['histo_id'].histogram, 0.1)
                    logger.info('[LVP Main  ] Clock.schedule_interval(...[self.layer]...histogram, 0.1)')
                else:
                    lumaview.ids['imagesettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        else: # if the button is 'normal' meaning not active
            # In active channel, and turn off LED
            if lumaview.scope.led:
                lumaview.scope.leds_off()
                logger.info('[LVP Main  ] lumaview.scope.leds_off()')
            else:
                logger.warning('[LVP Main  ] LED controller not available.')

        # update gain to currently selected settings
        # -----------------------------------------------------
        state = settings[self.layer]['auto_gain']
        lumaview.scope.set_auto_gain(state)

        if not(state):
            gain = settings[self.layer]['gain']
            lumaview.scope.set_gain(gain)

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = settings[self.layer]['exp']
        lumaview.scope.set_exposure_time(exposure)

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
            self.ids['apply_btn'].background_down = './data/icons/ToggleRW.png'

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        for i in np.arange(0.1, 2, 0.1):
            Clock.schedule_once(self.update_shader, i)

    def update_shader(self, dt):
        # logger.info('[LVP Main  ] LayerControl.update_shader()')
        if self.ids['false_color'].active:
            lumaview.ids['viewer_id'].update_shader(self.layer)
        else:
            lumaview.ids['viewer_id'].update_shader('none')

# Z Stack functions class
# ---------------------------------------------------------------------
class ZStack(CompositeCapture):
    def set_steps(self):
        logger.info('[LVP Main  ] ZStack.set_steps()')

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
        logger.info('[LVP Main  ] ZStack.aquire_zstack()')
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
        current_pos = lumaview.scope.get_current_position('Z')

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
        lumaview.scope.move_absolute_position('Z', self.positions[self.n_pos])

        if self.ids['ztack_aqr_btn'].state == 'down':
            logger.info('[LVP Main  ] Clock.schedule_interval(self.zstack_iterate, 0.01)')
            Clock.schedule_interval(self.zstack_iterate, 0.01)
            self.ids['ztack_aqr_btn'].text = 'Acquiring ZStack'

        else:
            self.ids['ztack_aqr_btn'].text = 'Acquire'
            # self.zstack_event.cancel()
            logger.info('[LVP Main  ] Clock.unschedule(self.zstack_iterate)')
            Clock.unschedule(self.zstack_iterate)

    def zstack_iterate(self, dt):
        logger.info('[LVP Main  ] ZStack.zstack_iterate()')

        if lumaview.scope.get_target_status('Z'):
            logger.info('[LVP Main  ] Z at target')
            self.live_capture()
            self.n_pos += 1

            if self.n_pos < len(self.positions):
                lumaview.scope.move_absolute_position('Z', self.positions[self.n_pos])
            else:
                self.ids['ztack_aqr_btn'].text = 'Acquire'
                self.ids['ztack_aqr_btn'].state = 'normal'
                logger.info('[LVP Main  ] Clock.unschedule(self.zstack_iterate)')
                Clock.unschedule(self.zstack_iterate)

# Button the triggers 'filechooser.open_file()' from plyer
class FileChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info('[LVP Main  ] FileChooseBTN.choose()')
        # Call plyer filechooser API to run a filechooser Activity.
        self.context = context
        if self.context == 'load_settings':
            filechooser.open_file(on_selection=self.handle_selection, filters = ["*.json"])   
        elif self.context == 'load_protocol':
            filechooser.open_file(on_selection=self.handle_selection, filters = ["*.tsv"])

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection()

    def on_selection(self, *a, **k):
        logger.info('[LVP Main  ] FileChooseBTN.on_selection()')
        global lumaview
        
        if self.selection:
            if self.context == 'load_settings':
                lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].load_settings(self.selection[0])

            elif self.context == 'load_protocol':
                lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(filepath = self.selection[0])
        else:
            return

# Button the triggers 'filechooser.choose_dir()' from plyer
class FolderChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info('[LVP Main  ] FolderChooseBTN.choose()')
        self.context = context
        filechooser.choose_dir(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FolderChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection()

    def on_selection(self, *a, **k):
        logger.info('[LVP Main  ] FolderChooseBTN.on_selection()')
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
        logger.info('[LVP Main  ] FileSaveBTN.choose()')
        self.context = context
        if self.context == 'save_settings':
            filechooser.save_file(on_selection=self.handle_selection, filters = ["*.json"])
        elif self.context == 'saveas_protocol':
            filechooser.save_file(on_selection=self.handle_selection, filters = ["*.tsv"])

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileSaveBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection()

    def on_selection(self, *a, **k):
        logger.info('[LVP Main  ] FileSaveBTN.on_selection()')
        global lumaview
        
        if self.context == 'save_settings':
            if self.selection:
                lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].save_settings(self.selection[0])
                logger.info('[LVP Main  ] Saving Settings to File:' + self.selection[0])

        elif self.context == 'saveas_protocol':
            if self.selection:
                lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].save_protocol(filepath = self.selection[0])
                logger.info('[LVP Main  ] Saving Protocol to File:' + self.selection[0])


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def on_start(self):
        logger.info('[LVP Main  ] LumaViewProApp.on_start()')
        lumaview.scope.xyhome()
        # if profiling:
        #     self.profile = cProfile.Profile()
        #     self.profile.enable()
        # Clock.schedule_once(lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].ids['stage_widget_id'].draw_labware, 5)

    def build(self):
        current_time = time.strftime("%m/%d/%Y", time.localtime())
        logger.info('[LVP Main  ] LumaViewProApp.build()')

        logger.info('[LVP Main  ] -----------------------------------------')
        logger.info('[LVP Main  ] Code Compiled On: %s', current_time)
        logger.info('[LVP Main  ] Run Time: ' + time.strftime("%Y %m %d %H:%M:%S"))
        logger.info('[LVP Main  ] -----------------------------------------')

        global Window
        global lumaview
        self.icon = './data/icons/icon.png'

        try:
            from kivy.core.window import Window
            #Window.bind(on_resize=self._on_resize)
            lumaview = MainDisplay()
            #Window.maximize()
        except:
            logger.exception('[LVP Main  ] Cannot open main display.')
            raise

        # load settings file
        if os.path.exists("./data/current.json"):
            lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].load_settings("./data/current.json")
        elif os.path.exists("./data/settings.json"):
            lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].load_settings("./data/settings.json")
        else:
            if not os.path.isdir('./data'):
                raise FileNotFoundError("Cound't find 'data' directory.")
            else:
                raise FileNotFoundError('No settings files found.')
        
        # Continuously update image of stage and protocol
        Clock.schedule_interval(lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].ids['stage_widget_id'].draw_labware, 0.1)
        Clock.schedule_interval(lumaview.ids['motionsettings_id'].ids['xy_stagecontrol_id'].update_gui, 0.1) # Includes text boxes, not just stage

        try:
            filepath = settings['protocol']['filepath']
            lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(filepath=filepath)
        except:
            logger.exception('[LVP Main  ] Unable to load protocol at startup')

        lumaview.ids['imagesettings_id'].ids['BF'].apply_settings()
        if lumaview.scope.led:
            lumaview.scope.leds_off()
            logger.info('[LVP Main  ] lumaview.scope.leds_off()')
        else:
            logger.warning('[LVP Main  ] LED controller not available.')

        return lumaview

    def _on_resize(self, window, w, h):
        pass
        #Clock.schedule_once(lumaview.ids['motionsettings_id'].check_settings, 0.1)
        #Clock.schedule_once(lumaview.ids['imagesettings_id'].check_settings, 0.1)

    def on_stop(self):
        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')
        # if profiling:
        #     self.profile.disable()
        #     self.profile.dump_stats('./logs/LumaViewProApp.profile')
        #     stats = pstats.Stats('./logs/LumaViewProApp.profile')
        #     stats.sort_stats('cumulative').print_stats(30)
        #     stats.sort_stats('cumulative').dump_stats('./logs/LumaViewProApp.stats')

        global lumaview

        if lumaview.scope.led:
            lumaview.scope.leds_off()
            logger.info('[LVP Main  ] lumaview.scope.leds_off()')
        else:
            logger.warning('[LVP Main  ] LED controller not available.')

        lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].save_settings("./data/current.json")

LumaViewProApp().run()
