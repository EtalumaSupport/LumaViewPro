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
import logging
import datetime
import io
import os
import pathlib
import numpy as np
import pandas as pd
import csv
import time
import json
import sys
import glob
from lvp_logger import logger
from tkinter import filedialog as tkinter_filedialog
from plyer import filechooser

if getattr(sys, 'frozen', False):
    import pyi_splash
    pyi_splash.update_text("")

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
from kivy.uix.accordion import AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.label import Label

# Video Related
from kivy.graphics.texture import Texture

# User Interface Custom Widgets
from custom_widgets.range_slider import RangeSlider
from custom_widgets.progress_popup import show_popup

#post processing
from image_stitcher import image_stitcher
from modules.video_builder import VideoBuilder

from modules.tiling_config import TilingConfig
import modules.common_utils as common_utils

from modules.stitcher import Stitcher
from modules.color_channels import ColorChannel
from modules.composite_generation import CompositeGeneration
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.zstack_config import ZStackConfig
from modules.json_helper import CustomJSONizer

import cv2
import skimage

# Hardware
from labware import WellPlate
import lumascope_api
import post_processing

import image_utils

global lumaview
global settings
global cell_count_content

global ENGINEERING_MODE
ENGINEERING_MODE = False

global debug_counter
debug_counter = 0

PROTOCOL_DATA_DIR_NAME = "ProtocolData"

abspath = os.path.abspath(__file__)
basename = os.path.basename(__file__)
source_path = abspath[:-len(basename)]
print(source_path)

start_str = time.strftime("%Y %m %d %H_%M_%S")
start_str = str(int(round(time.time() * 1000)))

global focus_round
focus_round = 0


auto_gain_countdown = 0

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


def scope_leds_off():
    global lumaview
    if lumaview.scope.led:
        lumaview.scope.leds_off()
        logger.info('[LVP Main  ] lumaview.scope.leds_off()')
    else:
        logger.warning('[LVP Main  ] LED controller not available.')


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
        self.use_bullseye = False
        self.use_crosshairs = False
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


    @staticmethod
    def add_crosshairs(image):
        height, width = image.shape[0], image.shape[1]

        if image.ndim == 3:
            is_color = True
        else:
            is_color = False

        center_x = round(width/2)
        center_y = round(height/2)

        # Crosshairs - 2 pixels wide
        if is_color:
            image[:,center_x-1:center_x+1,:] = 255
            image[center_y-1:center_y+1,:,:] = 255
        else:
            image[:,center_x-1:center_x+1] = 255
            image[center_y-1:center_y+1,:] = 255

        # Radiating circles
        num_circles = 4
        minimum_dimension = min(height, width)
        circle_spacing = round(minimum_dimension/ 2 / num_circles)
        for i in range(num_circles):
            radius = (i+1) * circle_spacing
            rr, cc = skimage.draw.circle_perimeter(center_y, center_x, radius=radius, shape=image.shape)
            image[rr, cc] = 255

            # To make circles 2 pixel wide...
            rr, cc = skimage.draw.circle_perimeter(center_y, center_x, radius=radius+1, shape=image.shape)
            image[rr, cc] = 255

        return image
    

    @staticmethod
    def transform_to_bullseye(image):
        image_bullseye = np.zeros((*image.shape, 3), dtype=np.uint8)

        # The range is defined by (start_value, end_value]
        # key: [start_value, end_value, RGB Value]
        color_map = {
            0:  [ -1,   5,   0,   0,   0],
            1:  [  5,  15,   0, 255,   0],
            2:  [ 15,  25,   0,   0,   0],
            3:  [ 25,  35,   0, 255,   0],
            4:  [ 35,  45,   0,   0,   0],
            5:  [ 45,  55,   0, 255,   0],
            6:  [ 55,  65,   0,   0,   0],
            7:  [ 65,  75,   0, 255,   0],
            8:  [ 75,  85,   0,   0,   0],
            9:  [ 85,  95,   0, 255,   0],
            10: [ 95, 105,   0,   0,   0],
            11: [105, 115,   0, 255,   0],
            12: [115, 125,   0,   0,   0],
            13: [125, 135,   0,   0, 255],
            14: [135, 145,   0,   0,   0],
            15: [145, 155,   0, 255,   0],
            16: [155, 165,   0,   0,   0],
            17: [165, 175,   0, 255,   0],
            18: [175, 185,   0,   0,   0],
            19: [185, 195,   0, 255,   0],
            20: [195, 205,   0,   0,   0],
            21: [205, 215,   0, 255,   0],
            22: [215, 225,   0,   0,   0],
            23: [225, 235,   0, 255,   0],
            24: [235, 245,   0,   0,   0],
            25: [245, 255, 255,   0,   0]
        }

        for key in color_map.keys():
            start, end, *_rgb = color_map[key]
            boolean_array = np.logical_and(image > start, image <= end)
            image_bullseye[boolean_array] = _rgb

        return image_bullseye
    

    def update_scopedisplay(self, dt=0):
        global lumaview
        global debug_counter

        if lumaview.scope.camera.active != False:
            image = lumaview.scope.get_image()
            if image is False:
                return
            
            if ENGINEERING_MODE == True:

                debug_counter += 1
                if debug_counter == 30:
                    debug_counter = 0

                if debug_counter % 10 == 0:
                    mean = round(np.mean(a=image), 2)
                    stddev = round(np.std(a=image), 2)
                    open_layer = None
                    for layer in common_utils.get_layers():
                        accordion = layer + '_accordion'
                        if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                            open_layer = layer
                            break
                    
                    if open_layer is not None:
                        lumaview.ids['imagesettings_id'].ids[open_layer].ids['image_stats_mean_id'].text = f"Mean: {mean}"
                        lumaview.ids['imagesettings_id'].ids[open_layer].ids['image_stats_stddev_id'].text = f"StdDev: {stddev}"

                if debug_counter % 3 == 0:
                    if self.use_bullseye:
                        image_bullseye = self.transform_to_bullseye(image=image)

                        if self.use_crosshairs:
                            image_bullseye = self.add_crosshairs(image=image_bullseye)

                        texture = Texture.create(size=(image_bullseye.shape[1],image_bullseye.shape[0]), colorfmt='rgb')
                        texture.blit_buffer(image_bullseye.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                        self.texture = texture
                
            if not self.use_bullseye:
                if self.use_crosshairs:
                    image = self.add_crosshairs(image=image)

                # Convert to texture for display (using OpenGL)
                texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')
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

        return current_labware.get_well_label(x=x_target, y=y_target)


    def live_capture(self):
        print("Live capture")
        logger.info('[LVP Main  ] CompositeCapture.live_capture()')
        global lumaview

        save_folder = pathlib.Path(settings['live_folder']) / "Manual"
        save_folder.mkdir(parents=True, exist_ok=True)
        file_root = 'live_'
        color = 'BF'
        well_label = self.get_well_label()

        for layer in common_utils.get_layers():
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:

                append = f'{well_label}_{layer}'

                if lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active:
                    color = layer
                    
                break
            
        # lumaview.scope.get_image()
        return lumaview.scope.save_live_image(save_folder, file_root, append, color)
    

    def custom_capture(
        self,
        save_folder,
        color,
        illumination,
        gain,
        auto_gain,
        exposure,
        false_color = True,
        tile_label = None,
        z_height_idx = None,
        scan_count = None,
        custom_name = None,
        well_label = None
    ):
        print("Custom capture")
        logger.info('[LVP Main  ] CompositeCapture.custom_capture()')
        global lumaview
        global settings
        
        # Set gain and exposure
        if not auto_gain:
            lumaview.scope.set_gain(gain)
            lumaview.scope.set_exposure_time(exposure)
    
        # Save Settings
        # file_root = settings[color]['file_root']

        name = common_utils.generate_default_step_name(
            well_label=well_label,
            color=color,
            z_height_idx=z_height_idx,
            scan_count=scan_count,
            custom_name_prefix=custom_name,
            tile_label=tile_label
        )
        # if custom_name is None:

        #     if well_label is None:
        #         well_label = self.get_well_label()

        #     name = common_utils.generate_default_step_name(
        #         well_label=well_label,
        #         color=color,
        #         z_height_idx=z_height_idx,
        #         scan_count=scan_count,
        #         tile_label=tile_label
        #     )
        # else:
        #     DESIRED_SCAN_COUNT_DIGITS = 4
        #     name = f"{custom_name}_{scan_count:0>{DESIRED_SCAN_COUNT_DIGITS}}"

        # Illuminate
        if lumaview.scope.led:
            channel = lumaview.scope.color2ch(color)
            lumaview.scope.led_on(channel, illumination)
            logger.info(f'[LVP Main  ] lumaview.scope.led_on({channel}, {illumination})')
        else:
            logger.warning('LED controller not available.')

        # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        # Grab image and save
        time.sleep(2*exposure/1000+0.2)
        
        use_color = color if false_color else 'BF'

        if self.enable_image_saving == True:
            image_filepath = lumaview.scope.save_live_image(
                save_folder=save_folder,
                file_root=None,
                append=name,
                color=use_color,
                tail_id_mode=None
            )
        else:
            image_filepath = None

        # Turn off LEDs and LED toggle buttons
        scope_leds_off()
        for layer in common_utils.get_layers():
            lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'

        return image_filepath


    # capture and save a composite image using the current settings
    def composite_capture(self):
        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')
        global lumaview

        if lumaview.scope.camera.active == False:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        img = np.zeros((settings['frame']['height'], settings['frame']['width'], 3))

        for layer in common_utils.get_layers():
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
                scope_leds_off()

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

            scope_leds_off()

            # turn off all LED toggle buttons and histograms
            lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'
            Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        # lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'
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


class AccordionItemXyStageControl(AccordionItem):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def update_gui(self):
        self.ids['xy_stagecontrol_id'].update_gui()


class MotionSettings(BoxLayout):
    settings_width = dp(300)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('[LVP Main  ] MotionSettings.__init__()')
        self._accordion_item_xystagecontrol = AccordionItemXyStageControl()
        self._accordion_item_xystagecontrol_visible = False
        Clock.schedule_once(self._init_ui, 0)

       
    def _init_ui(self, dt=0):
        self.enable_ui_features_for_engineering_mode()


    def enable_ui_features_for_engineering_mode(self):
        if ENGINEERING_MODE == True:
            # for layer in common_utils.get_layers():
            ps = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
            ps.ids['protocol_disable_image_saving_box_id'].opacity = 1
            ps.ids['protocol_disable_image_saving_box_id'].height = '30dp'
            ps.ids['protocol_disable_image_saving_id'].height = '30dp'
            ps.ids['protocol_disable_image_saving_label_id'].height = '30dp'

            lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].ids['enable_bullseye_box_id'].height = '30dp'
            lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].ids['enable_bullseye_box_id'].opacity = 1
                

    def set_xystage_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_xystage_control()
        else:
            self._hide_xystage_control()


    def _show_xystage_control(self):
        if not self._accordion_item_xystagecontrol_visible:
            self._accordion_item_xystagecontrol_visible = True
            self.ids['motionsettings_accordion_id'].add_widget(self._accordion_item_xystagecontrol, 2)


    def _hide_xystage_control(self):
        if self._accordion_item_xystagecontrol_visible:
            self._accordion_item_xystagecontrol_visible = False
            self.ids['motionsettings_accordion_id'].remove_widget(self._accordion_item_xystagecontrol)


    def set_turret_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['verticalcontrol_id']
        for turret_id in ('turret_selection_label', 'turret_btn_box'):
            vert_control.ids[turret_id].visible = visible

    
    def set_tiling_control_visibility(self, visible: bool) -> None:
        vert_control = self.ids['protocol_settings_id']
        for tiling_id in ('tiling_box_layout_id',):
            vert_control.ids[tiling_id].visible = visible

        if not visible:
            vert_control.ids['tiling_size_spinner'].text = '1x1'


    # Hide (and unhide) motion settings
    def toggle_settings(self):
        logger.info('[LVP Main  ] MotionSettings.toggle_settings()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        self.ids['verticalcontrol_id'].update_gui()
        self.update_xy_stage_control_gui()

        # move position of motion control
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

        if scope_display.play == True:
            scope_display.start()

    
    def update_xy_stage_control_gui(self, *args):
        self._accordion_item_xystagecontrol.update_gui()


    def check_settings(self, *args):
        logger.info('[LVP Main  ] MotionSettings.check_settings()')
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

class StitchControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        global stitch_controls
        super().__init__(**kwargs)
        stitch_controls = self


    @show_popup
    def run_stitcher(self, popup, path):
        status_map = {
            True: "Success",
            False: "FAILED"
        }
        popup.title = "Stitcher"
        popup.text = "Generating stitched images..."
        stitcher = Stitcher()
        result = stitcher.load_folder(path=pathlib.Path(path))
        final_text = f"Generating stitched images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
        popup.text = final_text
        
        time.sleep(2)
        self.done = True


class CompositeGenControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        global composite_gen_controls
        super().__init__(**kwargs)
        composite_gen_controls = self


    @show_popup
    def run_composite_gen(self, popup, path):
        status_map = {
            True: "Success",
            False: "FAILED"
        }
        popup.title = "Composite Image Generation"
        popup.text = "Generating composite images..."
        composite_gen = CompositeGeneration()
        result = composite_gen.load_folder(path=pathlib.Path(path))
        final_text = f"Generating composite images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
        popup.text = final_text
        
        time.sleep(2)
        self.done = True


class VideoCreationControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        global video_creation_controls
        super().__init__(**kwargs)
        logger.info('LVP Main: VideoCreationControls.__init__()')
        self._post = post_processing.PostProcessing()
        video_creation_controls = self
        self._first_open = False
        self._input_images_loc = None
        self._output_file_loc = None

    
    def activate(self):
        if self._first_open is False:
            self._first_open = True


    def deactivate(self):
        pass


    def set_input_images_loc(self, directory: str | pathlib.Path) -> None:
        self._input_images_loc = pathlib.Path(directory)

    
    def set_output_file_loc(self, file_loc: str | pathlib.Path) -> None:
        self._output_file_loc = pathlib.Path(file_loc)


    @show_popup
    def create_video(self, popup) -> None:
        status_map = {
            True: "Success",
            False: "FAILED"
        }

        popup.title = "Video Builder"
        popup.text = "Generating video..."

        if self._input_images_loc is None:
            popup.text = f"{popup.text} {status_map[False]} - Set Image Folder"
            time.sleep(2)
            self.done = True
            return

        if self._output_file_loc is None:
            self._output_file_loc = self._input_images_loc.joinpath("movie.avi")

        video_builder = VideoBuilder()
        status = video_builder.create_video_from_directory(
            input_directory=self._input_images_loc,
            frames_per_sec=10,
            output_file_loc=self._output_file_loc,
        )

        popup.text = f"{popup.text} {status_map[status]}\n- Output: {self._output_file_loc}"
        time.sleep(2)
        self.done = True
        self._launch_video()

    
    def _launch_video(self) -> None:
        try:
            os.startfile(self._output_file_loc)
        except Exception as e:
            logger.error(f"Unable to launch video {self._output_file_loc}:\n{e}")


class CellCountControls(BoxLayout):

    ENABLE_PREVIEW_AUTO_REFRESH = False

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: CellCountControls.__init__()')
        self._preview_source_image = None
        self._preview_image = None
        self._post = post_processing.PostProcessing()
        self._settings = self._get_init_settings()
        self._set_ui_to_settings(self._settings)


    def _get_init_settings(self):
        return {
            'context': {
                'pixels_per_um': 1.0,       # TODO Isn't this supposed to be 0.5 ?
                'fluorescent_mode': True
            },
            'segmentation': {
                'algorithm': 'initial',
                'parameters': {
                    'threshold': 20,
                }
            },
            'filters': {
                'area': {
                    'min': 0,
                    'max': 100
                },
                'perimeter': {
                    'min': 0,
                    'max': 100
                },
                'sphericity': {
                    'min': 0.0,
                    'max': 1.0
                },
                'intensity': {
                    'min': {
                        'min': 0,
                        'max': 100
                    },
                    'mean': {
                        'min': 0,
                        'max': 100
                    },
                    'max': {
                        'min': 0,
                        'max': 100
                    }
                }
            }
        }

    def apply_method_to_preview_image(self):
        self._regenerate_image_preview()

    
    # Decorate function to show popup and run the code below in a thread
    @show_popup
    def apply_method_to_folder(self, popup, path):
        popup.title = 'Processing Cell Count Method'
        pre_text = f'Applying method to folder: {path}'
        popup.text = pre_text
        
        popup.progress = 0
        total_images = self._post.get_num_images_in_folder(path=path)
        image_count = 0
        for image_process in self._post.apply_cell_count_to_folder(path=path, settings=self._settings):
            filename = image_process['filename']
            image_count += 1
            popup.progress = int(100 * image_count / total_images)
            popup.text = f"{pre_text}\n- {image_count}/{total_images}: {filename}"

        popup.progress = 100
        popup.text = 'Done'
        time.sleep(1)
        self.done = True

    def set_post_processing_module(self, post_processing_module):
        self._post = post_processing_module

    def get_current_settings(self):
        return self._settings


    @staticmethod
    def _validate_method_settings_metadata(settings):
        if 'metadata' not in settings:
            raise Exception(f"No valid metadata found")
        
        metadata = settings['metadata']
        
        for key in ('type', 'version'):
            if key not in metadata:
                raise Exception(f"No {key} found in metadata")
                

    def _add_method_settings_metadata(self):
        self._settings['metadata'] = {
            'type': 'cell_count_method',
            'version': '1'
        }


    def load_settings(self, settings):
        self._validate_method_settings_metadata(settings=settings)
        self._settings = settings
        self._set_ui_to_settings(settings)


    def _area_range_slider_values_to_physical(self, slider_values):
        if self._preview_source_image is None:
            return slider_values
        
        xp = [0, 30, 60, 100]
        max = self.calculate_area_filter_max(image=self._preview_source_image)
        if max < 10001:
            max = 10001

        fp = [0, 1000, 10000, max]
        fg = np.interp(slider_values, xp, fp)
        return fg[0], fg[1]
    
    def _area_range_slider_physical_to_values(self, physical_values):
        if self._preview_source_image is None:
            return physical_values

        max = self.calculate_area_filter_max(image=self._preview_source_image)
        if max < 10001:
            max = 10001

        xp = [0, 1000, 10000, max]
        fp = [0, 30, 60, 100]
        fg = np.interp(physical_values, xp, fp)
        return fg[0], fg[1]

    def _perimeter_range_slider_values_to_physical(self, slider_values):
        if self._preview_source_image is None:
            return slider_values

        xp = [0, 50, 100]

        max = self.calculate_perimeter_filter_max(image=self._preview_source_image)
        if max < 101:
            max = 101

        fp = [0, 100, max]
        fg = np.interp(slider_values, xp, fp)
        return fg[0], fg[1]
    
    def _perimeter_range_slider_physical_to_values(self, physical_values):
        if self._preview_source_image is None:
            return physical_values
        
        max = self.calculate_perimeter_filter_max(image=self._preview_source_image)
        if max < 101:
            max = 101

        xp = [0, 100, max]
        fp = [0, 50, 100]
        fg = np.interp(physical_values, xp, fp)
        return fg[0], fg[1]


    def _set_ui_to_settings(self, settings):
        self.ids.text_cell_count_pixels_per_um_id.text = str(settings['context']['pixels_per_um'])
        self.ids.cell_count_fluorescent_mode_id.active = settings['context']['fluorescent_mode']
        self.ids.slider_cell_count_threshold_id.value = settings['segmentation']['parameters']['threshold']
        self.ids.slider_cell_count_area_id.value = self._area_range_slider_physical_to_values(
            (settings['filters']['area']['min'], settings['filters']['area']['max'])
        )
        
        self.ids.slider_cell_count_perimeter_id.value = self._perimeter_range_slider_physical_to_values(
            (settings['filters']['perimeter']['min'], settings['filters']['perimeter']['max'])
        )
        self.ids.slider_cell_count_sphericity_id.value = (settings['filters']['sphericity']['min'], settings['filters']['sphericity']['max'])
        self.ids.slider_cell_count_min_intensity_id.value = (settings['filters']['intensity']['min']['min'], settings['filters']['intensity']['min']['max'])
        self.ids.slider_cell_count_mean_intensity_id.value = (settings['filters']['intensity']['mean']['min'], settings['filters']['intensity']['mean']['max'])
        self.ids.slider_cell_count_max_intensity_id.value = (settings['filters']['intensity']['max']['min'], settings['filters']['intensity']['max']['max'])

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()
        self._regenerate_image_preview()


    def set_preview_source_file(self, file) -> None:
        image = image_utils.image_file_to_image(image_file=file)
        if image is None:
            return
            
        self.set_preview_source(image=image)


    def calculate_area_filter_max(self, image):
        pixels_per_um = self._settings['context']['pixels_per_um']

        max_area_pixels = image.shape[0] * image.shape[1]
        max_area_um2 = max_area_pixels / (pixels_per_um**2)
        return max_area_um2
    

    def calculate_perimeter_filter_max(self, image):
        pixels_per_um = self._settings['context']['pixels_per_um']

        # Assume max perimeter will never need to be larger than 2x frame size border
        # The 2x is to provide margin for handling various curvatures
        max_perimeter_pixels = 2*((2*image.shape[0])+(2*image.shape[1]))
        max_perimeter_um = max_perimeter_pixels / pixels_per_um
        return max_perimeter_um
    

    def update_filter_max(self, image):
        max_area_um2 = self. calculate_area_filter_max(image=image)
        max_perimeter_um = self.calculate_perimeter_filter_max(image=image)
        
        self.ids.slider_cell_count_area_id.max = int(self._area_range_slider_physical_to_values(physical_values=(0,max_area_um2))[1])
        self.ids.slider_cell_count_perimeter_id.max = int(self._perimeter_range_slider_physical_to_values(physical_values=(0,max_perimeter_um))[1])

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()


    def set_preview_source(self, image) -> None:
        self._preview_source_image = image
        self._preview_image = image
        self.ids['cell_count_image_id'].texture = image_utils.image_to_texture(image=image)
        self.update_filter_max(image=image)
        self._regenerate_image_preview()


    # Save settings to JSON file
    def save_method_as(self, file="./data/cell_count_method.json"):
        logger.info(f'[LVP Main  ] CellCountContent.save_method_as({file})')
        os.chdir(source_path)
        self._add_method_settings_metadata()
        with open(file, "w") as write_file:
            json.dump(self._settings, write_file, indent = 4, cls=CustomJSONizer)

    
    def load_method_from_file(self, file):
        logger.info(f'[LVP Main  ] CellCountContent.load_method_from_file({file})')
        with open(file, "r") as f:
            method_settings = json.load(f)
        
        self.load_settings(settings=method_settings)

    
    def _regenerate_image_preview(self):
        if self._preview_source_image is None:
            return

        image, _ = self._post.preview_cell_count(
            image=self._preview_source_image,
            settings=self._settings
        )

        self._preview_image = image

        cell_count_content.ids['cell_count_image_id'].texture = image_utils.image_to_texture(image=image)


    def slider_adjustment_threshold(self):
        self._settings['segmentation']['parameters']['threshold'] = self.ids['slider_cell_count_threshold_id'].value

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_area(self):
        low, high = self._area_range_slider_values_to_physical(
            (self.ids['slider_cell_count_area_id'].value[0], self.ids['slider_cell_count_area_id'].value[1])
        )

        self._settings['filters']['area']['min'], self._settings['filters']['area']['max'] = low, high

        self.ids['label_cell_count_area_id'].text = f"{int(low)}-{int(high)} μm²"

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_perimeter(self):
        low, high = self._perimeter_range_slider_values_to_physical(
            (self.ids['slider_cell_count_perimeter_id'].value[0], self.ids['slider_cell_count_perimeter_id'].value[1])
        )

        self._settings['filters']['perimeter']['min'], self._settings['filters']['perimeter']['max'] = low, high

        self.ids['label_cell_count_perimeter_id'].text = f"{int(low)}-{int(high)} μm"

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_sphericity(self):
        self._settings['filters']['sphericity']['min'] = self.ids['slider_cell_count_sphericity_id'].value[0]
        self._settings['filters']['sphericity']['max'] = self.ids['slider_cell_count_sphericity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_min_intensity(self):
        self._settings['filters']['intensity']['min']['min'] = self.ids['slider_cell_count_min_intensity_id'].value[0]
        self._settings['filters']['intensity']['min']['max'] = self.ids['slider_cell_count_min_intensity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_mean_intensity(self):
        self._settings['filters']['intensity']['mean']['min'] = self.ids['slider_cell_count_mean_intensity_id'].value[0]
        self._settings['filters']['intensity']['mean']['max'] = self.ids['slider_cell_count_mean_intensity_id'].value[1]
        
        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_max_intensity(self):
        self._settings['filters']['intensity']['max']['min'] = self.ids['slider_cell_count_max_intensity_id'].value[0]
        self._settings['filters']['intensity']['max']['max'] = self.ids['slider_cell_count_max_intensity_id'].value[1]
        
        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def flourescent_mode_toggle(self):
        self._settings['context']['fluorescent_mode'] = self.ids['cell_count_fluorescent_mode_id'].active
        
        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def pixel_conversion_adjustment(self):

        def _validate(value_str):
            try:
                value = float(value_str)
            except:
                return False, -1

            if value <= 0:
                return False, -1
                
            return True, value

        value_str = cell_count_content.ids['text_cell_count_pixels_per_um_id'].text

        valid, value = _validate(value_str)
        if not valid:
            return
        
        if self._preview_image is None:
            return
        
        self._settings['context']['pixels_per_um'] = value
        self.update_filter_max(image=self._preview_image)
      


class PostProcessingAccordion(BoxLayout):

    def __init__(self, **kwargs):
        #super(PostProcessingAccordion,self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self.post = post_processing.PostProcessing()
        #global settings
        #stitching params (see more info in image_stitcher.py):
        #self.raw_images_folder = settings['save_folder'] # I'm guessing not ./capture/ because that would have frames over time already (to make video)
        self.raw_images_folder = './capture/' # I'm guessing not ./capture/ because that would have frames over time already (to make video)
        self.combine_colors = False #True if raw images are in separate red/green/blue channels and need to be first combined
        self.ext = "tiff" #or read it from settings?
        #self.stitching_method = "features" # "features" - Low method, "position" - based on position information
        self.stitching_method = "position" # "features" - Low method, "position" - based on position information
        self.stitched_save_name = "last_composite_img.tiff"
        #self.positions_file = None #relevant if stitching method is position, will read positions from that file
        self.positions_file = "./capture/2x2.tsv" #relevant if stitching method is position, will read positions from that file
        self.pos2pix = 2630 # relevant if stitching method is position. The scale conversion for pos info into pixels
        
        
        # self.tiling_target = []
        self.tiling_min = {
            "x": 120000,
            "y": 80000
        }

        self.tiling_max = {
            "x": 0,
            "y": 0
        }

        self.tiling_count = {
            "x": 1,
            "y": 1
        }

        self.accordion_item_states = {
            'cell_count_accordion_id': None,
            'stitch_accordion_id': None,
            'open_last_save_folder_accordion_id': None,
            'create_avi_accordion_id': None
        }

        self.init_cell_count()


    @staticmethod
    def accordion_item_state(accordion_item):
        if accordion_item.collapse == True:
            return 'closed'
        return 'open'
     

    def get_accordion_item_states(self):
        return {
            'cell_count_accordion_id': self.accordion_item_state(self.ids['cell_count_accordion_id']),
            'stitch_accordion_id': self.accordion_item_state(self.ids['stitch_accordion_id']),
            'open_last_save_folder_accordion_id': self.accordion_item_state(self.ids['open_last_save_folder_accordion_id']),
            'create_avi_accordion_id': self.accordion_item_state(self.ids['create_avi_accordion_id']),
        }


    def accordion_collapse(self):
        
        new_accordion_item_states = self.get_accordion_item_states()

        changed_items = []
        for accordion_item_id, prev_accordion_item_state in self.accordion_item_states.items():
            if new_accordion_item_states[accordion_item_id] == prev_accordion_item_state:
                # No change
                continue
            
            # Update state and add state change to list
            self.accordion_item_states[accordion_item_id] = self.accordion_item_state(self.ids[accordion_item_id])
            changed_items.append(accordion_item_id)

        # TODO not currently needed to detect accordion item state changes, but an example is shown below
        # if 'cell_count_accordion_id' in changed_items:
        #     if self.accordion_item_states['cell_count_accordion_id'] == 'open':
        #         cell_count_content.activate()
        #     else:
        #         cell_count_content.deactivate()


    

    
    def init_cell_count(self):
        self._cell_count_popup = None
        

    def convert_to_avi(self):
        logger.debug('[LVP Main  ] PostProcessingAccordian.convert_to_avi() not yet implemented')
     
        
    def open_folder(self):
        logger.debug('[LVP Main  ] PostProcessing.open_folder() not yet implemented')

    def open_cell_count(self):
        global cell_count_content
        if self._cell_count_popup is None:
            cell_count_content.set_post_processing_module(self.post)
            self._cell_count_popup = Popup(
                title="Post Processing - Cell Count",
                content=cell_count_content,
                size_hint=(0.85,0.85),
                auto_dismiss=True
            )

        self._cell_count_popup.open()


class CellCountDisplay(FloatLayout):

    def __init__(self, **kwargs):
        super(CellCountDisplay,self).__init__(**kwargs)




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
        Clock.schedule_once(self._init_ui, 0)
    

    def _init_ui(self, dt=0):
        self.assign_led_button_down_images()
        self.accordion_collapse()
        self.set_layer_exposure_range()
        self.enable_image_stats_if_needed()


    def enable_image_stats_if_needed(self):
        global ENGINEERING_MODE
        if ENGINEERING_MODE == True:
            for layer in common_utils.get_layers():
                lumaview.ids['imagesettings_id'].ids[layer].ids['image_stats_mean_id'].height = '30dp'
                lumaview.ids['imagesettings_id'].ids[layer].ids['image_stats_stddev_id'].height = '30dp'
        

    def set_layer_exposure_range(self):
        for layer in common_utils.get_fluorescence_layers():
            lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].max = 1000

        for layer in common_utils.get_transmitted_layers():
            lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].max = 200


    def assign_led_button_down_images(self):
        led_button_down_background_map = {
            'Red': './data/icons/ToggleRR.png',
            'Green': './data/icons/ToggleRG.png',
            'Blue': './data/icons/ToggleRB.png',
        }

        for layer in common_utils.get_layers():
            button_down_image = led_button_down_background_map.get(layer, './data/icons/ToggleRW.png')
            self.ids[layer].ids['enable_led_btn'].background_down = button_down_image


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

            for layer in common_utils.get_layers():
                Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
                logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')
        else:
            self.pos = lumaview.width - self.settings_width, 0
 
        if scope_display.play == True:
            scope_display.start()

    def update_transmitted(self):
        for layer in common_utils.get_transmitted_layers():
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
        scope_leds_off()

        # turn off all LED toggle buttons and histograms
        for layer in common_utils.get_layers():
            layer_obj = lumaview.ids['imagesettings_id'].ids[layer]
            layer_is_collapsed = lumaview.ids['imagesettings_id'].ids[f"{layer}_accordion"].collapse

            if layer_is_collapsed:
                layer_obj.ids['enable_led_btn'].state = 'normal'
                continue

            layer_obj.apply_settings()

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


def set_histogram_layer(active_layer):   
    for layer in common_utils.get_layers():
        Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
        if layer == active_layer:
            Clock.schedule_interval(lumaview.ids['imagesettings_id'].ids[active_layer].ids['histo_id'].histogram, 0.5)
            logger.info(f'[LVP Main  ] Clock.schedule_interval(...[{active_layer}]...histogram, 0.5)')


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


    # @classmethod
    # def set_active_layer(self, active_layer):     
    #     Clock.unschedule(self.histogram) 
    #     for layer in common_utils.get_layers():
    #         if layer == active_layer:
    #             Clock.schedule_interval(self.histogram, 0.5)
    #             logger.info(f'[LVP Main  ] Clock.schedule_interval(...[{active_layer}]...histogram, 0.5)')


    def histogram(self, *args):
        # logger.info('[LVP Main  ] Histogram.histogram()')
        global lumaview
        bins = 128

        if lumaview.scope.camera != False and lumaview.scope.camera.active != False:
            #image = lumaview.scope.get_image()
            image = lumaview.scope.image_buffer
            hist = np.histogram(image, bins=bins,range=(0,256))
            '''
            if self.hist_range_set:
                edges = self.edges
            else:
                edges = np.histogram_bin_edges(image, bins=1)
                edges[0] = self.stablize*self.edges[0] + (1 - self.stablize)*edges[0]
                edges[1] = self.stablize*self.edges[1] + (1 - self.stablize)*edges[1]
            '''
            # mean = np.mean(hist[1],hist[0])
            lumaview.ids['viewer_id'].black = 0.0 # float(edges[0])/255.
            lumaview.ids['viewer_id'].white = 1.0 # float(edges[1])/255.

            # UPDATE SHADER
            self.canvas.clear()
            r, b, g, a = self.bg_color
            self.hist = hist
            #self.edges = edges
            with self.canvas:
                x = self.x
                y = self.y
                w = self.width
                h = self.height
                #Color(r, b, g, a/12)
                #Rectangle(pos=(x, y), size=(256, h))
                #Color(r, b, g, a/4)
                #Rectangle(pos=(x + edges[0], y), size=(edges[1]-edges[0], h))
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
                        bin_size= self.width/bins
                        Rectangle(pos=(x+max(i*bin_size-1, 1), y), size=(bin_size, counts))
                        #self.line = Line(points=(x+i, y, x+i, y+counts), width=1)


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

            if not self.ids['x_pos_id'].focus:
                self.ids['x_pos_id'].text = format(max(0, stage_x), '.2f') # display coordinate in mm

            if not self.ids['y_pos_id'].focus:  
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
        logger.info(f'[LVP Main  ] X pos {x_pos} Stage X {stage_x}')

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

        

        self._protocol_df = self.create_empty_protocol()
        self.curr_step = 0   # TODO isn't step 1 indexed? Why is is 0?

        self.custom_step_count = 0
        
        self.tiling_config = TilingConfig()
        self.tiling_min = {
            "x": 120000,
            "y": 80000
        }
        self.tiling_max = {
            "x": 0,
            "y": 0
        }

        self.tiling_count = self.tiling_config.get_mxn_size(self.tiling_config.default_config())

        self.scan_count = 0
        self.autofocus_was_used = False
        self.scan_in_progress = False
        self.separate_folder_per_channel = False
        self.enable_image_saving = True

        self.exposures = 1  # 1 indexed
        Clock.schedule_once(self._init_ui, 0)


    def _init_ui(self, dt=0):
        self.ids['tiling_size_spinner'].values = self.tiling_config.available_configs()
        self.ids['tiling_size_spinner'].text = self.tiling_config.default_config()


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
    def select_labware(self, labware : str = None):
        global settings
        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        if labware is None:
            spinner = self.ids['labware_spinner']
            spinner.values = list(self.labware['Wellplate'].keys())
            settings['protocol']['labware'] = spinner.text
        else:
            spinner = self.ids['labware_spinner']
            spinner.values = list('Center Plate',)
            settings['protocol']['labware'] = labware


    def set_labware_selection_visibility(self, visible):
        labware_spinner = self.ids['labware_spinner']
        labware_spinner.visible = visible
        labware_spinner.size_hint_y = None if visible else 0
        labware_spinner.height = '30dp' if visible else 0
        labware_spinner.opacity = 1 if visible else 0
        labware_spinner.disabled = not visible
    
    
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
        
    
    def apply_tiling(self):
        logger.info('[LVP Main  ] Apply tiling to protocol')
        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])
        current_labware.set_positions()
        
        tiles = self.tiling_config.get_tile_centers(
            config_label=self.ids['tiling_size_spinner'].text,
            focal_length=settings['objective']['focal_length'],
            frame_size=settings['frame'],
            fill_factor=TilingConfig.DEFAULT_FILL_FACTORS['position']
        )

        if len(tiles) == 1: # No tiling
            return

        # Get existing max tile group ID to start from
        # existing_max_tile_group_id = -1

        # for row_idx in range(len(self._protocol_df)):
        #     step = self._protocol_df.iloc[row_idx]
        #     tile_group_id = step['Tile']
        #     if tile_group_id != "":
        #         existing_max_tile_group_id = max(tile_group_id, existing_max_tile_group_id)
        existing_max_tile_group_id = self._protocol_df['Tile Group ID'].max()

        tile_group_id = existing_max_tile_group_id + 1

        new_steps = list()
        for row_idx in range(len(self._protocol_df)):
            orig_step_df = self._protocol_df.iloc[row_idx]
            orig_step_dict = orig_step_df.to_dict()

            # If already a tile, copy it over to the new protocol
            if orig_step_df['Tile'] not in (None, ""):
                new_steps.append(orig_step_dict)
                continue
            
            x = orig_step_df["X"]
            y = orig_step_df["Y"]

            # If not a tile, tile it.  
            for tile_label, tile_position in tiles.items():   
                
                x_tile = round(x + tile_position["x"]/1000, common_utils.max_decimal_precision('x')) # in 'plate' coordinates
                y_tile = round(y + tile_position["y"]/1000, common_utils.max_decimal_precision('y')) # in 'plate' coordinates

                # if orig_step_df["Custom Step"]:
                #     new_step_name = common_utils.generate_default_step_name(
                #         custom_name_prefix=orig_step_df['Name'],
                #         well_label=orig_step_df['Well'],
                #         color=orig_step_df['Color'],
                #         z_height_idx=None,
                #         scan_count=None,
                #         tile_label=tile_label
                #     )
                # else:
                #     new_step_name = common_utils.generate_default_step_name(
                #         well_label=orig_step_df['Well'],
                #         color=orig_step_df['Color'],
                #         z_height_idx=orig_step_df['Z-Slice'],
                #         scan_count=None,
                #         tile_label=tile_label
                #     )
                
                new_step_dict = self.create_step_dict(
                    name=orig_step_df['Name'],
                    x=x_tile,
                    y=y_tile,
                    z=orig_step_df['Z'],
                    af=orig_step_df['Auto_Focus'],
                    color=orig_step_df['Color'],
                    fc=orig_step_df['False_Color'],
                    ill=orig_step_df['Illumination'],
                    gain=orig_step_df['Gain'],
                    auto_gain=orig_step_df['Auto_Gain'],
                    exp=orig_step_df['Exposure'],
                    objective=orig_step_df['Objective'],
                    well=orig_step_df['Well'],
                    tile=tile_label,
                    zslice=orig_step_df['Z-Slice'],
                    custom_step=orig_step_df['Custom Step'],
                    tile_group_id=tile_group_id,
                    zstack_group_id=orig_step_df['Z-Stack Group ID']
                )

                new_steps.append(new_step_dict)
            
            tile_group_id += 1

        self._protocol_df = pd.DataFrame.from_dict(new_steps)
        self.update_step_ui()


    def update_step_ui(self):
        # Number of Steps
        length = len(self._protocol_df)
              
        step = self.get_curr_step()
        if length > 0:
            self.ids['step_name_input'].text = step["Name"]
            if step['Name'] == '':
                self.ids['step_name_input'].hint_text = self.get_default_name_for_curr_step()
            self.ids['step_number_input'].text = str(self.curr_step+1)
        else:
            self.ids['step_number_input'].text = '0'
            self.ids['step_name_input'].text = ''
            self.ids['step_name_input'].hint_text = 'Step Name'

        self.ids['step_total_input'].text = str(length)
        # settings['protocol']['filepath'] = ''        
        # self.ids['protocol_filename'].text = ''
      
    @staticmethod
    def create_empty_protocol() -> pd.DataFrame:
        dtypes = np.dtype(
            [
                ("Name", str),
                ("X", float),
                ("Y", float),
                ("Z", float),
                ("Auto_Focus", bool),
                ("Color", str),
                ("False_Color", bool),
                ("Illumination", float),
                ("Gain", float),
                ("Auto_Gain", bool),
                ("Exposure", float),
                ("Objective", str),
                ("Well", str),
                ("Tile", str),
                ("Z-Slice", int),
                ("Custom Step", bool),
                ("Tile Group ID", int),
                ("Z-Stack Group ID", int)
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df
    

    @staticmethod
    def create_step_dict(
        name,
        x,
        y,
        z,
        af,
        color,
        fc,
        ill,
        gain,
        auto_gain,
        exp,
        objective,
        well,
        tile,
        zslice,
        custom_step,
        tile_group_id,
        zstack_group_id
    ):
        return {
            "Name": name,
            "X": x,
            "Y": y,
            "Z": z,
            "Auto_Focus": af,
            "Color": color,
            "False_Color": fc,
            "Illumination": ill,
            "Gain": gain,
            "Auto_Gain": auto_gain,
            "Exposure": exp,
            "Objective": objective,
            "Well": well,
            "Tile": tile,
            "Z-Slice": zslice,
            "Custom Step": custom_step,
            "Tile Group ID": tile_group_id,
            "Z-Stack Group ID": zstack_group_id
        }
    

    def add_steps_to_protocol(
        self,
        steps: list[dict]
    ):
        steps_df = pd.DataFrame(steps)
        self._protocol_df = pd.concat([self._protocol_df, steps_df], ignore_index=True).reset_index(drop=True)


    # Create New Protocol
    def new_protocol(self):
        logger.info('[LVP Main  ] ProtocolSettings.new_protocol()')

        self.custom_step_count = 0
        steps = []

        os.chdir(source_path)
        current_labware = WellPlate()
        current_labware.load_plate(settings['protocol']['labware'])
        current_labware.set_positions()
        
        tiles = self.tiling_config.get_tile_centers(
            config_label=self.ids['tiling_size_spinner'].text,
            focal_length=settings['objective']['focal_length'],
            frame_size=settings['frame'],
            fill_factor=TilingConfig.DEFAULT_FILL_FACTORS['position']
        )
        
        self._protocol_df = self.create_empty_protocol()

        # Z-stack related
        def _zstack_positions() -> list[float]:
            zstack_settings = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['zstack_id']
            range = float(zstack_settings.ids['zstack_range_id'].text)
            step_size = float(zstack_settings.ids['zstack_stepsize_id'].text)
            z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
                text_label=zstack_settings.ids['zstack_spinner'].text
            )

            current_pos = lumaview.scope.get_current_position('Z')

            zstack_config = ZStackConfig(
                range=range,
                step_size=step_size,
                current_z_reference=z_reference,
                current_z_value=current_pos
            )

            if zstack_config.number_of_steps() <= 0:
                return []

            # begin moving to the first position
            return zstack_config.step_positions()

        use_zstacking = self.ids['acquire_zstack_id'].active
        if use_zstacking:
            zstack_positions = _zstack_positions()
        else:
            zstack_positions = {None: None}

        tile_group_id = 0
        zstack_group_id = 0

        # Iterate through all the positions in the scan
        for pos in current_labware.pos_list:
            for tile_label, tile_position in tiles.items():
                for zstack_slice, zstack_position in zstack_positions.items():
                    # Iterate through all the colors to create the steps
                    for layer in common_utils.get_layers():
                        if settings[layer]['acquire'] == False:
                            continue
                        
                        x = round(pos[0] + tile_position["x"]/1000, common_utils.max_decimal_precision('x')) # in 'plate' coordinates
                        y = round(pos[1] + tile_position["y"]/1000, common_utils.max_decimal_precision('y')) # in 'plate' coordinates

                        if use_zstacking:
                            z = zstack_position
                        else:
                            z = settings[layer]['focus']

                        z = round(z, common_utils.max_decimal_precision('z'))

                        af = settings[layer]['autofocus']
                        fc = settings[layer]['false_color']
                        ill = round(settings[layer]['ill'], common_utils.max_decimal_precision('illumination'))
                        gain = round(settings[layer]['gain'], common_utils.max_decimal_precision('gain'))
                        auto_gain = common_utils.to_bool(settings[layer]['auto_gain'])
                        exp = round(settings[layer]['exp'], common_utils.max_decimal_precision('exposure'))
                        objective = settings['objective']['ID']
                        custom_step = False
                        well_label = current_labware.get_well_label(x=pos[0], y=pos[1])

                        if zstack_slice in ("", None):
                            zstack_slice_label = -1
                        else:
                            zstack_slice_label = zstack_slice

                        if tile_label == "":
                            tile_group_id_label = -1
                        else:
                            tile_group_id_label = tile_group_id

                        if zstack_slice is None:
                            zstack_group_id_label = -1
                        else:
                            zstack_group_id_label = zstack_group_id
                        
                        step_dict = self.create_step_dict(
                            name="",
                            x=x,
                            y=y,
                            z=z,
                            af=af,
                            color=layer,
                            fc=fc,
                            ill=ill,
                            gain=gain,
                            auto_gain=auto_gain,
                            exp=exp,
                            objective=objective,
                            well=well_label,
                            tile=tile_label,
                            zslice=zstack_slice_label,
                            custom_step=custom_step,
                            tile_group_id=tile_group_id_label,
                            zstack_group_id=zstack_group_id_label
                        )
                        steps.append(step_dict)
                
                if zstack_slice is not None:
                    zstack_group_id += 1

            if tile_label != "":
                tile_group_id += 1

        self.add_steps_to_protocol(steps=steps)

        self.curr_step = 0 # start at the first step
        self.update_step_ui()
        settings['protocol']['filepath'] = ''        
        self.ids['protocol_filename'].text = ''


    def _validate_labware(self, labware: str):
        scope_configs = lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]

        # If XY motion is available, any type of labware is acceptable
        if selected_scope_config['XYStage'] is True:
            return True, labware
        
        # If XY motion is not available, only Center Plate
        if labware == "Center Plate":
            return True, labware
        else:
            return False, "Center Plate"
            

    # Load Protocol from File
    def load_protocol(self, filepath="./data/new_default_protocol.tsv"):
        logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

        # Load protocol
        with open(filepath, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t') # access the file using the CSV library
            verify = next(csvreader)
            if not (verify[0] == 'LumaViewPro Protocol'):
                return
            
            version_row = next(csvreader)
            if version_row[0] == "Version":
                version = int(version_row[1])
                period_row = next(csvreader)
            else:
                period_row = version_row
 
            period = float(period_row[1])
            duration = next(csvreader)
            duration = float(duration[1])
            labware = next(csvreader)
            labware = labware[1]

            orig_labware = labware
            labware_valid, labware = self._validate_labware(labware=orig_labware)
            if not labware_valid:
                logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

            # Search for "Steps" to indicate start of steps
            while True:
                tmp = next(csvreader)
                if len(tmp) == 0:
                    continue

                if tmp[0] == "Steps":
                    break

            table_lines = []
            for line in fp:
                table_lines.append(line)
            
            table_str = ''.join(table_lines)
            new_protocol_df = pd.read_csv(io.StringIO(table_str), sep='\t', lineterminator='\n')

            # Since there is currently no versioning in the protocol file, this is a workaround to add 'Objective'
            # to the protocol file, and still be able to load older protocol files which do not contain an 'Objective' column
            # for row in csvreader:

            #     for column in ('Objective', 'Tile', 'Custom Step', 'Well', 'Z-Slice', 'Tile Group ID', 'Z-Stack Group ID'):
            #         if column not in header:
            #             row.append(0)

        self._protocol_df = new_protocol_df.fillna('')
        
        # Extract tiling config from step names      
        tiling_config_label = self.tiling_config.determine_tiling_label_from_names(names=self._protocol_df['Name'])
        if tiling_config_label is not None:
            self.ids['tiling_size_spinner'].text = tiling_config_label
        else:
            self.ids['tiling_size_spinner'].text = self.tiling_config.no_tiling_label()

        settings['protocol']['filepath'] = filepath
        self.ids['protocol_filename'].text = os.path.basename(filepath)

        # Update GUI
        self.curr_step = 0 # start at first step

        step = self.get_curr_step()
        if len(self._protocol_df) > 0:
            self.ids['step_name_input'].text = step['Name']
            if step['Name'] == '':
                self.ids['step_name_input'].hint_text = self.get_default_name_for_curr_step()
            self.ids['step_number_input'].text = str(self.curr_step+1)
        else:
            self.ids['step_number_input'].text = '0'
            self.ids['step_name_input'].text = ''
            self.ids['step_name_input'].hint_text = 'Step Name'

        self.ids['step_total_input'].text = str(len(self._protocol_df))
        self.ids['capture_period'].text = str(period)
        self.ids['capture_dur'].text = str(duration)
       
        # Update Protocol
        settings['protocol']['period'] = period
        settings['protocol']['duration'] = duration
        settings['protocol']['labware'] = labware
        
        # Update Labware Selection in Spinner
        self.ids['labware_spinner'].text = settings['protocol']['labware']

        self.go_to_step()
    

    def get_default_name_for_curr_step(self):
        step = self.get_curr_step()
        return common_utils.generate_default_step_name(
            well_label=step["Well"],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            tile_label=step['Tile']
        )


    # Save Protocol to File
    def save_protocol(self, filepath='', update_protocol_filepath: bool = True):
        logger.info('[LVP Main  ] ProtocolSettings.save_protocol()')

        # Gather information
        period = settings['protocol']['period']
        duration = settings['protocol']['duration']
        labware = settings['protocol']['labware']

        if (type(filepath) == str) and len(filepath)==0:
            # If there is no current file path, "save" button will act as "save as" 
            if len(settings['protocol']['filepath']) == 0:
                FileSaveBTN_instance=FileSaveBTN()
                FileSaveBTN_instance.choose('saveas_protocol')
                return
            filepath = settings['protocol']['filepath']
        else:

            if (type(filepath) == str) and (filepath[-4:].lower() != '.tsv'):
                filepath = filepath+'.tsv'

            if update_protocol_filepath:
                settings['protocol']['filepath'] = filepath

        if (type(filepath) == str) and (filepath[-4:].lower() != '.tsv'):
            filepath = filepath+'.tsv'

        self.ids['protocol_filename'].text = os.path.basename(filepath)

        version = 2

        # Write a TSV file
        with open(filepath, 'w') as fp:
            csvwriter = csv.writer(fp, delimiter='\t', lineterminator='\n') # access the file using the CSV library

            csvwriter.writerow(['LumaViewPro Protocol'])
            csvwriter.writerow(['Version', version])
            csvwriter.writerow(['Period', period])
            csvwriter.writerow(['Duration', duration])
            csvwriter.writerow(['Labware', labware])
            
            fp.write('\nSteps\n')

            protocol_table_str = self._protocol_df.to_csv(
                sep='\t',
                lineterminator='\n',
                index=False
            )
            fp.write(protocol_table_str)

    #
    # Multiple Exposures
    # ------------------------------
    #
    # # increase exposure count
    # def exposures_down_button(self):
    #     logger.info('[LVP Main  ] ProtocolSettings.exposures_up_button()')
    #     self.exposures = max(self.exposures-1,1)
    #     self.ids['exposures_number_input'].text = str(self.exposures)
        
    # # increase exposure count
    # def exposures_up_button(self):
    #     logger.info('[LVP Main  ] ProtocolSettings.exposures_up_button()')
    #     self.exposures = self.exposures+1
    #     self.ids['exposures_number_input'].text = str(self.exposures)
    
    #
    # Edit steps
    # ------------------------------
    #
    # Goto to Previous Step
    def prev_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        if len(self._protocol_df) <= 0:
            return
        self.curr_step = max(self.curr_step - 1, 0)
        self.ids['step_number_input'].text = str(self.curr_step+1)
        self.go_to_step()
 
    # Go to Next Step
    def next_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        if len(self._protocol_df) <= 0:
            return
        self.curr_step = min(self.curr_step + 1, len(self._protocol_df)-1)
        self.ids['step_number_input'].text = str(self.curr_step+1)
        self.go_to_step()
    
    # Go to Input Step
    def go_to_step(self, ignore_auto_gain: bool = False):
        logger.info('[LVP Main  ] ProtocolSettings.go_to_step()')

        if len(self._protocol_df) <= 0:
            self.ids['step_number_input'].text = '0'
            return
        
        # Get the Current Step Number
        self.curr_step = int(self.ids['step_number_input'].text)-1

        if self.curr_step < 0 or self.curr_step > len(self._protocol_df):
            self.ids['step_number_input'].text = '0'
            return
        
        step = self.get_curr_step()
        self.ids['step_name_input'].text = step["Name"]
        if step['Name'] == '':
            self.ids['step_name_input'].hint_text = self.get_default_name_for_curr_step()

        # Convert plate coordinates to stage coordinates
        sx, sy = self.plate_to_stage(step["X"], step["Y"])

        # Move into position
        if lumaview.scope.motion.driver:
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', step["Z"])
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

        color = step['Color']
        layer  = lumaview.ids['imagesettings_id'].ids[color]

        # open ImageSettings
        lumaview.ids['imagesettings_id'].ids['toggle_imagesettings'].state = 'down'
        lumaview.ids['imagesettings_id'].toggle_settings()
        
        
        # set accordion item to corresponding channel
        id = f"{color}_accordion"
        lumaview.ids['imagesettings_id'].ids[id].collapse = False

        # set autofocus checkbox
        logger.info(f'[LVP Main  ] autofocus: {step["Auto_Focus"]}')
        settings[color]['autofocus'] = step['Auto_Focus']
        layer.ids['autofocus'].active = step['Auto_Focus']
        
        # set false_color checkbox
        logger.info(f'[LVP Main  ] false_color: {step["False_Color"]}')
        settings[color]['false_color'] = step['False_Color']
        layer.ids['false_color'].active = step['False_Color']

        # set illumination settings, text, and slider
        logger.info(f'[LVP Main  ] ill: {step["Illumination"]}')
        settings[color]['ill'] = step["Illumination"]
        layer.ids['ill_text'].text = str(step["Illumination"])
        layer.ids['ill_slider'].value = float(step["Illumination"])

        # set gain settings, text, and slider
        logger.info(f'[LVP Main  ] gain: {step["Gain"]}')
        settings[color]['gain'] = step["Gain"]
        layer.ids['gain_text'].text = str(step["Gain"])
        layer.ids['gain_slider'].value = float(step["Gain"])

        # set auto_gain checkbox
        logger.info(f'[LVP Main  ] auto_gain: {step["Auto_Gain"]}')
        settings[color]['auto_gain'] = step["Auto_Gain"]
        layer.ids['auto_gain'].active = step["Auto_Gain"]

        # set exposure settings, text, and slider
        logger.info(f'[LVP Main  ] exp: {step["Exposure"]}')
        settings[color]['exp'] = step["Exposure"]
        layer.ids['exp_text'].text = str(step["Exposure"])
        layer.ids['exp_slider'].value = float(step["Exposure"])

        # update position in stage control
        lumaview.ids['motionsettings_id'].update_xy_stage_control_gui()

        layer.apply_settings(ignore_auto_gain=ignore_auto_gain)


    # Delete Current Step of Protocol
    def delete_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.delete_step()')

        if len(self._protocol_df) < 1:
            return
        
        self._protocol_df.drop(index=self.curr_step, axis=0, inplace=True)
        self._protocol_df.reset_index(drop=True, inplace=True)
        self.curr_step = max(self.curr_step-1, 0)
 
        # Update total number of steps to GUI
        self.ids['step_total_input'].text = str(len(self._protocol_df))

        if len(self._protocol_df) == 0:
            self.ids['step_number_input'].text = '0'

        self.next_step()

    # Modify Current Step of Protocol
    def modify_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if len(self._protocol_df) < 1:
            return

        self._protocol_df.at[self.curr_step, "Name"] = self.ids['step_name_input'].text

        # Determine and update plate position
        if lumaview.scope.motion.driver:
            sx = lumaview.scope.get_current_position('X')
            sy = lumaview.scope.get_current_position('Y')
            px, py = self.stage_to_plate(sx, sy)

            self._protocol_df.at[self.curr_step, "X"] = round(px, common_utils.max_decimal_precision('x'))
            self._protocol_df.at[self.curr_step, "Y"] = round(py, common_utils.max_decimal_precision('y'))
            self._protocol_df.at[self.curr_step, "Z"] = round(lumaview.scope.get_current_position('Z'), common_utils.max_decimal_precision('z'))
        else:
            logger.warning('[LVP Main  ] Motion controller not availabble.')

        c_layer = False

        for layer in common_utils.get_layers():
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                c_layer = layer
                break

        if c_layer == False:
            mssg = "No layer currently selected"
            return mssg

        layer_id = lumaview.ids['imagesettings_id'].ids[c_layer]
        
        self._protocol_df.at[self.curr_step, "Auto_Focus"] = layer_id.ids['autofocus'].active
        self._protocol_df.at[self.curr_step, "Color"] = c_layer
        self._protocol_df.at[self.curr_step, "False_Color"] = layer_id.ids['false_color'].active
        self._protocol_df.at[self.curr_step, "Illumination"] = round(layer_id.ids['ill_slider'].value, common_utils.max_decimal_precision('illumination'))
        self._protocol_df.at[self.curr_step, "Gain"] = round(layer_id.ids['gain_slider'].value, common_utils.max_decimal_precision('gain'))
        self._protocol_df.at[self.curr_step, "Auto_Gain"] = layer_id.ids['auto_gain'].active
        self._protocol_df.at[self.curr_step, "Exposure"] = round(layer_id.ids['exp_slider'].value, common_utils.max_decimal_precision('exposure'))
        self._protocol_df.at[self.curr_step, "Objective"] = settings['objective']['ID']


    # Insert Current Step to Protocol at Current Position
    def insert_step(self):
        
        logger.info('[LVP Main  ] ProtocolSettings.insert_step()')

         # Determine Values
        name = f"custom{self.custom_step_count}"
        self.ids['step_name_input'].text = name
        # if step['Name'] == '':
        #         self.ids['step_name_input'].hint_text = self.get_default_name_for_curr_step()
        self.custom_step_count += 1
        c_layer = False

        for layer in common_utils.get_layers():
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

        well = ""
        tile = "" # Manually inserted step is not a tile
        zslice = -1
        custom_step = True
        tile_group_id = -1
        zstack_group_id = -1
        z = lumaview.scope.get_current_position('Z')

        step_dict = self.create_step_dict(
            name=name,
            x=round(px, common_utils.max_decimal_precision('x')),
            y=round(py, common_utils.max_decimal_precision('y')),
            z=round(z, common_utils.max_decimal_precision('z')),
            af=layer_id.ids['autofocus'].active,
            color=c_layer,
            fc=layer_id.ids['false_color'].active,
            ill=round(layer_id.ids['ill_slider'].value, common_utils.max_decimal_precision('illumination')),
            gain=round(layer_id.ids['gain_slider'].value, common_utils.max_decimal_precision('gain')),
            auto_gain=layer_id.ids['auto_gain'].active,
            exp=round(layer_id.ids['exp_slider'].value, common_utils.max_decimal_precision('exposure')),
            objective=settings['objective']['ID'],
            well=well,
            tile=tile,
            zslice=zslice,
            custom_step=custom_step,
            tile_group_id=tile_group_id,
            zstack_group_id=zstack_group_id
        )

        line = pd.DataFrame(data=step_dict, index=[self.curr_step-0.5])
        self._protocol_df = pd.concat([self._protocol_df, line], ignore_index=False, axis=0)
        self._protocol_df = self._protocol_df.sort_index().reset_index(drop=True)

        self.ids['step_total_input'].text = str(len(self._protocol_df))

        # Handle special case for inserting a step from an empty protocol
        if len(self._protocol_df) == 1:
            self.ids['step_number_input'].text = '1'
            self.go_to_step()


    def update_acquire_zstack(self):
        pass
        # self.determine_and_set_run_autofocus_scan_allow()


    def update_tiling_selection(self):
        pass
        # self.determine_and_set_run_autofocus_scan_allow()


    def determine_and_set_run_autofocus_scan_allow(self):
        tiling = self.ids['tiling_size_spinner'].text
        zstack = self.ids['acquire_zstack_id'].active
        if (zstack == True) and (tiling != '1x1'):
            self.set_run_autofocus_scan_allow(allow=False)
        else:
            self.set_run_autofocus_scan_allow(allow=True)


    # Run one scan of protocol, autofocus at each step, and update protocol
    def run_zstack_scan(self):
        logger.debug('[LVP Main  ] ProtocolSettings.run_zstack_scan() not yet implemented')
        #logger.info('[LVP Main  ] ProtocolSettings.run_zstack_scan()')
        # TODO


    def set_run_autofocus_scan_allow(self, allow: bool):
        if allow:
            self.ids['run_autofocus_btn'].disabled = False
        else:
            self.ids['run_autofocus_btn'].disabled = True

        
    # Run one scan of protocol, autofocus at each step, and update protocol
    def run_autofocus_scan(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_autofocus_scan()')

        # If there are no steps, do not continue
        if len(self._protocol_df) < 1:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
            return

        # If the toggle button is in the down position: start autofocus scan
        if self.ids['run_autofocus_btn'].state == 'down':
            self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # begin at current step (curr_step = 0)
            self.curr_step = 0
            self.ids['step_number_input'].text = str(self.curr_step+1)

            step = self._protocol_df.iloc[self.curr_step]
 
            # Convert plate coordinates to stage coordinates
            sx, sy = self.plate_to_stage(step["X"], step["Y"])

            # Move into position
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', step["Z"])

            logger.info('[LVP Main  ] Clock.schedule_interval(self.autofocus_scan_iterate, 0.1)')
            Clock.schedule_interval(self.autofocus_scan_iterate, 0.1)

        # If the toggle button is in the up position: Stop Running Autofocus Scan
        else:  # self.ids['run_autofocus_btn'].state =='normal'
            self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'

            # toggle all LEDs AND TOGGLE BUTTONS OFF
            scope_leds_off()

            for layer in common_utils.get_layers():
                lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'

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
            self._protocol_df.at[self.curr_step, "Z"] = lumaview.scope.get_current_position('Z')

            # increment to the next step
            self.curr_step += 1

            # determine and go to next positions
            if self.curr_step < len(self._protocol_df):
                # Update Step number text
                self.ids['step_number_input'].text = str(self.curr_step+1)
                self.go_to_step()

            # if all positions have already been reached
            else:
                logger.info('[LVP Main  ] Autofocus Scan Complete')
                self.ids['run_autofocus_btn'].state = 'normal'
                self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'
                scope_leds_off()


                logger.info('[LVP Main  ] Clock.unschedule(self.autofocus_scan_iterate)')
                Clock.unschedule(self.autofocus_scan_iterate) # unschedule all copies of scan iterate
            
            return

        # Check if at desired position 
        x_status = lumaview.scope.get_target_status('X')
        y_status = lumaview.scope.get_target_status('Y')
        z_status = lumaview.scope.get_target_status('Z')

        # If target location has been reached
        if (not x_status) or (not y_status) or (not z_status) or lumaview.scope.get_overshoot():
            return
        
        logger.info(f"[LVP Main  ] Autofocus Scan Step ({self.curr_step}): {self._protocol_df.iloc[self.curr_step]['Name']}")

        step = self.get_curr_step()
        
        # set camera settings and turn on LED
        lumaview.scope.leds_off()
        lumaview.scope.led_on(step['Color'], step['Illumination'])
        lumaview.scope.set_gain(step['Gain'])
        lumaview.scope.set_auto_gain(step['Auto_Gain'], target_brightness=settings['protocol']['autogain']['target_brightness'])
        lumaview.scope.set_exposure_time(step['Exposure'])

        # Begin autofocus routine
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['autofocus_id'].state = 'down'
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].autofocus()
        return

    def _initialize_protocol_data_folder(self):
        if os.path.basename(settings['protocol']['filepath']) == "":
            protocol_filename = "unsaved_protocol.tsv"
        else:
            protocol_filename = os.path.basename(settings['protocol']['filepath'])

        # Create the folder to save the protocol captures and protocol itself
        save_folder = pathlib.Path(settings['live_folder']) / PROTOCOL_DATA_DIR_NAME
        save_folder.mkdir(parents=True, exist_ok=True)
        self.protocol_run_dir = self._create_protocol_run_folder(parent_dir=save_folder)
        protocol_filepath = self.protocol_run_dir / protocol_filename
        self.save_protocol(
            filepath=protocol_filepath,
            update_protocol_filepath=False
        )

        protocol_record_filepath = self.protocol_run_dir / ProtocolExecutionRecord.DEFAULT_FILENAME
        self.protocol_execution_record = ProtocolExecutionRecord(outfile=protocol_record_filepath)


    def get_curr_step(self):
        if len(self._protocol_df) == 0:
            return None
        
        return self._protocol_df.iloc[self.curr_step]


    # Run one scan of the protocol
    def run_scan(self, protocol = False):
        logger.info('[LVP Main  ] ProtocolSettings.run_scan()')

        # If there are no steps, do not continue
        if len(self._protocol_df) < 1:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'
            return

        # If the toggle button is in the down position: Start Running Scan
        if self.ids['run_scan_btn'].state == 'down' or protocol == True:
            self.ids['run_scan_btn'].text = 'Running Scan'
            
            # This handles the case where the interval between scans is too short
            # for the amount of steps in a protocol.  If the previous scan is not
            # complete when the interval time occurs, this will at least increment
            # the scan count so that images have the correct sequence number
            if protocol == True:
                if self.scan_in_progress == True:
                    logger.warning('[LVP Main  ] Next scan in protocol started before previous scan completed.')
                    self.scan_count += 1

            self.scan_in_progress = True 

            # When only running a single scan (instead of a protocol)
            # do similar setup as is done for protocol
            if protocol is False:
                self.separate_folder_per_channel = self.ids['protocol_channel_per_folder_id'].active
                self._initialize_protocol_data_folder()
                
            # TODO: shut off live updates

            # reset the is_complete flag on autofocus
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

            # begin at current step set to 0 (curr_step = 0)
            self.curr_step = 0
            self.ids['step_number_input'].text = str(self.curr_step+1)
            self.go_to_step()
            
            step = self.get_curr_step()
   
            # Convert plate coordinates to stage coordinates
            sx, sy = self.plate_to_stage(step["X"], step["Y"])

            # Move into position
            lumaview.scope.move_absolute_position('X', sx)
            lumaview.scope.move_absolute_position('Y', sy)
            lumaview.scope.move_absolute_position('Z', step["Z"])

            logger.info('[LVP Main  ] Clock.schedule_interval(self.scan_iterate, 0.1)')
            Clock.schedule_interval(self.scan_iterate, 0.1)

        # If the toggle button is in the up position: Stop Running Scan
        else:  # self.ids['run_scan_btn'].state =='normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'

            # toggle all LEDs AND TOGGLE BUTTONS OFF
            scope_leds_off()

            for layer in common_utils.get_layers():
                lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.protocol_execution_record.complete()

    
    def perform_grease_redistribution(self):
        z_orig = lumaview.scope.get_current_position('Z')
        logger.info('[LVP Main  ] Performing Z-axis grease redistribution')
        lumaview.scope.move_absolute_position('Z', 0)
        z_status = False
        while not z_status:
            z_status = lumaview.scope.get_target_status('Z')
            time.sleep(0.1)
        lumaview.scope.move_absolute_position('Z', z_orig)
        z_status = False
        while not z_status:
            z_status = lumaview.scope.get_target_status('Z')
            time.sleep(0.1)
        logger.info('[LVP Main  ] Grease redistribution complete')
        

    def scan_iterate(self, dt):
        global lumaview
        global settings
        global auto_gain_countdown

        # If the autofocus is currently active, leave the function before continuing step
        is_autofocus = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_autofocus
        if is_autofocus:
            return

        # Identify if an autofocus cycle completed
        autofocus_is_complete = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete

        # Check if at desired position
        x_status = lumaview.scope.get_target_status('X')
        y_status = lumaview.scope.get_target_status('Y')
        z_status = lumaview.scope.get_target_status('Z')

        # Check if target location has not been reached yet
        if (not x_status) or (not y_status) or (not z_status) or lumaview.scope.get_overshoot():
            return
        
        step = self.get_curr_step()
        logger.info(f"[LVP Main  ] Scan Step: {step['Name']}")
        
        if step["Auto_Focus"]:
            self.autofocus_was_used = True

        # Set camera settings
        lumaview.scope.set_auto_gain(step['Auto_Gain'], target_brightness=settings['protocol']['autogain']['target_brightness'])
        lumaview.scope.led_on(lumaview.scope.color2ch(step['Color']), step['Illumination'])

        if not step['Auto_Gain']:
            lumaview.scope.set_gain(step['Gain'])
            # 2023-12-18 Instead of using only auto gain, now it's auto gain + exp. If auto gain is enabled, then don't set exposure time
            lumaview.scope.set_exposure_time(step['Exposure'])

        
        if step['Auto_Gain'] and auto_gain_countdown > 0:
            auto_gain_countdown -= 0.1
        
        # If the autofocus is selected, is not currently running and has not completed, begin autofocus
        if step['Auto_Focus'] and not autofocus_is_complete:
            # turn on LED
            # lumaview.scope.leds_off()
            # lumaview.scope.led_on(ch, ill)

            # Begin autofocus routine
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['autofocus_id'].state = 'down'
            lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].autofocus()
            
            return
        
        # Check if autogain has time-finished after auto-focus so that they can run in parallel
        if step['Auto_Gain'] and auto_gain_countdown > 0:
            return
        else:
            auto_gain_countdown = settings['protocol']['autogain']['max_duration_seconds']
        
        # reset the is_complete flag on autofocus
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False

        # z_slice = common_utils.get_z_slice_from_name(name=self._protocol_df.iloc[self.curr_step]['Name'])
        # z_slice = step['Z-Slice']

        # tile_label = common_utils.get_tile_label_from_name(name=self._protocol_df.iloc[self.curr_step]['Name'])

        # if common_utils.is_custom_name(name=step['N']):
        #     custom_name = step_name
        # else:
        #     custom_name = None

        # if step['Name'] in (None, ""):


        # well_label = common_utils.get_well_label_from_name(name=step_name)

        if self.separate_folder_per_channel:
            save_folder = self.protocol_run_dir / ColorChannel(step["Channel"]).name
            save_folder.mkdir(parents=True, exist_ok=True)
        else:
            save_folder = self.protocol_run_dir

        # capture image
        image_filepath = self.custom_capture(
            save_folder=save_folder,
            color=step['Color'],
            illumination=step['Illumination'],
            gain=step['Gain'],
            auto_gain=step['Auto_Gain'],
            exposure=step['Exposure'],
            false_color=step['False_Color'],
            tile_label=step['Tile'],
            z_height_idx=step['Z-Slice'],
            scan_count=self.scan_count,
            custom_name=step['Name'],
            well_label=step['Well']
        )

        if self.enable_image_saving == True:
            if self.separate_folder_per_channel:
                image_filepath_name = pathlib.Path(ColorChannel(step['Channel']).name) / image_filepath.name
            else:
                image_filepath_name = image_filepath.name
        else:
            image_filepath_name = "unsaved"

        self.protocol_execution_record.add_step(
            image_file_name=image_filepath_name,
            step_name=step['Name'],
            step_index=self.curr_step,
            scan_count=self.scan_count,
            timestamp=datetime.datetime.now()
        )

        # increment to the next step
        self.curr_step += 1

        # Disable autogain when moving between steps
        if step['Auto_Gain']:
            lumaview.scope.set_auto_gain(state=False)

        if self.curr_step < len(self._protocol_df):

            # Update Step number text
            self.ids['step_number_input'].text = str(self.curr_step+1)
            self.go_to_step(ignore_auto_gain=True)

        # if all positions have already been reached
        else:
            # At the end of a scan, if autofocus was used, cycle the Z-axis to re-distribute grease
            if self.autofocus_was_used == True:
                self.perform_grease_redistribution()
                self.autofocus_was_used = False

            self.scan_count += 1
            
            logger.info('[LVP Main  ] Scan Complete')
            self.ids['run_scan_btn'].state = 'normal'
            self.ids['run_scan_btn'].text = 'Run One Scan'

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.scan_in_progress = False

    # Run protocol without xy movement
    def run_stationary(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_stationary()')

        if self.ids['run_stationary_btn'].state == 'down':
            self.ids['run_stationary_btn'].text = 'State == Down'
        else:
            self.ids['run_stationary_btn'].text = 'Run Stationary Protocol' # 'normal'


    @staticmethod
    def _create_protocol_run_folder(parent_dir: str | pathlib.Path):
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")
        parent_dir = pathlib.Path(parent_dir)
        protocol_run_dir = parent_dir / time_string
        protocol_run_dir.mkdir(exist_ok=True)
        return protocol_run_dir


    # Run the complete protocol 
    def run_protocol(self):
        global auto_gain_countdown

        logger.info('[LVP Main  ] ProtocolSettings.run_protocol()')
        self.n_scans = int(float(settings['protocol']['duration'])*60 / float(settings['protocol']['period']))
        self.scan_count = 0
        self.autofocus_was_used = False
        self.scan_in_progress = False
        self.start_t = time.time() # start of cycle in seconds

        if self.ids['run_protocol_btn'].state == 'down':

            if ENGINEERING_MODE == True:
                if lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].ids['protocol_disable_image_saving_id'].active == True:
                    self.enable_image_saving = False
                else:
                    self.enable_image_saving = True

            self.separate_folder_per_channel = self.ids['protocol_channel_per_folder_id'].active
            lumaview.scope.camera.update_auto_gain_target_brightness(settings['protocol']['autogain']['target_brightness'])
            auto_gain_countdown = settings['protocol']['autogain']['max_duration_seconds']
            self._initialize_protocol_data_folder()

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            self.run_scan(protocol = True)
            logger.info('[LVP Main  ] Clock.schedule_interval(self.protocol_iterate, 1)')

            # Move to first step when starting run
            self.curr_step = 0
            self.ids['step_number_input'].text = str(self.curr_step+1)
            self.go_to_step()

            Clock.schedule_interval(self.protocol_iterate, 1)

        else:
            self.ids['run_protocol_btn'].text = 'Run Full Protocol' # 'normal'

            logger.info('[LVP Main  ] Clock.unschedule(self.scan_iterate)')
            Clock.unschedule(self.scan_iterate) # unschedule all copies of scan iterate
            logger.info('[LVP Main  ] Clock.unschedule(self.protocol_iterate)')
            Clock.unschedule(self.protocol_iterate) # unschedule all copies of protocol iterate
            # self.protocol_event.cancel()
            scope_leds_off()
 

    def protocol_iterate(self, dt):
        logger.info('[LVP Main  ] ProtocolSettings.protocol_iterate()')

        # Don't start the next scan if the current scan is in progress
        if self.scan_in_progress:
            return

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
        self.ids['run_protocol_btn'].text = f"{n_scans} scans remaining. Press to ABORT"

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
               scope_leds_off()

               self.protocol_execution_record.complete()
               

# Widget for displaying Microscope Stage area, labware, and current position 
class Stage(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)
        logger.info('[LVP Main  ] Stage.__init__()')
        self.ROI_min = [0,0]
        self.ROI_max = [0,0]
        self._motion_enabled = True
        self.ROIs = []
        
    def append_ROI(self, x_min, y_min, x_max, y_max):
        self.ROI_min = [x_min, y_min]
        self.ROI_max = [x_max, y_max]
        self.ROIs.append([self.ROI_min, self.ROI_max])
    
    def set_motion_capability(self, enabled: bool):
        self._motion_enabled = enabled

    def on_touch_down(self, touch):
        logger.info('[LVP Main  ] Stage.on_touch_down()')

        if not self._motion_enabled:
            return

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
            lumaview.ids['motionsettings_id'].update_xy_stage_control_gui()
    

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

            # Needed for green cicles, cross hairs and roi
            # ------------------
            protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']

            # Get target position
            # Outline of Stage Area from Above
            # ------------------
            Color(.2, .2, .2 , 0.5)                # dark grey
            Rectangle(pos=(x+(x_max-stage_w-stage_x)*scale_x, y+stage_y*scale_y),
                           size=(stage_w*scale_x, stage_h*scale_y))

            # Outline of Plate from Above
            # ------------------
            Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
            Line(points=(x, y, x, y+h-15), width = 1)          # Left
            Line(points=(x+w, y, x+w, y+h), width = 1)         # Right
            Line(points=(x, y, x+w, y), width = 1)             # Bottom
            Line(points=(x+15, y+h, x+w, y+h), width = 1)      # Top
            Line(points=(x, y+h-15, x+15, y+h), width = 1)     # Diagonal

            # ROI rectangle
            # ------------------
            if self.ROI_max[0] > self.ROI_min[0]:
                roi_min_x, roi_min_y = protocol_settings.stage_to_pixel(self.ROI_min[0], self.ROI_min[1], scale_x, scale_y)
                roi_max_x, roi_max_y = protocol_settings.stage_to_pixel(self.ROI_max[0], self.ROI_max[1], scale_x, scale_y)
                Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
                Line(rectangle=(x+roi_min_x, y+roi_min_y, roi_max_x - roi_min_x, roi_max_y - roi_min_y))
            
            # Draw all ROI rectangles
            # ------------------
            # TODO (for each step)
            '''
            for ROI in self.ROIs:
                if self.ROI_max[0] > self.ROI_min[0]:
                    roi_min_x, roi_min_y = protocol_settings.stage_to_pixel(self.ROI_min[0], self.ROI_min[1], scale_x, scale_y)
                    roi_max_x, roi_max_y = protocol_settings.stage_to_pixel(self.ROI_max[0], self.ROI_max[1], scale_x, scale_y)
                    Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
                    Line(rectangle=(x+roi_min_x, y+roi_min_y, roi_max_x - roi_min_x, roi_max_y - roi_min_y))
            '''
            
            # Draw all wells
            # ------------------
            cols = current_labware.plate['columns']
            rows = current_labware.plate['rows']
            
            Color(0.4, 0.4, 0.4, 0.5)
            well_spacing_x = current_labware.plate['spacing']['x']
            well_spacing_y = current_labware.plate['spacing']['y']
            well_spacing_pixel_x = well_spacing_x
            well_spacing_pixel_y = well_spacing_y

            well_diameter = current_labware.plate['diameter']
            if well_diameter == -1:
                well_radius_pixel_x = well_spacing_pixel_x
                well_radius_pixel_y = well_spacing_pixel_y
            else:
                well_radius = well_diameter / 2
                well_radius_pixel_x = well_radius * scale_x
                well_radius_pixel_y = well_radius * scale_y

            
            for i in range(cols):
                for j in range(rows):                   
                    well_plate_x, well_plate_y = current_labware.get_well_position(i, j)
                    well_pixel_x, well_pixel_y = protocol_settings.plate_to_pixel(
                        px=well_plate_x,
                        py=well_plate_y,
                        scale_x=scale_x,
                        scale_y=scale_y
                    )
                    x_center = int(x+well_pixel_x) # on screen center
                    y_center = int(y+well_pixel_y) # on screen center
                    Ellipse(pos=(x_center-well_radius_pixel_x, y_center-well_radius_pixel_y), size=(well_radius_pixel_x*2, well_radius_pixel_y*2))

            try:
                target_stage_x = lumaview.scope.get_target_position('X')
                target_stage_y = lumaview.scope.get_target_position('Y')
            except:
                logger.exception('[LVP Main  ] Error talking to Motor board.')
                raise
                
            target_plate_x, target_plate_y = protocol_settings.stage_to_plate(target_stage_x, target_stage_y)

            target_i, target_j = current_labware.get_well_index(target_plate_x, target_plate_y)
            target_well_plate_x, target_well_plate_y = current_labware.get_well_position(target_i, target_j)
            target_well_pixel_x, target_well_pixel_y = protocol_settings.plate_to_pixel(
                px=target_well_plate_x,
                py=target_well_plate_y,
                scale_x=scale_x,
                scale_y=scale_y
            )
            target_well_center_x = int(x+target_well_pixel_x) # on screen center
            target_well_center_y = int(y+target_well_pixel_y) # on screen center
    
            # Green selection circle
            Color(0., 1., 0., 1.)
            Line(circle=(target_well_center_x, target_well_center_y, well_radius_pixel_x))
            
            #  Red Crosshairs
            # ------------------
            if self._motion_enabled:
                x_current = lumaview.scope.get_current_position('X')
                x_current = np.clip(x_current, 0, 120000) # prevents crosshairs from leaving the stage area
                y_current = lumaview.scope.get_current_position('Y')
                y_current = np.clip(y_current, 0, 80000) # prevents crosshairs from leaving the stage area

                # Convert stage coordinates to relative pixel coordinates
                pixel_x, pixel_y = protocol_settings.stage_to_pixel(x_current, y_current, scale_x, scale_y)
                x_center = x+pixel_x
                y_center = y+pixel_y

                Color(1., 0., 0., 1.)
                Line(points=(x_center-10, y_center, x_center+10, y_center), width = 1) # horizontal line
                Line(points=(x_center, y_center-10, x_center, y_center+10), width = 1) # vertical line


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

                if 'autogain' not in settings['protocol']:
                    settings['protocol']['autogain'] = {
                        'max_duration_seconds': 1.0,
                        'target_brightness': 0.3
                    }

                # update GUI values from JSON data:
                self.ids['scope_spinner'].text = settings['microscope']
                self.ids['objective_spinner'].text = settings['objective']['ID']
                # TODO self.ids['objective_spinner'].text = settings['objective']['description']
                self.ids['magnification_id'].text = str(settings['objective']['magnification'])
                self.ids['frame_width_id'].text = str(settings['frame']['width'])
                self.ids['frame_height_id'].text = str(settings['frame']['height'])

                protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
                protocol_settings.ids['capture_period'].text = str(settings['protocol']['period'])
                protocol_settings.ids['capture_dur'].text = str(settings['protocol']['duration'])
                protocol_settings.ids['labware_spinner'].text = settings['protocol']['labware']
                protocol_settings.select_labware()

                zstack_settings = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['zstack_id']
                zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']
                zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
                zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])

                z_reference = common_utils.convert_zstack_reference_position_setting_to_config(text_label=settings['zstack']['position'])
                
                zstack_config = ZStackConfig(
                    range=settings['zstack']['range'],
                    step_size=settings['zstack']['step_size'],
                    current_z_reference=z_reference,
                    current_z_value=None
                )

                zstack_settings.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

                for layer in common_utils.get_layers():
                    lumaview.ids['imagesettings_id'].ids[layer].ids['ill_slider'].value = settings[layer]['ill']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['gain_slider'].value = settings[layer]['gain']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].value = settings[layer]['exp']
                    # lumaview.ids['imagesettings_id'].ids[layer].ids['exp_slider'].value = float(np.log10(settings[layer]['exp']))
                    lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active = settings[layer]['false_color']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['acquire'].active = settings[layer]['acquire']
                    lumaview.ids['imagesettings_id'].ids[layer].ids['autofocus'].active = settings[layer]['autofocus']

                lumaview.scope.set_frame_size(settings['frame']['width'], settings['frame']['height'])
            except:
                logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')
        
        self.set_ui_features_for_scope()


    def update_bullseye_state(self):
        if self.ids['enable_bullseye_btn_id'].state == 'down':
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye = True
        else:
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye = False

    
    def update_crosshairs_state(self):
        if self.ids['enable_crosshairs_btn'].state == 'down':
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_crosshairs = True
        else:
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_crosshairs = False


    # Save settings to JSON file
    def save_settings(self, file="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.save_settings()')
        global settings

        if (type(file) == str) and (file[-5:].lower() != '.json'):
                file = file+'.json'

        os.chdir(source_path)
        with open(file, "w") as write_file:
            json.dump(settings, write_file, indent = 4, cls=CustomJSONizer)

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

        self.set_ui_features_for_scope()


    def set_ui_features_for_scope(self) -> None:
        scope_configs = lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]
        motion_settings =  lumaview.ids['motionsettings_id']
        motion_settings.set_turret_control_visibility(visible=selected_scope_config['Turret'])
        motion_settings.set_xystage_control_visibility(visible=selected_scope_config['XYStage'])
        motion_settings.set_tiling_control_visibility(visible=selected_scope_config['XYStage'])

        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        protocol_settings.set_labware_selection_visibility(visible=selected_scope_config['XYStage'])

        if selected_scope_config['XYStage'] is False:
            protocol_settings.select_labware(labware="Center Plate")

        protocol_settings.ids['stage_widget_id'].set_motion_capability(enabled=selected_scope_config['XYStage'])
        protocol_settings.ids['stage_widget_id'].draw_labware()

           
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

        fov_size = common_utils.get_field_of_view(
            focal_length=settings['objective']['focal_length'],
            frame_size=settings['frame']
        )
        self.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
        self.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))


    def frame_size(self):
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        global lumaview
        global settings

        w = int(self.ids['frame_width_id'].text)
        h = int(self.ids['frame_height_id'].text)

        width = int(min(w, lumaview.scope.get_max_width())/4)*4
        height = int(min(h, lumaview.scope.get_max_height())/4)*4

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame_width_id'].text = str(width)
        self.ids['frame_height_id'].text = str(height)

        fov_size = common_utils.get_field_of_view(
            focal_length=settings['objective']['focal_length'],
            frame_size=settings['frame']
        )
        self.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
        self.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))

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
        Clock.schedule_once(self._init_ui, 0)
    
    
    def _init_ui(self, dt=0):
        self.update_auto_gain()

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

    def update_auto_gain(self):
        logger.info('[LVP Main  ] LayerControl.update_auto_gain()')
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


    def update_led_state(self):
        enabled = True if self.ids['enable_led_btn'].state == 'down' else False
        illumination = settings[self.layer]['ill']

        self.set_led_state(enabled=enabled, illumination=illumination)
        
        # self.apply_settings()

    
    def set_led_state(self, enabled: bool, illumination: float):
        if not lumaview.scope.led:
            logger.warning('[LVP Main  ] LED controller not available.')
            return

        channel=lumaview.scope.color2ch(self.layer)
        if not enabled:
            lumaview.scope.led_off(channel=channel)
        else:
            logger.info(f'[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch({self.layer}), {illumination})')
            lumaview.scope.led_on(channel=channel, mA=illumination)          


    def apply_settings(self, ignore_auto_gain=False):
        logger.info('[LVP Main  ] LayerControl.apply_settings()')
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        set_histogram_layer(active_layer=self.layer)
        self.update_led_state()
        if self.ids['enable_led_btn'].state == 'down': # if the button is down
            for layer in common_utils.get_layers():
                if layer != self.layer:
                    lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'

        # update gain to currently selected settings
        # -----------------------------------------------------
        auto_gain_enabled = settings[self.layer]['auto_gain']
        auto_gain_target_brightness = settings['protocol']['autogain']['target_brightness']
        if not ignore_auto_gain:
            lumaview.scope.set_auto_gain(auto_gain_enabled, target_brightness=auto_gain_target_brightness)


        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = settings[self.layer]['exp']
        gain = settings[self.layer]['gain']

        if not auto_gain_enabled:
            lumaview.scope.set_gain(gain)
            lumaview.scope.set_exposure_time(exposure)
        
        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        self.update_shader(dt=0)

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

        settings['zstack']['step_size'] = float(self.ids['zstack_stepsize_id'].text)
        settings['zstack']['range'] = float(self.ids['zstack_range_id'].text)

        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=self.ids['zstack_spinner'].text
        )

        zstack_config = ZStackConfig(
            range=settings['zstack']['range'],
            step_size=settings['zstack']['step_size'],
            current_z_reference=z_reference,
            current_z_value=None
        )

        self.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())


    def set_position(self):
        settings['zstack']['position'] = self.ids['zstack_spinner'].text


    def aquire_zstack(self):
        logger.info('[LVP Main  ] ZStack.aquire_zstack()')
        global lumaview

        range = float(self.ids['zstack_range_id'].text)
        step_size = float(self.ids['zstack_stepsize_id'].text)
        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=self.ids['zstack_spinner'].text
        )

        current_pos = lumaview.scope.get_current_position('Z')

        zstack_config = ZStackConfig(
            range=range,
            step_size=step_size,
            current_z_reference=z_reference,
            current_z_value=current_pos
        )

        if zstack_config.number_of_steps() <= 0:
            return False

        # begin moving to the first position
        self.positions = zstack_config.step_positions()
        self.n_pos = 0
        lumaview.scope.move_absolute_position('Z', self.positions[self.n_pos])

        if self.ids['zstack_aqr_btn'].state == 'down':
            logger.info('[LVP Main  ] Clock.schedule_interval(self.zstack_iterate, 0.01)')
            Clock.schedule_interval(self.zstack_iterate, 0.01)
            self.ids['zstack_aqr_btn'].text = 'Acquiring ZStack'

        else:
            self.ids['zstack_aqr_btn'].text = 'Acquire'
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
                self.ids['zstack_aqr_btn'].text = 'Acquire'
                self.ids['zstack_aqr_btn'].state = 'normal'
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
        elif self.context == 'load_cell_count_input_image':
            filechooser.open_file(on_selection=self.handle_selection, filters = ["*.tif?","*.jpg","*.bmp","*.png","*.gif"])
        elif self.context == 'load_cell_count_method':
            filechooser.open_file(on_selection=self.handle_selection, filters = ["*.json"]) 

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        logger.info('[LVP Main  ] FileChooseBTN.on_selection_function()')
        global lumaview
        
        if self.selection:
            if self.context == 'load_settings':
                lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].load_settings(self.selection[0])

            elif self.context == 'load_protocol':
                lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(filepath = self.selection[0])
            
            elif self.context == 'load_cell_count_input_image':
                cell_count_content.set_preview_source_file(file=self.selection[0])

            elif self.context == 'load_cell_count_method':
                cell_count_content.load_method_from_file(file=self.selection[0])

        else:
            return

# Button the triggers 'filechooser.choose_dir()' from plyer
class FolderChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info(f'[LVP Main  ] FolderChooseBTN.choose({context})')
        self.context = context

        # Show previously selected/default folder
        if self.context == "apply_stitching_to_folder":
            selected_path = pathlib.Path(settings['live_folder']) / PROTOCOL_DATA_DIR_NAME
            if selected_path.exists() is False:
                selected_path = pathlib.Path(settings['live_folder'])
            
            selected_path = str(selected_path)
        else:
            selected_path = settings['live_folder']


        # Note: Could likely use tkinter filedialog for all platforms
        # works on windows and MacOSX
        # but needs testing on Linux
        if sys.platform in ('win32','darwin'):
            # Tested for Windows/Mac platforms
            selection = tkinter_filedialog.askdirectory(
                initialdir=selected_path
            )

            # Nothing selected/cancel
            if selection == '':
                return
            
            self.handle_selection(selection=[selection])     
        else:
            filechooser.choose_dir(
                on_selection=self.handle_selection
                # path=selected_path
            )
            return
        

    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FolderChooseBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()


    def on_selection_function(self, *a, **k):
        global settings
        logger.info('[LVP Main  ] FolderChooseBTN.on_selection_function()')
        if self.selection:
            path = self.selection[0]
        else:
            return

        if self.context == 'live_folder':
            settings['live_folder'] = path

        elif self.context == 'video_input_images_folder':
            video_creation_controls.set_input_images_loc(directory=path)

        elif self.context == 'apply_cell_count_method_to_folder':
            cell_count_content.apply_method_to_folder(
                path=path
            )
        elif self.context == 'apply_stitching_to_folder':
            stitch_controls.run_stitcher(path=pathlib.Path(path))
        elif self.context == 'apply_composite_gen_to_folder':
            composite_gen_controls.run_composite_gen(path=pathlib.Path(path))
        else:
            raise Exception(f"on_selection_function(): Unknown selection {self.context}")


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
        elif self.context == 'saveas_cell_count_method':
            filechooser.save_file(on_selection=self.handle_selection, filters = ["*.json"])
        elif self.context == 'video_output_path':
            filechooser.save_file(on_selection=self.handle_selection, filters = ["*.avi"])


    def handle_selection(self, selection):
        logger.info('[LVP Main  ] FileSaveBTN.handle_selection()')
        if selection:
            self.selection = selection
            self.on_selection_function()

    def on_selection_function(self, *a, **k):
        logger.info('[LVP Main  ] FileSaveBTN.on_selection_function()')
        global lumaview
        
        if self.context == 'save_settings':
            if self.selection:
                lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].save_settings(self.selection[0])
                logger.info('[LVP Main  ] Saving Settings to File:' + self.selection[0])

        elif self.context == 'saveas_protocol':
            if self.selection:
                lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].save_protocol(filepath = self.selection[0])
                logger.info('[LVP Main  ] Saving Protocol to File:' + self.selection[0])
        
        elif self.context == 'saveas_cell_count_method':
            if self.selection:
                logger.info('[LVP Main  ] Saving Cell Count Method to File:' + self.selection[0])
                filename = self.selection[0]
                if os.path.splitext(filename)[1] == "":
                    filename += ".json"
                cell_count_content.save_method_as(file=filename)
        
        elif self.context == 'video_output_path':
            if self.selection:
                logger.info('[LVP Main  ] Set video output path to file:' + self.selection[0])
                filepath = pathlib.Path(self.selection[0])
                if filepath.suffix == "":
                    filepath = filepath.with_suffix(".avi")
                video_creation_controls.set_output_file_loc(file_loc=filepath)


def load_log_level():
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                log_level = logging.getLevelName(data['logging']['default']['level'])
                logger.setLevel(level=log_level)          
                return
            except:
                pass


def load_mode():
    global ENGINEERING_MODE
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                mode = data['mode']
                if mode == 'engineering':
                    logger.info(f"Enabling engineering mode")
                    ENGINEERING_MODE = True       
                    return
            except:
                pass

        ENGINEERING_MODE = False
             

# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def on_start(self):
        load_log_level()
        load_mode()
        logger.info('[LVP Main  ] LumaViewProApp.on_start()')
        lumaview.scope.xyhome()
        # if profiling:
        #     self.profile = cProfile.Profile()
        #     self.profile.enable()

    def build(self):
        current_time = time.strftime("%m/%d/%Y", time.localtime())
        logger.info('[LVP Main  ] LumaViewProApp.build()')

        logger.info('[LVP Main  ] -----------------------------------------')
        logger.info('[LVP Main  ] Code Compiled On: %s', current_time)
        logger.info('[LVP Main  ] Run Time: ' + time.strftime("%Y %m %d %H:%M:%S"))
        logger.info('[LVP Main  ] -----------------------------------------')

        global Window
        global lumaview
        global cell_count_content
        global video_creation_controls
        global stitch_controls
        global composite_gen_controls
        self.icon = './data/icons/icon.png'

        version = ""
        try:
            with open("version.txt") as f:
                version = f.readlines()[0]
        except:
            pass
        
        self.title = f'LumaViewPro {version}'

        try:
            from kivy.core.window import Window
            #Window.bind(on_resize=self._on_resize)
            lumaview = MainDisplay()
            cell_count_content = CellCountControls()
            #Window.maximize()
        except:
            logger.exception('[LVP Main  ] Cannot open main display.')
            raise

        if getattr(sys, 'frozen', False):
            pyi_splash.close()

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
        Clock.schedule_interval(lumaview.ids['motionsettings_id'].update_xy_stage_control_gui, 0.1) # Includes text boxes, not just stage

        try:
            filepath = settings['protocol']['filepath']
            lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].load_protocol(filepath=filepath)
        except:
            logger.exception('[LVP Main  ] Unable to load protocol at startup')
            # If protocol file is missing or incomplete, file name and path are cleared from memory. 
            filepath=''	
            settings['protocol']['filepath']=''

        lumaview.ids['imagesettings_id'].ids['BF'].apply_settings()
        scope_leds_off()

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

        scope_leds_off()

        lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].save_settings("./data/current.json")

LumaViewProApp().run()
