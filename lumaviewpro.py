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
import copy
import logging
import datetime
import math
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json
import subprocess
import sys
import typing

from lvp_logger import logger
import tkinter
from tkinter import filedialog, Tk
from plyer import filechooser

import modules.profiling_utils as profiling_utils
global profiling_helper
profiling_helper = None


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
Config.set('kivy', 'exit_on_escape', '0')

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
from kivy.input.motionevent import MotionEvent
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

import labware
from modules.autofocus_executor import AutofocusExecutor
from modules.stitcher import Stitcher
from modules.composite_generation import CompositeGeneration
from modules.contrast_stretcher import ContrastStretcher
import modules.coord_transformations as coord_transformations
import modules.labware_loader as labware_loader
import modules.objectives_loader as objectives_loader
from modules.protocol import Protocol
from modules.sequenced_capture_executor import SequencedCaptureExecutor
from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
from modules.stack_builder import StackBuilder
from modules.zstack_config import ZStackConfig
from modules.json_helper import CustomJSONizer
import modules.imagej_helper as imagej_helper
import modules.zprojector as zprojector

import cv2
import skimage

# Hardware
import lumascope_api
import post_processing

import image_utils
import image_utils_kivy

global lumaview
global settings
global cell_count_content

global wellplate_loader
wellplate_loader = None

global objective_helper
objective_helper = None

global coordinate_transformer
coordinate_transformer = None

global ij_helper
ij_helper = None

global sequenced_capture_executor
sequenced_capture_executor = None

# global autofocus_executor
# autofocus_executor = None

global last_save_folder
last_save_folder = None
global stage
stage = None

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


def set_last_save_folder(dir: pathlib.Path | None):
    if dir is None:
        return
    
    global last_save_folder
    last_save_folder=dir


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


def _handle_ui_for_leds_off():
    global lumaview
    for layer in common_utils.get_layers():
        lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = 'normal'


def _handle_ui_for_led(layer: str, enabled: bool, **kwargs):
    global lumaview
    if enabled:
        state = "down"
    else:
        state = "normal"

    lumaview.ids['imagesettings_id'].ids[layer].ids['enable_led_btn'].state = state


def scope_leds_off():
    global lumaview

    if not lumaview.scope.led:
        logger.warning('[LVP Main  ] LED controller not available.')
        return
    
    lumaview.scope.leds_off()
    logger.info('[LVP Main  ] lumaview.scope.leds_off()')
    _handle_ui_for_leds_off()


def is_image_saving_enabled() -> bool:
    if ENGINEERING_MODE == True:
        if lumaview.ids['motionsettings_id'].ids['protocol_settings_id'].ids['protocol_disable_image_saving_id'].active:
            return False
    
    return True


def _update_step_number_callback(step_num: int):
    protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
    protocol_settings.curr_step = step_num-1
    protocol_settings.update_step_ui()


def go_to_step(
    protocol: Protocol,
    step_idx: int,
    ignore_auto_gain: bool = False,
    include_move: bool = True
):
    num_steps = protocol.num_steps()
    protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
    if num_steps <= 0:
        protocol_settings.curr_step = -1
        protocol_settings.update_step_ui()
        return

    if (step_idx < 0) or (step_idx >= num_steps):
        protocol_settings.curr_step = -1
        protocol_settings.update_step_ui()
        return
    
    step = protocol.step(idx=step_idx)
    protocol_settings.ids['step_name_input'].text = step["Name"]
    if step['Name'] == '':
        step_name = common_utils.generate_default_step_name(
            well_label=step["Well"],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            tile_label=step['Tile']
        )
        protocol_settings.ids['step_name_input'].hint_text = step_name

    # Convert plate coordinates to stage coordinates
    if include_move:
        _, labware = get_selected_labware()
        sx, sy = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=step["X"],
            py=step["Y"]
        )

        # Move into position
        if lumaview.scope.motion.driver:
            move_absolute_position('X', sx)
            move_absolute_position('Y', sy)
            move_absolute_position('Z', step["Z"])
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

    layer.apply_settings(ignore_auto_gain=ignore_auto_gain)


def get_binning_from_ui() -> int:
    return int(lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].ids['binning_spinner'].text)


def get_zstack_positions() -> tuple[bool, dict]:
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
        return False, {None: None}

    return True, zstack_config.step_positions()


def get_layer_configs(
    specific_layers: list | None = None,
) -> dict[dict]:
    layer_configs = {}
    for layer in common_utils.get_layers():

        if (specific_layers is not None) and (layer not in specific_layers):
            continue

        layer_configs[layer] = {}
        layer_settings = settings[layer]

        acquire = layer_settings['acquire']
        autofocus = layer_settings['autofocus']
        false_color = layer_settings['false_color']
        illumination = round(layer_settings['ill'], common_utils.max_decimal_precision('illumination'))
        gain = round(layer_settings['gain'], common_utils.max_decimal_precision('gain'))
        auto_gain = common_utils.to_bool(layer_settings['auto_gain'])
        exposure = round(layer_settings['exp'], common_utils.max_decimal_precision('exposure'))
        focus = layer_settings['focus']

        layer_configs[layer] = {
            'acquire': acquire,
            'autofocus': autofocus,
            'false_color': false_color,
            'illumination': illumination,
            'gain': gain,
            'auto_gain': auto_gain,
            'exposure': exposure,
            'focus': focus
        }

    return layer_configs


def get_active_layer_config() -> tuple[str, dict]:
    c_layer = None
    for layer in common_utils.get_layers():
        accordion = layer + '_accordion'
        if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
            c_layer = layer
            break

    if c_layer is None:
        raise Exception("No layer currently selected")
    
    layer_configs = get_layer_configs(
        specific_layers=[c_layer]
    )
    
    return c_layer, layer_configs[c_layer]


def get_current_plate_position():
    if not lumaview.scope.motion.driver:
        logger.error(f"Cannot retrieve current plate position")
        return {
            'x': 0,
            'y': 0,
            'z': 0
        }
    
    pos = lumaview.scope.get_current_position(axis=None)
    _, labware = get_selected_labware()
    px, py = coordinate_transformer.stage_to_plate(
        labware=labware,
        stage_offset=settings['stage_offset'],
        sx=pos['X'],
        sy=pos['Y'],
    )

    return {
        'x': round(px, common_utils.max_decimal_precision('x')),
        'y': round(py, common_utils.max_decimal_precision('y')),
        'z': round(pos['Z'], common_utils.max_decimal_precision('z'))
    }


def get_current_frame_dimensions() -> dict:
    microscope_settings = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']
    try:
        frame_width = int(microscope_settings.ids['frame_width_id'].text)
        frame_height = int(microscope_settings.ids['frame_height_id'].text)
    except:
        raise ValueError(f"Invalid value for frame width/height")
    
    frame = {
        'width': frame_width,
        'height': frame_height
    }
    return frame


def get_protocol_time_params() -> dict:
    protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
    try:
        period = float(protocol_settings.ids['capture_period'].text)
    except:
        period = 1

    period = datetime.timedelta(minutes=period)
    try:
        duration = float(protocol_settings.ids['capture_dur'].text)
    except:
        duration = 1

    duration = datetime.timedelta(hours=duration)

    return {
        'period': period,
        'duration': duration
    }


def get_selected_labware() -> tuple[str, labware.WellPlate]:
    protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
    labware_id = protocol_settings.ids['labware_spinner'].text
    labware = wellplate_loader.get_plate(plate_key=labware_id)
    return labware_id, labware


def get_image_capture_config_from_ui() -> dict:
    microscope_settings = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']
    output_format = {
        'live': microscope_settings.ids['live_image_output_format_spinner'].text,
        'sequenced': microscope_settings.ids['sequenced_image_output_format_spinner'].text,
    }
    use_full_pixel_depth = lumaview.ids['viewer_id'].ids['scope_display_id'].use_full_pixel_depth
    return {
        'output_format': output_format,
        'use_full_pixel_depth': use_full_pixel_depth,
    }

def get_sequenced_capture_config_from_ui() -> dict:
    objective_id, _ = get_current_objective_info()
    time_params = get_protocol_time_params()
    labware_id, _ = get_selected_labware()
    protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
    tiling = protocol_settings.ids['tiling_size_spinner'].text
    use_zstacking = protocol_settings.ids['acquire_zstack_id'].active
    frame_dimensions = get_current_frame_dimensions()
    zstack_positions_valid, zstack_positions = get_zstack_positions()

    layer_configs = get_layer_configs()

    config = {
        'labware_id': labware_id,
        'objective_id': objective_id,
        'zstack_positions': zstack_positions,
        'zstack_positions_valid': zstack_positions_valid,
        'use_zstacking': use_zstacking,
        'tiling': tiling,
        'layer_configs': layer_configs,
        'period': time_params['period'],
        'duration': time_params['duration'],
        'frame_dimensions': frame_dimensions,
        'binning_size': get_binning_from_ui(),
    }

    return config


def get_auto_gain_settings() -> dict:
    autogain_settings = settings['protocol']['autogain'].copy()
    autogain_settings['max_duration'] = datetime.timedelta(seconds=autogain_settings['max_duration_seconds'])
    del autogain_settings['max_duration_seconds']
    return autogain_settings


def create_hyperstacks_if_needed():
    image_capture_config = get_image_capture_config_from_ui()
    if image_capture_config['output_format']['sequenced'] == 'ImageJ Hyperstack':
        _, objective = get_current_objective_info()
        stack_builder = StackBuilder()
        stack_builder.load_folder(
            path=sequenced_capture_executor.run_dir(),
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            binning_size=get_binning_from_ui(),
            focal_length=objective['focal_length'],
        )


def get_current_objective_info() -> tuple[str, dict]:
    objective_id = settings['objective_id']
    objective = objective_helper.get_objective_info(objective_id=objective_id)
    return objective_id, objective


def _handle_ui_update_for_axis(axis: str):
    axis = axis.upper()
    if axis == 'Z':
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].update_gui()
    elif axis in ('X', 'Y', 'XY'):
        lumaview.ids['motionsettings_id'].update_xy_stage_control_gui()


# Wrapper function when moving to update UI position
def move_absolute_position(
    axis: str,
    pos: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True
):
    lumaview.scope.move_absolute_position(
        axis=axis,
        pos=pos,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled
    )

    _handle_ui_update_for_axis(axis=axis)


# Wrapper function when moving to update UI position
def move_relative_position(
    axis: str,
    um: float,
    wait_until_complete: bool = False,
    overshoot_enabled: bool = True
):
    lumaview.scope.move_relative_position(
        axis=axis,
        um=um,
        wait_until_complete=wait_until_complete,
        overshoot_enabled=overshoot_enabled
    )

    _handle_ui_update_for_axis(axis=axis)


def move_home(axis: str):
    axis = axis.upper()

    if axis == 'Z':
        lumaview.scope.zhome()
    elif axis == 'XY':
        lumaview.scope.xyhome()

    _handle_ui_update_for_axis(axis=axis)


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
        self.use_live_image_histogram_equalization = False

        self._contrast_stretcher = ContrastStretcher(
            window_len=3,
            bottom_pct=0.3,
            top_pct=0.3,
        )
        
        self.use_full_pixel_depth = False
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


    def touch(self, target: Widget, event: MotionEvent):
        if event.is_touch and (event.device == 'mouse') and (event.button == 'right'):
            norm_texture_width, norm_texture_height = self.norm_image_size
            norm_texture_x_min = self.center_x - norm_texture_width/2
            norm_texture_x_max = self.center_x + norm_texture_width/2
            norm_texture_y_min = self.center_y - norm_texture_height/2
            norm_texture_y_max = self.center_y + norm_texture_height/2

            click_pos_x = event.pos[0]
            click_pos_y = event.pos[1]

            # Check if click occurred within texture
            if (click_pos_x >= norm_texture_x_min) and (click_pos_x <= norm_texture_x_max) and \
               (click_pos_y >= norm_texture_y_min) and (click_pos_y <= norm_texture_y_max):
                norm_texture_click_pos_x = click_pos_x - norm_texture_x_min
                norm_texture_click_pos_y = click_pos_y - norm_texture_y_min
                texture_width, texture_height = self.texture_size

                # Scale to image pixels
                texture_click_pos_x = norm_texture_click_pos_x * texture_width / norm_texture_width
                texture_click_pos_y = norm_texture_click_pos_y * texture_height / norm_texture_height

                # Distance from center
                x_dist_pixel = texture_click_pos_x - texture_width/2 # Positive means to the right of center
                y_dist_pixel = texture_click_pos_y - texture_height/2 # Positive means above center

                _, objective = get_current_objective_info()
                pixel_size_um = common_utils.get_pixel_size(
                    focal_length=objective['focal_length'],
                    binning_size=get_binning_from_ui(),
                )

                x_dist_um = x_dist_pixel * pixel_size_um
                y_dist_um = y_dist_pixel * pixel_size_um

                move_relative_position(axis='X', um=x_dist_um)
                move_relative_position(axis='Y', um=y_dist_um)


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

        if lumaview.scope.camera.active == False:
            self.source = "./data/icons/camera to USB.png"
            return

        image = lumaview.scope.get_image(force_to_8bit=True)
        if (image is False) or (image.size == 0):
            return

        if ENGINEERING_MODE == True:
            debug_counter += 1
            if debug_counter == 30:
                debug_counter = 0

            if debug_counter % 10 == 0:
                mean = round(np.mean(a=image), 2)
                stddev = round(np.std(a=image), 2)
                af_score = lumaview.scope.focus_function(
                    image=image,
                    include_logging=False
                )

                open_layer = None
                for layer in common_utils.get_layers():
                    accordion = layer + '_accordion'
                    if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                        open_layer = layer
                        break
                
                if open_layer is not None:
                    lumaview.ids['imagesettings_id'].ids[open_layer].ids['image_stats_mean_id'].text = f"Mean: {mean}"
                    lumaview.ids['imagesettings_id'].ids[open_layer].ids['image_stats_stddev_id'].text = f"StdDev: {stddev}"
                    lumaview.ids['imagesettings_id'].ids[open_layer].ids['image_af_score_id'].text = f"AF Score: {af_score}"

            if debug_counter % 3 == 0:
                if self.use_bullseye:
                    image_bullseye = self.transform_to_bullseye(image=image)

                    if self.use_crosshairs:
                        image_bullseye = self.add_crosshairs(image=image_bullseye)

                    texture = Texture.create(size=(image_bullseye.shape[1],image_bullseye.shape[0]), colorfmt='rgb')
                    texture.blit_buffer(image_bullseye.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                    self.texture = texture
            
        if not self.use_bullseye:
            if self.use_live_image_histogram_equalization:
                image = self._contrast_stretcher.update(image)
                # image=cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            if self.use_crosshairs:
                image = self.add_crosshairs(image=image)

            # Convert to texture for display (using OpenGL)
            texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
            texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')
            self.texture = texture

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
        _, labware = get_selected_labware()

        # Get target position
        try:
            x_target = lumaview.scope.get_target_position('X')
            y_target = lumaview.scope.get_target_position('Y')
        except:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            raise

        x_target, y_target = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=settings['stage_offset'],
            sx=x_target,
            sy=y_target
        )

        return labware.get_well_label(x=x_target, y=y_target)


    def live_capture(self):
        logger.info('[LVP Main  ] CompositeCapture.live_capture()')
        global lumaview

        file_root = 'live_'
        color = 'BF'
        well_label = self.get_well_label()

        use_full_pixel_depth = lumaview.ids['viewer_id'].ids['scope_display_id'].use_full_pixel_depth
        force_to_8bit_pixel_depth = not use_full_pixel_depth

        for layer in common_utils.get_layers():
            accordion = layer + '_accordion'
            if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:

                append = f'{well_label}_{layer}'

                if lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active:
                    color = layer
                    
                break
        
        save_folder = pathlib.Path(settings['live_folder']) / "Manual"
        separate_folder_per_channel = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']._seperate_folder_per_channel
        if separate_folder_per_channel:
            save_folder = save_folder / layer

        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        if ENGINEERING_MODE is False:
            return lumaview.scope.save_live_image(
                save_folder,
                file_root,
                append,
                color,
                force_to_8bit=force_to_8bit_pixel_depth,
                output_format=settings['image_output_format']['live']
            )
        
        else:
            use_bullseye = lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye
            use_crosshairs = lumaview.ids['viewer_id'].ids['scope_display_id'].use_crosshairs

            if not use_bullseye and not use_crosshairs:
                return lumaview.scope.save_live_image(
                    save_folder,
                    file_root,
                    append,
                    color,
                    force_to_8bit=force_to_8bit_pixel_depth,
                    output_format=settings['image_output_format']
                )
            
            image_orig = lumaview.scope.get_image(force_to_8bit=force_to_8bit_pixel_depth)
            if image_orig is False:
                return 
            
            # If not in 8-bit mode, generate an 8-bit copy of the image for visualization
            if use_full_pixel_depth:
                image = image_utils.convert_12bit_to_8bit(image_orig)
            else:
                image = image_orig

            if use_bullseye:
                bullseye_image = lumaview.ids['viewer_id'].ids['scope_display_id'].transform_to_bullseye(image)

                # Swap red/blue channels to match required format
                red = bullseye_image[:,:,0].copy()
                blue = bullseye_image[:,:,2].copy()
                bullseye_image[:,:,0] = blue
                bullseye_image[:,:,2] = red
            else:
                bullseye_image = image

            if use_crosshairs:
                crosshairs_image = lumaview.ids['viewer_id'].ids['scope_display_id'].add_crosshairs(bullseye_image)
            else:
                crosshairs_image = bullseye_image

            # Save both versions of the image (unaltered and overlayed)
            now = datetime.datetime.now()
            time_string = now.strftime("%Y%m%d_%H%M%S")
            append = f"{append}_{time_string}"

            # Overlay image is in 8-bits
            lumaview.scope.save_image(
                array=crosshairs_image,
                save_folder=save_folder,
                file_root=file_root,
                append=f"{append}_overlay",
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']
            )

            # Original image may be in 8 or 12-bit
            lumaview.scope.save_image(
                array=image_orig,
                save_folder=save_folder,
                file_root=file_root,
                append=append,
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']
            )


    # capture and save a composite image using the current settings
    def composite_capture(self):
        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')
        global lumaview

        if lumaview.scope.camera.active == False:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        use_full_pixel_depth = scope_display.use_full_pixel_depth

        if use_full_pixel_depth:
            dtype = np.uint16
        else:
            dtype = np.uint8

        img = np.zeros((settings['frame']['height'], settings['frame']['width'], 3), dtype=dtype)

        for layer in common_utils.get_fluorescence_layers():
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

                # Florescent capture
                if lumaview.scope.led:
                    lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)
                    logger.info('[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)')
                else:
                    logger.warning('LED controller not available.')

                # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
                time.sleep(2*exposure/1000+0.2)
                img_gray = lumaview.scope.get_image(force_to_8bit=not use_full_pixel_depth)

                # buffer the images
                if layer == 'Blue':
                    img[:,:,0] = img_gray
                elif layer == 'Green':
                    img[:,:,1] = img_gray
                elif layer == 'Red':
                    img[:,:,2] = img_gray

            scope_leds_off()

            Clock.unschedule(lumaview.ids['imagesettings_id'].ids[layer].ids['histo_id'].histogram)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        lumaview.ids['composite_btn'].state = 'normal'

        append = f'{self.get_well_label()}'

        save_folder = pathlib.Path(settings['live_folder']) / "Manual"
        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)
        
        lumaview.scope.save_image(
            array=img,
            save_folder=save_folder,
            file_root='composite_',
            append=append,
            color=None,
            tail_id_mode='increment',
            output_format=settings['image_output_format']
        )


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

        Window.bind(on_key_up=self._key_up)
        Window.bind(on_key_down=self._key_down)

        self._track_keys = ['ctrl', 'shift']
        self._active_key_presses = set()


    def _key_up(self, *args):
        if len(args) < 5: # No modifiers present
            self._active_key_presses.clear()
            return
        
        modifiers = args[4]
        for key in self._track_keys:
            if (key not in modifiers) and (key in self._active_key_presses):
                self._active_key_presses.remove(key)
        

    def _key_down(self, *args):
        modifiers = args[4]
        for key in self._track_keys:
            if (key in modifiers) and (key not in self._active_key_presses):
                self._active_key_presses.add(key)


    def on_touch_down(self, touch, *args):
        logger.info('[LVP Main  ] ShaderViewer.on_touch_down()')
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:

            if 'ctrl' in self._active_key_presses:
                # Focus control
                vertical_control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
                overshoot_enabled = False
                if touch.button == 'scrolldown':
                    if 'shift' in self._active_key_presses:
                        vertical_control.coarse_up(overshoot_enabled=overshoot_enabled)
                    else:
                        vertical_control.fine_up(overshoot_enabled=overshoot_enabled)
                elif touch.button == 'scrollup':
                    if 'shift' in self._active_key_presses:
                        vertical_control.coarse_down(overshoot_enabled=overshoot_enabled)
                    else:
                        vertical_control.fine_down(overshoot_enabled=overshoot_enabled)

            else:
                # Digital zoom control
                if touch.button == 'scrolldown':
                    if self.scale < 100:
                        self.scale = self.scale * 1.1
                elif touch.button == 'scrollup':
                    if self.scale > 1:
                        self.scale = max(1, self.scale * 0.8)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ShaderViewer, self).on_touch_down(touch)


    def current_false_color(self) -> str:
        return self._false_color
    

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

    
    def update_gui(self, full_redraw: bool = False):
        self.ids['xy_stagecontrol_id'].update_gui(full_redraw=full_redraw)


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
                
    def accordion_collapse(self):
        logger.info('[LVP Main  ] MotionSettings.accordion_collapse()')

        # Handles removing/adding the stage display depending on whether or not the accordion item is visible
        protocol_accordion_item = self.ids['motionsettings_protocol_accordion_id']
        protocol_stage_widget_parent = self.ids['protocol_settings_id'].ids['protocol_stage_holder_id']
        xystage_widget_parent = self._accordion_item_xystagecontrol.ids['xy_stagecontrol_id'].ids['xy_stage_holder_id']

        if (protocol_accordion_item.collapse is True) or (self._accordion_item_xystagecontrol.collapse is True):
            stage.remove_parent()
   
        if protocol_accordion_item.collapse is False:
            stage.pos_hint = {'center_x':0.5, 'center_y':0.5}
            protocol_stage_widget_parent.add_widget(stage)
            stage.full_redraw()
        elif self._accordion_item_xystagecontrol.collapse is False:
            stage.pos_hint = {'center_x':0.5, 'center_y':0.5}
            xystage_widget_parent.add_widget(stage)
            stage.full_redraw()
        

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

        if visible:
            vert_control.ids['tiling_size_spinner'].disabled = False
            vert_control.ids['tiling_size_spinner'].opacity = 1
            vert_control.ids['tiling_size_apply_id'].disabled = False
            vert_control.ids['tiling_size_apply_id'].opacity = 1
            vert_control.ids['tiling_box_label_id'].opacity = 1
        else:
            vert_control.ids['tiling_size_spinner'].text = '1x1'
            vert_control.ids['tiling_size_spinner'].disabled = True
            vert_control.ids['tiling_size_spinner'].opacity = 0
            vert_control.ids['tiling_size_apply_id'].disabled = True
            vert_control.ids['tiling_size_apply_id'].opacity = 0
            vert_control.ids['tiling_box_label_id'].opacity = 0
            

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        logger.info('[LVP Main  ] MotionSettings.toggle_settings()')
        global lumaview
        scope_display = lumaview.ids['viewer_id'].ids['scope_display_id']
        scope_display.stop()
        self.ids['verticalcontrol_id'].update_gui()
        self.ids['protocol_settings_id'].select_labware()

        # move position of motion control
        if self.ids['toggle_motionsettings'].state == 'normal':
            self.pos = -self.settings_width+30, 0
        else:
            self.pos = 0, 0

        if scope_display.play == True:
            scope_display.start()

    
    def update_xy_stage_control_gui(self, *args, full_redraw: bool=False):
        self._accordion_item_xystagecontrol.update_gui(full_redraw=full_redraw)


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
        result = stitcher.load_folder(
            path=pathlib.Path(path),
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )
        final_text = f"Generating stitched images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            time.sleep(5)
            self.done = True
            return

        popup.text = final_text
        time.sleep(2)
        self.done = True


class ZProjectionControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        global zprojection_controls
        super().__init__(**kwargs)
        zprojection_controls = self
        Clock.schedule_once(self._init_ui, 0)
    

    def _init_ui(self, dt=0):
        self.ids['zprojection_method_spinner'].values = zprojector.ZProjector.methods()
        self.ids['zprojection_method_spinner'].text = zprojector.ZProjector.methods()[1]


    @show_popup
    def run_zprojection(self, popup, path):
        status_map = {
            True: "Success",
            False: "FAILED"
        }
        popup.title = "Z-Projection"
        popup.text = "Generating Z-Projection images..."
        zproj = zprojector.ZProjector(ij_helper=ij_helper)
        result = zproj.load_folder(
            path=pathlib.Path(path),
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            method=self.ids['zprojection_method_spinner'].text
        )
        final_text = f"Generating Z-Projection images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            time.sleep(5)
            self.done = True
            return

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
        result = composite_gen.load_folder(
            path=pathlib.Path(path),
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )
        final_text = f"Generating composite images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            time.sleep(5)
            self.done = True
            return
        
        popup.text = final_text
        time.sleep(2)
        self.done = True


class VideoCreationControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        global video_creation_controls
        super().__init__(**kwargs)
        video_creation_controls = self


    @show_popup
    def run_video_gen(self, popup, path) -> None:
        status_map = {
            True: "Success",
            False: "FAILED"
        }

        popup.title = "Video Builder"
        popup.text = "Generating video(s)..."

        try:
            fps = int(self.ids['video_gen_fps_id'].text)
        except:
            fps = 5
            logger.error(f"Could not retrieve valid FPS for video generation. Using {fps} fps.")

        ts_overlay_btn = self.ids['enable_timestamp_overlay_btn']
        enable_timestamp_overlay = True if ts_overlay_btn.state == 'down' else False

        if fps < 1:
            msg = "Video generation frames/second must be >= 1 fps"
            final_text = f"Generating video(s) - {status_map[False]}"
            final_text += f"\n{msg}"
            popup.text = final_text
            logger.error(f"{msg}")
            time.sleep(5)
            self.done = True

        video_builder = VideoBuilder()
        result = video_builder.load_folder(
            path=pathlib.Path(path),
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
            frames_per_sec=fps,
            enable_timestamp_overlay=enable_timestamp_overlay
        )
        final_text = f"Generating video(s) - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            time.sleep(5)
            self.done = True
            return
        
        popup.text = final_text
        time.sleep(2)
        self.done = True
        # self._launch_video()       

    
    # def _launch_video(self) -> None:
    #     try:
    #         os.startfile(self._output_file_loc)
    #     except Exception as e:
    #         logger.error(f"Unable to launch video {self._output_file_loc}:\n{e}")


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
        self.ids['cell_count_image_id'].texture = image_utils_kivy.image_to_texture(image=image)
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

        cell_count_content.ids['cell_count_image_id'].texture = image_utils_kivy.image_to_texture(image=image)


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

        OS_FOLDER_MAP = {
            'win32': 'explorer',
            'darwin': 'open',
            'linux': 'xdg-open'
        }

        if sys.platform not in OS_FOLDER_MAP:
            logger.info(f'[LVP Main  ] PostProcessing.open_folder() not yet implemented for {sys.platform} platform')
            return
        
        command = OS_FOLDER_MAP[sys.platform]
        if last_save_folder is None:
            subprocess.Popen([command, str(pathlib.Path(settings['live_folder']).resolve())])
        else:
            subprocess.Popen([command, str(last_save_folder)])


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
                lumaview.ids['imagesettings_id'].ids[layer].ids['image_af_score_id'].height = '30dp'


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
            image = lumaview.scope.image_buffer
            if image is None:
                return
            
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
        self.record_autofocus_to_file = False


    def update_gui(self):
        try:
            set_pos = lumaview.scope.get_target_position('Z')  # Get target value
        except:
            return

        self.ids['obj_position'].value = max(0, set_pos)
        self.ids['z_position_id'].text = format(max(0, set_pos), '.2f')


    def coarse_up(self, overshoot_enabled: bool = True):
        logger.info('[LVP Main  ] VerticalControl.coarse_up()')
        _, objective = get_current_objective_info()
        coarse = objective['z_coarse']
        move_relative_position('Z', coarse, overshoot_enabled=overshoot_enabled)


    def fine_up(self, overshoot_enabled: bool = True):
        logger.info('[LVP Main  ] VerticalControl.fine_up()')
        _, objective = get_current_objective_info()
        fine = objective['z_fine']
        move_relative_position('Z', fine, overshoot_enabled=overshoot_enabled)


    def fine_down(self, overshoot_enabled: bool = True):
        logger.info('[LVP Main  ] VerticalControl.fine_down()')
        _, objective = get_current_objective_info()
        fine = objective['z_fine']
        move_relative_position('Z', -fine, overshoot_enabled=overshoot_enabled)


    def coarse_down(self, overshoot_enabled: bool = True):
        logger.info('[LVP Main  ] VerticalControl.coarse_down()')
        _, objective = get_current_objective_info()
        coarse = objective['z_coarse']
        move_relative_position('Z', -coarse, overshoot_enabled=overshoot_enabled)


    def set_position(self, pos):
        logger.info('[LVP Main  ] VerticalControl.set_position()')
        try:
            pos = float(pos)
        except:
            return
        
        move_absolute_position('Z', pos)


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
        move_absolute_position('Z', pos)

    def home(self):
        logger.info('[LVP Main  ] VerticalControl.home()')
        move_home(axis='Z')


    def _reset_run_autofocus_button(self, **kwargs):
        self.ids['autofocus_id'].state = 'normal'
        self.ids['autofocus_id'].text = 'Autofocus'

    
    def _set_run_autofocus_button(self, **kwargs):
        self.ids['autofocus_id'].state = 'down'
        self.ids['autofocus_id'].text = 'Focusing...'


    def _cleanup_at_end_of_autofocus(self):
        sequenced_capture_executor.reset()
        self._reset_run_autofocus_button()


    def _autofocus_run_complete(self, **kwargs):
        self._reset_run_autofocus_button()


    def run_autofocus_from_ui(self):
        logger.info('[LVP Main  ] VerticalControl.run_autofocus_from_ui()')

        if ENGINEERING_MODE == True:
            save_autofocus_data = True
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / "Autofocus Characterization"
        else:
            save_autofocus_data = False
            parent_dir = None

        trigger_source = 'autofocus'
        run_complete_func = self._autofocus_run_complete
        run_not_started_func = self._reset_run_autofocus_button

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start autofocus. Run already in progress from {run_trigger_source}")
            return
        
        if self.ids['autofocus_id'].state == 'normal':
            self._cleanup_at_end_of_autofocus()
            return
        
        self._set_run_autofocus_button()

        objective_id, _ = get_current_objective_info()
        labware_id, _ = get_selected_labware()
        active_layer, active_layer_config = get_active_layer_config()
        active_layer_config['autofocus'] = True
        active_layer_config['acquire'] = True
        curr_position = get_current_plate_position()
        curr_position.update({'name': 'AF'})

        positions = [
            curr_position,
        ]

        tiling_config = TilingConfig(
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        )

        config = {
            'labware_id': labware_id,
            'positions': positions,
            'objective_id': objective_id,
            'zstack_positions': {None: None},
            'zstack_positions_valid': True,
            'use_zstacking': False,
            'tiling': tiling_config.no_tiling_label(),
            'layer_configs': {active_layer: active_layer_config},
            'period': None,
            'duration': None,
            'frame_dimensions': get_current_frame_dimensions(),
            'binning_size': get_binning_from_ui(),
        }
      
        autofocus_sequence = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        )

        autogain_settings = get_auto_gain_settings()

        callbacks = {
            'move_position': _handle_ui_update_for_axis,
            'update_scope_display': lumaview.ids['viewer_id'].ids['scope_display_id'].update_scopedisplay,
            'scan_iterate_post': run_complete_func,
            'run_complete': run_complete_func,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
        }

        sequenced_capture_executor.run(
            protocol=autofocus_sequence,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='af',
            parent_dir=parent_dir,
            image_capture_config=get_image_capture_config_from_ui(),
            enable_image_saving=False,
            disable_saving_artifacts=True,
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            return_to_position=None,
            save_autofocus_data=save_autofocus_data,
            leds_state_at_end="return_to_original",
        )


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


    def turret_select(self, selected_position):
        #TODO check if turret has been HOMED turret first
        lumaview.scope.turret_id = selected_position - 1
        angle = 90*lumaview.scope.turret_id
        lumaview.scope.tmove(
            degrees=angle
        )
        
        for available_position in range(1,5):
            if selected_position == available_position:
                state = 'down'
            else:
                state = 'normal'
            
            self.ids[f'turret_pos_{available_position}_btn'].state = state


class XYStageControl(BoxLayout):

    def update_gui(self, dt=0, full_redraw: bool = False):
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
            _, labware = get_selected_labware()
            stage_x, stage_y = coordinate_transformer.stage_to_plate(
                labware=labware,
                stage_offset=settings['stage_offset'],
                sx=x_target,
                sy=y_target
            )

            if not self.ids['x_pos_id'].focus:
                self.ids['x_pos_id'].text = format(max(0, stage_x), '.2f') # display coordinate in mm

            if not self.ids['y_pos_id'].focus:  
                self.ids['y_pos_id'].text = format(max(0, stage_y), '.2f') # display coordinate in mm


    def fine_left(self):
        logger.info('[LVP Main  ] XYStageControl.fine_left()')
        _, objective = get_current_objective_info()
        fine = objective['xy_fine']
        move_relative_position('X', -fine)  # Move LEFT fine step

    def fine_right(self):
        logger.info('[LVP Main  ] XYStageControl.fine_right()')
        _, objective = get_current_objective_info()
        fine = objective['xy_fine']
        move_relative_position('X', fine)  # Move RIGHT fine step

    def coarse_left(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_left()')
        _, objective = get_current_objective_info()
        coarse = objective['xy_coarse']
        move_relative_position('X', -coarse)  # Move LEFT coarse step

    def coarse_right(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_right()')
        _, objective = get_current_objective_info()
        coarse = objective['xy_coarse']
        move_relative_position('X', coarse)  # Move RIGHT

    def fine_back(self):
        logger.info('[LVP Main  ] XYStageControl.fine_back()')
        _, objective = get_current_objective_info()
        fine = objective['xy_fine']
        move_relative_position('Y', -fine)  # Move BACK 

    def fine_fwd(self):
        logger.info('[LVP Main  ] XYStageControl.fine_fwd()')
        _, objective = get_current_objective_info()
        fine = objective['xy_fine']
        move_relative_position('Y', fine)  # Move FORWARD 

    def coarse_back(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_back()')
        _, objective = get_current_objective_info()
        coarse = objective['xy_coarse']
        move_relative_position('Y', -coarse)  # Move BACK

    def coarse_fwd(self):
        logger.info('[LVP Main  ] XYStageControl.coarse_fwd()')
        _, objective = get_current_objective_info()
        coarse = objective['xy_coarse']
        move_relative_position('Y', coarse)  # Move FORWARD 

    def set_xposition(self, x_pos):
        logger.info('[LVP Main  ] XYStageControl.set_xposition()')
        global lumaview
        try:
            x_pos = float(x_pos)
        except:
            return

        # x_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=x_pos,
            py=0
        )

        logger.info(f'[LVP Main  ] X pos {x_pos} Stage X {stage_x}')

        # Move to x-position
        move_absolute_position('X', stage_x)  # position in text is in mm


    def set_yposition(self, y_pos):
        logger.info('[LVP Main  ] XYStageControl.set_yposition()')
        global lumaview

        try:
            y_pos = float(y_pos)
        except:
            return

        # y_pos is the the plate position in mm
        # Find the coordinates for the stage
        _, labware = get_selected_labware()
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=0,
            py=y_pos
        )

        # Move to y-position
        move_absolute_position('Y', stage_y)  # position in text is in mm


    def set_xbookmark(self):
        logger.info('[LVP Main  ] XYStageControl.set_xbookmark()')
        global lumaview

        # Get current stage x-position in um
        x_pos = lumaview.scope.get_current_position('X')
 
        # Save plate x-position to settings
        _, labware = get_selected_labware()
        plate_x, _ = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=settings['stage_offset'],
            sx=x_pos,
            sy=0
        )
        
        settings['bookmark']['x'] = plate_x

    def set_ybookmark(self):
        logger.info('[LVP Main  ] XYStageControl.set_ybookmark()')
        global lumaview

        # Get current stage y-position in um
        y_pos = lumaview.scope.get_current_position('Y')  

        # Save plate y-position to settings
        _, labware = get_selected_labware()
        _, plate_y = coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=settings['stage_offset'],
            sx=0,
            sy=y_pos
        )

        settings['bookmark']['y'] = plate_y

    def goto_xbookmark(self):
        logger.info('[LVP Main  ] XYStageControl.goto_xbookmark()')
        global lumaview

        # Get bookmark plate x-position in mm
        x_pos = settings['bookmark']['x']

        # Move to x-position
        _, labware = get_selected_labware()
        stage_x, _ = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=x_pos,
            py=0
        )
        move_absolute_position('X', stage_x)  # set current x position in um

    def goto_ybookmark(self):
        logger.info('[LVP Main  ] XYStageControl.goto_ybookmark()')
        global lumaview

        # Get bookmark plate y-position in mm
        y_pos = settings['bookmark']['y']

        # Move to y-position
        _, labware = get_selected_labware()
        _, stage_y = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=0,
            py=y_pos
        )
        move_absolute_position('Y', stage_y)  # set current y position in um

    # def calibrate(self):
    #     logger.info('[LVP Main  ] XYStageControl.calibrate()')
    #     global lumaview
    #     x_pos = lumaview.scope.get_current_position('X')  # Get current x position in um
    #     y_pos = lumaview.scope.get_current_position('Y')  # Get current x position in um

    #     _, labware = get_selected_labware()
    #     x_plate_offset = labware.plate['offset']['x']*1000
    #     y_plate_offset = labware.plate['offset']['y']*1000

    #     settings['stage_offset']['x'] = x_plate_offset-x_pos
    #     settings['stage_offset']['y'] = y_plate_offset-y_pos
    #     self.update_gui()

    def home(self):
        logger.info('[LVP Main  ] XYStageControl.home()')
        global lumaview

        if lumaview.scope.motion.driver: # motor controller is actively connected
            move_home(axis='XY')
            
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

# Protocol settings tab
class ProtocolSettings(CompositeCapture):
    global settings

    done = BooleanProperty(False)

    def __init__(self, **kwargs):

        super(ProtocolSettings, self).__init__(**kwargs)
        logger.info('[LVP Main  ] ProtocolSettings.__init__()')

        os.chdir(source_path)
        try:
            read_file = open('./data/labware.json', "r")
        except:
            logger.exception("[LVP Main  ] Error reading labware definition file 'data/labware.json'")
            if not os.path.isdir('./data'):
                raise FileNotFoundError("Couldn't find 'data' directory.")
            else:
                raise
        else:
            self.labware = json.load(read_file)
            read_file.close()

        self.curr_step = -1

        
        self.tiling_config = TilingConfig(
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )

        self.tiling_min = {
            "x": 120000,
            "y": 80000
        }
        self.tiling_max = {
            "x": 0,
            "y": 0
        }

        self.tiling_count = self.tiling_config.get_mxn_size(self.tiling_config.default_config())

        self._protocol = None

        self.exposures = 1  # 1 indexed
        Clock.schedule_once(self._init_ui, 0)


    def _init_ui(self, dt=0):
        self.ids['tiling_size_spinner'].values = self.tiling_config.available_configs()
        self.ids['tiling_size_spinner'].text = self.tiling_config.default_config()

        protocol_config = get_sequenced_capture_config_from_ui()
        self._protocol = Protocol.create_empty(
            config=protocol_config,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        )
        self.select_labware()
        self.update_step_ui()


    # Update Protocol Period   
    def update_period(self):
        logger.info('[LVP Main  ] ProtocolSettings.update_period()')
        try:
            settings['protocol']['period'] = float(self.ids['capture_period'].text)
        except:
            logger.exception('[LVP Main  ] Update Period is not an acceptable value')

        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

    # Update Protocol Duration   
    def update_duration(self):
        logger.info('[LVP Main  ] ProtocolSettings.update_duration()')
        try:
            settings['protocol']['duration'] = float(self.ids['capture_dur'].text)
        except:
            logger.warning('[LVP Main  ] Update Duration is not an acceptable value')

        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )


    def step_name_validation(self, text: str):
        self.ids['step_name_input'].text = Protocol.sanitize_step_name(input=text)


    # Labware Selection
    def select_labware(self, labware: str = None):
        global settings
        logger.info('[LVP Main  ] ProtocolSettings.select_labware()')
        if labware is None:
            spinner = self.ids['labware_spinner']
            spinner.values = wellplate_loader.get_plate_list()
            settings['protocol']['labware'] = spinner.text
        else:
            center_plate_str = 'Center Plate'
            spinner = self.ids['labware_spinner']
            spinner.values = list(center_plate_str,)
            spinner.text = center_plate_str
            settings['protocol']['labware'] = labware

        labware_id, labware = get_selected_labware()
        lumaview.scope.set_labware(labware=labware)

        if self._protocol is not None:
            self._protocol.modify_labware(labware_id=labware_id)

        stage.full_redraw()


    def set_labware_selection_visibility(self, visible):
        labware_spinner = self.ids['labware_spinner']
        labware_spinner.visible = visible
        labware_spinner.size_hint_y = None if visible else 0
        labware_spinner.height = '30dp' if visible else 0
        labware_spinner.opacity = 1 if visible else 0
        labware_spinner.disabled = not visible

        if not visible:
            labware_spinner.text = 'Center Plate'

    
    def set_show_protocol_step_locations_visibility(self, visible: bool) -> None:
        if visible:
            self.ids['show_step_locations_id'].disabled = False
            self.ids['show_step_locations_id'].opacity = 1
            self.ids['show_step_locations_label_id'].opacity = 1
        else:
            self.ids['show_step_locations_id'].disabled = True
            self.ids['show_step_locations_id'].opacity = 0
            self.ids['show_step_locations_label_id'].opacity = 0


    def apply_tiling(self):
        logger.info('[LVP Main  ] Apply tiling to protocol')
        objective_id, _ = get_current_objective_info()

        self._protocol.apply_tiling(
            tiling=self.ids['tiling_size_spinner'].text,
            frame_dimensions=get_current_frame_dimensions(),
            objective_id=objective_id,
            binning_size=get_binning_from_ui(),
        )
        
        self._protocol.optimize_step_ordering()
        stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step()


    def apply_zstacking(self):
        logger.info('[LVP Main  ] Apply Z-Stacking to protocol')
        zstack_valid, zstack_positions = get_zstack_positions()

        if not zstack_valid:
            return
        
        self._protocol.apply_zstacking(
            zstack_positions=zstack_positions,
        )
       
        self._protocol.optimize_step_ordering()
        stage.set_protocol_steps(df=self._protocol.steps())
        self.update_step_ui()
        self.go_to_step()


    def update_step_ui(self):
        num_steps = self._protocol.num_steps()

        self.ids['step_number_input'].text = str(self.curr_step+1)
        self.ids['step_total_input'].text = str(num_steps)

        if num_steps > 0:
            step = self.get_curr_step()
            self.ids['step_name_input'].text = step["Name"]
            if step['Name'] == '':
                self.ids['step_name_input'].hint_text = self.get_default_name_for_curr_step()
        else:
            self.ids['step_name_input'].text = ''
            self.ids['step_name_input'].hint_text = 'Step Name'


    def new_protocol(self):
        logger.info('[LVP Main  ] ProtocolSettings.new_protocol()')

        config = get_sequenced_capture_config_from_ui()

        self._protocol = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )

        stage.set_protocol_steps(df=self._protocol.steps())

        settings['protocol']['filepath'] = ''        
        self.ids['protocol_filename'].text = ''
        self.curr_step = 0
        self.update_step_ui()
        self.go_to_step()


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


    @show_popup
    def _show_popup_message(self, popup, title, message, delay_sec):
        popup.title = title
        popup.text = message
        time.sleep(delay_sec)
        self.done = True


    # Load Protocol from File
    def load_protocol(self, filepath="./data/new_default_protocol.tsv"):
        logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

        if not pathlib.Path(filepath).exists():
            return
        
        self._protocol = Protocol.from_file(
            file_path=filepath,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        )

        settings['protocol']['filepath'] = filepath
        self.ids['protocol_filename'].text = os.path.basename(filepath)

        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            self.curr_step = -1
        else:
            self.curr_step = 0

        self.update_step_ui()
        self.go_to_step()

        period = round(self._protocol.period().total_seconds() / 60, 2)
        duration = round(self._protocol.duration().total_seconds() / 3600, 2)
        labware = self._protocol.labware()

        scope_configs = lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]

        # If the scope has no XY stage, then don't allow the protocol to modify the labware
        if selected_scope_config['XYStage'] == False:
            labware = "Center Plate"

        self.ids['capture_period'].text = str(period)
        self.ids['capture_dur'].text = str(duration)
       
        settings['protocol']['period'] = period
        settings['protocol']['duration'] = duration
        settings['protocol']['labware'] = labware
        self.ids['labware_spinner'].text = settings['protocol']['labware']

        # Make steps available for drawing locations
        stage.set_protocol_steps(df=self._protocol.steps())

        self.update_step_ui()
        if lumaview.scope.has_xyhomed():
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

        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration']
        )

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

        self._protocol.to_file(
            file_path=filepath
        )

        self.ids['protocol_filename'].text = os.path.basename(filepath)


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
    def handle_step_ui_input_change(
        self
    ):
        obj = self.ids['step_number_input']
        try:
            val = int(obj.text)
        except:
            return
        
        num_steps = self._protocol.num_steps()
        if num_steps < 1:
            val = 0
            obj.text = f"{val}"
        elif val < 1:
            val = 1
            obj.text = f"{val}"
        elif val > num_steps:
            val = num_steps
            obj.text = f"{val}"

        self.curr_step = val-1
        self.go_to_step()


    def go_to_step(
        self,
    ):
        go_to_step(
            protocol=self._protocol,
            step_idx=self.curr_step,
            ignore_auto_gain=False,
            include_move=True
        )

    # Goto to Previous Step
    def prev_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.prev_step()')
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            self.curr_step = -1
            self.update_step_ui()
            return
        
        self.curr_step = max(self.curr_step-1, 0)
        self.update_step_ui()
        self.go_to_step()
 
    # Go to Next Step
    def next_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.next_step()')
        num_steps = self._protocol.num_steps()
        if num_steps <= 0:
            return
        
        self.curr_step = min(self.curr_step+1, num_steps-1)
        self.update_step_ui()
        self.go_to_step()


    # Delete Current Step of Protocol
    def delete_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.delete_step()')
 
        if self._protocol.num_steps() <= 0:
            return
    
        self._protocol.delete_step(
            step_idx=self.curr_step
        )
        
        stage.set_protocol_steps(df=self._protocol.steps())

        if self._protocol.num_steps() <= 0:
            self.curr_step = -1
        else:
            self.curr_step = max(self.curr_step-1, 0)
 
        self.update_step_ui()
        self.go_to_step()


    def modify_step(self):
        logger.info('[LVP Main  ] ProtocolSettings.modify_step()')

        if self._protocol.num_steps() < 1:
            return
        
        active_layer, active_layer_config = get_active_layer_config()
        plate_position = get_current_plate_position()
        objective_id, _ = get_current_objective_info()

        step_name = self.ids['step_name_input'].text

        self._protocol.modify_step(
            step_idx=self.curr_step,
            step_name=step_name,
            layer=active_layer,
            layer_config=active_layer_config,
            plate_position=plate_position,
            objective_id=objective_id,
        )

        stage.set_protocol_steps(df=self._protocol.steps())


    def insert_step(self, after_current_step: bool = True):
        
        logger.info('[LVP Main  ] ProtocolSettings.insert_step()')

        active_layer, active_layer_config = get_active_layer_config()
        plate_position = get_current_plate_position()
        objective_id, _ = get_current_objective_info()

        if after_current_step:
            after_step = self.curr_step
            before_step = None
        else:
            after_step = None
            before_step = self.curr_step

        step_name = self._protocol.insert_step(
            step_name=None,
            layer=active_layer,
            layer_config=active_layer_config,
            plate_position=plate_position,
            objective_id=objective_id,
            before_step=before_step,
            after_step=after_step
        )

        if after_current_step or (self.curr_step < 0):
            self.curr_step += 1

        self.update_step_ui()
        stage.set_protocol_steps(df=self._protocol.steps())
        self.go_to_step()


    def update_acquire_zstack(self):
        pass


    def update_show_step_locations(self):
        stage.show_protocol_steps(enable=self.ids['show_step_locations_id'].active)


    def update_tiling_selection(self):
        pass


    def determine_and_set_run_autofocus_scan_allow(self):
        tiling = self.ids['tiling_size_spinner'].text
        zstack = self.ids['acquire_zstack_id'].active
        if (zstack == True) and (tiling != '1x1'):
            self.set_run_autofocus_scan_allow(allow=False)
        else:
            self.set_run_autofocus_scan_allow(allow=True)


    def set_run_autofocus_scan_allow(self, allow: bool):
        if allow:
            self.ids['run_autofocus_btn'].disabled = False
        else:
            self.ids['run_autofocus_btn'].disabled = True


    def get_curr_step(self):
        if self._protocol.num_steps() == 0:
            return None
        
        return self._protocol.step(idx=self.curr_step)


    def _reset_run_autofocus_scan_button(self, **kwargs):
        self.ids['run_autofocus_btn'].state = 'normal'
        self.ids['run_autofocus_btn'].text = 'Scan and Autofocus All Steps'


    def _reset_run_scan_button(self, **kwargs):
        self.ids['run_scan_btn'].state = 'normal'
        self.ids['run_scan_btn'].text = 'Run One Scan'

    
    def _reset_run_protocol_button(self, **kwargs):
        self.ids['run_protocol_btn'].state = 'normal'
        self.ids['run_protocol_btn'].text = 'Run Full Protocol'
        

    def _is_protocol_valid(self) -> bool:
        if self._protocol.num_steps() == 0:
            logger.warning('[LVP Main  ] Protocol has no steps.')
            return False
        
        return True


    def _autofocus_run_complete_callback(self, **kwargs):
        self._reset_run_autofocus_scan_button()

        # Copy the Z-heights from the autofocus scan into the protocol
        focused_protocol = kwargs['protocol']
        self._protocol.steps()['Z'] = focused_protocol.steps()['Z']


    def run_autofocus_scan_from_ui(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_autofocus_scan_from_ui()')
        trigger_source = 'autofocus_scan'
        run_not_started_func = self._reset_run_autofocus_scan_button

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start autofocus scan. Run already in progress from {run_trigger_source}")
            return
        
        if not self._is_protocol_valid():
            run_not_started_func()
            return
        
        if self.ids['run_autofocus_btn'].state == 'normal':
            self._cleanup_at_end_of_protocol(autofocus_scan=True)
            return
        
        self.ids['run_autofocus_btn'].text = 'Running Autofocus Scan'

        callbacks = {
            'move_position': _handle_ui_update_for_axis,
            'update_scope_display': lumaview.ids['viewer_id'].ids['scope_display_id'].update_scopedisplay,
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'scan_iterate_post': run_not_started_func,
            'update_step_number': _update_step_number_callback,
            'run_complete': self._autofocus_run_complete_callback,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
        }

        initial_position = get_current_plate_position()

        autogain_settings = get_auto_gain_settings()

        sequence = copy.deepcopy(self._protocol)
        sequence.modify_autofocus_all_steps(enabled=True)

        sequenced_capture_executor.run(
            protocol=sequence,
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='af_scan',
            parent_dir=None,
            image_capture_config=get_image_capture_config_from_ui(),
            enable_image_saving=False,
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            return_to_position=initial_position,
            update_z_pos_from_autofocus=True,
            leds_state_at_end="off",
        )


    def _scan_run_complete(self, **kwargs):
        self._reset_run_scan_button()
        create_hyperstacks_if_needed()


    def run_scan_from_ui(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_scan_from_ui()')
        trigger_source = 'scan'
        run_complete_func = self._scan_run_complete
        run_not_started_func = self._reset_run_scan_button

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start scan. Run already in progress from {run_trigger_source}")
            return
        
        if not self._is_protocol_valid():
            run_not_started_func()
            return
        
        if self.ids['run_scan_btn'].state == 'normal':
            self._cleanup_at_end_of_protocol(autofocus_scan=False)
            return
        
        self.ids['run_scan_btn'].text = 'Running Scan'

        callbacks = {
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'scan_iterate_post': run_not_started_func,
            'run_complete': run_complete_func,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
        }

        self.run_sequenced_capture(
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            run_trigger_source=trigger_source,
            max_scans=1,
            callbacks=callbacks,
        )


    def _protocol_run_complete(self, **kwargs):
        self._reset_run_protocol_button()
        create_hyperstacks_if_needed()


    def run_protocol_from_ui(self):
        logger.info('[LVP Main  ] ProtocolSettings.run_protocol_from_ui()')
        trigger_source = 'protocol'
        run_complete_func = self._protocol_run_complete
        run_not_started_func = self._reset_run_protocol_button

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start protocol run. Run already in progress from {run_trigger_source}")
            return
        
        if not self._is_protocol_valid():
            run_not_started_func()
            return
        
        if self.ids['run_protocol_btn'].state == 'normal':
            self._cleanup_at_end_of_protocol(autofocus_scan=False)
            return
        
        # Note: This will be quickly overwritten by the remaining number of scans
        self.ids['run_protocol_btn'].text = 'Running Protocol'

        callbacks = {
            'protocol_iterate_pre': self._update_protocol_run_button_status,
            'run_scan_pre': self._run_scan_pre_callback,
            'autofocus_in_progress': self._autofocus_in_progress_callback,
            'autofocus_complete': self._autofocus_complete_callback,
            'run_complete': run_complete_func,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
        }

        time_params = get_protocol_time_params()
        self._protocol.modify_time_params(
            period=time_params['period'],
            duration=time_params['duration'],
        )

        self.run_sequenced_capture(
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            run_trigger_source=trigger_source,
            max_scans=None,
            callbacks=callbacks,
        )


    def _update_protocol_run_button_status(
        self,
        **kwargs,
    ):
        remaining_scans = kwargs['remaining_scans']
        scan_word = "scan" if remaining_scans == 1 else "scans"
        self.ids['run_protocol_btn'].text = f"{remaining_scans} {scan_word} remaining. Press to ABORT"


    def _run_scan_pre_callback(self):
        global lumaview
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False
        self.update_step_ui()


    def _autofocus_in_progress_callback(self):
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']._set_run_autofocus_button()


    def _autofocus_complete_callback(self):
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']._reset_run_autofocus_button()
        lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].is_complete = False


    def run_sequenced_capture(
        self,
        run_mode: SequencedCaptureRunMode,
        run_trigger_source: str,
        max_scans: int | None,
        callbacks: dict[str, typing.Callable],
        disable_saving_artifacts: bool = False,
        return_to_position: dict | None = None,
    ):

        logger.info('[LVP Main  ] ProtocolSettings.run_sequenced_capture()')

        callbacks.update(
            {
                'move_position': _handle_ui_update_for_axis,
                'leds_off': _handle_ui_for_leds_off,
                'led_state': _handle_ui_for_led,
                'update_step_number': _update_step_number_callback,
                'go_to_step': go_to_step,
                'update_scope_display': lumaview.ids['viewer_id'].ids['scope_display_id'].update_scopedisplay,
            }
        )

        parent_dir = pathlib.Path(settings['live_folder']).resolve() / "ProtocolData"

        sequence_name = self.ids['protocol_filename'].text

        image_capture_config = get_image_capture_config_from_ui()
        autogain_settings = get_auto_gain_settings()

        sequenced_capture_executor.run(
            protocol=self._protocol,
            run_mode=run_mode,
            run_trigger_source=run_trigger_source,
            max_scans=max_scans,
            sequence_name=sequence_name,
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=is_image_saving_enabled(),
            separate_folder_per_channel=lumaview.ids['motionsettings_id'].ids['microscope_settings_id']._seperate_folder_per_channel,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            disable_saving_artifacts=disable_saving_artifacts,
            return_to_position=return_to_position,
            leds_state_at_end="off",
        )
        
        set_last_save_folder(dir=sequenced_capture_executor.run_dir())

        if run_mode == SequencedCaptureRunMode.FULL_PROTOCOL:
            self._update_protocol_run_button_status(
                remaining_scans=sequenced_capture_executor.remaining_scans()
            )


    def _cleanup_at_end_of_protocol(self, autofocus_scan: bool):
        sequenced_capture_executor.reset()
        self._reset_run_protocol_button()
        self._reset_run_scan_button()
        self._reset_run_autofocus_scan_button()

        if not autofocus_scan:
            create_hyperstacks_if_needed()


# Widget for displaying Microscope Stage area, labware, and current position 
class Stage(Widget):

    def full_redraw(self, *args):
        self.draw_labware(full_redraw=True)

    
    def remove_parent(self):
        if self.parent is not None:
            self.parent.remove_widget(stage)

    
    def get_id(self):
        return id(self)


    def __init__(self, **kwargs):
        super(Stage, self).__init__(**kwargs)
        logger.info('[LVP Main  ] Stage.__init__()')
        self.ROI_min = [0,0]
        self.ROI_max = [0,0]
        self._motion_enabled = True
        self.ROIs = []

        self.full_redraw()
        self.bind(
            pos=self.full_redraw,
            size=self.full_redraw
        )
        self._protocol_step_locations_df = None
        self._protocol_step_redraw = False
        self._protocol_step_locations_show = False


    def show_protocol_steps(self, enable: bool):
        self._protocol_step_locations_show = enable
        self._protocol_step_redraw = True


    def set_protocol_steps(self, df):
        # Filter to only keep the X/Y locations
        df = df.copy()
        df = df[['X','Y']]
        self._protocol_step_locations_df = df.drop_duplicates()
        self._protocol_step_redraw = True


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
            _, labware = get_selected_labware()

            # Get labware dimensions
            dim_max = labware.get_dimensions()

            # Scale from pixels to mm (from the bottom left)
            scale_x = dim_max['x'] / self.width
            scale_y = dim_max['y'] / self.height

            # Convert to plate position in mm (from the top left)
            plate_x = mouse_x*scale_x
            plate_y = dim_max['y'] - mouse_y*scale_y

            # Convert from plate position to stage position
            _, labware = get_selected_labware()
            stage_x, stage_y = coordinate_transformer.plate_to_stage(
                labware=labware,
                stage_offset=settings['stage_offset'],
                px=plate_x,
                py=plate_y
            )

            move_absolute_position('X', stage_x)
            move_absolute_position('Y', stage_y)


    def draw_labware(
        self,
        *args,
        full_redraw: bool = False
    ): # View the labware from front and above
        global lumaview
        global settings

        if self.parent is None:
            return
        
        if 'settings' not in globals():
            return
        
        # Create current labware instance
        _, labware = get_selected_labware()

        if full_redraw:
            self.canvas.clear()
        else:
            self.canvas.remove_group('crosshairs')
            self.canvas.remove_group('selected_well')

        if self._protocol_step_redraw:
            self.canvas.remove_group('steps')

        with self.canvas:
            w = self.width
            h = self.height
            x = self.x
            y = self.y

            # Get labware dimensions
            dim_max = labware.get_dimensions()

            # mm to pixels scale
            scale_x = w/dim_max['x']
            scale_y = h/dim_max['y']

            # Stage Coordinates (120x80 mm)
            stage_w = 120
            stage_h = 80

            stage_x = settings['stage_offset']['x']/1000
            stage_y = settings['stage_offset']['y']/1000

            # Get target position
            # Outline of Stage Area from Above
            # ------------------
            if full_redraw:
                Color(.2, .2, .2 , 0.5)                # dark grey
                Rectangle(pos=(x+(dim_max['x']-stage_w-stage_x)*scale_x, y+stage_y*scale_y),
                            size=(stage_w*scale_x, stage_h*scale_y), group='outline')

                # Outline of Plate from Above
                # ------------------
                Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
                Line(points=(x, y, x, y+h-15), width = 1, group='outline')          # Left
                Line(points=(x+w, y, x+w, y+h), width = 1, group='outline')         # Right
                Line(points=(x, y, x+w, y), width = 1, group='outline')             # Bottom
                Line(points=(x+15, y+h, x+w, y+h), width = 1, group='outline')      # Top
                Line(points=(x, y+h-15, x+15, y+h), width = 1, group='outline')     # Diagonal

                # ROI rectangle
                # ------------------
                if self.ROI_max[0] > self.ROI_min[0]:
                    roi_min_x, roi_min_y = coordinate_transformer.stage_to_pixel(
                        labware=labware,
                        stage_offset=settings['stage_offset'],
                        sx=self.ROI_min[0],
                        sy=self.ROI_min[1],
                        scale_x=scale_x,
                        scale_y=scale_y
                    )
                
                    roi_max_x, roi_max_y = coordinate_transformer.stage_to_pixel(
                        labware=labware,
                        stage_offset=settings['stage_offset'],
                        sx=self.ROI_max[0],
                        sy=self.ROI_max[1],
                        scale_x=scale_x,
                        scale_y=scale_y
                    )

                    Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
                    Line(rectangle=(x+roi_min_x, y+roi_min_y, roi_max_x - roi_min_x, roi_max_y - roi_min_y), group='outline')
            
            # Draw all ROI rectangles
            # ------------------
            # TODO (for each step)
            '''
            for ROI in self.ROIs:
                if self.ROI_max[0] > self.ROI_min[0]:
                    roi_min_x, roi_min_y = coordinate_transformer.stage_to_pixel(self.ROI_min[0], self.ROI_min[1], scale_x, scale_y)
                    roi_max_x, roi_max_y = coordinate_transformer.stage_to_pixel(self.ROI_max[0], self.ROI_max[1], scale_x, scale_y)
                    Color(50/255, 164/255, 206/255, 1.)                # kivy aqua
                    Line(rectangle=(x+roi_min_x, y+roi_min_y, roi_max_x - roi_min_x, roi_max_y - roi_min_y))
            '''
            
            # Draw all wells
            # ------------------
            cols = labware.config['columns']
            rows = labware.config['rows']
            
            well_spacing_x = labware.config['spacing']['x']
            well_spacing_y = labware.config['spacing']['y']
            well_spacing_pixel_x = well_spacing_x
            well_spacing_pixel_y = well_spacing_y

            well_diameter = labware.config['diameter']
            if well_diameter == -1:
                well_radius_pixel_x = well_spacing_pixel_x
                well_radius_pixel_y = well_spacing_pixel_y
            else:
                well_radius = well_diameter / 2
                well_radius_pixel_x = well_radius * scale_x
                well_radius_pixel_y = well_radius * scale_y

            if full_redraw:
                Color(0.4, 0.4, 0.4, 0.5)
            
                for i in range(cols):
                    for j in range(rows):                   
                        well_plate_x, well_plate_y = labware.get_well_position(i, j)
                        well_pixel_x, well_pixel_y = coordinate_transformer.plate_to_pixel(
                            labware=labware,
                            px=well_plate_x,
                            py=well_plate_y,
                            scale_x=scale_x,
                            scale_y=scale_y
                        )
                        x_center = int(x+well_pixel_x) # on screen center
                        y_center = int(y+well_pixel_y) # on screen center
                        Ellipse(pos=(x_center-well_radius_pixel_x, y_center-well_radius_pixel_y), size=(well_radius_pixel_x*2, well_radius_pixel_y*2), group='wells')

            if full_redraw or self._protocol_step_redraw:
                self._protocol_step_redraw = False

                if  (self._protocol_step_locations_show == True) and \
                    (self._protocol_step_locations_df is not None):

                    Color(1., 1., 0., 1.)
                    half_size = 2
                    for _, step in self._protocol_step_locations_df.iterrows():
                        pixel_x, pixel_y = coordinate_transformer.plate_to_pixel(
                            labware=labware,
                            px=step['X'],
                            py=step['Y'],
                            scale_x=scale_x,
                            scale_y=scale_y
                        )
                        
                        x_center = x+pixel_x
                        y_center = y+pixel_y

                        Line(points=(x_center-half_size, y_center, x_center+half_size, y_center), width = 1, group='steps') # horizontal line
                        Line(points=(x_center, y_center-half_size, x_center, y_center+half_size), width = 1, group='steps') # vertical line

            try:
                target_stage_x = lumaview.scope.get_target_position('X')
                target_stage_y = lumaview.scope.get_target_position('Y')
            except:
                logger.exception('[LVP Main  ] Error talking to Motor board.')
                raise
                
            _, labware = get_selected_labware()
            target_plate_x, target_plate_y = coordinate_transformer.stage_to_plate(
                labware=labware,
                stage_offset=settings['stage_offset'],
                sx=target_stage_x,
                sy=target_stage_y
            )

            target_i, target_j = labware.get_well_index(target_plate_x, target_plate_y)
            target_well_plate_x, target_well_plate_y = labware.get_well_position(target_i, target_j)
            target_well_pixel_x, target_well_pixel_y = coordinate_transformer.plate_to_pixel(
                labware=labware,
                px=target_well_plate_x,
                py=target_well_plate_y,
                scale_x=scale_x,
                scale_y=scale_y
            )
            target_well_center_x = int(x+target_well_pixel_x) # on screen center
            target_well_center_y = int(y+target_well_pixel_y) # on screen center
    
            # Green selection circle
            Color(0., 1., 0., 1., group='selected_well')
            Line(circle=(target_well_center_x, target_well_center_y, well_radius_pixel_x), group='selected_well')

            #  Red Crosshairs
            # ------------------
            if self._motion_enabled:
                x_current = lumaview.scope.get_current_position('X')
                x_current = np.clip(x_current, 0, 120000) # prevents crosshairs from leaving the stage area
                y_current = lumaview.scope.get_current_position('Y')
                y_current = np.clip(y_current, 0, 80000) # prevents crosshairs from leaving the stage area

                # Convert stage coordinates to relative pixel coordinates
                pixel_x, pixel_y = coordinate_transformer.stage_to_pixel(
                    labware=labware,
                    stage_offset=settings['stage_offset'],
                    sx=x_current,
                    sy=y_current,
                    scale_x=scale_x,
                    scale_y=scale_y
                )
                
                x_center = x+pixel_x
                y_center = y+pixel_y

                Color(1., 0., 0., 1., group='crosshairs')
                Line(points=(x_center-10, y_center, x_center+10, y_center), width = 1, group='crosshairs') # horizontal line
                Line(points=(x_center, y_center-10, x_center, y_center+10), width = 1, group='crosshairs') # vertical line


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

        # try:
        #     os.chdir(source_path)
        #     with open('./data/objectives.json', "r") as read_file:
        #         self.objectives = json.load(read_file)
        # except:
        #     logger.exception('[LVP Main  ] Unable to open objectives.json.')
        #     raise


    # def get_objective_info(self, objective_id: str) -> dict:
    #     return self.objectives[objective_id]


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

                if settings['profiling']['enabled']:
                    global profiling_helper
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    profiling_save_path = f'./logs/profile/{ts}'
                    profiling_helper = profiling_utils.ProfilingHelper(save_path=profiling_save_path)
                    Clock.schedule_interval(profiling_helper.restart, 30)

                if 'autogain' not in settings['protocol']:
                    settings['protocol']['autogain'] = {
                        'max_duration_seconds': 1.0,
                        'target_brightness': 0.3,
                        'min_gain': 0.0,
                        'max_gain': 20.0,
                    }

                settings['live_folder'] = str(pathlib.Path(settings['live_folder']).resolve())
                
                # update GUI values from JSON data:
               
                # Scope auto-detection
                detected_model = lumaview.scope.get_microscope_model()
                if detected_model in self.scopes.keys():
                    logger.info(f'[LVP Main  ] Auto-detected scope as {detected_model}')
                    self.ids['scope_spinner'].text = detected_model
                else:
                    logger.info(f'[LVP Main  ] Using scope selection from {filename}')
                    self.ids['scope_spinner'].text = settings['microscope']

                if settings['use_full_pixel_depth'] == True:
                    self.ids['enable_full_pixel_depth_btn'].state = 'down'
                else:
                    self.ids['enable_full_pixel_depth_btn'].state = 'normal'
                self.update_full_pixel_depth_state()

                if 'separate_folder_per_channel' in settings:
                    if settings['separate_folder_per_channel'] == True:
                        self.ids['separate_folder_per_channel_id'].state = 'down'
                    else:
                        self.ids['separate_folder_per_channel_id'].state = 'normal'
                self.update_separate_folders_per_channel()

                self.ids['live_image_output_format_spinner'].text = settings['image_output_format']['live']
                self.select_live_image_output_format()

                self.ids['sequenced_image_output_format_spinner'].text = settings['image_output_format']['sequenced']
                self.select_sequenced_image_output_format()

                self.ids['frame_width_id'].text = str(settings['frame']['width'])
                self.ids['frame_height_id'].text = str(settings['frame']['height'])
                lumaview.scope.set_frame_size(settings['frame']['width'], settings['frame']['height'])

                self.ids['binning_spinner'].text = str(settings['binning']['size'])
                self.select_binning_size()
                lumaview.scope.set_stage_offset(stage_offset=settings['stage_offset'])

                objective_id = settings['objective_id']
                self.ids['objective_spinner'].text = objective_id
                objective = objective_helper.get_objective_info(objective_id=objective_id)
                self.ids['magnification_id'].text = f"{objective['magnification']}"
                lumaview.scope.set_objective(objective_id=objective_id)

                if settings['scale_bar']['enabled'] == True:
                    self.ids['enable_scale_bar_btn'].state = 'down'
                else:
                    self.ids['enable_scale_bar_btn'].state = 'normal'
                self.update_scale_bar_state()

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

            except:
                logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')

        self.set_ui_features_for_scope()


    def update_separate_folders_per_channel(self):
        global settings

        if self.ids['separate_folder_per_channel_id'].state == 'down':
            self._seperate_folder_per_channel = True
        else:
            self._seperate_folder_per_channel = False

        settings['separate_folder_per_channel'] = self._seperate_folder_per_channel


    def update_bullseye_state(self):
        if self.ids['enable_bullseye_btn_id'].state == 'down':
            lumaview.ids['viewer_id'].update_shader(false_color='BF')
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye = True
        else:
            for layer in common_utils.get_layers():
                accordion = layer + '_accordion'
                if lumaview.ids['imagesettings_id'].ids[accordion].collapse == False:
                    if lumaview.ids['imagesettings_id'].ids[layer].ids['false_color'].active:
                        lumaview.ids['viewer_id'].update_shader(false_color=layer)
                        
                    break

            lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye = False


    def update_full_pixel_depth_state(self):
        global settings

        if self.ids['enable_full_pixel_depth_btn'].state == 'down':
            use_full_pixel_depth = True
        else:
            use_full_pixel_depth = False

        lumaview.ids['viewer_id'].ids['scope_display_id'].use_full_pixel_depth = use_full_pixel_depth

        if use_full_pixel_depth:
            lumaview.scope.camera.set_pixel_format('Mono12')
        else:
            lumaview.scope.camera.set_pixel_format('Mono8')

        settings['use_full_pixel_depth'] = use_full_pixel_depth

    
    def select_live_image_output_format(self):
        global settings
        settings['image_output_format']['live'] = self.ids['live_image_output_format_spinner'].text


    def select_sequenced_image_output_format(self):
        global settings
        settings['image_output_format']['sequenced'] = self.ids['sequenced_image_output_format_spinner'].text

    
    def update_binning_size(self, size: int):
        global settings
        # size = int(self.ids['binning_spinner'].text)
        lumaview.scope.set_binning_size(size=size)
        settings['binning']['size'] = size


    def update_scale_bar_state(self):
        if self.ids['enable_scale_bar_btn'].state == 'down':
            enabled = True  
        else:
            enabled = False
            
        lumaview.scope.set_scale_bar(enabled=enabled)
        settings['scale_bar']['enabled'] = enabled

    def update_crosshairs_state(self):
        if self.ids['enable_crosshairs_btn'].state == 'down':
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_crosshairs = True
        else:
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_crosshairs = False


    def update_live_image_histogram_equalization(self):
        if self.ids['enable_live_image_histogram_equalization_btn'].state == 'down':
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_live_image_histogram_equalization = True
        else:
            lumaview.ids['viewer_id'].ids['scope_display_id'].use_live_image_histogram_equalization = False


    # Save settings to JSON file
    def save_settings(self, file="./data/current.json"):
        logger.info('[LVP Main  ] MicroscopeSettings.save_settings()')
        global settings

        if (type(file) == str) and (file[-5:].lower() != '.json'):
                file = file+'.json'

        os.chdir(source_path)
        with open(file, "w") as write_file:
            json.dump(settings, write_file, indent = 4, cls=CustomJSONizer)

    
    def load_binning_sizes(self):
        spinner = self.ids['binning_spinner']
        sizes = (1,2,4)
        spinner.values = list(map(str,sizes))


    def select_binning_size(self):
        global settings

        orig_binning_size = lumaview.scope.get_binning_size()
        orig_frame_size = get_current_frame_dimensions()

        new_binning_size = int(self.ids['binning_spinner'].text)
        self.update_binning_size(size=new_binning_size)
        ratio = new_binning_size / orig_binning_size
        new_frame_size = {
            'width': math.floor(orig_frame_size['width'] / ratio),
            'height': math.floor(orig_frame_size['height'] / ratio),
        }
        self.ids['frame_width_id'].text = str(new_frame_size['width'])
        self.ids['frame_height_id'].text = str(new_frame_size['height'])
        self.frame_size()


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
        stage.full_redraw()


    def set_ui_features_for_scope(self) -> None:
        scope_configs = lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].scopes
        selected_scope_config = scope_configs[settings['microscope']]
        motion_settings =  lumaview.ids['motionsettings_id']
        motion_settings.set_turret_control_visibility(visible=selected_scope_config['Turret'])
        motion_settings.set_xystage_control_visibility(visible=selected_scope_config['XYStage'])
        motion_settings.set_tiling_control_visibility(visible=selected_scope_config['XYStage'])

        protocol_settings = lumaview.ids['motionsettings_id'].ids['protocol_settings_id']
        protocol_settings.set_labware_selection_visibility(visible=selected_scope_config['XYStage'])
        protocol_settings.set_show_protocol_step_locations_visibility(visible=selected_scope_config['XYStage'])

        if selected_scope_config['XYStage'] is False:
            stage.remove_parent()
            protocol_settings.select_labware(labware="Center Plate")

        stage.set_motion_capability(enabled=selected_scope_config['XYStage'])

           
    def load_objectives(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_objectives()')
        spinner = self.ids['objective_spinner']
        spinner.values = objective_helper.get_objectives_list()


    def select_objective(self):
        logger.info('[LVP Main  ] MicroscopeSettings.select_objective()')
        global lumaview
        global settings

        objective_id = self.ids['objective_spinner'].text
        objective = objective_helper.get_objective_info(objective_id=objective_id)
        settings['objective_id'] = objective_id
        microscope_settings_id = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['magnification_id'].text = f"{objective['magnification']}"

        lumaview.scope.set_objective(objective_id=objective_id)

        fov_size = common_utils.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=settings['frame'],
            binning_size=get_binning_from_ui(),
        )
        self.ids['field_of_view_width_id'].text = str(round(fov_size['width'],0))
        self.ids['field_of_view_height_id'].text = str(round(fov_size['height'],0))
        
    def frame_size(self):
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        global lumaview
        global settings

        if not lumaview.scope.camera_is_connected():
            return

        try:
            current_frame_size = get_current_frame_dimensions()
        except ValueError:
            current_frame_size = {
                'width': settings['frame']['width'],
                'height': settings['frame']['height'],
            }

        width = int(min(current_frame_size['width'], lumaview.scope.get_max_width()))
        height = int(min(current_frame_size['height'], lumaview.scope.get_max_height()))

        try:
            min_frame_size = lumaview.scope.camera.get_min_frame_size()
            width = max(width, min_frame_size['width'])
            height = max(height, min_frame_size['height'])

            max_frame_size = lumaview.scope.camera.get_max_frame_size()
            width = min(width, max_frame_size['width'])
            height = min(height, max_frame_size['height'])
        except:
            pass

        settings['frame']['width'] = width
        settings['frame']['height'] = height

        self.ids['frame_width_id'].text = str(width)
        self.ids['frame_height_id'].text = str(height)

        objective_id = settings['objective_id']
        objective = objective_helper.get_objective_info(objective_id=objective_id)

        fov_size = common_utils.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=settings['frame'],
            binning_size=get_binning_from_ui(),
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
        try:
            ill_val = float(self.ids['ill_text'].text)
        except:
            return
        
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
        try:
            gain_val = float(self.ids['gain_text'].text)
        except:
            return
        
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

        try:
            exp_val = float(self.ids['exp_text'].text)
        except:
            return
        
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
        move_absolute_position('Z', pos)  # set current z height in usteps


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

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = settings[self.layer]['exp']
        gain = settings[self.layer]['gain']

        lumaview.scope.set_gain(gain)
        lumaview.scope.set_exposure_time(exposure)

        # update gain to currently selected settings
        # -----------------------------------------------------
        auto_gain_enabled = settings[self.layer]['auto_gain']

        if not ignore_auto_gain:
            autogain_settings = get_auto_gain_settings()
            lumaview.scope.set_auto_gain(auto_gain_enabled, settings=autogain_settings)

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        if lumaview.ids['viewer_id'].ids['scope_display_id'].use_bullseye is False:
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

        try:
            settings['zstack']['step_size'] = float(self.ids['zstack_stepsize_id'].text)
            settings['zstack']['range'] = float(self.ids['zstack_range_id'].text)
        except:
            return

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


    def _reset_run_zstack_acquire_button(self, **kwargs):
        self.ids['zstack_aqr_btn'].state = 'normal'
        self.ids['zstack_aqr_btn'].text = 'Acquire'


    def _cleanup_at_end_of_acquire(self):
        sequenced_capture_executor.reset()
        self._reset_run_zstack_acquire_button()


    def _zstack_run_complete(self, **kwargs):
        self._reset_run_zstack_acquire_button()
        create_hyperstacks_if_needed()


    def run_zstack_acquire_from_ui(self):
        logger.info('[LVP Main  ] ZStack.run_zstack_acquire_from_ui()')

        trigger_source = 'zstack'
        run_not_started_func = self._reset_run_zstack_acquire_button
        run_complete_func = self._zstack_run_complete

        run_trigger_source = sequenced_capture_executor.run_trigger_source()
        if sequenced_capture_executor.run_in_progress() and \
            (run_trigger_source != trigger_source):
            run_not_started_func()
            logger.warning(f"Cannot start Z-Stack acquire. Run already in progress from {run_trigger_source}")
            return
        
        if self.ids['zstack_aqr_btn'].state == 'normal':
            self._cleanup_at_end_of_acquire()
            return
        
        # Note: This will be quickly overwritten by the remaining number of scans
        self.ids['zstack_aqr_btn'].text = 'Running Z-Stack'

        config = get_sequenced_capture_config_from_ui()

        labware_id, _ = get_selected_labware()
        objective_id, _ = get_current_objective_info()
        zstack_positions_valid, zstack_positions = get_zstack_positions()
        active_layer, active_layer_config = get_active_layer_config()
        active_layer_config['acquire'] = True

        if not config['zstack_positions_valid']:
            logger.info('[LVP Main  ] ZStack.acquire_zstack() -> No Z-Stack positions configured')
            run_not_started_func()
            return
        
        curr_position = get_current_plate_position()
        curr_position.update({'name': 'ZStack'})

        positions = [
            curr_position,
        ]
        
        tiling_config = TilingConfig(
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        )
        
        config = {
            'labware_id': labware_id,
            'positions': positions,
            'objective_id': objective_id,
            'zstack_positions': zstack_positions,
            'zstack_positions_valid': zstack_positions_valid,
            'use_zstacking': True,
            'tiling': tiling_config.no_tiling_label(),
            'layer_configs': {active_layer: active_layer_config},
            'period': None,
            'duration': None,
            'frame_dimensions': get_current_frame_dimensions(),
            'binning_size': get_binning_from_ui(),
        }
        
        zstack_sequence = Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        )

        autogain_settings = get_auto_gain_settings()

        callbacks = {
            'move_position': _handle_ui_update_for_axis,
            'update_scope_display': lumaview.ids['viewer_id'].ids['scope_display_id'].update_scopedisplay,
            'run_complete': run_complete_func,
            'leds_off': _handle_ui_for_leds_off,
            'led_state': _handle_ui_for_led,
        }

        parent_dir = pathlib.Path(settings['live_folder']).resolve() / "Manual" / "Z-Stacks"

        initial_position = get_current_plate_position()
        image_capture_config = get_image_capture_config_from_ui()

        sequenced_capture_executor.run(
            protocol=zstack_sequence,
            run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
            run_trigger_source=trigger_source,
            max_scans=1,
            sequence_name='zstack',
            parent_dir=parent_dir,
            image_capture_config=image_capture_config,
            enable_image_saving=is_image_saving_enabled(),
            separate_folder_per_channel=False,
            autogain_settings=autogain_settings,
            callbacks=callbacks,
            return_to_position=initial_position,
            leds_state_at_end="off",
        )

        set_last_save_folder(dir=sequenced_capture_executor.run_dir())


class FileChooseBTN(Button):
    context  = StringProperty()
    selection = ListProperty([])

    def choose(self, context):
        logger.info(f'[LVP Main  ] FileChooseBTN.choose({context})')
        # Call plyer filechooser API to run a filechooser Activity.
        self.context = context

        # Show previously selected/default folder
        selected_path = None
        filetypes = None
        filetypes_tk = None
        if self.context == "load_protocol":
            selected_path = str(pathlib.Path(settings['live_folder']))
            filetypes = ["*.tsv"]
            filetypes_tk = [('TSV', '.tsv')]
        elif self.context == "load_settings":
            filetypes=["*.json"]
            filetypes_tk = [('JSON', '.json')]
        elif self.context == "load_cell_count_input_image":
            filetypes=["*.tif?"]
            filetypes_tk = [('TIFF', '.tif .tiff')]
        elif self.context == "load_cell_count_method":
            filetypes_tk = [('JSON', '.json')]
            filetypes=["*.json"]
        else:
            logger.exception(f"Unsupported handling for {self.context}")
            return

        if sys.platform in ('win32', 'darwin'):
            # Tested for Windows/Mac platforms

            # Use root with attributes to keep filedialog on top
            # Ref: https://stackoverflow.com/questions/3375227/how-to-give-tkinter-file-dialog-focus
            root = Tk()
            root.attributes('-alpha', 0.0)
            root.attributes('-topmost', True)
            selection = filedialog.askopenfilename(
                parent=root,
                initialdir=selected_path,
                filetypes=filetypes_tk
            )
            root.destroy()

            # Nothing selected/cancel
            if selection == '':
                return
            
            self.handle_selection(selection=[selection])
            return
        
        else:
            filechooser.open_file(
                on_selection=self.handle_selection,
                filters=filetypes
            )
            return


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
        if self.context in (
            "apply_stitching_to_folder",
            "apply_composite_gen_to_folder",
            "apply_video_gen_to_folder",
        ):
            selected_path = pathlib.Path(settings['live_folder']) / PROTOCOL_DATA_DIR_NAME
            if not selected_path.exists():
                selected_path = pathlib.Path(settings['live_folder'])
            
            selected_path = str(selected_path)
        elif self.context in (
            "apply_zprojection_to_folder",
        ):
            # Special handling for Z-Projections since they can either be from protocols or
            # from manually-acquired Z-Stacks
            if last_save_folder is not None:
                selected_path = pathlib.Path(last_save_folder)
                if not selected_path.exists():
                    selected_path = pathlib.Path(settings['live_folder'])
            else:
                selected_path = pathlib.Path(settings['live_folder'])

            selected_path = str(selected_path)
            
        else:
            selected_path = settings['live_folder']


        # Note: Could likely use tkinter filedialog for all platforms
        # works on windows and MacOSX
        # but needs testing on Linux
        if sys.platform in ('win32','darwin'):
            # Tested for Windows/Mac platforms

            # Use root with attributes to keep filedialog on top
            # Ref: https://stackoverflow.com/questions/3375227/how-to-give-tkinter-file-dialog-focus
            root = Tk()
            root.attributes('-alpha', 0.0)
            root.attributes('-topmost', True)
            selection = filedialog.askdirectory(
                parent=root,
                initialdir=selected_path
            )
            root.destroy()

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
            settings['live_folder'] = str(pathlib.Path(path).resolve())
        elif self.context == 'apply_cell_count_method_to_folder':
            cell_count_content.apply_method_to_folder(
                path=path
            )
        elif self.context == 'apply_stitching_to_folder':
            stitch_controls.run_stitcher(path=pathlib.Path(path))
        elif self.context == 'apply_composite_gen_to_folder':
            composite_gen_controls.run_composite_gen(path=pathlib.Path(path))
        elif self.context == 'apply_video_gen_to_folder':
            video_creation_controls.run_video_gen(path=pathlib.Path(path))
        elif self.context == 'apply_zprojection_to_folder':
            zprojection_controls.run_zprojection(path=pathlib.Path(path))
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
            filetypes = [('JSON', '.json')]
        elif self.context == 'saveas_protocol':
            filetypes = [('TSV', '.tsv')]
        elif self.context == 'saveas_cell_count_method':
            filetypes = [('JSON', '.json')]
        else:
            logger.exception(f"Unsupported handling for {self.context}")
            return
        
        selected_path = settings['live_folder']
        
        # Use root with attributes to keep filedialog on top
        # Ref: https://stackoverflow.com/questions/3375227/how-to-give-tkinter-file-dialog-focus
        root = Tk()
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', True)
        selection = filedialog.asksaveasfilename(
            parent=root,
            initialdir=selected_path,
            filetypes=filetypes
        )
        root.destroy()

        # Nothing selected/cancel
        if selection == '':
            return
        
        self.handle_selection(selection=[selection])     


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
        os.chdir(source_path)
        load_log_level()
        load_mode()
        logger.info('[LVP Main  ] LumaViewProApp.on_start()')
        move_home(axis='XY')


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
        global zprojection_controls
        global composite_gen_controls
        global stage
        global wellplate_loader
        global coordinate_transformer
        global objective_helper
        global ij_helper
        global sequenced_capture_executor
        # global autofocus_executor
        self.icon = './data/icons/icon.png'

        stage = Stage()

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

        # load labware file
        wellplate_loader = labware_loader.WellPlateLoader()
        coordinate_transformer = coord_transformations.CoordinateTransformer()

        objective_helper = objectives_loader.ObjectiveLoader()
        
        ij_helper = imagej_helper.ImageJHelper()

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
            
        autofocus_executor = AutofocusExecutor(
            scope=lumaview.scope,
            use_kivy_clock=True,
        )

        sequenced_capture_executor = SequencedCaptureExecutor(
            scope=lumaview.scope,
            stage_offset=settings['stage_offset'],
            autofocus_executor=autofocus_executor,
        )
        
        # Continuously update image of stage and protocol
        Clock.schedule_interval(stage.draw_labware, 0.1)
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

        if getattr(sys, 'frozen', False):
            pyi_splash.close()

        return lumaview

    def _on_resize(self, window, w, h):
        pass
        #Clock.schedule_once(lumaview.ids['motionsettings_id'].check_settings, 0.1)
        #Clock.schedule_once(lumaview.ids['imagesettings_id'].check_settings, 0.1)

    def on_stop(self):
        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')
        if profiling_helper is not None:
            profiling_helper.stop()

        global lumaview

        scope_leds_off()

        lumaview.ids['motionsettings_id'].ids['microscope_settings_id'].save_settings("./data/current.json")


LumaViewProApp().run()
