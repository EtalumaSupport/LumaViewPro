#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

# Python version check — must run before any imports that require 3.11+
import sys
if sys.version_info < (3, 11):
    _ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _msg = (
        f"LumaViewPro requires Python 3.11 or later.\n"
        f"You are running Python {_ver}.\n\n"
        f"Supported versions: 3.11, 3.12, 3.13"
    )
    try:
        # Try GUI dialog first
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Unsupported Python Version", _msg)
        root.destroy()
    except Exception:
        pass
    print(f"ERROR: {_msg}", file=sys.stderr)
    sys.exit(1)

# General

import copy
import logging
import datetime
from datetime import datetime as date_time
import functools
import math
import os
import pathlib
import matplotlib
matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter
import numpy as np
import pandas as pd
import time
import json
import subprocess
import sys
import typing
import shutil
import userpaths
import threading
# import faulthandler

import tkinter
from tkinter import filedialog, Tk
from plyer import filechooser


if __name__ == "__main__":
    # Enable faulthandler to catch segfaults and native crashes
    # faulthandler.enable()

    disable_homing = False
    simulate_mode = '--simulate' in sys.argv
    if simulate_mode:
        sys.argv.remove('--simulate')

    ############################################################################
    #---------------------Directory Initialization-----------------------------#
    ############################################################################


    """Main application entry point"""
    # All the initialization code goes here
    global version, windows_machine, num_cores
    global lumaview, settings, cell_count_content, graphing_controls
    global max_exposure, wellplate_loader, objective_helper
    global coordinate_transformer, sequenced_capture_executor
    global show_tooltips, protocol_running_global, live_histo_setting
    global last_save_folder, stage, ENGINEERING_MODE
    global focus_round
    global io_executor, camera_executor, temp_ij_executor, protocol_executor, file_io_executor, autofocus_thread_executor, stage_executor, turret_executor, reset_executor
    global cpu_pool
    # motorboard_lock, ledboard_lock, camera_lock — removed (never assigned or used)
    global ij_helper
    global live_view_fps

    global use_multiprocessing

    cpu_pool = None
    use_multiprocessing = False

    live_view_fps = 10

    ij_helper = None

    # Directory initialization
    abspath = os.path.abspath(__file__)
    basename = os.path.basename(__file__)
    script_path = abspath[:-len(basename)]

    print(f"Script Location: {script_path}")

    os.chdir(script_path)
    # The version.txt file is in the same directory as the actual script, so making sure it can find the version file.


    windows_machine = False

    if os.name == "nt":
        windows_machine = True


    # Python version check
    if sys.version_info < (3, 10):
        print(f"ERROR: LumaViewPro requires Python 3.10 or later. You are running {sys.version}")
        sys.exit(1)
    if sys.version_info >= (3, 13):
        print(f"WARNING: Python {sys.version_info.major}.{sys.version_info.minor} has not been tested with LumaViewPro. "
              f"Recommended: Python 3.10-3.12.")

    version = ""
    try:
        with open("version.txt") as f:
            version = f.readlines()[0].strip()
    except Exception:
        pass

    # Get git commit hash for build identification (e.g. "4.0.0-beta (eda766e)")
    build_hash = ""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=script_path,
        )
        if result.returncode == 0:
            build_hash = result.stdout.strip()
    except Exception:
        pass

    PROTOCOL_DATA_DIR_NAME = "ProtocolData"

    try:
        with open("marker.lvpinstalled") as f:
            lvp_installed = True
    except Exception:
        lvp_installed = False

    if windows_machine and lvp_installed:
        print("Machine-Type - WINDOWS")
        documents_folder = userpaths.get_my_documents()
        os.chdir(documents_folder)
        lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {version}")

        if os.path.exists(lvp_appdata):
            pass
        else:
            os.mkdir(lvp_appdata)

        source_path = lvp_appdata
        print(f"Data Location: {source_path}")

        os.chdir(source_path)

        if os.path.exists(os.path.join(lvp_appdata, "data")):
            pass
        else:
            shutil.copytree(os.path.join(script_path, "data"), os.path.join(lvp_appdata, "data"))

        if os.path.exists(os.path.join(lvp_appdata, "logs")):
            pass
        else:
            shutil.copytree(os.path.join(script_path, "logs"), os.path.join(lvp_appdata, "logs"))

    elif windows_machine and not lvp_installed:
        print("Machine-Type - WINDOWS (not installed)")
        source_path = script_path
    else:
        print("Machine-Type - NON-WINDOWS")
        source_path = script_path

    num_cores = os.cpu_count()
    print(f"Num cores identified as {num_cores}")


    ############################################################################
    #---------------------Module Imports---------------------------------------#
    ############################################################################

    global DEBUG_MODE

    from lvp_logger import logger, debug

    DEBUG_MODE = debug

    if build_hash:
        version = f"{version} ({build_hash})"
    print(f"LumaViewPro {version}")
    logger.info(f"[LVP Main  ] LumaViewPro {version}")

    if DEBUG_MODE:
        logger.info("[LVP Main  ] Debug mode is enabled.")

    try:
        from modules.settings_init import load_lvp_settings

        load_lvp_settings(logger, source_path)

        from modules.settings_init import settings as initialized_settings

        settings = initialized_settings

    except Exception as e:
        logger.exception(f"[LVP Main  ] Failed to load settings. {e}")

    import modules.profiling_utils as profiling_utils

    from modules.memory_profiler import MemoryLeakProfiler

    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

    #post processing
    from modules.video_builder import VideoBuilder

    from modules.tiling_config import TilingConfig
    import modules.common_utils as common_utils

    import modules.labware as labware
    from modules.autofocus_executor import AutofocusExecutor
    import modules.autofocus_functions as autofocus_functions
    from modules.stitcher import Stitcher
    import modules.binning as binning
    from modules.composite_generation import CompositeGeneration
    from modules.contrast_stretcher import ContrastStretcher
    import modules.coord_transformations as coord_transformations
    import modules.labware_loader as labware_loader
    import modules.lvp_lock as lvp_lock
    import modules.objectives_loader as objectives_loader
    from modules.protocol import Protocol, ProtocolFormatError
    from modules.sequenced_capture_executor import SequencedCaptureExecutor
    from modules.sequenced_capture_run_modes import SequencedCaptureRunMode
    from modules.stack_builder import StackBuilder
    from modules.zstack_config import ZStackConfig
    from modules.json_helper import CustomJSONizer
    from modules.timedelta_formatter import strfdelta
    import modules.imagej_helper as imagej_helper
    import modules.zprojector as zprojector
    from modules.video_writer import VideoWriter
    from modules.debounce import debounce
    from modules.sequential_io_executor import IOTask, SequentialIOExecutor
    import modules.config_helpers as config_helpers
    import modules.scope_commands as scope_commands
    from modules.scope_session import ScopeSession
    import modules.app_context as app_context
    from modules.app_context import AppContext

    import cv2
    import skimage

    # Hardware
    import lumascope_api
    import modules.post_processing as post_processing

    import modules.image_utils as image_utils


    global profiling_helper
    profiling_helper = None


    if getattr(sys, 'frozen', False):
        import pyi_splash # type: ignore
        pyi_splash.update_text("")

    # Disable Kivy's own file logging (LVP has its own RotatingFileHandler)
    os.environ["KIVY_NO_CONSOLELOG"] = "1"
    os.environ["KIVY_NO_FILELOG"] = "1"

    # Kivy configurations
    # Configurations must be set before Kivy is imported
    from kivy.config import Config
    Config.set('input', 'mouse', 'mouse, disable_multitouch')
    Config.set('graphics', 'resizable', True) # this seemed to have no effect so may be unnessesary
    Config.set('kivy', 'exit_on_escape', '0')

    # Maximized at launch — works correctly on macOS, Windows, and Linux
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
    from kivy.graphics import Line, Color, Rectangle, Ellipse, Mesh, InstructionGroup
    # Matplotlib-to-Kivy bridge (replaces kivy-garden.matplotlib)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from kivy.graphics.texture import Texture as KivyTexture
    from kivy.uix.image import Image as KivyImage

    class FigureCanvasKivyAgg(KivyImage):
        """Render a matplotlib figure as a Kivy Image widget using the Agg backend."""

        def __init__(self, figure, **kwargs):
            super().__init__(**kwargs)
            self.figure = figure
            self._canvas_agg = FigureCanvasAgg(figure)
            self.draw()

        def draw(self):
            self._canvas_agg.draw()
            w, h = self._canvas_agg.get_width_height()
            buf = self._canvas_agg.buffer_rgba()
            texture = KivyTexture.create(size=(w, h), colorfmt='rgba')
            texture.blit_buffer(bytes(buf), colorfmt='rgba', bufferfmt='ubyte')
            texture.flip_vertical()
            self.texture = texture

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
    from kivy.graphics import Fbo

    # Video Related
    from kivy.graphics.texture import Texture


    # User Interface Custom Widgets
    from ui.range_slider import RangeSlider
    from ui.progress_popup import show_popup
    from ui.notification_popup import show_notification_popup, show_confirmation_popup, show_confirmation_w_ack_popup
    from ui.hover_behavior import HoverBehavior

    from kivy.uix.togglebutton import ToggleButton

    class RoundedButton(HoverBehavior, Button):
        """Button with rounded corners and hover highlighting."""
        pass

    class RoundedToggleButton(HoverBehavior, ToggleButton):
        """Toggle button with rounded corners and hover highlighting."""
        pass

    import modules.image_utils_kivy as image_utils_kivy




    wellplate_loader = None


    objective_helper = None


    coordinate_transformer = None




    sequenced_capture_executor = None


    show_tooltips = False


    protocol_running_global = threading.Event()  # thread-safe protocol state

    # global autofocus_executor
    # autofocus_executor = None

    live_histo_setting = False


    # -------------------------------------------------------------------------
    # SCROLLVIEW MEMORY CLEANUP UTILITY
    # -------------------------------------------------------------------------
    def cleanup_scrollview_viewport(scrollview):
        """
        Clean up ScrollView viewport textures to prevent memory accumulation.
        This is called after accordion collapse events to release viewport resources.
        """
        try:
            if not isinstance(scrollview, ScrollView):
                return

            # Clear viewport canvas
            if hasattr(scrollview, '_viewport') and scrollview._viewport:
                if hasattr(scrollview._viewport, 'canvas'):
                    scrollview._viewport.canvas.ask_update()

            # Clear effect textures (primary source of memory accumulation)
            for effect in [scrollview.effect_x, scrollview.effect_y]:
                if effect and hasattr(effect, '_texture'):
                    effect._texture = None

            # Clear viewport texture reference
            if hasattr(scrollview, '_viewport_texture'):
                scrollview._viewport_texture = None

            logger.debug('[LVP Main  ] ScrollView viewport cleanup completed')
        except Exception as e:
            logger.warning(f'[LVP Main  ] ScrollView cleanup error: {e}')

    last_save_folder = None
    stage = None

    ENGINEERING_MODE = False

    # Flag to prevent apply_settings during app initialization
    _app_initializing = True

    focus_round = 0



    # Executors and scope_session are created in LumaViewProApp.build()
    io_executor = None
    camera_executor = None
    temp_ij_executor = None
    protocol_executor = None
    file_io_executor = None
    autofocus_thread_executor = None
    scope_display_thread_executor = None
    stage_executor = None
    turret_executor = None
    reset_executor = None
    scope_session = None
    ctx = None  # AppContext — created in build() after all services initialized

    if use_multiprocessing:
        import multiprocessing
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)

        # Import existing writer for process pool
        from modules.sequenced_capture_writer import write_capture, worker_initializer, _noop

        # def worker_initializer():
        #     """Initialize worker process to prevent Kivy initialization."""
        #     import os
        #     import sys

        #     # Set environment variables to prevent Kivy initialization
        #     os.environ['KIVY_NO_CONSOLELOG'] = '1'
        #     os.environ['KIVY_NO_ARGS'] = '1'
        #     os.environ['KIVY_NO_CONFIG'] = '1'
        #     os.environ['KIVY_LOGGER_LEVEL'] = 'critical'

        #     # Prevent pygame sound initialization
        #     os.environ['SDL_AUDIODRIVER'] = 'dummy'

        #     print(f"Worker process {os.getpid()} initialized with Kivy isolation")

        from lvp_logger import lvp_appdata as lvp_appdata_logger

        # Create ProcessPoolExecutor with worker initializer
        cpu_pool = ProcessPoolExecutor(
            max_workers=num_cores-1,
            initializer=worker_initializer,
            initargs=(lvp_appdata_logger,)
        )

        # Warm up the pool
        futures = [cpu_pool.submit(_noop, i) for i in range(num_cores-1)]

        for f in futures:
            f.result()

        logger.info("LVP Main] All processes warmup complete")



else:
    # Minimal dummy class definitions for subprocess compatibility
    class App:
        def __init__(self, **kwargs): pass
        def build(self): pass
        def run(self): pass
        def on_start(self): pass
        def on_stop(self): pass

    class Widget:
        def __init__(self, **kwargs):
            self.ids = {}
            self.parent = None
        def add_widget(self, widget): pass
        def remove_widget(self, widget): pass

    class BoxLayout(Widget): pass
    class FloatLayout(Widget): pass
    class Scatter(Widget): pass
    class Image(Widget): pass
    class Button(Widget): pass
    class ToggleButton(Widget): pass
    class Label(Widget): pass
    class RoundedButton(Button): pass
    class RoundedToggleButton(ToggleButton): pass
    class Slider(Widget): pass
    class ScrollView(Widget): pass
    class Popup(Widget): pass
    class AccordionItem(Widget): pass

    # Graphics classes
    class RenderContext: pass
    class Line: pass
    class Color: pass
    class Rectangle: pass
    class Ellipse: pass
    class Texture: pass

    # Properties - FIXED to accept arguments
    class StringProperty:
        def __init__(self, default_value="", **kwargs):
            self.default_value = default_value
        def __get__(self, obj, objtype): return self.default_value
        def __set__(self, obj, value): pass

    class ObjectProperty:
        def __init__(self, default_value=None, **kwargs):
            self.default_value = default_value
        def __get__(self, obj, objtype): return self.default_value
        def __set__(self, obj, value): pass

    class BooleanProperty:
        def __init__(self, default_value=False, **kwargs):
            self.default_value = default_value
        def __get__(self, obj, objtype): return self.default_value
        def __set__(self, obj, value): pass

    class ListProperty:
        def __init__(self, default_value=None, **kwargs):
            self.default_value = default_value or []
        def __get__(self, obj, objtype): return self.default_value
        def __set__(self, obj, value): pass

    # Other classes
    class MotionEvent: pass
    class Factory: pass

    # Clock dummy
    class Clock:
        @staticmethod
        def schedule_once(func, timeout): pass
        @staticmethod
        def schedule_interval(func, interval): pass
        @staticmethod
        def unschedule(func): pass

    # Metrics
    def dp(value): return value

    # Custom widgets dummies
    class RangeSlider(Widget): pass
    class FigureCanvasKivyAgg(Widget): pass

    # Custom widget functions
    def show_popup(*args, **kwargs): pass
    def show_notification_popup(*args, **kwargs): pass

    # Module dummy
    class image_utils_kivy:
        @staticmethod
        def any_method(*args, **kwargs): pass



# ============================================================================
# Utility / Helper Functions
# ============================================================================

from modules.ui_helpers import (  # noqa: E402 — extracted functions
    move_absolute_position, move_relative_position, move_home, move_home_cb,
    scope_leds_off, set_recording_title, set_writing_title, reset_title,
    live_histo_off, live_histo_reverse, set_last_save_folder,
    reset_acquire_ui, reset_stim_ui,
    _handle_ui_for_leds_off, _handle_ui_for_led,
    _update_step_number_callback, _handle_ui_update_for_axis,
    _handle_autofocus_ui, update_autofocus_selection_after_protocol,
    focus_log, find_nearest_step,
)


# ============================================================================
# Protocol Step Navigation
# ============================================================================

def go_to_step(
    protocol: Protocol,
    step_idx: int,
    ignore_auto_gain: bool = False,
    include_move: bool = True,
    called_from_protocol: bool = True
):
    num_steps = protocol.num_steps()
    protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
    if num_steps <= 0:
        protocol_settings.curr_step = -1
        Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    if (step_idx < 0) or (step_idx >= num_steps):
        protocol_settings.curr_step = -1
        Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)
        return

    step = protocol.step(idx=step_idx)
    protocol_settings.curr_step = step_idx

    Clock.schedule_once(lambda dt: protocol_settings.generate_step_name_input(), 0)
    Clock.schedule_once(lambda dt: protocol_settings.update_step_ui(), 0)


    # Convert plate coordinates to stage coordinates
    if include_move:
        _, labware = get_selected_labware()
        sx, sy = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=settings['stage_offset'],
            px=step["X"],
            py=step["Y"]
        )

        turret_pos = None
        if lumaview.scope.has_turret():
            step_objective_id = step["Objective"]
            turret_pos = lumaview.scope.get_turret_position_for_objective_id(
                objective_id=step_objective_id,
            )

            if turret_pos is None:

                logger.error(f"Cannot move turret for step {step_idx}. No position found with objective {step_objective_id}")

                error_msg = f"Cannot move turret to step {step_idx}. No objective position found matching step's objective: {step_objective_id}. Please check objective settings."
                Clock.schedule_once(lambda dt: show_notification_popup(title="Protocol Objective Not Set", message=error_msg), 0)

        # Move into position
        if lumaview.scope.motion.driver:
            if not called_from_protocol:
                if turret_pos is not None:
                    io_executor.put(IOTask(action=move_absolute_position,kwargs={"axis":'T',"pos": turret_pos,"protocol": False}))
                    Clock.schedule_once(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
                io_executor.put(IOTask(action=move_absolute_position,kwargs={"axis":'X',"pos": sx,"protocol": False}))
                io_executor.put(IOTask(
                        action=move_absolute_position,
                        kwargs={
                            "axis":'Y',
                            "pos": sy,
                            "protocol": False
                        }
                    ))
                io_executor.put(IOTask(
                        action=move_absolute_position,
                        kwargs={
                            "axis":'Z',
                            "pos": step['Z'],
                            "protocol": False
                        }
                    ))
            else:
                if turret_pos is not None:
                    move_absolute_position(axis='T', pos=turret_pos, protocol=True)
                    Clock.schedule_once(lambda dt: ctx.motion_settings.ids['verticalcontrol_id'].update_turret_gui(turret_pos), 0)
                move_absolute_position('X', sx, protocol=True)
                move_absolute_position('Y', sy, protocol=True)
                move_absolute_position('Z', step["Z"], protocol=True, wait_until_complete=True)
        else:
            logger.warning('[LVP Main  ] Motion controller not available.')

        # Update settings to correspond with step — batch write for thread safety
        color = step['Color']
        settings[color].update({
            'autofocus': step['Auto_Focus'],
            'false_color': step['False_Color'],
            'ill': step["Illumination"],
            'gain': step["Gain"],
            'auto_gain': step["Auto_Gain"],
            'exp': step["Exposure"],
            'sum': step["Sum"],
            'acquire': step['Acquire'],
        })

        layer_obj = ctx.image_settings.layer_lookup(layer=color)

        def temp():
            layer_obj.ids['enable_led_btn'].state = 'down'
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)

        if not called_from_protocol and settings['protocol_led_on']:
            scope_commands.led_on(lumaview.scope, io_executor, color, step['Illumination'])
            Clock.schedule_once(lambda dt: temp(), 0)
        else:
            layer_obj.apply_settings(ignore_auto_gain=ignore_auto_gain, protocol=True)



        Clock.schedule_once(lambda dt: go_to_step_update_ui(step), 0)



def go_to_step_update_ui(step):

    color = step['Color']
    layer_obj = ctx.image_settings.layer_lookup(layer=color)

    # open ImageSettings
    ctx.image_settings.ids['toggle_imagesettings'].state = 'down'
    ctx.image_settings.toggle_settings()

    # set accordion item to corresponding channel (skip during protocol to prevent memory leaks)
    if not protocol_running_global.is_set():
        accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=color)
        accordion_item_obj.collapse = False



    # set autofocus checkbox
    logger.info(f'[LVP Main  ] autofocus: {step["Auto_Focus"]}')

    layer_obj.ids['autofocus'].active = step['Auto_Focus']

    # set false_color checkbox
    logger.info(f'[LVP Main  ] false_color: {step["False_Color"]}')

    layer_obj.ids['false_color'].active = step['False_Color']

    # set illumination settings, text, and slider
    logger.info(f'[LVP Main  ] ill: {step["Illumination"]}')

    layer_obj.ids['ill_text'].text = str(step["Illumination"])
    layer_obj.ids['ill_slider'].value = float(step["Illumination"])

    # set gain settings, text, and slider
    logger.info(f'[LVP Main  ] gain: {step["Gain"]}')

    layer_obj.ids['gain_text'].text = str(step["Gain"])
    layer_obj.ids['gain_slider'].value = float(step["Gain"])

    # set auto_gain checkbox
    logger.info(f'[LVP Main  ] auto_gain: {step["Auto_Gain"]}')

    layer_obj.ids['auto_gain'].active = step["Auto_Gain"]

    # set exposure settings, text, and slider
    logger.info(f'[LVP Main  ] exp: {step["Exposure"]}')

    layer_obj.ids['exp_text'].text = str(step["Exposure"])
    layer_obj.ids['exp_slider'].value = float(step["Exposure"])

    # set sum count settings, text, and slider
    logger.info(f'[LVP Main  ] sum: {step["Sum"]}')

    layer_obj.ids['sum_text'].text = str(step["Sum"])
    layer_obj.ids['sum_slider'].value = int(step["Sum"])

    # set video config (e.g., duration) controls
    if 'Video Config' in step and isinstance(step['Video Config'], dict):
        vc = step['Video Config']
        settings[color]['video_config'] = copy.deepcopy(vc)
        if 'duration' in vc:
            layer_obj.ids['video_duration_text'].text = str(vc['duration'])
            layer_obj.ids['video_duration_slider'].value = float(vc['duration'])

    # Set stim configuration for each channel
    if 'Stim_Config' in step:
        if isinstance(step['Stim_Config'], dict):
            # Update each channel's stim config
            for layer in step['Stim_Config']:
                stim_config = step['Stim_Config'][layer]
                settings[layer]['stim_config'] = copy.deepcopy(stim_config)

                stim_layer_obj = ctx.image_settings.layer_lookup(layer=layer)

                if stim_config['enabled']:
                    stim_layer_obj.ids['stim_enable_btn'].active = True
                    stim_layer_obj.ids['stim_disable_btn'].active = False
                else:
                    stim_layer_obj.ids['stim_disable_btn'].active = True
                    stim_layer_obj.ids['stim_enable_btn'].active = False

                stim_layer_obj.update_stim_controls_visibility()

                stim_layer_obj.ids['stim_freq_text'].text = str(stim_config['frequency'])
                stim_layer_obj.ids['stim_freq_slider'].value = float(stim_config['frequency'])
                stim_layer_obj.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                stim_layer_obj.ids['stim_pulse_width_slider'].value = float(stim_config['pulse_width'])
                stim_layer_obj.ids['stim_pulse_count_text'].text = str(stim_config['pulse_count'])
                stim_layer_obj.ids['stim_pulse_count_slider'].value = int(stim_config['pulse_count'])

                # layer_obj.ids['enable_stim_btn'].state = 'down'
                # layer_obj.ids['stim_text'].text = str(stim_config['illumination'])
                # layer_obj.ids['stim_slider'].value = float(stim_config['illumination'])

    # acquire type

    for acquire_sel in ('acquire_video', 'acquire_image', 'acquire_none'):
        layer_obj.ids[acquire_sel].active = False

    if step['Acquire'] == 'video':
        layer_obj.ids['acquire_video'].active = True
    elif step['Acquire'] == 'image':
        layer_obj.ids['acquire_image'].active = True
    else:
        layer_obj.ids['acquire_none'].active = True

    # Set LED button state to match: 'down' if LED is on, 'normal' if off.
    # The actual LED hardware command is sent in go_to_step() when
    # protocol_led_on is True and not called_from_protocol.
    if settings['protocol_led_on'] and not protocol_running_global.is_set():
        layer_obj.ids['enable_led_btn'].state = 'down'




from modules.config_getters import (  # noqa: E402 — extracted functions
    get_binning_from_ui, get_zstack_params, get_zstack_positions,
    get_layer_configs, get_active_layer_config, get_stim_configs,
    get_enabled_stim_configs, get_current_plate_position,
    get_current_frame_dimensions, get_selected_labware,
    get_image_capture_config_from_ui, get_sequenced_capture_config_from_ui,
    get_auto_gain_settings, get_current_objective_info,
    get_protocol_time_params, is_image_saving_enabled,
    create_hyperstacks_if_needed,
)


# Motion, Window Title, and Histogram helpers → modules/ui_helpers.py

def log_system_metrics(dt=None):
    config_helpers.log_system_metrics(settings)

    

from ui.scope_display import ScopeDisplay  # noqa: E402 — extracted widget

# ============================================================================
# CompositeCapture — Shared Capture Capabilities
# ============================================================================

class CompositeCapture(FloatLayout):

    _capturing = False  # Guard against rapid double-clicks

    def __init__(self, **kwargs):
        super(CompositeCapture,self).__init__(**kwargs)

    # Gets the current well label (ex. A1, C2, ...)
    def get_well_label(self):
        _, labware = get_selected_labware()

        # Get target position
        try:
            x_target = lumaview.scope.get_target_position('X')
            y_target = lumaview.scope.get_target_position('Y')
        except Exception:
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
        if CompositeCapture._capturing:
            logger.warning('[LVP Main  ] Capture already in progress, ignoring')
            return
        CompositeCapture._capturing = True
        try:
            self._live_capture_impl()
        finally:
            CompositeCapture._capturing = False

    def _live_capture_impl(self):
        logger.info('[LVP Main  ] CompositeCapture.live_capture()')
        global lumaview

        file_root = 'live_'
        color = 'BF'
        well_label = self.get_well_label()

        use_full_pixel_depth = ctx.scope_display.use_full_pixel_depth
        force_to_8bit_pixel_depth = not use_full_pixel_depth

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            accordion_item_obj =  ctx.image_settings.accordion_item_lookup(layer=layer)
            if not accordion_item_obj.collapse:
                append = f'{well_label}_{layer}'
                if layer_obj.ids['false_color'].active:
                    color = layer

                break

        save_folder = pathlib.Path(settings['live_folder']) / "Manual"
        separate_folder_per_channel = ctx.motion_settings.ids['microscope_settings_id']._seperate_folder_per_channel
        if separate_folder_per_channel:
            save_folder = save_folder / layer

        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        sum_iteration_callback = ctx.scope_display.update_scopedisplay

        layer_configs = get_layer_configs(specific_layers=layer)
        sum_delay_s=layer_configs[layer]['exposure']/1000
        sum_count=layer_configs[layer]['sum']

        if ENGINEERING_MODE is False:
            return lumaview.scope.save_live_image(
                save_folder,
                file_root,
                append,
                color,
                force_to_8bit=force_to_8bit_pixel_depth,
                output_format=settings['image_output_format']['live'],
                sum_count=sum_count,
                sum_delay_s=sum_delay_s,
                sum_iteration_callback=sum_iteration_callback,
                turn_off_all_leds_after=False,
            )

        else:
            use_bullseye = ctx.scope_display.use_bullseye
            use_crosshairs = ctx.scope_display.use_crosshairs

            if not use_bullseye and not use_crosshairs:
                return lumaview.scope.save_live_image(
                    save_folder,
                    file_root,
                    append,
                    color,
                    force_to_8bit=force_to_8bit_pixel_depth,
                    output_format=settings['image_output_format']['live'],
                    sum_count=sum_count,
                    sum_delay_s=sum_delay_s,
                    sum_iteration_callback=sum_iteration_callback,
                    turn_off_all_leds_after=False,
                )

            image_orig = lumaview.scope.get_image(force_to_8bit=force_to_8bit_pixel_depth)
            if image_orig is False:
                return

            # Save both versions of the image (unaltered and overlayed)
            now = datetime.datetime.now()
            time_string = now.strftime("%Y%m%d_%H%M%S")
            append = f"{append}_{time_string}"

            # If not in 8-bit mode, generate an 8-bit copy of the image for visualization
            if use_full_pixel_depth:
                image = image_utils.convert_12bit_to_8bit(image_orig)
            else:
                image = image_orig

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

            if use_bullseye:
                bullseye_image = ctx.scope_display.transform_to_bullseye(image)
            else:
                bullseye_image = image

            if use_crosshairs:
                crosshairs_image = ctx.scope_display.add_crosshairs(bullseye_image)
            else:
                crosshairs_image = bullseye_image

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


    # capture and save a composite image using the current settings
    def composite_capture(self):
        if CompositeCapture._capturing:
            logger.warning('[LVP Main  ] Composite capture already in progress, ignoring')
            return
        CompositeCapture._capturing = True

        z_stage_present = not disable_homing

        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')
        global lumaview

        initial_layer = common_utils.get_opened_layer(ctx.image_settings)

        if lumaview.scope.get_led_state(initial_layer)['enabled']:
            led_restore_state = True
        else:
            led_restore_state = False

        acquired_channel_count = 0
        most_recent_aq_channel = None

        live_histo_off()

        if lumaview.scope.camera.active is None:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        use_full_pixel_depth = scope_display.use_full_pixel_depth

        # Run hardware-blocking work on a background thread to avoid freezing UI
        t = threading.Thread(
            target=self._composite_capture_worker,
            kwargs={
                'z_stage_present': z_stage_present,
                'initial_layer': initial_layer,
                'led_restore_state': led_restore_state,
                'use_full_pixel_depth': use_full_pixel_depth,
            },
            daemon=True,
            name='CompositeCapture',
        )
        t.start()

    def _composite_capture_worker(
        self,
        z_stage_present,
        initial_layer,
        led_restore_state,
        use_full_pixel_depth,
    ):
        """Runs on background thread — performs hardware I/O without blocking UI."""
        global lumaview

        # Snapshot settings at entry for thread safety — avoids seeing partial
        # updates from the UI thread during the capture sequence.
        all_layers = (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
            *common_utils.get_luminescence_layers(),
        )
        layer_settings = {layer: dict(settings[layer]) for layer in all_layers}
        frame_settings = dict(settings['frame'])
        live_folder = settings['live_folder']
        image_output_format = dict(settings['image_output_format'])

        acquired_channel_count = 0
        most_recent_aq_channel = None

        if use_full_pixel_depth:
            dtype = np.uint16
        else:
            dtype = np.uint8

        img = np.zeros((frame_settings['height'], frame_settings['width'], 3), dtype=dtype)
        transmitted_present = False

        for trans_layer in common_utils.get_transmitted_layers():
            trans_layer_obj = ctx.image_settings.layer_lookup(layer=trans_layer)
            if layer_settings[trans_layer]["acquire"] == "image":
                transmitted_present = True
                acquired_channel_count += 1
                most_recent_aq_channel = trans_layer

                if z_stage_present:
                    # Move to focus position via io_executor (blocks until arrived)
                    focus_pos = layer_settings[trans_layer]['focus']
                    scope_commands.move_absolute_sync(
                        lumaview.scope, io_executor, 'Z', focus_pos,
                        wait_until_complete=True,
                    )

                # set the gain and exposure via camera_executor
                gain = layer_settings[trans_layer]['gain']
                scope_commands.set_gain_sync(lumaview.scope, camera_executor, gain)
                exposure = layer_settings[trans_layer]['exp']
                scope_commands.set_exposure_sync(lumaview.scope, camera_executor, exposure)

                # update illumination to currently selected settings
                illumination = layer_settings[trans_layer]['ill']

                # Transmitted channel capture — route LED through io_executor
                scope_commands.led_on_sync(
                    lumaview.scope, io_executor,
                    lumaview.scope.color2ch(trans_layer), illumination,
                )

                transmitted_channel = scope_commands.capture_and_wait_sync(
                    lumaview.scope, camera_executor,
                    force_to_8bit=not use_full_pixel_depth,
                )
                scope_commands.leds_off_sync(lumaview.scope, io_executor)

                img = np.array(transmitted_channel, dtype=dtype)

                # Init mask to keep track of changed pixels
                # Set all values in the mask for changed to False
                mask_transmitted_changed = np.zeros(img.shape[:2], dtype=bool)

                # Prep transmitted channel to have 3 channels for RGB value manipulation
                img = np.repeat(transmitted_channel[:, :, None], 3, axis=2)

                # Can only use one transmitted channel per composite
                break


        layer_map = {
            'Red': 0,
            'Green': 1,
            'Blue': 2,
            'Lumi': 2,
        }

        scope_commands.leds_off_sync(lumaview.scope, io_executor)

        for layer in (*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers()):
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            if layer_settings[layer]['acquire'] == "image":
                acquired_channel_count += 1
                most_recent_aq_channel = layer

                if z_stage_present:
                    # Move to focus position via io_executor (blocks until arrived)
                    focus_pos = layer_settings[layer]['focus']
                    scope_commands.move_absolute_sync(
                        lumaview.scope, io_executor, 'Z', focus_pos,
                        wait_until_complete=True,
                    )

                # set the gain and exposure via camera_executor
                gain = layer_settings[layer]['gain']
                scope_commands.set_gain_sync(lumaview.scope, camera_executor, gain)
                exposure = layer_settings[layer]['exp']
                scope_commands.set_exposure_sync(lumaview.scope, camera_executor, exposure)
                sum_count=layer_settings[layer]['sum']
                sum_iteration_callback = ctx.scope_display.update_scopedisplay

                # Set brightness threshold for composites dealing with transmitted channels
                # If given in percentage, convert to 8 or 16 bit value
                if not use_full_pixel_depth:
                    brightness_threshold = layer_settings[layer]["composite_brightness_threshold"] / 100 * 255
                else:
                    brightness_threshold = layer_settings[layer]["composite_brightness_threshold"] / 100 * 4095

                # update illumination to currently selected settings
                illumination = layer_settings[layer]['ill']

                # Fluorescence capture — route LED through io_executor
                # Check to make sure we are not capturing from a luminescence layer which doesn't use an LED
                if layer not in common_utils.get_transmitted_layers():
                    scope_commands.led_on_sync(
                        lumaview.scope, io_executor,
                        lumaview.scope.color2ch(layer), illumination,
                    )

                img_gray = scope_commands.capture_and_wait_sync(
                    lumaview.scope, camera_executor,
                    force_to_8bit=not use_full_pixel_depth,
                    sum_count=sum_count,
                    sum_delay_s=exposure/1000,
                    sum_iteration_callback=sum_iteration_callback,
                )
                scope_commands.leds_off_sync(lumaview.scope, io_executor)

                img_gray = np.array(img_gray)

                if transmitted_present:
                    # Create mask of every pixel > brightness threshold in channel image
                    channel_above_threshold_mask = img_gray > brightness_threshold

                    # Create masks for pixels that correspond to changed/unchanged pixels in the transmitted image
                    not_changed_mask = channel_above_threshold_mask & (~mask_transmitted_changed)
                    changed_mask = channel_above_threshold_mask & mask_transmitted_changed

                    # Find channel index value
                    channel_index = layer_map[layer]

                    # For not-yet changed pixels, set every other channel to 0, then the desired color channel value
                    # Allows desired channel to show up fully
                    img[not_changed_mask, 0] = 0
                    img[not_changed_mask, 1] = 0
                    img[not_changed_mask, 2] = 0

                    img[not_changed_mask, channel_index] = img_gray[not_changed_mask]

                    # Update changed pixels
                    mask_transmitted_changed[not_changed_mask] = True

                    # For already changed pixels, only update the current channel value (allows stacking of RGB values)
                    img[changed_mask, channel_index] = img_gray[changed_mask]


                else:
                    # No transmitted channel present
                    # buffer the images
                    if layer == 'Red':
                        img[:,:,0] = img_gray
                    elif layer == 'Green':
                        img[:,:,1] = img_gray
                    elif layer in ('Blue', 'Lumi'):
                        img[:,:,2] = img_gray

            scope_commands.leds_off_sync(lumaview.scope, io_executor)

            Clock.schedule_once(lambda dt, lo=layer_obj: Clock.unschedule(lo.ids['histo_id'].histogram), 0)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        # File saving can run on this thread (no UI dependency)
        append = f'{self.get_well_label()}'

        save_folder = pathlib.Path(live_folder) / "Manual"
        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        if acquired_channel_count != 1 and acquired_channel_count != 0:
            lumaview.scope.save_image(
                array=img,
                save_folder=save_folder,
                file_root='composite_',
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live']
            )
        elif acquired_channel_count != 0:
            lumaview.scope.save_image(
                array=img,
                save_folder=save_folder,
                file_root=f"{most_recent_aq_channel}_Image_",
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live']
            )
        else:
            logger.info("[Composite Capture  ] No image saved as no channels were selected")

        # UI updates must happen on the main thread
        def _restore_ui(dt):
            lumaview.ids['composite_btn'].state = 'normal'
            live_histo_reverse()
            opened_layer_obj = common_utils.get_opened_layer_obj(ctx.image_settings)
            if led_restore_state:
                opened_layer_obj.ids['enable_led_btn'].state = 'down'
            else:
                opened_layer_obj.ids['enable_led_btn'].state = 'normal'
            opened_layer_obj.apply_settings(update_led=True)

        CompositeCapture._capturing = False
        Clock.schedule_once(_restore_ui, 0)
        # # Reverse to settings of the channel that were originally selected
        # if initial_layer is not None:
        #     gain = settings[initial_layer]['gain']
        #     lumaview.scope.set_gain(gain)
        #     exposure = settings[initial_layer]['exp']
        #     lumaview.scope.set_exposure_time(exposure)
        #     sum_count=settings[initial_layer]['sum']
        #     sum_iteration_callback = ctx.scope_display.update_scopedisplay


# ============================================================================
# MainDisplay — Primary Application Display (Recording, Camera, Fit/Zoom)
# ============================================================================

class MainDisplay(CompositeCapture): # i.e. global lumaview

    def __init__(self, **kwargs):
        super(MainDisplay,self).__init__(**kwargs)
        self.scope = lumascope_api.Lumascope(camera_type=initialized_settings['camera_type'], simulate=simulate_mode)
        self.camera_temps_event = None
        self.recording = threading.Event()
        self.recording.clear()
        self.video_writing = threading.Event()  # Track if video is being written
        self.video_writing.clear()
        self.recording_check = None
        self.recording_event = None
        self.recording_complete_event = None
        self.recording_title_update = None
        self.writing_progress_update = None
        self.video_writing_progress = 0
        self.video_writing_total_frames = 0
        self.led_on_before_pause = False

    def log_camera_temps(self):
        if self.scope.camera_is_connected():
            temps = self.scope.get_camera_temps()
            for source, temp in temps.items():
                logger.info(f'[CAM Class ] Camera {source} Temperature : {temp:.2f} °C')
        else:
            if self.camera_temps_event is not None:
                Clock.unschedule(self.camera_temps_event)

    def cam_toggle(self):
        logger.info('[LVP Main  ] MainDisplay.cam_toggle()')
        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        if self.scope.camera.active is None:
            return

        if scope_display.play:
            scope_display.play = False
            scope_display.stop()
            if self.scope.led:
                self.led_on_before_pause = self.scope.get_led_state(color=common_utils.get_opened_layer(ctx.image_settings))['enabled']
                scope_commands.leds_off(self.scope, io_executor)
                layer_obj = ctx.image_settings.layer_lookup(layer=common_utils.get_opened_layer(ctx.image_settings))
                layer_obj.update_led_toggle_ui()
        else:
            if self.led_on_before_pause:
                opened_layer = common_utils.get_opened_layer(ctx.image_settings)
                io_executor.put(IOTask(
                    action=self.scope.led_on,
                    kwargs={'channel': self.scope.color2ch(opened_layer), 'mA': settings[opened_layer]['ill']}
                ))
                layer_obj = ctx.image_settings.layer_lookup(layer=opened_layer)
                layer_obj.update_led_toggle_ui()

            scope_display.play = True
            scope_display.start()

    def record_button(self):
        if self.recording.is_set():
            return

        # Check if video is currently being written
        if self.video_writing.is_set():
            logger.warning('[LVP Main  ] Cannot start recording - video is being written')
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Video Being Written",
                message="Please wait for the current video to finish writing before starting a new recording."
            ), 0)
            # Reset button state
            try:
                self.ids['record_btn'].state = 'normal'
            except Exception:
                pass
            return

        camera_executor.put(IOTask(self.record_init))

    def open_save_folder_button(self):
        open_last_save_folder()

    def record_init(self):
        logger.info('[LVP Main  ] MainDisplay.record()')

        # Guard against race condition: if another record_init() already started, abort
        if self.recording.is_set():
            logger.warning('[LVP Main  ] Recording already in progress, ignoring duplicate record_init()')
            return

        if self.scope.camera.active is None:
            return

        # Atomically claim the recording operation
        self.recording.set()

        self.video_as_frames = settings['video_as_frames']

        color = None

        for layer in common_utils.get_layers():
            layer_accordion_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            if not layer_accordion_obj.collapse:

                if layer_obj.ids['false_color'].active:
                    color = layer

                break

        if color is not None:
            self.video_false_color = color
        else:
            self.video_false_color = None

        if "manual_video" in settings:
            max_fps = settings["manual_video"]["max_fps"]
            max_duration = settings["manual_video"]["max_duration"]
        else:
            max_fps = 40
            max_duration = 30

        # Clamp the FPS to be no faster than the exposure rate
        frame_size = self.scope.camera.get_frame_size()
        exposure = self.scope.camera.get_exposure_t()
        exposure_freq = 1.0 / (exposure / 1000)
        video_fps = min(exposure_freq, max_fps)

        max_frames = math.ceil(video_fps * max_duration)

        start_time = datetime.datetime.now()
        self.start_time_str = start_time.strftime("%Y-%m-%d_%H.%M.%S")

        if self.video_as_frames:
            save_folder = pathlib.Path(settings['live_folder']) / "Manual" / f"Video_{self.start_time_str}"
        else:
            save_folder = pathlib.Path(settings['live_folder']) / "Manual"

        self.video_save_folder = save_folder

        self.start_ts = time.time()
        self.stop_ts = self.start_ts + max_duration
        seconds_per_frame = 1.0 / video_fps

        self.memmap_location = pathlib.Path(settings['live_folder']) / "recording_temp.dat"

        if not settings['use_full_pixel_depth'] or not settings['video_as_frames']:
            dtype = 'uint8'
        else:
            dtype = 'uint16'

        # Calculate expected file size and shape
        if (color is None) or (dtype == 'uint16'):
            required_shape = (max_frames, frame_size["height"], frame_size["width"])
        else:
            required_shape = (max_frames, frame_size["height"], frame_size["width"], 3)

        bytes_per_element = 1 if dtype == 'uint8' else 2
        expected_size = int(np.prod(required_shape, dtype=np.int64)) * bytes_per_element

        # Check if we can reuse existing file (fast path - no truncation needed)
        reuse_existing = False
        if self.memmap_location.exists():
            try:
                actual_size = self.memmap_location.stat().st_size
                if actual_size == expected_size:
                    logger.info('[LVP Main  ] Reusing existing memmap file (same size)')
                    reuse_existing = True
                else:
                    logger.info(f'[LVP Main  ] Memmap size changed ({actual_size} -> {expected_size}), recreating')
                    # Try to delete old file, but don't block if it fails
                    try:
                        self.memmap_location.unlink()
                    except (OSError, PermissionError) as e:
                        logger.warning(f'[LVP Main  ] Could not remove old memmap: {e}, will overwrite')
            except Exception as e:
                logger.warning(f'[LVP Main  ] Could not check memmap file: {e}')

        # Create or reuse memmap
        try:
            # Use mode="r+" to reuse existing file without truncation (fast)
            # Use mode="w+" only when creating new file or size changed (requires truncation)
            memmap_mode = "r+" if reuse_existing else "w+"

            if (color is None) or (dtype == 'uint16'):
                self.current_video_frames = np.memmap(str(self.memmap_location), dtype=dtype, mode=memmap_mode, shape=(max_frames, frame_size["height"], frame_size["width"]))
            else:
                self.current_video_frames = np.memmap(str(self.memmap_location), dtype=dtype, mode=memmap_mode, shape=(max_frames, frame_size["height"], frame_size["width"], 3))
        except (OSError, IOError) as e:
            logger.error(f'[LVP Main  ] Failed to create memmap file: {e}')
            logger.error(f'[LVP Main  ] If this persists, manually delete: {self.memmap_location}')
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Recording Failed",
                message=f"Could not create recording file. The file may be locked from a previous crash.\n\nTry manually deleting:\n{self.memmap_location.name}"
            ), 0)
            raise

        self.current_captured_frames = 0
        self.timestamps = []

        logger.info(f"Manual-Video] Capturing video...")

        # Schedule title updates to show recording progress
        self.recording_title_update = Clock.schedule_interval(self.update_recording_title, 0.1)  # Update every 100ms
        self.recording_check = Clock.schedule_interval(self.check_recording_state, seconds_per_frame)
        self.recording_event = Clock.schedule_interval(self._enqueue_recording_frame, seconds_per_frame)

    def _enqueue_recording_frame(self, dt=None):
        """Enqueue a recording frame task without creating closure."""
        camera_executor.put(IOTask(self.record_helper))

    def check_recording_state(self, dt=None):
        # Over the max duration, stop video
        if time.time() >= self.stop_ts:
            Clock.unschedule(self.recording_check)
            Clock.unschedule(self.recording_event)
            if hasattr(self, 'recording_title_update') and self.recording_title_update:
                Clock.unschedule(self.recording_title_update)
            self.video_duration = time.time() - self.start_ts
            self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)
            self.ids['record_btn'].state = 'normal'

        # Button not clicked yet, keep recording
        if self.ids['record_btn'].state == 'down':
            return

        # Button clicked, stop recording
        Clock.unschedule(self.recording_check)
        Clock.unschedule(self.recording_event)
        if hasattr(self, 'recording_title_update') and self.recording_title_update:
            Clock.unschedule(self.recording_title_update)
        self.video_duration = time.time() - self.start_ts
        self.recording_complete_event = Clock.schedule_once(self._enqueue_recording_complete, 0)

    def update_recording_title(self, dt=None):
        """Update window title with recording elapsed time."""
        if self.recording.is_set():
            elapsed = time.time() - self.start_ts
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {version}   |   Recording Manual Video: {elapsed:.1f}s")

    def update_writing_progress(self, dt=None):
        """Update window title with video writing progress percentage."""
        if self.video_writing_total_frames > 0:
            progress_pct = (self.video_writing_progress / self.video_writing_total_frames) * 100
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {version}   |   Writing Manual Video: {progress_pct:.0f}%")

    def _enqueue_recording_complete(self, dt=None):
        """Enqueue recording finalization task on camera executor."""
        camera_executor.put(IOTask(self._finalize_recording_state))

    def _finalize_recording_state(self, dt=None):
        """Run on camera executor: Capture final state quickly and hand off to file writer."""
        try:
            logger.info("Manual-Video] Finalizing recording state...")

            # Capture state (atomic with respect to camera thread, as we are ON camera thread)
            captured_frames = self.current_captured_frames if hasattr(self, 'current_captured_frames') else 0
            timestamps = self.timestamps[:] if hasattr(self, 'timestamps') else []
            video_frames = self.current_video_frames if hasattr(self, 'current_video_frames') else None
            video_duration = self.video_duration if hasattr(self, 'video_duration') else 0
            video_save_folder = self.video_save_folder if hasattr(self, 'video_save_folder') else None
            start_time_str = self.start_time_str if hasattr(self, 'start_time_str') else ""
            video_as_frames = self.video_as_frames if hasattr(self, 'video_as_frames') else False
            video_false_color = self.video_false_color if hasattr(self, 'video_false_color') else None
            memmap_path = self.memmap_location if hasattr(self, 'memmap_location') else None

            # Release memmap reference from MainDisplay so file_io_executor has exclusive ownership
            self.current_video_frames = None

            # Clear recording event immediately - camera is now free
            if not self.recording.is_set():
                logger.warning("Manual-Video] Recording already cleared in finalize")
            else:
                self.recording.clear()

            # Set video writing event to block new recordings
            self.video_writing.set()

            # Initialize progress tracking
            self.video_writing_progress = 0
            self.video_writing_total_frames = max(1, captured_frames)

            # Schedule progress updates
            self.writing_progress_update = Clock.schedule_interval(self.update_writing_progress, 0.1)

            # Prepare kwargs for file IO
            kwargs = {
                'captured_frames': captured_frames,
                'timestamps': timestamps,
                'video_frames': video_frames,
                'video_duration': video_duration,
                'video_save_folder': video_save_folder,
                'start_time_str': start_time_str,
                'video_as_frames': video_as_frames,
                'memmap_path': memmap_path,
                'video_false_color': video_false_color,
            }

            # Hand off to file IO executor (doesn't block camera)
            file_io_executor.put(IOTask(
                self.recording_complete,
                kwargs=kwargs,
                callback=self._recording_cleanup_callback,
                pass_result=True
            ))

        except Exception as e:
            logger.exception(f"Manual-Video] Error in finalize_recording: {e}")
            # Ensure cleanup happens even if error
            Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path if 'memmap_path' in locals() else None), 0)

    def _recording_cleanup_callback(self, dt=None, result=None, exception=None):
        """Callback after file writing completes - run cleanup on GUI thread."""
        memmap_path = result
        Clock.schedule_once(lambda dt: self._recording_cleanup_gui(memmap_path=memmap_path), 0)

    def recording_complete(self, **kwargs):
        """Run on file_io_executor: Do heavy file writing without blocking camera."""
        # Retrieve captured state passed from _finalize_recording_state
        captured_frames = kwargs.get('captured_frames', 0)
        timestamps = kwargs.get('timestamps', [])
        video_frames = kwargs.get('video_frames', None)
        video_duration = kwargs.get('video_duration', 0)
        video_save_folder = kwargs.get('video_save_folder', None)
        start_time_str = kwargs.get('start_time_str', "")
        video_as_frames = kwargs.get('video_as_frames', False)
        memmap_path = kwargs.get('memmap_path', None)
        video_false_color = kwargs.get('video_false_color', None)

        try:
            # Defensive check
            if video_frames is None:
                logger.error("Manual-Video] recording_complete called with no video frames")
                return memmap_path

            # Prevent division by zero
            if video_duration <= 0:
                video_duration = 0.1
                logger.warning("Manual-Video] Video duration was 0, using 0.1s")

            if captured_frames == 0:
                logger.error("Manual-Video] No frames captured, aborting video write")
                return memmap_path

            calculated_fps = captured_frames // video_duration

            logger.info(f"Manual-Video] Images present in video array: {len(video_frames) > 0 if video_frames is not None else 0}")
            logger.info(f"Manual-Video] Captured Frames: {captured_frames}")
            logger.info(f"Manual-Video] Video FPS: {calculated_fps}")
            logger.info("Manual-Video] Writing video...")

            color, active_layer_config = get_active_layer_config()

            include_hyperstack_generation = False

            if video_as_frames:

                image_capture_config = get_image_capture_config_from_ui()

                if image_capture_config['output_format']['sequenced'] == 'ImageJ Hyperstack':
                    include_hyperstack_generation = True
                    _, objective = get_current_objective_info()
                    stack_builder = StackBuilder(
                        has_turret=lumaview.scope.has_turret(),
                    )
                    frame_metadata = []

                save_folder = video_save_folder

                if not save_folder.exists():
                    save_folder.mkdir(exist_ok=True, parents=True)

                for frame_num in range(captured_frames):

                    image = video_frames[frame_num]
                    ts = timestamps[frame_num] if frame_num < len(timestamps) else datetime.datetime.now()
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    image = image_utils.add_timestamp(image=image, timestamp_str=ts_str)

                    frame_name = f"ManualVideo_Frame_{frame_num:04}"

                    output_file_loc = save_folder / f"{frame_name}.tiff"

                    metadata = {
                                "datetime": ts.strftime("%Y:%m:%d %H:%M:%S"),
                                "timestamp": ts.strftime("%Y:%m:%d %H:%M:%S.%f"),
                                "frame_num": frame_num
                            }

                    if include_hyperstack_generation:
                        current_position = lumaview.scope.get_current_position()
                        frame_metadata.append(
                            {
                                'Filepath': output_file_loc.name,
                                'Scan Count': frame_num,
                                'Color': color,
                                'Z-Slice': 0,
                                'X': current_position['X'],
                                'Y': current_position['Y'],
                                'Z': current_position['Z'],
                            }
                        )

                    try:
                        image_utils.write_tiff(
                            data=image,
                            metadata=metadata,
                            file_loc=output_file_loc,
                            video_frame=True,
                            ome=False,
                            color=color
                        )
                    except Exception as e:
                        logger.exception(f"Protocol-Video] Failed to write frame {frame_num}: {e}")

                    # Update progress after writing each frame
                    self.video_writing_progress = frame_num + 1

                logger.info("Manual-Video] Video frames written to disk.")


                if include_hyperstack_generation:
                    logger.info("Manual-Video] Creating hyperstack...")

                    _, objective = get_current_objective_info()
                    frame_metadata_df = pd.DataFrame(frame_metadata)
                    stack_builder.create_single_recording_stack(
                        df=frame_metadata_df,
                        path=save_folder,
                        output_file_loc=save_folder / f"ManualVideo_Frame_HyperStack.ome.tiff",
                        focal_length=objective['focal_length'],
                        binning_size=get_binning_from_ui(),
                    )

                    logger.info(f"Manual-Video] Hyperstack created at {save_folder / f'ManualVideo_Frame_HyperStack.ome.tiff'}")

            else:
                if not video_save_folder.exists():
                    video_save_folder.mkdir(exist_ok=True, parents=True)

                output_file_loc = video_save_folder / f"Video_{start_time_str}.mp4v"

                video_writer = VideoWriter(
                    output_file_loc=output_file_loc,
                    fps=calculated_fps,
                    include_timestamp_overlay=True
                )

                for frame_num in range(captured_frames):
                    try:
                        ts = timestamps[frame_num] if frame_num < len(timestamps) else datetime.datetime.now()
                        video_writer.add_frame(image=video_frames[frame_num], timestamp=ts)
                    except Exception:
                        logger.exception("Manual-Video] FAILED TO WRITE FRAME")

                    # Update progress after adding each frame
                    self.video_writing_progress = frame_num + 1

                video_writer.finish()
                logger.info(f"Manual-Video] Mp4 written to {output_file_loc}")

            logger.info("Manual-Video] Video writing finished.")

        finally:
            # Cleanup memmap - must explicitly close the underlying mmap object
            # This MUST run even if we return early (e.g., no frames captured)
            if video_frames is not None:
                try:
                    # Explicitly close the memory-mapped file
                    # Note: No need to flush() before close - close() handles any pending writes
                    if hasattr(video_frames, '_mmap') and video_frames._mmap is not None:
                        video_frames._mmap.close()
                    del video_frames  # Delete the reference
                except Exception as e:
                    logger.warning(f'[LVP Main  ] Error closing memmap: {e}')

            # NOTE: We intentionally do NOT delete the memmap file here because:
            # 1. Windows file deletion can block for several seconds even after closing
            # 2. This causes "Not Responding" freezes in the application
            # 3. The file will be automatically reused on the next recording (see record_init)
            # 4. Reusing the file is actually faster than creating a new one
            logger.info('[LVP Main  ] Memmap file closed and ready for reuse')

        # Return memmap_path so cleanup callback knows which path to remove from tracking
        return memmap_path

    def _recording_cleanup_gui(self, memmap_path=None):
        """Final cleanup on GUI thread after video writing completes."""
        try:
            # Unschedule progress updates
            if hasattr(self, 'writing_progress_update') and self.writing_progress_update:
                Clock.unschedule(self.writing_progress_update)

            # Unschedule recording complete event if it exists
            if hasattr(self, 'recording_complete_event') and self.recording_complete_event:
                Clock.unschedule(self.recording_complete_event)

            # Set last save folder
            if hasattr(self, 'video_save_folder'):
                set_last_save_folder(self.video_save_folder)

            # Clear video writing state - new recordings can now start
            self.video_writing.clear()

            # Reset window title
            from kivy.core.window import Window
            Window.set_title(f"Lumaview Pro {version}")

            logger.info("Manual-Video] Recording cleanup complete")
        except Exception as e:
            logger.exception(f"Manual-Video] Error during GUI cleanup: {e}")

    def record_helper(self, dt=None):

        if not settings['use_full_pixel_depth'] or not settings['video_as_frames']:
            force_to_8bit = True
        else:
            force_to_8bit = False

        image = self.scope.get_image(force_to_8bit=force_to_8bit)

        if isinstance(image, np.ndarray):

            if image.dtype == np.uint16:
                image = image_utils.convert_12bit_to_16bit(image)

            # Note: Currently, if image is 12/16-bit, then we ignore false coloring for video captures.
            if (image.dtype != np.uint16) and (self.video_false_color is not None):
                image = image_utils.add_false_color(array=image, color=self.video_false_color)

            image = np.flip(image, 0)

            self.current_video_frames[self.current_captured_frames] = image
            self.timestamps.append(datetime.datetime.now())

            # self.current_video_frames.append((image, datetime.datetime.now()))

            self.current_captured_frames += 1


    def fit_image(self):
        logger.info('[LVP Main  ] MainDisplay.fit_image()')
        if self.scope.camera.active is None:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        logger.info('[LVP Main  ] MainDisplay.one2one_image()')
        if self.scope.camera.active is None:
            return
        w = self.width
        h = self.height
        scale_hor = float(lumaview.scope.get_width()) / float(w)
        scale_ver = float(lumaview.scope.get_height()) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (int((w-scale*w)/2),int((h-scale*h)/2))

from ui.shader import ShaderViewer, ShaderEditor  # noqa: E402 — extracted widgets


from ui.image_settings import (  # noqa: E402 — extracted widgets
    AccordionItemXyStageControl, AccordionItemImageSettingsBase,
    AccordionItemImageSettingsLumiControl, AccordionItemImageSettingsDfControl,
    AccordionItemImageSettingsRedControl, AccordionItemImageSettingsGreenControl,
    AccordionItemImageSettingsBlueControl, ImageSettings, set_histogram_layer,
)

from ui.motion_settings import MotionSettings, XYStageControl  # noqa: E402 — extracted widgets

from ui.post_processing import (  # noqa: E402 — extracted widgets
    StitchControls, ZProjectionControls, CompositeGenControls,
    VideoCreationControls, GraphingControls, CellCountControls,
    PostProcessingAccordion, CellCountDisplay,
    open_last_save_folder,
)

from ui.histogram import Histogram  # noqa: E402 — extracted widget

from ui.vertical_control import VerticalControl  # noqa: E402 — extracted widget


from ui.protocol_settings import ProtocolSettings  # noqa: E402 — extracted widget

from ui.stage import Stage  # noqa: E402 — extracted widget


from ui.microscope_settings import MicroscopeSettings  # noqa: E402 — extracted widget

# ============================================================================
# ModSlider — Custom Slider with on_release Event
# ============================================================================

from ui.mod_slider import ModSlider  # noqa: E402 — extracted widget

# ============================================================================
# LayerControl — Per-Channel LED, Exposure, Gain, and Stimulation Controls
# ============================================================================

class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    illumination_support = BooleanProperty(True)
    stimulation_support = BooleanProperty(False)
    show_stim_controls = BooleanProperty(False)
    autogain_support = BooleanProperty(True)
    exposure_summing_support = BooleanProperty(False)
    show_camera_controls = BooleanProperty(True)
    show_cbt = BooleanProperty(True)

    global settings



    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)

        logger.debug('[LVP Main  ] LayerControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

        # Flag to prevent apply_settings during initialization
        self._initializing = True

        self.apply_gain_slider = Clock.create_trigger(lambda dt: self.apply_settings(), 0.1)
        self.apply_exp_slider = Clock.create_trigger(lambda dt: self.apply_settings(), 0.1)
        self.apply_ill_slider = Clock.create_trigger(lambda dt: self.apply_settings(), 0.1)
        Clock.schedule_once(self._init_ui, 0)


    def _init_ui(self, dt=0):

        if self.layer in ['Red', 'Green', 'Blue'] and settings['stimulation_enabled']:
            self.stimulation_support = True
            self.show_stim_controls = True
        else:
            self.stimulation_support = False
            self.show_stim_controls = False

        self.update_stim_controls_visibility()

        # Don't apply settings during initial UI setup - will be done after load_settings
        # Skip initialization of autogain and apply_settings here


        self.init_acquire()
        self.init_autofocus()


    def cleanup_scrollviews(self):
        """
        Clean up ScrollView viewport resources in this LayerControl.
        Called when accordion is collapsed to prevent memory accumulation.
        """
        for child in self.walk():
            if isinstance(child, ScrollView):
                cleanup_scrollview_viewport(child)

    def update_stim_controls_visibility(self):
        if self.ids['stim_enable_btn'].active:
            self.show_stim_controls = True
            self.show_camera_controls = False
            self.hide_camera_controls()
        else:
            self.show_stim_controls = False
            self.show_camera_controls = True

    def hide_camera_controls(self):
        self.show_camera_controls = False
        settings[self.layer]['acquire'] = None
        self.ids['acquire_none'].active = True

    def ill_slider(self):
        if protocol_running_global.is_set():
            return
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.ill_slider()')
        illumination = round(self.ids['ill_slider'].value)  # Round to integer (step=1)
        settings[self.layer]['ill'] = illumination

        if 'stim_config' in settings[self.layer]:
            settings[self.layer]['stim_config']['illumination'] = illumination

        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(illumination)
        if self.ids['ill_text'].text != new_text:
            self.ids['ill_text'].text = new_text
        if not self._initializing:
            self.apply_ill_slider()


    def ill_text(self):
        logger.info('[LVP Main  ] LayerControl.ill_text()')
        ill_min = self.ids['ill_slider'].min
        if self.layer == "BF":
            ill_max = 500
        else:
            ill_max = self.ids['ill_slider'].max
        try:
            ill_val = float(self.ids['ill_text'].text)
        except Exception:
            return

        illumination = float(np.clip(ill_val, ill_min, ill_max))

        settings[self.layer]['ill'] = illumination
        self.ids['ill_slider'].value = float(np.clip(illumination, ill_min, self.ids['ill_slider'].max))
        self.ids['ill_text'].text = str(illumination)

        if 'stim_config' in settings[self.layer]:
            settings[self.layer]['stim_config']['illumination'] = illumination

        self.apply_settings()


    def sum_slider(self):
        logger.info('[LVP Main  ] LayerControl.sum_slider()')
        sum = int(self.ids['sum_slider'].value)
        settings[self.layer]['sum'] = sum
        self.apply_settings()


    def sum_text(self):
        logger.info('[LVP Main  ] LayerControl.sum_text()')
        sum_min = self.ids['sum_slider'].min
        sum_max = self.ids['sum_slider'].max
        try:
            sum_val = int(self.ids['sum_text'].text)
        except Exception:
            return

        sum = int(np.clip(sum_val, sum_min, sum_max))

        settings[self.layer]['sum'] = sum
        self.ids['sum_slider'].value = sum
        self.ids['sum_text'].text = str(sum)

        self.apply_settings()


    def video_duration_slider(self):
        logger.info('[LVP Main  ] LayerControl.video_duration_slider()')
        duration = self.ids['video_duration_slider'].value
        settings[self.layer]['video_config']['duration'] = duration
        self.apply_settings()

    def video_duration_text(self):
        logger.info('[LVP Main  ] LayerControl.video_duration_text()')
        duration_min = self.ids['video_duration_slider'].min
        duration_max = self.ids['video_duration_slider'].max
        try:
            duration_val = int(self.ids['video_duration_text'].text)
        except Exception:
            return

        duration = int(np.clip(duration_val, duration_min, duration_max))

        settings[self.layer]['video_config']['duration'] = duration
        self.ids['video_duration_slider'].value = duration
        self.ids['video_duration_text'].text = str(duration)

        self.apply_settings()

    def update_auto_gain(self, init: bool = False):
        logger.info('[LVP Main  ] LayerControl.update_auto_gain()')
        if self.ids['auto_gain'].state == 'down':
            state = True
        else:
            state = False

        for item in ('gain_slider', 'gain_text', 'exp_slider', 'exp_text'):
            self.ids[item].disabled = state

        # When transitioning out of auto-gain, keep last auto-gain settings to apply
        camera_executor.put(IOTask(
            action = LayerControl.get_gain_exposure,
            args=(self, init, state),
            callback=LayerControl.update_auto_gain_cb,
            cb_args=(self),
            pass_result=True
        ))

        # actual_gain = lumaview.scope.camera.get_gain()
        # actual_exp = lumaview.scope.camera.get_exposure_t()


    def get_gain_exposure(self, init, state):
        actual_gain = lumaview.scope.camera.get_gain()
        actual_exp = lumaview.scope.camera.get_exposure_t()

        return (init, state, actual_gain, actual_exp)

    def update_auto_gain_cb(self, result=None, exception=None):
        try:

            if exception is not None:
                logger.error(f"LVP Main] Update_auto_gain error: {exception}")
                return

            init = result[0]
            state = result[1]
            gain = result[2]
            exp = result[3]

            if self.ids['auto_gain'].state == 'down':
                state = True
            else:
                state = False

            # If being called on program initialization, we don't want to
            # inadvertantly load the settings from the scope hardware into the software maintained settings
            # print("AUTOGAIN")
            # print(f"init: {init}    state: {state}")
            # print(f"Gain: {gain}    Exp: {exp}")

            if (not init) and (not state):
                settings[self.layer]['gain'] = gain
                settings[self.layer]['exp'] = exp

            settings[self.layer]['auto_gain'] = state
            self.apply_settings()

        except Exception as e:
            logger.error(f"LVP Main] Update_auto_gain error: {e}")
            return

    def gain_slider(self):
        if protocol_running_global.is_set():
            return
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.gain_slider()')
        gain = round(self.ids['gain_slider'].value, 1)  # Round to 1 decimal (step=0.1)
        settings[self.layer]['gain'] = gain
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(gain)
        if self.ids['gain_text'].text != new_text:
            self.ids['gain_text'].text = new_text
        if not self.ids['gain_slider'].disabled and not self._initializing:
            self.apply_gain_slider()
        ####

    def gain_text(self):
        logger.info('[LVP Main  ] LayerControl.gain_text()')
        gain_min = self.ids['gain_slider'].min
        gain_max = self.ids['gain_slider'].max
        try:
            gain_val = float(self.ids['gain_text'].text)
        except Exception:
            return

        gain = float(np.clip(gain_val, gain_min, gain_max))

        settings[self.layer]['gain'] = gain
        self.ids['gain_slider'].value = gain
        self.ids['gain_text'].text = str(gain)

        self.apply_gain_slider()

    def composite_threshold_slider(self):
        logger.info('[LVP Main  ] LayerControl.composite_threshold_slider()')
        composite_threshold = self.ids['composite_threshold_slider'].value
        settings[self.layer]['composite_brightness_threshold'] = composite_threshold

    def composite_threshold_text(self):
        logger.info('[LVP Main  ] LayerControl.composite_threshold_text()')
        composite_threshold_min = self.ids['composite_threshold_slider'].min
        composite_threshold_max = self.ids['composite_threshold_slider'].max
        try:
            composite_threshold_val = float(self.ids['composite_threshold_text'].text)
        except Exception:
            return

        composite_threshold = float(np.clip(composite_threshold_val, composite_threshold_min, composite_threshold_max))

        settings[self.layer]['composite_brightness_threshold'] = composite_threshold
        self.ids['composite_threshold_slider'].value = composite_threshold
        self.ids['composite_threshold_text'].text = str(composite_threshold)

    def exp_slider(self):
        if protocol_running_global.is_set():
            return
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.exp_slider()')
        exposure = round(self.ids['exp_slider'].value, 2)  # Round to 2 decimals (step=0.01)
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp'] = exposure        # exposure in ms
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(exposure)
        if self.ids['exp_text'].text != new_text:
            self.ids['exp_text'].text = new_text
        if not self.ids['exp_slider'].disabled and not self._initializing:
            self.apply_exp_slider()

    def exp_text(self):
        logger.info('[LVP Main  ] LayerControl.exp_text()')
        exp_min = self.ids['exp_slider'].min
        #exp_max = self.ids['exp_slider'].max
        if self.layer == "BF":
            exp_max = 1000
        else:
            exp_max = self.ids['exp_slider'].max

        try:
            exp_val = float(self.ids['exp_text'].text)
        except Exception:
            return

        exposure = float(np.clip(exp_val, exp_min, exp_max))

        settings[self.layer]['exp'] = exposure
        self.ids['exp_slider'].value = float(np.clip(exposure, exp_min, self.ids['exp_slider'].max))
        # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
        self.ids['exp_text'].text = str(exposure)

        self.apply_exp_slider()

    def stim_freq_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_freq_slider()')
        frequency = self.ids['stim_freq_slider'].value
        try:
            settings[self.layer]['stim_config']['frequency'] = frequency
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_freq_slider() -> {e}")
        self.apply_settings()

    def stim_pulse_count_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_slider()')
        pulse_count = self.ids['stim_pulse_count_slider'].value
        try:
            settings[self.layer]['stim_config']['pulse_count'] = pulse_count
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_count_slider() -> {e}")
        self.apply_settings()

    def stim_pulse_width_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_slider()')
        pulse_width = self.ids['stim_pulse_width_slider'].value
        try:
            settings[self.layer]['stim_config']['pulse_width'] = pulse_width
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_width_slider() -> {e}")
        self.apply_settings()

    def stim_freq_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_freq_text()')

        freq_min = self.ids['stim_freq_slider'].min
        freq_max = self.ids['stim_freq_slider'].max

        try:
            frequency = float(self.ids['stim_freq_text'].text)
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_freq_text() -> {e}")
            return

        frequency = round(float(np.clip(frequency, freq_min, freq_max)), 2)

        self.ids['stim_freq_slider'].value = frequency
        self.ids['stim_freq_text'].text = str(frequency)
        try:
            settings[self.layer]['stim_config']['frequency'] = frequency
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_freq_text() -> {e}")
        self.apply_settings()

    def stim_pulse_count_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_text()')

        pulse_count_min = self.ids['stim_pulse_count_slider'].min
        pulse_count_max = self.ids['stim_pulse_count_slider'].max

        try:
            pulse_count = float(self.ids['stim_pulse_count_text'].text)
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_count_text() -> {e}")
            return

        pulse_count = int(np.clip(pulse_count, pulse_count_min, pulse_count_max))

        self.ids['stim_pulse_count_slider'].value = pulse_count
        self.ids['stim_pulse_count_text'].text = str(pulse_count)
        try:
            settings[self.layer]['stim_config']['pulse_count'] = pulse_count
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_count_text() -> {e}")
        self.apply_settings()

    def stim_pulse_width_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_text()')

        pulse_width_min = self.ids['stim_pulse_width_slider'].min
        pulse_width_max = self.ids['stim_pulse_width_slider'].max

        try:
            pulse_width = float(self.ids['stim_pulse_width_text'].text)
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_width_text() -> {e}")
            return

        pulse_width = int(np.clip(pulse_width, pulse_width_min, pulse_width_max))

        self.ids['stim_pulse_width_slider'].value = pulse_width
        self.ids['stim_pulse_width_text'].text = str(pulse_width)

        try:
            settings[self.layer]['stim_config']['pulse_width'] = pulse_width
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_width_text() -> {e}")
        self.apply_settings()

    def false_color(self):
        logger.info('[LVP Main  ] LayerControl.false_color()')
        settings[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def init_acquire(self):
        if settings[self.layer]['acquire'] == "image":
            self.ids['acquire_image'].state = 'down'
        elif settings[self.layer]['acquire'] == 'video':
            self.ids['acquire_video'].state = 'down'
        else:
            self.ids['acquire_none'].state = 'down'

    def update_acquire(self):
        logger.info('[LVP Main  ] LayerControl.update_acquire()')

        if self.ids['acquire_image'].active:
            settings[self.layer]['acquire'] = "image"
            if "stim_config" in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False

        elif self.ids['acquire_video'].active:
            settings[self.layer]['acquire'] = "video"
            if "stim_config" in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
                self.ids['stim_disable_btn'].active = True
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False
        else:
            settings[self.layer]['acquire'] = None

        if "stim_config" in settings[self.layer]:
            self.update_stim_controls_visibility()

    def update_stim_enable(self):
        logger.info('[LVP Main  ] LayerControl.update_stim_enable()')
        if self.ids['stim_enable_btn'].active:
            if "stim_config" in settings[self.layer]:
                if settings[self.layer]['stim_config'] is not None:
                    settings[self.layer]['stim_config']['enabled'] = True
            settings[self.layer]['acquire'] = None
            self.ids['acquire_none'].active = True
            self.ids['acquire_none'].state = 'down'
        else:
            if "stim_config" in settings[self.layer]:
                if settings[self.layer]['stim_config'] is not None:
                    settings[self.layer]['stim_config']['enabled'] = False

        self.update_stim_controls_visibility()

    def init_autofocus(self):
        if not settings[self.layer]['autofocus']:
            self.ids['autofocus'].state = 'normal'
        else:
            self.ids['autofocus'].state = 'down'

    def update_autofocus(self):
        logger.info('[LVP Main  ] LayerControl.update_autofocus()')
        settings[self.layer]['autofocus'] = self.ids['autofocus'].active

    def save_focus(self):
        logger.info('[LVP Main  ] LayerControl.save_focus()')
        io_executor.put(IOTask(
            action=self.execute_save_focus
        ))

    def execute_save_focus(self):
        pos = lumaview.scope.get_current_position('Z')
        settings[self.layer]['focus'] = pos


    def goto_focus(self):
        logger.info('[LVP Main  ] LayerControl.goto_focus()')
        io_executor.put(IOTask(
            action=self.execute_goto_focus,
        ))

    def execute_goto_focus(self):
        pos = settings[self.layer]['focus']
        move_absolute_position('Z', pos)  # set current z height in usteps

    def update_led_state(self, apply_settings=True):
        enabled = True if self.ids['enable_led_btn'].state == 'down' else False
        illumination = settings[self.layer]['ill']

        if apply_settings:
            self.apply_settings(update_led=False)

        camera_executor.put(IOTask(
            action=self.set_led_state,
            kwargs= {
                "enabled": enabled,
                "illumination": illumination
            }
        ))
        #self.set_led_state(enabled=enabled, illumination=illumination)

        # self.apply_settings()


    def set_led_state(self, enabled: bool, illumination: float):
        channel = lumaview.scope.color2ch(self.layer)
        if not enabled:
            scope_commands.led_off(lumaview.scope, io_executor, channel)
        else:
            logger.info(f'[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch({self.layer}), {illumination})')
            scope_commands.led_on(lumaview.scope, io_executor, channel, illumination)

    def update_led_toggle_ui(self):
        if lumaview.scope.led:
            led_state = lumaview.scope.get_led_state(color=self.layer)
            if led_state['enabled']:
                self.ids['enable_led_btn'].state = 'down'
            else:
                self.ids['enable_led_btn'].state = 'normal'


    def apply_settings(self, ignore_auto_gain=False, update_led=True, protocol=False):

        # Skip apply_settings if layer is still initializing
        if getattr(self, '_initializing', False):
            return

        logger.info(f'[LVP Main  ] {self.layer}_LayerControl.apply_settings()')
        global lumaview

        def update_shader(dt=None):
            if not ctx.scope_display.paused.is_set():
                if ctx.scope_display.use_bullseye is False:
                    self.update_shader(dt=0)

        def disable_leds_for_other_layers(dt=None):
            if self.ids['enable_led_btn'].state == 'down': # if the button is down
                for layer in common_utils.get_layers():
                    if layer != self.layer:
                        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                        layer_obj.ids['enable_led_btn'].state = 'normal'



        if protocol or protocol_running_global.is_set():
            Clock.schedule_once(disable_leds_for_other_layers, 0)
            Clock.schedule_once(update_shader, 0)
            return

        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        if not protocol:
            set_histogram_layer(active_layer=self.layer)


        # Queue IO task and update UI after completing IO
        if update_led:
            if not protocol_running_global.is_set():
                self.update_led_state(apply_settings=False)





        disable_leds_for_other_layers()


        # update exposure to currently selected settings
        # -----------------------------------------------------

        exposure = settings[self.layer]['exp']
        gain = settings[self.layer]['gain']

        if not protocol_running_global.is_set():
            camera_executor.put(IOTask(action=lumaview.scope.set_gain, args=(gain)))
            camera_executor.put(IOTask(action=lumaview.scope.set_exposure_time, args=(exposure)))
        #lumaview.scope.set_gain(gain)
        #lumaview.scope.set_exposure_time(exposure)

        # update gain to currently selected settings
        # -----------------------------------------------------
        auto_gain_enabled = settings[self.layer]['auto_gain']

        if not ignore_auto_gain:
            if not protocol_running_global.is_set():
                autogain_settings = get_auto_gain_settings()
                camera_executor.put(IOTask(
                    action=lumaview.scope.set_auto_gain,
                    args=(auto_gain_enabled),
                    kwargs={
                        "settings": autogain_settings
                    }
                )
                )
                #lumaview.scope.set_auto_gain(auto_gain_enabled, settings=autogain_settings)

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        update_shader()




    def update_shader(self, dt):
        # logger.info('[LVP Main  ] LayerControl.update_shader()')
        if self.ids['false_color'].active:
            ctx.viewer.update_shader(self.layer)
        else:
            ctx.viewer.update_shader('none')

# reset_acquire_ui, reset_stim_ui → modules/ui_helpers.py

# ============================================================================
# ZStack — Z-Stack Acquisition Controls
# ============================================================================

from ui.zstack import ZStack  # noqa: E402 — extracted widget


# ============================================================================
# File / Folder Chooser Buttons
# ============================================================================

from ui.file_dialogs import FileChooseBTN, FolderChooseBTN, FileSaveBTN  # noqa: E402 — extracted widgets


# ============================================================================
# Application Initialization Helpers
# ============================================================================

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
            except Exception:
                pass


def get_lvp_lock_port() -> int:
    DEFAULT_LVP_LOCK_PORT = 43101
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                return data['lvp_lock_port']
            except Exception:
                pass
        
    return DEFAULT_LVP_LOCK_PORT


def load_autofocus_log_enable():
    global autofocus_executor
    for settings_file in ("./data/current.json", "./data/settings.json"):
        if not os.path.exists(settings_file):
            continue

        with open(settings_file, 'r') as fp:
            data = json.load(fp)

            try:
                if data['logging']['autofocus']:
                    autofocus_functions.enable_af_score_logging(enable=True)
                return
            except Exception:
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
            except Exception:
                pass

        ENGINEERING_MODE = False


def block_wait_for_threads(futures: list, log_loc="LVP") -> None:
    config_helpers.block_wait_for_threads(futures, log_loc)

def init_ij():
    import imagej.doctor
    import imagej
    import scyjava

    imagej.doctor.checkup()
    global ij_helper
    ij_helper = imagej_helper.ImageJHelper()
    return

def process_ij_helper():
    return imagej_helper.ImageJHelper()

# ============================================================================
# LumaViewProApp — Main Application Class (Build, Start, Stop, Tooltips)
# ============================================================================

class LumaViewProApp(App):
    kv_file = 'ui/lumaviewpro.kv'

    def on_start(self):
        global _app_initializing

        # Continuously update image of stage and protocol
        Clock.schedule_interval(stage.draw_labware, 0.1)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1) # Includes text boxes, not just stage
        Clock.schedule_once(functools.partial(ctx.image_settings.set_expanded_layer, 'BF'), 0.2)

        # Clear app initialization flag and apply settings for the default opened layer
        def complete_initialization(dt):
            global _app_initializing
            _app_initializing = False
            if ctx is not None:
                ctx.ready = True

            # Check if a protocol is loaded and has steps
            protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
            if hasattr(protocol_settings, '_protocol') and protocol_settings._protocol is not None:
                if protocol_settings._protocol.num_steps() > 0:
                    # Go to the first step of the protocol
                    protocol_settings.go_to_step(protocol=False)
                    return

            # If no protocol, just apply settings for the default BF layer
            ctx.image_settings.accordion_collapse()

        Clock.schedule_once(complete_initialization, 0.3)

        # Executor health watchdog: logs queue depths periodically and prunes stale display backlog
        def _executor_watchdog(dt):
            try:
                io_q = io_executor.queue_size() if hasattr(io_executor, 'queue_size') else -1
                cam_q = camera_executor.queue_size() if hasattr(camera_executor, 'queue_size') else -1
                prot_q = protocol_executor.queue_size() if hasattr(protocol_executor, 'protocol_queue_size') else -1
                file_q = file_io_executor.queue_size() if hasattr(file_io_executor, 'protocol_queue_size') else -1
                af_q = autofocus_thread_executor.queue_size() if hasattr(autofocus_thread_executor, 'protocol_queue_size') else -1
                sd_q = scope_display_thread_executor.queue_size() if hasattr(scope_display_thread_executor, 'queue_size') else -1
                stage_q = 0   # consolidated into io_executor
                turret_q = 0  # consolidated into io_executor
                reset_q = reset_executor.queue_size() if hasattr(reset_executor, 'queue_size') else -1
                io_pq = io_executor.protocol_queue_size() if hasattr(io_executor, 'protocol_queue_size') else -1
                cam_pq = camera_executor.protocol_queue_size() if hasattr(camera_executor, 'protocol_queue_size') else -1
                prot_pq = protocol_executor.protocol_queue_size() if hasattr(protocol_executor, 'protocol_queue_size') else -1
                file_pq = file_io_executor.protocol_queue_size() if hasattr(file_io_executor, 'protocol_queue_size') else -1
                af_pq = autofocus_thread_executor.protocol_queue_size() if hasattr(autofocus_thread_executor, 'protocol_queue_size') else -1
                sd_pq = scope_display_thread_executor.protocol_queue_size() if hasattr(scope_display_thread_executor, 'protocol_queue_size') else -1
                stage_pq = 0  # consolidated into io_executor
                turret_pq = 0 # consolidated into io_executor
                reset_pq = reset_executor.protocol_queue_size() if hasattr(reset_executor, 'protocol_queue_size') else -1

                logger.error(f"[Watchdog] Queues - IO:{io_q} CAM:{cam_q} PROT:{prot_q} FILE:{file_q} AF:{af_q} SD:{sd_q} STAGE:{stage_q} TURRET:{turret_q} RESET:{reset_q}")
                logger.error(f"[Watchdog] Protocol Queues - IO:{io_pq} CAM:{cam_pq} PROT:{prot_pq} FILE:{file_pq} AF:{af_pq} SD:{sd_pq} STAGE:{stage_pq} TURRET:{turret_pq} RESET:{reset_pq}")

                # If scopedisplay backlog is growing and appears stale, prune it to keep UI responsive
                if sd_q is not None and sd_q > 20:
                    try:
                        scope_display_thread_executor.clear_pending()
                        logger.warning("[Watchdog] Cleared ScopeDisplay pending queue to prevent backlog")
                    except Exception:
                        pass
            except Exception:
                pass

        #Clock.schedule_interval(_executor_watchdog, 60)

        os.chdir(source_path)


        load_log_level()
        load_autofocus_log_enable()
        # load_mode() and engineering_mode assignment moved to build() for correct _init_ui timing
        logger.info('[LVP Main  ] LumaViewProApp.on_start()')

        if lumaview.scope.no_hardware:
            Clock.schedule_once(lambda dt: show_notification_popup(
                title="Hardware Connection Failed",
                message="No hardware detected. Please connect LED board, motor board, and camera."
            ), 0)

        if not disable_homing:
            # Note: If the scope has a turret, this also performs a T homing
            task = IOTask(
                move_home,
                args=('XY')
            )
            stage_executor.put(task)


        if lumaview.scope.has_turret():
            objective_id = settings['objective_id']
            turret_position = lumaview.scope.get_turret_position_for_objective_id(objective_id=objective_id)

            if turret_position is None:
                DEFAULT_POSITION = 1
                logger.info(f"Turret position for set objective {objective_id} not in turret objectives configuration. Setting to position {DEFAULT_POSITION}")
                turret_position = DEFAULT_POSITION

            turret_executor.put(IOTask(
                move_absolute_position,
                kwargs= {
                    "axis": 'T',
                    "pos": turret_position,
                    "wait_until_complete": True
                } 
            ))
            #thread_pool.submit(move_absolute_position, axis='T', pos=turret_position, wait_until_complete=True)
            #move_absolute_position(axis='T', pos=turret_position, wait_until_complete=True)

        layer_obj = ctx.image_settings.layer_lookup(layer='BF')
        layer_obj.apply_settings()

        log_system_metrics() # Log once on startup

        Clock.schedule_interval(functools.partial(log_system_metrics), 14400)   # Log metrics every 4 hours

        if lumaview.scope.camera_is_connected():
            lumaview.log_camera_temps()  # Log once on startup
            lumaview.camera_temps_event = Clock.schedule_interval(lambda dt: lumaview.log_camera_temps(), 14400)  # Log every 4 hours

        scope_leds_off()

        if getattr(sys, 'frozen', False):
            pyi_splash.close()

    def shutdown_threads(self):
        logger.info('[LVP Main  ] Shutting down threads...')

        if profiling_helper is not None:
            profiling_helper.stop()

        if io_executor is not None:
            io_executor.shutdown(wait=False)

        if camera_executor is not None:
            camera_executor.shutdown(wait=False)

        if temp_ij_executor is not None:
            temp_ij_executor.shutdown(wait=False)

        if protocol_executor is not None:
            protocol_executor.shutdown(wait=False)

        if file_io_executor is not None:
            file_io_executor.shutdown(wait=False)

        if autofocus_thread_executor is not None:
            autofocus_thread_executor.shutdown(wait=False)

        if scope_display_thread_executor is not None:
            scope_display_thread_executor.shutdown(wait=False)

        # stage_executor and turret_executor are aliases for io_executor (already shut down above)

        if cpu_pool is not None:
            cpu_pool.shutdown(wait=True)

        if reset_executor is not None:
            reset_executor.shutdown(wait=False)

        logger.info('[LVP Main  ] Threads shut down.')


    def build(self):
        current_time = time.strftime("%m/%d/%Y", time.localtime())
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        logger.info('[LVP Main  ] -----------------------------------------')
        logger.info('[LVP Main  ] Code Compiled On: %s', current_time)
        logger.info('[LVP Main  ] Run Time: ' + time.strftime("%Y %m %d %H:%M:%S"))
        logger.info('[LVP Main  ] -----------------------------------------')

        self._lvp_lock = lvp_lock.LvpLock(lock_port=get_lvp_lock_port())
        if not self._lvp_lock.lock():
            error_msg = "Another instance of LVP may already be running. Exiting."
            logger.error(f'[LVP Lock ] {error_msg}')
            tkinter.messagebox.showerror(
                "LumaViewPro Error",
                error_msg
            )
            sys.exit(1)
        
        global Window
        global lumaview
        global cell_count_content
        global graphing_controls
        global video_creation_controls
        global stitch_controls
        global zprojection_controls
        global composite_gen_controls
        global stage
        global wellplate_loader
        global coordinate_transformer
        global objective_helper
        global ij_helper
        ij_helper = None
        global sequenced_capture_executor

        global autofocus_executor

        self.icon = './data/icons/icon.png'

        # Load engineering mode early so _init_ui() methods see the correct value
        global ENGINEERING_MODE
        load_mode()

        stage = Stage()


        try:
            from kivy.core.window import Window
            # Window min size uses SDL point coordinates — do NOT use dp()
            Window.minimum_width = 1024
            Window.minimum_height = 600
            Window.bind(on_resize=self._on_resize)
            Window.bind(on_request_close=self.on_request_close)
            lumaview = MainDisplay()
            lumaview.scope.engineering_mode = ENGINEERING_MODE
            cell_count_content = CellCountControls()
            graphing_controls = GraphingControls()
        except Exception:
            logger.exception('[LVP Main  ] Cannot open main display.')
            raise

        # load labware file
        wellplate_loader = labware_loader.WellPlateLoader()
        coordinate_transformer = coord_transformations.CoordinateTransformer()

        objective_helper = objectives_loader.ObjectiveLoader()

        # Create executors (previously at module level, moved here for init consolidation)
        global io_executor, camera_executor, temp_ij_executor, protocol_executor
        global file_io_executor, autofocus_thread_executor, scope_display_thread_executor
        global stage_executor, turret_executor, reset_executor
        io_executor = SequentialIOExecutor(name="IO")
        camera_executor = SequentialIOExecutor(name="CAMERA")
        temp_ij_executor = SequentialIOExecutor(name="IJ")
        protocol_executor = SequentialIOExecutor(name="PROTOCOL")
        file_io_executor = SequentialIOExecutor(name="FILE")
        autofocus_thread_executor = SequentialIOExecutor(name="AUTOFOCUS")
        scope_display_thread_executor = SequentialIOExecutor(name="SCOPEDISPLAY")
        stage_executor = io_executor    # consolidated: all motor serial I/O through one executor
        turret_executor = io_executor   # consolidated: prevents concurrent motor board access
        reset_executor = SequentialIOExecutor(name="RESET")

        # Create the GUI-independent scope session
        global scope_session
        scope_session = ScopeSession(
            settings=settings,
            scope=lumaview.scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
        )
        scope_session.protocol_running = protocol_running_global

        io_executor.start()
        camera_executor.start()
        temp_ij_executor.start()
        protocol_executor.start()
        file_io_executor.start()
        autofocus_thread_executor.start()
        scope_display_thread_executor.start()
        # stage_executor and turret_executor are aliases for io_executor (already started above)
        reset_executor.start()
        #ij_helper = imagej_helper.ImageJHelper()

        # temp_ij_executor.put(IOTask(
        #     action=imagej_helper.ImageJHelper,
        #     callback=ij_creation,
        #     pass_result=True
        # ))

        # load settings file
        ctx.motion_settings.ids['microscope_settings_id'].load_settings("./data/current.json")

        autofocus_executor = AutofocusExecutor(
            scope=lumaview.scope,
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=file_io_executor,
            autofocus_executor=autofocus_thread_executor,
            use_kivy_clock=True,
            ui_update_func=_handle_autofocus_ui
        )

        sequenced_capture_executor = SequencedCaptureExecutor(
            scope=lumaview.scope,
            stage_offset=settings['stage_offset'],
            autofocus_executor=autofocus_executor,
            io_executor=io_executor,
            protocol_executor=protocol_executor,
            file_io_executor=file_io_executor,
            camera_executor=camera_executor,
            autofocus_io_executor=autofocus_thread_executor,
            z_ui_update_func=_handle_autofocus_ui,
            cpu_pool=cpu_pool if use_multiprocessing else None
        )

        # Create AppContext — central service registry
        global ctx
        ctx = AppContext(
            scope=lumaview.scope,
            settings=settings,
            session=scope_session,
            io_executor=io_executor,
            camera_executor=camera_executor,
            protocol_executor=protocol_executor,
            file_io_executor=file_io_executor,
            autofocus_thread_executor=autofocus_thread_executor,
            scope_display_thread_executor=scope_display_thread_executor,
            reset_executor=reset_executor,
            temp_ij_executor=temp_ij_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            protocol_running=protocol_running_global,
            engineering_mode=ENGINEERING_MODE,
        )
        app_context.ctx = ctx  # Publish to module-level singleton for extracted modules

        # Wire UI components now that widget tree exists
        ctx.viewer = lumaview.ids['viewer_id']
        ctx.scope_display = ctx.viewer.ids['scope_display_id']
        ctx.image_settings = lumaview.ids['imagesettings_id']
        ctx.motion_settings = lumaview.ids['motionsettings_id']

        # Creates and manages Tooltips
        self.hidden = True
        self.tooltip_attr_widgets = self.find_widgets_with_tooltips(lumaview)
        self.widget_to_accordion_dict = self.create_widget_to_parent_dict(self.tooltip_attr_widgets)
        self.tt_widget = Tooltip()
        self.widget_being_described = None
        Window.bind(mouse_pos=self.mouse_moved)
        self.tt_shown = False
        self.tt_clock_event = None

        return lumaview

    def _on_resize(self, window, w, h):
        Clock.schedule_once(ctx.motion_settings.check_settings, 0.1)
        Clock.schedule_once(ctx.image_settings.check_settings, 0.1)

    def on_request_close(self, *args):
        """Handle window close request - show confirmation if protocol is running."""

        if protocol_running_global.is_set():
            Clock.schedule_once(lambda dt: (
                show_confirmation_popup(
                title='Confirm Exit',
                message='A protocol is currently running.\n\nAre you sure you want to exit?',
                confirm_text='Confirm Exit',
                cancel_text='Cancel',
                on_confirm=self.stop
                )
            ))

            return True  # Prevent window from closing

        # No protocol running - allow normal close
        return False

    def on_stop(self):
        global lumaview

        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')

        # Unschedule all recurring interval callbacks to prevent orphaned events
        try:
            Clock.unschedule(stage.draw_labware)
            Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
        except Exception:
            pass

        ctx.motion_settings.ids['protocol_settings_id'].cancel_all_protocols()

        self.shutdown_threads()


        logger.info("[LVP Main  ] lumaview.scope.leds_off()")
        lumaview.scope.leds_off()

        ctx.motion_settings.ids['microscope_settings_id'].save_settings("./data/current.json")

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})


    # Returns a list of widgets with tooltip_text attribute
    def find_widgets_with_tooltips(self, widget) -> list:
        widgets = []

        children = widget.children
        if hasattr(widget, 'tooltip_text'):
            if widget.tooltip_text != "":
                widgets.append(widget)
                return widgets
        for child in children:
            widgets += self.find_widgets_with_tooltips(child)
        return widgets

    # Helper function to find a widget's Accordion
    # Returns a list of all parents that are accordions. As list increments, accordions approach head of widget tree
    def find_accordion_parents(self, widget) -> list:
        return_list = []
        if widget.parent is None:
            return return_list
        if isinstance(widget.parent, kivy.uix.accordion.AccordionItem) or isinstance(widget.parent, AccordionItem):
            return return_list + [widget.parent] + self.find_accordion_parents(widget.parent)
        else:
            return self.find_accordion_parents(widget.parent)

    # Creates a dictionary to relate a widget to the Accordion(s) it is in
    def create_widget_to_parent_dict(self, tt_attr_widgets) -> dict:
        dict = {}
        for widget in tt_attr_widgets:
            dict[widget] = self.find_accordion_parents(widget)
        return dict


    # Called every time mouse is moved
    # Used to check if tooltip should be shown
    def mouse_moved(self, *args) -> None:
        delay_until_tooltip = 0.5   # In Seconds

        mouse_pos = args[1]
        self.mouse_pos = mouse_pos
        on_widget = False

        if show_tooltips:
            self.hidden = False

            # Hide tooltip on mouse movement if not colliding anymore (Put here to check asap after a change)
            if self.widget_being_described is not None:
                if not self.tt_collision(self.widget_being_described, mouse_pos[0], mouse_pos[1]):
                    self.hide_tooltip()
                if self.tt_clock_event is not None:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None

            # Checks collision and if tooltip is visible. If it isn't on any tooltip, hide the tooltip
            for widget in self.tooltip_attr_widgets:

                if widget.pos[0] > -100 and widget.pos[0] < Window.width and widget.pos[1] > 0 and widget.pos[1] < Window.height:

                    collision = self.tt_collision(widget, mouse_pos[0], mouse_pos[1])

                    if collision:
                        accordion_parents = self.widget_to_accordion_dict[widget]
                        self.widget_being_described = widget

                        # If widget is not in an Accordion, it is always visible, so show tooltip
                        if len(accordion_parents) < 1:

                            on_widget = True
                            if not self.tt_shown:
                                self.tt_widget.text = widget.tooltip_text
                                self.tt_clock_event = Clock.schedule_once(self.show_tooltip, delay_until_tooltip)
                            break

                        # If all accordions above the widget are not collapsed, show the widget
                        elif True not in [accordion.collapse for accordion in accordion_parents]:
                            on_widget = True
                            if not self.tt_shown:
                                self.tt_widget.text = widget.tooltip_text
                                self.tt_clock_event = Clock.schedule_once(self.show_tooltip, delay_until_tooltip)
                            break
                        else:
                            continue

                    else:
                        on_widget = False
                else:
                    on_widget = False

            if not on_widget:
                if self.tt_clock_event:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None

                self.hide_tooltip()
        else:
            # Hides tooltip one time if tooltips are turned off (else always remains on screen)
            if not self.hidden:
                self.hide_tooltip()
                if self.tt_clock_event is not None:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None
                self.hidden = True


    def tt_collision(self, widget, mouse_x: float, mouse_y: float) -> bool:
        # Shows hitboxes for tooltips.
        # Only seems to work for widgets not in channel control for some reason
        show_hitboxes = False

        true_widget_x = widget.to_window(*widget.pos)[0]
        true_widget_y = widget.to_window(*widget.pos)[1]

        if type(widget) is not Label:
            left = true_widget_x
            right = true_widget_x + widget.width
            bottom = true_widget_y
            top = true_widget_y + widget.height

            if show_hitboxes:
                with widget.canvas.after:
                    Color(1,0,0,1)
                    Line(rectangle=(left, bottom, right-left, top-bottom))

            return left <= mouse_x <= right and bottom <= mouse_y <= top

        else:
            # Widget is a Label
            # Hitbox is only on the text portion of the label, unless wrapping is present

            text_width = widget.texture_size[0]
            text_height = widget.texture_size[1]
            total_width = widget.width
            total_height = widget.height

            if text_width == total_width and text_height == total_height:
                text_width, text_height = self.calculate_label_text_size(widget)

            # Setting text_x and text_y to represent the bottom left corner of the label text

            if widget.halign == "left":
                text_x = true_widget_x
            elif widget.halign == "right":
                text_x = true_widget_x + (total_width - text_width)
            else:
                text_x = ((total_width - text_width) / 2) + true_widget_x

            if widget.valign == "top":
                text_y = true_widget_y + (total_height - text_height)
            else:
                text_y = ((total_height - text_height) / 2) + true_widget_y

            if show_hitboxes:
                with widget.canvas.after:  # Use canvas.after to draw on top of everything else
                # Optional: Set a color for the hitbox
                    Color(1, 0, 0, 1)  # Red color, fully opaque

                    # Draw a rectangle around the widget's bounding box
                    Line(rectangle=(text_x, text_y, text_width, text_height), width=1)

            return text_x <= mouse_x <= (text_x + text_width) and text_y <= mouse_y <= (text_y + text_height)

    # Used to calculate a label's text dimensions when the label size is preset (keeps collision only on the text)
    def calculate_label_text_size(self, widget) -> tuple:
        text = widget.text
        font_size = widget.font_size

        temp_label = Label(text=text, font_size=font_size,)

        temp_label.texture_update()

        text_width, text_height = temp_label.texture_size

        if text_width > widget.size[0]:
            temp_label.text_size[0] = widget.size[0]
            temp_label.texture_update()
            text_width, text_height = temp_label.texture_size

        return text_width, text_height


    def show_tooltip(self, *args) -> None:
        global show_tooltips

        if show_tooltips:
            if self.widget_being_described is not None:
                self.tt_widget._update_rect()
                # Default offsets
                vert_offset = 15
                horiz_offset = 15

                # If mouse is low on the screen
                low_screen_vert_offset = 7

                # If mouse is far right on the screen
                right_screen_horiz_offset = 7

                # If mouse is in lower quarter of screen, show tooltip above mouse instead of below
                if self.mouse_pos[1] < Window.height / 4:
                    lower_half = True
                else:
                    lower_half = False

                if self.mouse_pos[0] > Window.width - Window.width / 4:
                    far_right = True
                else:
                    far_right = False

                if not self.tt_shown:

                    # Remove and add the widget to ensure it shows up at the front of the screen
                    lumaview.remove_widget(self.tt_widget)
                    lumaview.add_widget(self.tt_widget)
                    self.tt_widget.size = Window.size

                    if lower_half:
                        tt_widget_y = self.mouse_pos[1] - self.tt_widget.height + low_screen_vert_offset + (Window.height / 2)
                        tt_widget_rect_y = self.mouse_pos[1] + low_screen_vert_offset/2 + (self.tt_widget.vert_padding / 2) - self.tt_widget.texture_size[1]/2 - self.tt_widget.vert_padding/2 + 1
                    else:
                        # Upper Half
                        tt_widget_y = self.mouse_pos[1] - self.tt_widget.height - vert_offset + (Window.height / 2)
                        tt_widget_rect_y = self.mouse_pos[1] - vert_offset/2 + (self.tt_widget.vert_padding / 2) - self.tt_widget.rect.size[1] - 2*self.tt_widget.vert_padding + self.tt_widget.texture_size[1]/2

                    if far_right:
                        tt_widget_x = self.mouse_pos[0] - right_screen_horiz_offset - (Window.width / 2) - (self.tt_widget.texture_size[0]/2)
                        tt_widget_rect_x = self.mouse_pos[0] - right_screen_horiz_offset - (self.tt_widget.horiz_padding / 2) - (self.tt_widget.texture_size[0])
                    else:
                        # Left Side
                        tt_widget_x = self.mouse_pos[0] + horiz_offset - (Window.width / 2) + (self.tt_widget.texture_size[0]/2)
                        tt_widget_rect_x = self.mouse_pos[0] + horiz_offset - (self.tt_widget.horiz_padding / 2)

                    self.tt_widget.pos = (tt_widget_x, tt_widget_y)
                    self.tt_widget.rect.pos = (tt_widget_rect_x, tt_widget_rect_y)

                    self.tt_widget.opacity = 1
                    self.tt_shown = True

    def hide_tooltip(self, *args) -> None:
        self.widget_being_described = None
        if self.tt_shown:
            self.tt_widget.opacity = 0
            self.tt_shown = False

# ============================================================================
# Tooltip — Hover Tooltip Widget
# ============================================================================

class Tooltip(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.horiz_padding = 4      #4
        self.vert_padding = 4       #4

        self.opacity = 0
        self.font_size = '15sp'
        self.color = [0, 0, 0, 1]  # Black text
        self.bind(size=self._update_rect, pos=self._update_rect)
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White background
            self.rect = Rectangle(size=(self.texture_size[0] + self.horiz_padding, self.texture_size[1] + self.vert_padding))

        self.opacity = 0  # Initially hidden

    def _update_rect(self, *args):
        self.rect.size = (self.texture_size[0] + self.horiz_padding, self.texture_size[1] + self.vert_padding)



if __name__ == "__main__":
    # Fixing Kivy issue that was leading to crazy memory accumulation due to tracebacks being stored in memory
    # For some reason kivy was calling a Python 2 method on newer Python 3 List objects, causing exceptions that would accumulate
    # These exceptions were CONSTANT because they were happening on each Main Display refresh and Histogram refresh
    from kivy.properties import ObservableReferenceList



    def patched_setslice(self, i, j, sequence, **kwargs):
        try:
            # Try the original assignment
            return original_setslicemethod(self, i, j, sequence)
        except AttributeError:
            # Getting attribute error if kivy is calling a deprecated method on a new Python 3 object
            # Call proper method
            return set_item_method(self, slice(i, j), sequence)
        except Exception as e:
            # If for some reason we get another error again, bite the bullet
            return original_setslicemethod(self, i, j, sequence)

# ============================================================================
# Application Entry Point
# ============================================================================

def dummy_function(PID: int):
    for _ in range(100):
        print(f"DUMMY FUNCTION {PID}")
    return

if __name__ == "__main__":

    original_setslicemethod = ObservableReferenceList.__setslice__
    set_item_method = ObservableReferenceList.__setitem__
    # Replace the original method with our patched version
    ObservableReferenceList.__setslice__ = patched_setslice
    LumaViewProApp().run()

#endregion
