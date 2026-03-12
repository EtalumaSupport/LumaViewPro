#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os

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
import numpy as np
import pandas as pd
import time
import json
import sys
import typing
import threading
# import faulthandler

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

    global version, windows_machine, num_cores
    global lumaview, settings, cell_count_content, graphing_controls
    global max_exposure, wellplate_loader, objective_helper
    global coordinate_transformer, sequenced_capture_executor
    global show_tooltips, protocol_running_global, live_histo_setting
    global last_save_folder, stage, ENGINEERING_MODE
    global focus_round
    global io_executor, camera_executor, temp_ij_executor, protocol_executor, file_io_executor, autofocus_thread_executor, stage_executor, turret_executor, reset_executor
    global cpu_pool
    global ij_helper
    global live_view_fps
    global use_multiprocessing

    cpu_pool = None
    use_multiprocessing = False
    live_view_fps = 30
    ij_helper = None

    # Environment setup — paths, version, platform detection
    from modules.app_environment import init_environment
    _env = init_environment(main_file=__file__)
    script_path = _env.script_path
    source_path = _env.source_path
    version = _env.version
    windows_machine = _env.windows_machine
    num_cores = _env.num_cores

    PROTOCOL_DATA_DIR_NAME = "ProtocolData"


    ############################################################################
    #---------------------Module Imports---------------------------------------#
    ############################################################################

    global DEBUG_MODE

    from lvp_logger import logger, debug

    DEBUG_MODE = debug

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
        logger.critical(f"[LVP Main  ] Failed to load settings — cannot continue. {e}")
        sys.exit(1)

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
    import modules.coord_transformations as coord_transformations
    import modules.labware_loader as labware_loader
    import modules.lvp_lock as lvp_lock
    import modules.objectives_loader as objectives_loader
    from modules.protocol import Protocol, ProtocolFormatError
    from modules.sequenced_capture_executor import SequencedCaptureExecutor
    from modules.sequenced_capture_executor import SequencedCaptureRunMode
    from modules.stack_builder import StackBuilder
    from modules.zstack_config import ZStackConfig
    from modules.common_utils import CustomJSONizer
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
    Config.set('graphics', 'minimum_width', '1024')
    Config.set('graphics', 'minimum_height', '600')

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
    # Matplotlib-to-Kivy bridge → ui/figure_canvas.py
    from ui.figure_canvas import FigureCanvasKivyAgg

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
    from ui.rounded_buttons import RoundedButton, RoundedToggleButton

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


    last_save_folder = None
    stage = None

    ENGINEERING_MODE = False

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
    # Subprocess/worker compatibility — Kivy not available
    from modules.subprocess_stubs import (  # noqa: F401
        App, Widget, BoxLayout, FloatLayout, Scatter, Image, Button,
        ToggleButton, Label, RoundedButton, RoundedToggleButton,
        Slider, ScrollView, Popup, AccordionItem,
        RenderContext, Line, Color, Rectangle, Ellipse, Texture,
        StringProperty, ObjectProperty, BooleanProperty, ListProperty,
        MotionEvent, Factory, Clock, dp,
        RangeSlider, FigureCanvasKivyAgg,
        show_popup, show_notification_popup, image_utils_kivy,
    )



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


# Protocol step navigation → modules/step_navigation.py
from modules.step_navigation import go_to_step, go_to_step_update_ui  # noqa: E402




from modules.config_ui_getters import (  # noqa: E402 — extracted functions
    get_binning_from_ui, get_zstack_params, get_zstack_positions,
    get_layer_configs, get_active_layer_config, get_stim_configs,
    get_enabled_stim_configs, get_current_plate_position,
    get_current_frame_dimensions, get_selected_labware,
    get_image_capture_config_from_ui, get_sequenced_capture_config_from_ui,
    get_auto_gain_settings, get_current_objective_info,
    get_protocol_time_params, is_image_saving_enabled,
    create_hyperstacks_if_needed,
)


# ScrollView cleanup → modules/ui_helpers.py
from modules.ui_helpers import cleanup_scrollview_viewport  # noqa: E402

from ui.scope_display import ScopeDisplay  # noqa: E402 — extracted widget

# CompositeCapture — shared capture capabilities → ui/composite_capture.py
from ui.composite_capture import CompositeCapture  # noqa: E402


# MainDisplay — primary application display → ui/main_display.py
from ui.main_display import MainDisplay  # noqa: E402


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

from ui.layer_control import LayerControl  # noqa: E402 — extracted widget

# ============================================================================
# ZStack — Z-Stack Acquisition Controls
# ============================================================================

from ui.zstack import ZStack  # noqa: E402 — extracted widget


# ============================================================================
# File / Folder Chooser Buttons
# ============================================================================

from ui.file_dialogs import FileChooseBTN, FolderChooseBTN, FileSaveBTN  # noqa: E402 — extracted widgets


# Application config loaders → modules/app_config.py
from modules.app_config import (  # noqa: E402
    load_log_level, get_lvp_lock_port, load_autofocus_log_enable,
    load_mode as _load_mode,
)

# ============================================================================
# LumaViewProApp — Main Application Class (Build, Start, Stop, Tooltips)
# ============================================================================

from ui.tooltip import Tooltip, TooltipMixin  # noqa: E402

class LumaViewProApp(TooltipMixin, App):
    kv_file = 'ui/lumaviewpro.kv'

    def on_start(self):
        # Continuously update image of stage and protocol
        Clock.schedule_interval(stage.draw_labware, 0.1)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 0.1) # Includes text boxes, not just stage
        Clock.schedule_once(functools.partial(ctx.image_settings.set_expanded_layer, 'BF'), 0.2)

        # Clear app initialization flag and apply settings for the default opened layer
        def complete_initialization(dt):
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

        # Set initial objective from settings (enables scale bar, pixel size calc)
        try:
            lumaview.scope.set_objective(objective_id=settings['objective_id'])
        except Exception as e:
            logger.warning(f'[LVP Main  ] Failed to set initial objective: {e}')

        layer_obj = ctx.image_settings.layer_lookup(layer='BF')
        layer_obj.apply_settings()

        config_helpers.log_system_metrics(settings)  # Log once on startup

        Clock.schedule_interval(lambda dt: config_helpers.log_system_metrics(settings), 14400)   # Log metrics every 4 hours

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
            print(f"ERROR: {error_msg}", file=sys.stderr)
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
        global ctx

        self.icon = './data/icons/icon.png'

        # Load engineering mode early so _init_ui() methods see the correct value
        global ENGINEERING_MODE
        ENGINEERING_MODE = _load_mode()

        stage = Stage()


        try:
            from kivy.core.window import Window
            # Window min size uses SDL point coordinates — do NOT use dp()
            Window.minimum_width = 1024
            Window.minimum_height = 600
            Window.bind(on_resize=self._on_resize)
            Window.bind(on_request_close=self.on_request_close)
            lumaview = MainDisplay(camera_type=settings['camera_type'], simulate=simulate_mode)
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
        ctx = AppContext(
            scope=lumaview.scope,
            lumaview=lumaview,
            settings=settings,
            session=scope_session,
            sequenced_capture_executor=sequenced_capture_executor,
            autofocus_executor=autofocus_executor,
            version=version,
            source_path=source_path,
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
            stage=stage,
            cell_count_content=cell_count_content,
            graphing_controls=graphing_controls,
            ij_helper=ij_helper,
            protocol_running=protocol_running_global,
            engineering_mode=ENGINEERING_MODE,
            show_tooltips=show_tooltips,
            live_histo_setting=live_histo_setting,
            last_save_folder=last_save_folder,
            disable_homing=disable_homing,
            simulate_mode=simulate_mode,
            live_view_fps=live_view_fps,
            focus_round=focus_round,
        )
        app_context.ctx = ctx  # Publish to module-level singleton for extracted modules

        # Wire UI components now that widget tree exists
        ctx.viewer = lumaview.ids['viewer_id']
        ctx.scope_display = ctx.viewer.ids['scope_display_id']
        ctx.image_settings = lumaview.ids['imagesettings_id']
        ctx.motion_settings = lumaview.ids['motionsettings_id']

        # load settings file (must be after motion_settings is wired)
        ctx.motion_settings.ids['microscope_settings_id'].load_settings("./data/current.json")

        # Copy post_processing self-registration values (set during widget tree construction
        # before ctx existed via `import lumaviewpro; lumaviewpro.X = self`)
        import lumaviewpro as _lvp_mod
        ctx.stitch_controls = getattr(_lvp_mod, 'stitch_controls', None)
        ctx.zprojection_controls = getattr(_lvp_mod, 'zprojection_controls', None)
        ctx.composite_gen_controls = getattr(_lvp_mod, 'composite_gen_controls', None)
        ctx.video_creation_controls = getattr(_lvp_mod, 'video_creation_controls', None)

        # Creates and manages Tooltips
        self.init_tooltips(lumaview)

        # Engineering plugin hook — adds engineering tab when installed
        try:
            import etaluma_engineering
            etaluma_engineering.register(ctx)
            logger.info('[LVP Main  ] Engineering plugin loaded')
        except ImportError:
            pass  # Expected — plugin not installed
        except Exception as e:
            logger.warning(f'[LVP Main  ] Engineering plugin failed to register: {e}')

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

        logger.info("[LVP Main  ] lumaview.scope.disconnect()")
        lumaview.scope.disconnect()

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})


    # Tooltip methods provided by TooltipMixin (ui/tooltip.py)



# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    LumaViewProApp().run()
