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

import atexit
import datetime
import functools
import json
import logging
import os
import sys
import threading
import time

import matplotlib
matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict
import numpy as np
import pandas as pd

if __name__ == "__main__":
    disable_homing = False
    simulate_mode = '--simulate' in sys.argv
    if simulate_mode:
        sys.argv.remove('--simulate')
    no_engineering = '--no-engineering' in sys.argv
    if no_engineering:
        sys.argv.remove('--no-engineering')

    ############################################################################
    #---------------------Directory Initialization-----------------------------#
    ############################################################################

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

    from concurrent.futures import ProcessPoolExecutor

    import modules.common_utils as common_utils
    import modules.labware as labware
    from modules.autofocus_executor import AutofocusExecutor
    import modules.autofocus_functions as autofocus_functions
    import modules.binning as binning
    import modules.coord_transformations as coord_transformations
    import modules.labware_loader as labware_loader
    import modules.lvp_lock as lvp_lock
    import modules.objectives_loader as objectives_loader
    from modules.protocol import Protocol
    from modules.sequenced_capture_executor import SequencedCaptureExecutor
    from modules.sequential_io_executor import IOTask, SequentialIOExecutor
    import modules.config_helpers as config_helpers
    from modules.scope_session import ScopeSession
    import modules.app_context as app_context
    from modules.app_context import AppContext
    import modules.post_processing as post_processing

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

    # Module-level state — assigned in build(), accessed throughout app lifetime.
    # All are registered on AppContext after creation.
    wellplate_loader = None
    objective_helper = None
    coordinate_transformer = None
    sequenced_capture_executor = None
    show_tooltips = False
    protocol_running_global = threading.Event()
    live_histo_setting = False
    last_save_folder = None
    stage = None
    ENGINEERING_MODE = False
    focus_round = 0

    # Executors — created in build(), registered on AppContext
    io_executor = None
    camera_executor = None
    protocol_executor = None
    file_io_executor = None
    autofocus_thread_executor = None
    scope_display_thread_executor = None
    stage_executor = None
    turret_executor = None
    reset_executor = None
    scope_session = None
    ctx = None

    if use_multiprocessing:
        import multiprocessing
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)

        # Import existing writer for process pool
        from modules.sequenced_capture_writer import write_capture, worker_initializer, _noop

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
# Imports — extracted modules (must be after Kivy init)
# ============================================================================

from modules.ui_helpers import (  # noqa: E402
    move_absolute_position, move_relative_position, move_home, move_home_cb,
    scope_leds_off, set_recording_title, set_writing_title, reset_title,
    live_histo_off, live_histo_reverse, set_last_save_folder,
    reset_acquire_ui, reset_stim_ui,
    _handle_ui_for_leds_off, _handle_ui_for_led,
    _update_step_number_callback, _handle_ui_update_for_axis,
    _handle_autofocus_ui, update_autofocus_selection_after_protocol,
    focus_log, find_nearest_step,
)

from modules.step_navigation import go_to_step, go_to_step_update_ui  # noqa: E402

from modules.config_ui_getters import (  # noqa: E402
    get_binning_from_ui, get_zstack_params, get_zstack_positions,
    get_layer_configs, get_active_layer_config, get_stim_configs,
    get_enabled_stim_configs, get_current_plate_position,
    get_current_frame_dimensions, get_selected_labware,
    get_image_capture_config_from_ui, get_sequenced_capture_config_from_ui,
    get_auto_gain_settings, get_current_objective_info,
    get_protocol_time_params, is_image_saving_enabled,
    create_hyperstacks_if_needed,
)

from ui.scope_display import ScopeDisplay  # noqa: E402
from ui.composite_capture import CompositeCapture  # noqa: E402
from ui.main_display import MainDisplay  # noqa: E402
from ui.shader import ShaderViewer, ShaderEditor  # noqa: E402

from ui.image_settings import (  # noqa: E402
    AccordionItemXyStageControl, AccordionItemImageSettingsBase,
    AccordionItemImageSettingsLumiControl, AccordionItemImageSettingsDfControl,
    AccordionItemImageSettingsRedControl, AccordionItemImageSettingsGreenControl,
    AccordionItemImageSettingsBlueControl, ImageSettings, set_histogram_layer,
)

from ui.motion_settings import MotionSettings, XYStageControl  # noqa: E402
from ui.post_processing import (  # noqa: E402
    StitchControls, ZProjectionControls, CompositeGenControls,
    VideoCreationControls, GraphingControls, CellCountControls,
    PostProcessingAccordion, CellCountDisplay,
)

from ui.histogram import Histogram  # noqa: E402
from ui.vertical_control import VerticalControl  # noqa: E402
from ui.protocol_settings import ProtocolSettings  # noqa: E402
from ui.stage import Stage  # noqa: E402
from ui.microscope_settings import MicroscopeSettings  # noqa: E402
from ui.mod_slider import ModSlider  # noqa: E402
from ui.layer_control import LayerControl  # noqa: E402
from ui.zstack import ZStack  # noqa: E402
from ui.file_dialogs import FileChooseBTN, FolderChooseBTN, FileSaveBTN  # noqa: E402
from modules.app_config import (  # noqa: E402
    load_log_level, get_lvp_lock_port, load_autofocus_log_enable,
    load_mode as _load_mode,
)

from ui.tooltip import Tooltip, TooltipMixin  # noqa: E402


class LumaViewProApp(TooltipMixin, App):
    """Main application class — build, start, stop, tooltips."""
    kv_file = 'ui/lumaviewpro.kv'

    def on_start(self):
        # Position listener: push-based UI updates on every move (immediate response).
        # Replaces 10Hz polling for crosshair/stage/position text during motion.
        def _on_position_change(axis, target, state):
            if axis in ('X', 'Y'):
                Clock.schedule_once(lambda dt: ctx.motion_settings.update_xy_stage_control_gui(), 0)
                Clock.schedule_once(lambda dt: stage.draw_labware(), 0)
            elif axis == 'Z':
                z_ctrl = ctx.motion_settings.ids.get('verticalcontrol_id')
                if z_ctrl:
                    Clock.schedule_once(lambda dt: z_ctrl._update_z_text(target), 0)
        lumaview.scope.add_position_listener(_on_position_change)

        # Slow idle refresh (1Hz) for display elements that may change without motion
        # (e.g., labware selection, stage offset changes)
        Clock.schedule_interval(stage.draw_labware, 1.0)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 1.0)
        Clock.schedule_once(functools.partial(ctx.image_settings.set_expanded_layer, 'BF'), 0.2)

        # Clear app initialization flag and apply settings for the default opened layer
        def complete_initialization(dt):
            if ctx is not None:
                ctx.ready = True

                # Log initial per-channel settings for debugging
                try:
                    settings = ctx.settings
                    for layer in ('BF', 'PC', 'DF', 'Red', 'Green', 'Blue', 'Lumi'):
                        ls = settings.get(layer, {})
                        logger.info(
                            f'[INIT      ] {layer:6s}: gain={ls.get("gain", "?"):>6}, '
                            f'exp={ls.get("exp", "?"):>8}ms, ill={ls.get("ill", "?"):>6}mA, '
                            f'af={ls.get("autofocus", "?")}, acquire={ls.get("acquire", "?")}'
                        )
                except Exception:
                    pass

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
                file_q = file_io_executor.queue_size() if hasattr(file_io_executor, 'queue_size') else -1
                af_q = autofocus_thread_executor.queue_size() if hasattr(autofocus_thread_executor, 'queue_size') else -1
                sd_q = scope_display_thread_executor.queue_size() if hasattr(scope_display_thread_executor, 'queue_size') else -1
                reset_q = reset_executor.queue_size() if hasattr(reset_executor, 'queue_size') else -1

                # Only log at warning level when any queue is backing up
                total_q = sum(q for q in [io_q, cam_q, prot_q, file_q, af_q, sd_q, reset_q] if q > 0)
                if total_q > 10:
                    logger.warning(f"[Watchdog  ] Queue backlog ({total_q} total) — IO:{io_q} CAM:{cam_q} PROT:{prot_q} FILE:{file_q} AF:{af_q} SD:{sd_q} RESET:{reset_q}")
                else:
                    logger.debug(f"[Watchdog  ] Queues — IO:{io_q} CAM:{cam_q} PROT:{prot_q} FILE:{file_q} AF:{af_q} SD:{sd_q} RESET:{reset_q}")

                # If scopedisplay backlog is growing and appears stale, prune it to keep UI responsive
                if sd_q is not None and sd_q > 20:
                    try:
                        scope_display_thread_executor.clear_pending()
                        logger.warning("[Watchdog  ] Cleared ScopeDisplay pending queue to prevent backlog")
                    except Exception:
                        pass
            except Exception:
                pass

        Clock.schedule_interval(_executor_watchdog, 60)

        load_log_level(source_path)
        load_autofocus_log_enable(source_path)
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

        # Objective and LEDs already set by scope.initialize() in load_settings().
        # BF apply_settings will be called by complete_initialization() → accordion_collapse().

        config_helpers.log_system_metrics(settings)  # Log once on startup

        Clock.schedule_interval(lambda dt: config_helpers.log_system_metrics(settings), 14400)   # Log metrics every 4 hours

        if lumaview.scope.camera_is_connected():
            lumaview.log_camera_temps()  # Log once on startup
            lumaview.camera_temps_event = Clock.schedule_interval(lambda dt: lumaview.log_camera_temps(), 14400)  # Log every 4 hours

        # Register emergency shutdown handler — ensures LEDs are off and serial
        # ports released even if the app crashes or is killed. This is safety-
        # critical: LEDs left on can overheat samples.
        def _emergency_shutdown():
            try:
                if lumaview and lumaview.scope:
                    lumaview.scope.leds_off()
                    lumaview.scope.disconnect()
                    logger.info('[LVP Main  ] atexit: emergency shutdown complete (LEDs off, disconnected)')
            except Exception:
                pass  # Best-effort — logging may already be torn down
        atexit.register(_emergency_shutdown)

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
            # Use a thread with timeout to prevent blocking forever on shutdown
            t = threading.Thread(target=lambda: cpu_pool.shutdown(wait=True), daemon=True)
            t.start()
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning('[LVP Main  ] cpu_pool.shutdown timed out after 5s')

        if reset_executor is not None:
            reset_executor.shutdown(wait=False)

        logger.info('[LVP Main  ] Threads shut down.')


    def build(self):
        current_time = time.strftime("%m/%d/%Y", time.localtime())
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        logger.info('[LVP Main  ] -----------------------------------------')
        logger.info(f'[LVP Main  ] Version: {version}')
        logger.info('[LVP Main  ] Run Time: ' + time.strftime("%Y %m %d %H:%M:%S"))
        logger.info('[LVP Main  ] -----------------------------------------')

        self._lvp_lock = lvp_lock.LvpLock(lock_port=get_lvp_lock_port(source_path))
        if not self._lvp_lock.lock():
            error_msg = "Another instance of LVP may already be running. Exiting."
            logger.error(f'[LVP Lock ] {error_msg}')
            print(f"ERROR: {error_msg}", file=sys.stderr)
            sys.exit(1)
        
        global Window
        global lumaview
        global cell_count_content
        global graphing_controls
        # video_creation_controls, stitch_controls, zprojection_controls,
        # composite_gen_controls — now registered directly on ctx by their __init__
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

        # Window title: version + build timestamp
        _title_version = version
        try:
            _build_ts = _env.build_timestamp
            if _build_ts:
                _title_version = f"{version} ({_build_ts})"
        except AttributeError:
            pass
        self.title = f'LumaViewPro {_title_version}'
        logger.info(f'[LVP Main  ] Window title: {self.title}')

        # Load engineering mode early so _init_ui() methods see the correct value
        global ENGINEERING_MODE
        ENGINEERING_MODE = _load_mode(source_path)

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
        wellplate_loader = labware_loader.WellPlateLoader(source_path=source_path)
        coordinate_transformer = coord_transformations.CoordinateTransformer()

        objective_helper = objectives_loader.ObjectiveLoader(source_path=source_path)

        # Create executors (previously at module level, moved here for init consolidation)
        global io_executor, camera_executor, protocol_executor
        global file_io_executor, autofocus_thread_executor, scope_display_thread_executor
        global stage_executor, turret_executor, reset_executor
        io_executor = SequentialIOExecutor(name="IO")
        camera_executor = SequentialIOExecutor(name="CAMERA")
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
        protocol_executor.start()
        file_io_executor.start()
        autofocus_thread_executor.start()
        scope_display_thread_executor.start()
        # stage_executor and turret_executor are aliases for io_executor (already started above)
        reset_executor.start()
        #ij_helper = imagej_helper.ImageJHelper()

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
        app_context.apply_early_registrations()  # Copy widgets registered during KV construction

        # Wire UI components now that widget tree exists
        ctx.viewer = lumaview.ids['viewer_id']
        ctx.scope_display = ctx.viewer.ids['scope_display_id']
        ctx.image_settings = lumaview.ids['imagesettings_id']
        ctx.motion_settings = lumaview.ids['motionsettings_id']

        # load settings file (must be after motion_settings is wired)
        ctx.motion_settings.ids['microscope_settings_id'].load_settings("./data/current.json")

        # Creates and manages Tooltips
        self.init_tooltips(lumaview)

        # Engineering plugin hook — adds engineering tab when installed
        try:
            import etaluma_engineering
            # Check version compatibility
            REQUIRED_PLUGIN_VERSION = "0.1.0"
            plugin_version = getattr(etaluma_engineering, '__version__', '0.0.0')
            if plugin_version < REQUIRED_PLUGIN_VERSION:
                logger.debug(f'[LVP Main  ] Engineering plugin {plugin_version} outdated, '
                             f'need {REQUIRED_PLUGIN_VERSION}. '
                             f'Please update: pip install -e path/to/etaluma-engineering')
            etaluma_engineering.register(ctx)
            # Auto-enable engineering mode when plugin is present
            # (unless --no-engineering was passed on command line)
            if not ENGINEERING_MODE and not no_engineering:
                ENGINEERING_MODE = True
                lumaview.scope.engineering_mode = True
                ctx.engineering_mode = True
                logger.info('[LVP Main  ] Engineering mode auto-enabled (plugin detected)')
            logger.info(f'[LVP Main  ] Engineering plugin v{plugin_version} loaded')
        except ImportError:
            pass  # Expected — plugin not installed
        except Exception as e:
            logger.warning(f'[LVP Main  ] Engineering plugin failed to register: {e}')

        # Enable engineering-only log files (autofocus.log, api.log)
        from lvp_logger import enable_engineering_logs
        enable_engineering_logs(ENGINEERING_MODE)

        # Wire NotificationCenter to UI popups
        from modules.notification_center import notifications, Severity

        def _ui_notification_bridge(n):
            from kivy.clock import Clock
            from ui.notification_popup import show_notification_popup
            Clock.schedule_once(lambda dt: show_notification_popup(title=n.title, message=n.message), 0)

        notifications.add_listener(_ui_notification_bridge,
            min_severity=Severity.DEBUG if ENGINEERING_MODE else Severity.WARNING)

        # CPU profiling — enabled via debug_mode in settings.json
        # On exit, dumps a .profile file to logs/profile/ that can be
        # viewed with: pip install snakeviz && snakeviz <file>.profile
        if settings.get('debug_mode', False):
            global profiling_helper
            profiling_helper = profiling_utils.ProfilingHelper()
            profiling_helper.enable()
            logger.info('[LVP Main  ] cProfile enabled (debug_mode=true) — will dump on exit')

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


        # Stop motors if any are moving
        try:
            if lumaview.scope.motor_connected:
                lumaview.scope.motion.exchange_command('STOP')
                logger.info('[LVP Main  ] Motors stopped')
        except Exception as e:
            logger.warning(f'[LVP Main  ] Motor stop failed during shutdown: {e}')

        logger.info("[LVP Main  ] lumaview.scope.leds_off()")
        try:
            # Use a thread with timeout to avoid blocking MainThread
            # if workers still hold _hw_lock
            t = threading.Thread(target=lumaview.scope.leds_off, daemon=True)
            t.start()
            t.join(timeout=2.0)  # Wait max 2 seconds
            if t.is_alive():
                logger.warning('[LVP Main  ] leds_off timed out during shutdown')
        except Exception as e:
            logger.warning(f'[LVP Main  ] leds_off failed during shutdown: {e}')

        # Save settings if hardware was connected this session.
        # Without hardware, slider defaults (0.01ms exposure, etc.) get written
        # to current.json, corrupting the user's real hardware settings.
        # TODO 4.1: Split settings save so non-hardware values (folder paths,
        # protocol config) are always saved, while hardware values (gain,
        # exposure) are only saved when hardware was connected.
        if lumaview.scope.camera_is_connected() or lumaview.scope.motor_connected or lumaview.scope.led_connected:
            ctx.motion_settings.ids['microscope_settings_id'].save_settings("./data/current.json")
        else:
            logger.info('[LVP Main  ] Skipping settings save - no hardware was connected')

        logger.info("[LVP Main  ] lumaview.scope.disconnect()")
        lumaview.scope.disconnect()

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})


    # Tooltip methods provided by TooltipMixin (ui/tooltip.py)



# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    LumaViewProApp().run()
