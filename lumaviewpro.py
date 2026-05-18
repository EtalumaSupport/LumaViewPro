#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os

# Python version check — must run before any imports that require 3.11+
import sys

if sys.version_info < (3, 11):  # noqa: UP036 -- runtime check is load-bearing UX (friendly error before deeper SyntaxError on Python 3.10).
    _ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    _msg = (
        f'LumaViewPro requires Python 3.11 or later.\n'
        f'You are running Python {_ver}.\n\n'
        f'Supported versions: 3.11, 3.12, 3.13'
    )
    try:
        # Try GUI dialog first
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror('Unsupported Python Version', _msg)
        root.destroy()
    except Exception:  # grain: ignore NAKED_EXCEPT
        pass
    print(f'ERROR: {_msg}', file=sys.stderr)
    sys.exit(1)

import functools
import threading

import matplotlib

matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict

if __name__ == '__main__':
    disable_homing = False
    simulate_mode = '--simulate' in sys.argv
    if simulate_mode:
        sys.argv.remove('--simulate')
    no_engineering = '--no-engineering' in sys.argv
    if no_engineering:
        sys.argv.remove('--no-engineering')

    ############################################################################
    # ---------------------Directory Initialization-----------------------------#
    ############################################################################

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

    PROTOCOL_DATA_DIR_NAME = 'ProtocolData'

    ############################################################################
    # ---------------------Module Imports---------------------------------------#
    ############################################################################

    from lvp_logger import debug, logger

    DEBUG_MODE = debug

    print(f'LumaViewPro {version}')
    logger.info(f'[LVP Main  ] LumaViewPro {version}')

    if DEBUG_MODE:
        logger.info('[LVP Main  ] Debug mode is enabled.')

    try:
        from modules.settings_init import load_lvp_settings

        load_lvp_settings(logger, source_path)

        from modules.settings_init import settings as initialized_settings

        settings = initialized_settings

    except Exception as e:  # grain: ignore NAKED_EXCEPT
        logger.critical(f'[LVP Main  ] Failed to load settings -- cannot continue. {e}')
        sys.exit(1)

    import modules.app_context as app_context
    import modules.common_utils as common_utils
    import modules.config_helpers as config_helpers
    import modules.coord_transformations as coord_transformations
    import modules.labware_loader as labware_loader
    import modules.lvp_lock as lvp_lock
    import modules.objectives_loader as objectives_loader
    import modules.profiling_utils as profiling_utils
    from modules.app_context import AppContext
    from modules.autofocus_runner import AutofocusRunner
    from modules.autofocus_thread import AutofocusThread
    from modules.scope_session import ScopeSession
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    global profiling_helper
    profiling_helper = None

    if getattr(sys, 'frozen', False):
        import pyi_splash  # type: ignore

        pyi_splash.update_text('')

    # Disable Kivy's own file logging (LVP has its own RotatingFileHandler)
    os.environ['KIVY_NO_CONSOLELOG'] = '1'
    os.environ['KIVY_NO_FILELOG'] = '1'

    # Single-instance lock check BEFORE any Kivy import.
    # When the check lived inside App.build(), Kivy had already
    # initialized SDL2 and opened a native window by the time the
    # loser reached sys.exit, producing duplicate visible Kivy
    # windows on double-launch. Run the check now while only
    # tkinter is alive (used above for the Python-version dialog).
    from modules.app_config import get_lvp_lock_port as _get_lvp_lock_port

    _lvp_lock_singleton = lvp_lock.LvpLock(
        lock_port=_get_lvp_lock_port(source_path)
    )
    if not _lvp_lock_singleton.lock():
        _msg = 'Another instance of LVP may already be running. Exiting.'
        logger.error(f'[LVP Lock ] {_msg}')
        print(f'ERROR: {_msg}', file=sys.stderr)
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb

            _root = _tk.Tk()
            _root.withdraw()
            _mb.showerror(
                'LumaViewPro: already running',
                'Another copy of LumaViewPro is already running.\n\n'
                'This copy will now close. Switch to the existing '
                'window, or close the other instance first before '
                'launching again.',
            )
            _root.destroy()
        except Exception as popup_err:  # grain: ignore NAKED_EXCEPT
            logger.warning(
                f'[LVP Lock ] Could not display already-running popup: {popup_err}'
            )
        # os._exit terminates immediately; sys.exit raises SystemExit
        # which downstream cleanup paths may swallow before any Kivy
        # import has the chance to spin up.
        os._exit(1)

    # Kivy configurations
    # Configurations must be set before Kivy is imported
    from kivy.config import Config

    Config.set('input', 'mouse', 'mouse, disable_multitouch')
    Config.set('graphics', 'resizable', True)
    Config.set('kivy', 'exit_on_escape', '0')
    Config.set('graphics', 'minimum_width', '1024')
    Config.set('graphics', 'minimum_height', '600')

    # Maximized at launch — works correctly on macOS, Windows, and Linux
    Config.set('graphics', 'window_state', 'maximized')

    import kivy

    kivy.require('2.1.0')

    from kivy.app import App
    from kivy.clock import Clock
    from kivy.factory import Factory
    from kivy.graphics import (
        Color,
        Ellipse,
        Line,
        Rectangle,
        RenderContext,
    )

    # Video Related
    from kivy.graphics.texture import Texture
    from kivy.input.motionevent import MotionEvent
    from kivy.metrics import dp
    from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, StringProperty

    # User Interface
    from kivy.uix.accordion import AccordionItem
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.image import Image
    from kivy.uix.label import Label
    from kivy.uix.popup import Popup
    from kivy.uix.scatter import Scatter
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.slider import Slider
    from kivy.uix.widget import Widget

    import ui.image_utils_kivy as image_utils_kivy

    # Matplotlib-to-Kivy bridge → ui/figure_canvas.py
    from ui.figure_canvas import FigureCanvasKivyAgg
    from ui.notification_popup import (
        show_confirmation_popup,
        show_notification_popup,
    )
    from ui.progress_popup import show_popup

    # User Interface Custom Widgets
    from ui.range_slider import RangeSlider
    from ui.rounded_buttons import RoundedButton, RoundedToggleButton

    # Most state lives on ctx (single source of truth). The few globals
    # that remain below are read by multiple methods (on_start / on_stop
    # / on_request_close / closures) and have not been lifted onto ctx
    # yet; build-only state lives as locals in build().
    show_tooltips = False

    # Executors are created in build() via ExecutorBundle and bound to
    # these named globals for backwards compat with existing readers.
    io_executor = None
    camera_executor = None
    protocol_thread = None
    file_io_executor = None
    autofocus_thread = None
    scope_display_thread = None
    worker_pool = None
    executor_bundle = None
    ctx = None

else:
    # Subprocess/worker compatibility — Kivy not available
    from modules.subprocess_stubs import (
        AccordionItem,
        App,
        BooleanProperty,
        BoxLayout,
        Button,
        Clock,
        Color,
        Ellipse,
        Factory,
        FigureCanvasKivyAgg,
        FloatLayout,
        Image,
        Label,
        Line,
        ListProperty,
        MotionEvent,
        ObjectProperty,
        Popup,
        RangeSlider,
        Rectangle,
        RenderContext,
        RoundedButton,
        RoundedToggleButton,
        Scatter,
        ScrollView,
        Slider,
        StringProperty,
        Texture,
        ToggleButton,
        Widget,
        dp,
        image_utils_kivy,
        show_notification_popup,
        show_popup,
    )

# ============================================================================
# Imports — extracted modules (must be after Kivy init)
# ============================================================================

from modules.app_config import (
    load_autofocus_log_enable,
    load_log_level,
)
from modules.app_config import (
    load_mode as _load_mode,
)

# Kivy Factory imports: the classes below are referenced from ui/lumaviewpro.kv
# (and other .kv files Kivy loads at startup). Kivy's Builder.apply() resolves
# class names via Factory.get(), which only finds a class once it has been
# imported into a Python module. ruff F401 cannot see .kv references and would
# strip these as unused; the per-file ignore in pyproject.toml ([tool.ruff.lint
# .per-file-ignores] "lumaviewpro.py" lists F401) silences that. Do not remove
# any of these without first grepping ui/*.kv for the class name.
from ui.composite_capture import CompositeCapture
from ui.file_dialogs import FileChooseBTN, FileSaveBTN, FolderChooseBTN
from ui.histogram import Histogram
from ui.image_settings import (
    AccordionItemImageSettingsBase,
    AccordionItemImageSettingsBlueControl,
    AccordionItemImageSettingsDfControl,
    AccordionItemImageSettingsGreenControl,
    AccordionItemImageSettingsLumiControl,
    AccordionItemImageSettingsRedControl,
    AccordionItemXyStageControl,
    ImageSettings,
)
from ui.layer_control import LayerControl
from ui.main_display import MainDisplay
from ui.microscope_settings import MicroscopeSettings
from ui.mod_slider import ModSlider
from ui.motion_settings import MotionSettings, XYStageControl
from ui.post_processing import (
    CellCountControls,
    CellCountDisplay,
    CompositeGenControls,
    GraphingControls,
    PostProcessingAccordion,
    StitchControls,
    VideoCreationControls,
    ZProjectionControls,
)
from ui.protocol_settings import ProtocolSettings
from ui.scope_display import ScopeDisplay
from ui.shader import ShaderEditor, ShaderViewer
from ui.stage import Stage
from ui.tooltip import Tooltip, TooltipMixin
from ui.ui_helpers import (
    _handle_autofocus_ui,
    _handle_ui_update_for_axis,
)
from ui.vertical_control import VerticalControl
from ui.zstack import ZStack


class LumaViewProApp(TooltipMixin, App):
    """Main application class — build, start, stop, tooltips."""

    kv_file = 'ui/lumaviewpro.kv'

    def on_start(self) -> None:
        """Kivy lifecycle hook: fires after build() and before the main loop runs."""
        # Read scope through ctx so widget rebuilds (LS850 <-> LS620) don't strand it.
        lumaview = ctx.lumaview

        # UI listener bridges live in modules/ui_listener_bridge.py so REST API and
        # headless tools can reuse them.
        from modules.ui_listener_bridge import UIListenerBridge

        ctx.ui_listener_bridge = UIListenerBridge(
            scope=lumaview.scope,
            ctx=ctx,
            stage=ctx.stage,
            ui_dispatcher=Clock.schedule_once,
        )
        ctx.ui_listener_bridge.register_all()

        # Slow idle refresh (1Hz) for display elements that may change without motion
        # (e.g., labware selection, stage offset changes)
        Clock.schedule_interval(ctx.stage.draw_labware, 1.0)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 1.0)
        Clock.schedule_once(functools.partial(ctx.image_settings.set_expanded_layer, 'BF'), 0.2)

        # Stage B1: publish Kivy-side layer state to scope_display_thread at
        # 30Hz. The thread cannot read Kivy widget attrs from a non-UI
        # thread (Rule 15). This callback reads
        # get_active_layer_config() + engineering-mode open-layer and
        # pushes them onto the thread; the thread reads under _config_lock
        # at each frame start. Staleness is bounded by 33ms (one tick).
        def _publish_layer_config(dt):
            if ctx is None or scope_display_thread is None:
                return
            active_layer = None
            active_layer_config = None
            open_layer = None
            try:
                from modules.config_ui_getters import get_active_layer_config
                active_layer, active_layer_config = get_active_layer_config()
            except Exception:
                pass
            if ctx.engineering_mode and ctx.image_settings is not None:
                import modules.common_utils as _cu
                for layer in _cu.get_layers():
                    accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
                    if not accordion_item_obj.collapse:
                        open_layer = layer
                        break
            scope_display_thread.update_layer_config(
                active_layer, active_layer_config, open_layer,
            )
        Clock.schedule_interval(_publish_layer_config, 1.0 / 30)

        # Clear app initialization flag and apply settings for the default opened layer
        def complete_initialization(dt):
            if ctx is not None:
                ctx.ready = True

                # Sync Z slider to actual motor position. Without this the
                # .kv hardcodes obj_position.value=0 and the first user
                # click snaps Z to 0 regardless of where the motor is.
                try:
                    _handle_ui_update_for_axis('Z')
                except Exception as e:  # grain: ignore NAKED_EXCEPT
                    logger.warning(f'[INIT      ] Z slider sync failed: {e}')

                try:
                    settings = ctx.settings
                    for layer in common_utils.get_layers():
                        ls = settings.get(layer, {})
                        logger.info(
                            f'[INIT      ] {layer:6s}: gain={ls.get("gain", "?"):>6}, '
                            f'exp={ls.get("exp", "?"):>8}ms, ill={ls.get("ill", "?"):>6}mA, '
                            f'af={ls.get("autofocus", "?")}, acquire={ls.get("acquire", "?")}'
                        )
                except Exception as e:  # grain: ignore NAKED_EXCEPT
                    logger.warning(f'[INIT      ] per-channel settings log skipped: {e}')

            # Apply transmitted-layer slider caps (50 mA on BF / PC / DF)
            # before either branch below fires. The .kv ships ill_slider
            # at max=500; without this call the cap stays unapplied until
            # the user first toggles the settings panel, leaving BF / PC
            # / DF channels exposed at slider-default 500 mA.
            try:
                ctx.image_settings.update_transmitted()
            except Exception as e:  # grain: ignore NAKED_EXCEPT
                logger.warning(f'[INIT      ] update_transmitted skipped: {e}')

            # Check if a protocol is loaded and has steps
            if ctx.protocol is not None and ctx.protocol.num_steps() > 0:
                protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
                protocol_settings.go_to_step(protocol=False)
                return

            # If no protocol, just apply settings for the default BF layer
            ctx.image_settings.accordion_collapse()

        Clock.schedule_once(complete_initialization, 0.3)

        # MetricsLogger owns the executor watchdog, system metrics, and camera-temp
        # logging so adding a new periodic metric only requires editing one module.

        load_log_level(source_path)
        load_autofocus_log_enable(source_path)
        logger.info('[LVP Main  ] LumaViewProApp.on_start()')

        if lumaview.scope.no_hardware:
            Clock.schedule_once(
                lambda dt: show_notification_popup(
                    title='No hardware detected',
                    message=(
                        'No microscope hardware was detected. You can continue in software-only '
                        'mode (live view + protocol design will work; capture will not). To '
                        'connect hardware, power on the scope and reconnect the USB cable, then '
                        'restart LumaViewPro.'
                    ),
                ),
                0,
            )

        # ScopeSession owns startup orchestration so REST API, headless tools, and
        # the reconnect handler in ui/microscope_settings.py all hit the same path.
        ctx.session.start_application_session(disable_homing=disable_homing)

        # Objective and LEDs are set by scope.initialize() during load_settings();
        # BF apply_settings fires from complete_initialization() -> accordion_collapse().

        # Once-per-startup environment + dependency fingerprint. Pairs
        # with the per-tick [PDH METRICS] / [BUFFER METRICS] surface so
        # post-mortem can correlate a problem against the exact host
        # state (OS, Python, Pylon SDK, Defender state, etc.) without
        # the noise of repeating those facts every tick.
        config_helpers.log_environment_once()

        # Lumascope.__init__ constructs the MetricsLogger so REST and headless
        # callers share the same surface. Here we register the executor bundle and
        # start the logger with a KivyClockScheduler; REST entry points wire a
        # ThreadingTimerScheduler instead.
        from modules.scheduler import KivyClockScheduler

        lumaview.scope.register_executor_bundle(executor_bundle, settings)
        ctx.metrics_logger = lumaview.scope.metrics_logger
        if ctx.metrics_logger is not None:
            # settings.profiling.metrics_interval_s overrides the default
            # 3600 s cadence. Set to 30-60 s for short-soak leak hunts
            # (gen2_depth + handle/thread counts are usable signals at
            # sub-minute granularity; hourly is fine for production).
            _prof = ctx.settings.get('profiling', {})
            _metrics_interval = _prof.get('metrics_interval_s', None)
            _start_kwargs = {}
            if _metrics_interval is not None:
                _start_kwargs['system_metrics_interval_s'] = float(_metrics_interval)
            ctx.metrics_logger.start(KivyClockScheduler(Clock), **_start_kwargs)

        # The atexit emergency-shutdown hook is registered in Lumascope.__init__
        # so every Lumascope user gets the same safety net automatically.

        if getattr(sys, 'frozen', False):
            pyi_splash.close()

    def shutdown_threads(self) -> None:
        """Stop profiling and shut down every executor in the bundle.

        Order matters: long-lived consumer threads (autofocus_thread,
        scope_display_thread) stop BEFORE the SequentialIOExecutor
        lanes they consume. Otherwise a consumer mid-iteration can
        find its lane already shut down and either hang waiting for
        a queue dispatch that never fires or surface a misleading
        post-shutdown exception. AF holds io_executor + camera_executor;
        scope_display holds camera_executor.
        """
        logger.info('[LVP Main  ] Shutting down threads...')

        if profiling_helper is not None:
            profiling_helper.stop()

        if autofocus_thread is not None:
            autofocus_thread.stop(timeout=2.0)

        if scope_display_thread is not None:
            scope_display_thread.stop()

        if protocol_thread is not None:
            protocol_thread.stop(timeout=2.0)

        if io_executor is not None:
            io_executor.shutdown(wait=False)

        if camera_executor is not None:
            camera_executor.shutdown(wait=False)

        if file_io_executor is not None:
            file_io_executor.shutdown(wait=False)

        if worker_pool is not None:
            worker_pool.shutdown(wait=False)

        logger.info('[LVP Main  ] Threads shut down.')

    def build(self) -> 'MainDisplay':
        """Kivy lifecycle hook: construct the widget tree and return the root widget."""
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        # Every entry point emits the same launch fingerprint so REST API, headless
        # test runner, and CLI tools all get identical environment lines.
        from lvp_logger import log_environment_banner

        log_environment_banner(source_path, version)

        # Lock was claimed in __main__ before any Kivy import (issue #559);
        # keep a strong ref here so the bound socket survives for the
        # lifetime of the app. The instance is a module-global below;
        # take it onto self for symmetry with the rest of the app state.
        self._lvp_lock = _lvp_lock_singleton

        # video_creation_controls, stitch_controls, zprojection_controls, and
        # composite_gen_controls register themselves on ctx in their __init__.
        global Window
        global ctx
        ij_helper = None

        # AppContext binds these three as kwargs below; declared as locals here
        # so the kwargs don't NameError at runtime.
        live_histo_setting = False
        last_save_folder = None
        focus_round = 0

        self.icon = './data/icons/icon.png'

        # Window title: version + build timestamp
        _title_version = version
        try:
            _build_ts = _env.build_timestamp
            if _build_ts:
                _title_version = f'{version} ({_build_ts})'
        except AttributeError:
            pass
        self.title = f'LumaViewPro {_title_version}'
        logger.info(f'[LVP Main  ] Window title: {self.title}')

        # Load engineering mode early so _init_ui() methods see the correct value.
        # ENGINEERING_MODE is build-only; intentionally a local, not a global.
        ENGINEERING_MODE = _load_mode(source_path)

        stage = Stage()

        # Wire NotificationCenter to UI popups BEFORE any hardware init.
        # MainDisplay() below constructs Lumascope -> LED/motor boards
        # -> connect(), which can fire notifications.error() for silent-
        # board detection or any other early hardware failure. If the
        # listener is registered AFTER hardware init, those early errors
        # go to the log but never reach the user as popups.
        from modules.notification_center import Severity, notifications

        def _ui_notification_bridge(n):
            from kivy.clock import Clock

            from ui.notification_popup import show_notification_popup

            Clock.schedule_once(
                lambda dt: show_notification_popup(title=n.title, message=n.message), 0
            )

        notifications.add_listener(
            _ui_notification_bridge,
            min_severity=Severity.DEBUG if ENGINEERING_MODE else Severity.WARNING,
        )

        try:
            from kivy.core.window import Window

            # Window min size uses SDL point coordinates — do NOT use dp()
            Window.minimum_width = 1024
            Window.minimum_height = 600
            Window.bind(on_resize=self._on_resize)
            Window.bind(on_request_close=self.on_request_close)
            # camera_type='auto' lets the registry pick by priority (Pylon -> IDS
            # -> FX2). The legacy settings['camera_type'] field is vestigial.
            lumaview = MainDisplay(camera_type='auto', simulate=simulate_mode)
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

        # ExecutorRegistry.create_default constructs all SequentialIOExecutor
        # lanes (plus stage and turret aliases) and the protocol_thread, then
        # starts them; every entry point shares this topology so the watchdog
        # snapshot and engineering plugin see one truth.
        global io_executor, camera_executor, protocol_thread
        global file_io_executor, autofocus_thread, scope_display_thread
        global worker_pool
        global executor_bundle
        # Clock.schedule_once is passed as the UI dispatcher so executors can post
        # callbacks to the Kivy main thread without importing Kivy themselves.
        from kivy.clock import Clock

        _ui = Clock.schedule_once

        # Also set the global dispatcher for kivy_utils.schedule_ui()
        from modules.kivy_utils import set_ui_dispatcher

        set_ui_dispatcher(_ui)

        from modules.executor_registry import create_default as _create_executors

        executor_bundle = _create_executors(_ui)
        io_executor = executor_bundle.io_executor
        camera_executor = executor_bundle.camera_executor
        protocol_thread = executor_bundle.protocol_thread
        file_io_executor = executor_bundle.file_io_executor
        scope_display_thread = executor_bundle.scope_display_thread
        worker_pool = executor_bundle.worker_pool

        # GUI-independent scope session; persisted to ctx.session and
        # ctx.protocol_running so other methods read off ctx.
        protocol_running_global = threading.Event()
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

        # Register executors so scope.X_async / scope.X_sync methods can
        # dispatch without callers passing executor handles.
        lumaview.scope.register_executors(
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=file_io_executor,
        )
        # Register source_path so scope.load_protocol / create_protocol
        # can resolve data/tiling.json without callers passing the path.
        lumaview.scope.register_source_path(source_path)

        autofocus_runner = AutofocusRunner(
            scope=lumaview.scope,
            camera_executor=camera_executor,
            io_executor=io_executor,
            file_io_executor=file_io_executor,
            ui_update_func=_handle_autofocus_ui,
        )

        # AutofocusThread owns the actual AF worker thread; AFE is the
        # per-iteration state machine the thread drives. Construct after
        # AFE so the wiring is one-way (thread holds AFE, AFE is unaware
        # of the thread except via the abort_event passed to run()).
        autofocus_thread = AutofocusThread(
            afe=autofocus_runner,
            ui_dispatcher=_ui,
        )
        autofocus_thread.start()

        sequenced_capture_runner = SequencedCaptureRunner(
            scope=lumaview.scope,
            stage_offset=settings['stage_offset'],
            autofocus_runner=autofocus_runner,
            io_executor=io_executor,
            protocol_thread=protocol_thread,
            file_io_executor=file_io_executor,
            camera_executor=camera_executor,
            autofocus_thread=autofocus_thread,
            z_ui_update_func=_handle_autofocus_ui,
        )

        # Create AppContext — central service registry
        ctx = AppContext(
            scope=lumaview.scope,
            lumaview=lumaview,
            settings=settings,
            session=scope_session,
            sequenced_capture_runner=sequenced_capture_runner,
            autofocus_runner=autofocus_runner,
            version=version,
            source_path=source_path,
            io_executor=io_executor,
            camera_executor=camera_executor,
            protocol_thread=protocol_thread,
            file_io_executor=file_io_executor,
            autofocus_thread=autofocus_thread,
            scope_display_thread=scope_display_thread,
            worker_pool=worker_pool,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            stage=stage,
            cell_count_content=cell_count_content,
            graphing_controls=graphing_controls,
            ij_helper=ij_helper,
            protocol_running=protocol_running_global,
            engineering_mode=ENGINEERING_MODE,
            no_engineering=no_engineering,
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

        # Start the display thread now that both widget reference and
        # thread instance are in ctx. Earlier start sites (widget __init__,
        # registry creation) run before one or the other field is wired,
        # so they cannot validly delegate to thread.start().
        ctx.scope_display.start()

        # load settings file (must be after motion_settings is wired)
        ctx.motion_settings.ids['microscope_settings_id'].load_settings('./data/current.json')

        # Creates and manages Tooltips
        self.init_tooltips(lumaview)

        # Discover plugins via entry_points group 'lvp.plugins'.
        # Engineering plugin (etaluma-engineering, dev/bench-only) loads
        # here; customer installs find nothing in the group.
        from modules.plugins import load_plugins
        load_plugins(ctx)

        # Register in-tree built-in plugins (Stitcher canary, plus
        # CompositeGeneration / ZProjector / VideoBuilder once they
        # retire into the namespace). Runs AFTER load_plugins so an
        # external package claiming the same name wins -- the built-in
        # registration then logs WARNING and continues, leaving the
        # legacy UI button paths still wired to the same Stitcher class.
        from modules.plugins.builtin import register_builtins
        register_builtins(ctx)

        # Attach UI-namespace plugin mounts now that the widget tree
        # exists. Each registered (name, mount, builder) tuple is
        # invoked here; builder() returns the Kivy widget which is
        # added to the named mount point.
        motionsettings_accordion = ctx.motion_settings.ids['motionsettings_accordion_id']
        for plugin_name, mount_point, builder in ctx.plugins.ui.mounts():
            if mount_point == 'left_sidebar.accordion':
                try:
                    motionsettings_accordion.add_widget(builder())
                    logger.info(f'[LVP Main  ] Mounted {plugin_name} at {mount_point}')
                except Exception as e:
                    logger.error(
                        f'[LVP Main  ] {plugin_name} mount failed: {e}', exc_info=True,
                    )

        # Enable engineering-only log files (autofocus.log, api.log).
        # Read from ctx since the engineering plugin's register(ctx)
        # may have flipped ctx.engineering_mode during load_plugins.
        from lvp_logger import enable_engineering_logs

        enable_engineering_logs(ctx.engineering_mode)

        # NotificationCenter -> UI popup bridge was registered at the
        # top of build(), BEFORE MainDisplay() / Lumascope() / hardware
        # init, so any early hardware errors surface as popups instead
        # of being logged-only.

        # CPU profiling — enabled via debug_mode in settings.json
        # On exit, dumps a .profile file to logs/profile/ that can be
        # viewed with: pip install snakeviz && snakeviz <file>.profile
        if settings.get('debug_mode', False):
            global profiling_helper
            profiling_helper = profiling_utils.ProfilingHelper()
            profiling_helper.enable()
            logger.info('[LVP Main  ] cProfile enabled (debug_mode=true) -- will dump on exit')

        return lumaview

    def _on_resize(self, window, w: int, h: int) -> None:
        """Kivy on_resize hook: re-evaluate motion / image settings layout."""
        Clock.schedule_once(ctx.motion_settings.check_settings, 0.1)
        Clock.schedule_once(ctx.image_settings.check_settings, 0.1)

    def on_request_close(self, *args) -> bool:
        """Kivy on_request_close hook: show a confirmation popup if a protocol is running.

        Returns:
            True to prevent window close (popup shown); False to allow close.
        """
        if ctx.protocol_running.is_set():
            Clock.schedule_once(
                lambda dt: show_confirmation_popup(
                    title='Confirm Exit',
                    message='A protocol is currently running.\n\nAre you sure you want to exit?',
                    confirm_text='Confirm Exit',
                    cancel_text='Cancel',
                    on_confirm=self.stop,
                )
            )

            return True  # Prevent window from closing

        # No protocol running - allow normal close
        return False

    def on_stop(self) -> None:
        """Kivy lifecycle hook: tear down hardware, save settings, exit cleanly."""
        lumaview = ctx.lumaview

        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')

        # Suppress notification-listener dispatch during shutdown so the user
        # doesn't see 30+ error toasts as queued IO tasks fail against
        # disconnecting hardware. Log lines still fire for post-mortem.
        try:
            from modules.notification_center import notifications

            notifications.set_shutting_down(True)
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] Failed to suppress notifications on shutdown: {e}')

        # Plugins released first: their listener subscriptions and file handles
        # need to drop before hardware tear-down so shutdown ordering matches
        # registration ordering.
        try:
            from modules.plugins import unload_plugins
            unload_plugins(ctx)
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] Plugin unload during shutdown raised: {e}')

        # Unschedule all recurring interval callbacks to prevent orphaned events
        try:
            Clock.unschedule(ctx.stage.draw_labware)
            Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.debug(f'[LVP Main  ] Clock.unschedule during shutdown raised: {e}')

        # Stop the periodic metrics logger so its Clock intervals and
        # the camera-temp tick don't survive into shutdown and try to
        # log against torn-down hardware.
        try:
            if ctx.metrics_logger is not None:
                ctx.metrics_logger.stop()
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] metrics_logger stop failed during shutdown: {e}')

        ctx.motion_settings.ids['protocol_settings_id'].cancel_all_protocols()

        # Stop the scope-display thread BEFORE the executor cascade --
        # otherwise the FPS-paced loop submits work against a half-
        # disconnected scope and floods the shutdown log.
        try:
            if ctx.scope_display is not None:
                ctx.scope_display.stop()
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] scope_display stop during shutdown failed: {e}')

        self.shutdown_threads()

        # Considered removing this stop_motion() call, since disconnect() below calls
        # stop_motion() as its first step; rejected because shutdown_threads ran BEFORE
        # disconnect, and any in-flight motion should stop before we tear down the
        # executors that own the move callbacks. Revisit if shutdown_threads and disconnect
        # are consolidated into one teardown.
        lumaview.scope.motion.stop_motion()

        logger.info('[LVP Main  ] lumaview.scope.leds_off()')
        try:
            # Run leds_off on a thread with timeout so MainThread doesn't block
            # if workers still hold _hw_lock during teardown.
            t = threading.Thread(target=lumaview.scope.leds_off, daemon=True)
            t.start()
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning('[LVP Main  ] leds_off timed out during shutdown')
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] leds_off failed during shutdown: {e}')

        # The hardware-presence gate lives inside MicroscopeSettings.save_settings
        # now, so every caller (engineering plugin, REST, scheduled save) gets
        # the same guard. Pass force=True only to override.
        ctx.motion_settings.ids['microscope_settings_id'].save_settings('./data/current.json')

        logger.info('[LVP Main  ] lumaview.scope.disconnect()')
        lumaview.scope.disconnect()

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})

    # Tooltip methods provided by TooltipMixin (ui/tooltip.py)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    LumaViewProApp().run()
