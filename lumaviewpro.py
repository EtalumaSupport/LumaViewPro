#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os

# Python version check -- must run before any imports that require 3.12+
import sys

if sys.version_info < (3, 12):  # noqa: UP036 -- runtime check is load-bearing UX (friendly error before a deeper SyntaxError from a dependency on an unsupported Python).
    _ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    _msg = (
        f'LumaViewPro requires Python 3.12 or later.\n'
        f'You are running Python {_ver}.\n\n'
        f'Supported versions: 3.12, 3.13'
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

# The IDS camera stack must initialize while the process is still nearly
# empty: its image-processing DLL fails initialization once the full DLL
# population (matplotlib->numpy, Kivy/SDL, cv2, pylon) is resident, so the
# preload precedes every heavy import.
from modules.app_environment import preload_camera_sdks

preload_camera_sdks()

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

    # Environment setup -- paths, version, platform detection
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

    from lvp_logger import debug, log_dir, logger

    DEBUG_MODE = debug

    # The installer logs to the user TEMP folder, which no support bundle
    # collects and Windows eventually sweeps -- so an install that failed
    # to replace a file cannot be told apart from an application defect.
    from modules.app_environment import capture_installer_logs

    capture_installer_logs(log_dir)

    print(f'LumaViewPro {version}')
    logger.info(f'[LVP Main  ] LumaViewPro {version}')

    if DEBUG_MODE:
        logger.info('[LVP Main  ] Debug mode is enabled.')

    # Start the memory profiler before the heavy module/driver/ui imports so
    # tracemalloc captures the resident baseline, not just post-startup growth.
    # No-op unless memory_profile_enabled is set in the live settings.
    from lib import memory_profile

    memory_profile.start(source_path)

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

    _lvp_lock_singleton = lvp_lock.LvpLock(lock_port=_get_lvp_lock_port(source_path))
    if not _lvp_lock_singleton.lock():
        _msg = 'Another instance of LVP may already be running. Exiting.'
        logger.error(f'[LVP Lock ] {_msg}')
        # Previously also printed to stderr; on a windowed PyInstaller
        # build (console=False) that print is dropped silently and the
        # tkinter messagebox below is the user-facing signal. The
        # log line above is the file-side record. Issue #559.
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb

            _root = _tk.Tk()
            _root.withdraw()
            # Force the dialog to the foreground. Without this the messagebox
            # can open behind the existing LVP window and get buried, so the
            # user never sees why the second launch silently did nothing.
            _root.attributes('-topmost', True)
            _root.lift()
            _root.focus_force()
            _mb.showerror(
                'LumaViewPro: already running',
                'Another copy of LumaViewPro is already running.\n\n'
                'This copy will now close. Switch to the existing '
                'window, or close the other instance first before '
                'launching again.',
                parent=_root,
            )
            _root.destroy()
        except Exception as popup_err:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Lock ] Could not display already-running popup: {popup_err}')
        # After the popup dismiss, if our parent process is a console
        # shell (cmd.exe / PowerShell), post WM_CLOSE to its window so
        # the user's shell doesn't stay orphaned next to a popup they
        # already acknowledged. Windowed PyInstaller builds detach from
        # the launching console; the parent shell has no idea LVP just
        # exited and won't self-close. Best-effort: any error logs +
        # falls through to os._exit unchanged.
        if windows_machine:
            try:
                import ctypes
                from ctypes import wintypes

                import psutil

                _parent = psutil.Process(os.getpid()).parent()
                _parent_name = (_parent.name() or '').lower() if _parent else ''
                if _parent_name in ('cmd.exe', 'powershell.exe', 'pwsh.exe'):
                    _user32 = ctypes.windll.user32
                    _enum_proc_type = ctypes.WINFUNCTYPE(
                        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
                    )
                    _parent_pid = _parent.pid
                    _hwnds: list[int] = []

                    def _enum_cb(hwnd, _lparam):
                        _pid = wintypes.DWORD()
                        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(_pid))
                        if _pid.value == _parent_pid and _user32.IsWindowVisible(hwnd):
                            _hwnds.append(hwnd)
                        return True

                    _user32.EnumWindows(_enum_proc_type(_enum_cb), 0)
                    _WM_CLOSE = 0x0010
                    for _h in _hwnds:
                        _user32.PostMessageW(_h, _WM_CLOSE, 0, 0)
                    if _hwnds:
                        logger.info(
                            f'[LVP Lock ] Closed parent {_parent_name} window '
                            f'(pid={_parent_pid}, hwnds={len(_hwnds)}) after popup.'
                        )
            except Exception as console_err:  # grain: ignore NAKED_EXCEPT
                logger.warning(f'[LVP Lock ] Could not close parent console window: {console_err}')
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

    # Maximized at launch -- works correctly on macOS, Windows, and Linux
    Config.set('graphics', 'window_state', 'maximized')

    # No multisampling. The live image is a photo with no aliased edges to
    # smooth, so MSAA (Kivy defaults to 2x) only adds per-draw GL cost on the
    # integrated GPU. Measured on the acceptance-floor machine: GPU 3d-engine
    # load dropped from ~55% to ~35% with this off, no visible quality loss.
    # Line/vector UI edges lose a little smoothing in exchange.
    Config.set('graphics', 'multisamples', '0')

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
    from kivy.properties import (
        BooleanProperty,
        ListProperty,
        ObjectProperty,
        StringProperty,
    )

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

    # Matplotlib-to-Kivy bridge -> ui/figure_canvas.py
    from ui.figure_canvas import FigureCanvasKivyAgg
    from ui.notification_popup import (
        show_confirmation_popup,
        show_notification_popup,
    )
    from ui.progress_popup import show_popup

    # Imported for its side effect: registers the global <Popup> on_open rule
    # that adds a close (X) button to every popup app-wide.
    import ui.popup_close  # registers the Etaluma popup-close button on every Popup

    # User Interface Custom Widgets
    from ui.range_slider import RangeSlider
    from ui.rounded_buttons import RoundedButton, RoundedToggleButton

    # Record the clock's effective frame-rate cap once at startup. A future run
    # then tells us whether a maxfps override actually took (clock holds the set
    # value) or was ignored -- distinguishing an import-ordering miss (Config set
    # after Kivy read it) from vsync governing the swap rate independently. Kept
    # as a permanent diagnostic for render-rate / on_draw-frequency work.
    logger.info(f'[LVP Main  ] Kivy effective maxfps={Clock._max_fps}')

    # Most state lives on ctx (single source of truth). The few globals
    # that remain below are read by multiple methods (on_start / on_stop
    # / on_request_close / closures); build-only state lives as locals in
    # build().
    show_tooltips = False

    # The executor handles are the single-source-of-truth on ctx: they are
    # locals in build() and read everywhere else (including shutdown) as
    # ctx.<name>. The bundle itself rides into the session at construction
    # (which registers it on the scope), so no build()->on_start() handoff
    # global exists anymore.
    ctx = None

else:
    # Subprocess/worker compatibility -- Kivy not available
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
# Imports -- extracted modules (must be after Kivy init)
# ============================================================================

from modules import gui_logger
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
from ui.file_dialogs import FileChooseBTN, FileOrFolderChooseBTN, FileSaveBTN, FolderChooseBTN
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
    QuickEnhanceControls,
    StitchControls,
    VideoCreationControls,
    ZProjectionControls,
)
from ui.protocol_settings import ProtocolSettings
from ui.scope_display import ScopeDisplay
from ui.shader import ShaderViewer
from ui.stage import Stage
from ui.tooltip import Tooltip, TooltipMixin
from ui.ui_helpers import (
    _handle_autofocus_ui,
    _handle_ui_update_for_axis,
)
from ui.vertical_control import VerticalControl
from ui.zstack import ZStack


# current.json is written at clean shutdown (on_stop). A hard kill or crash
# would otherwise leave no runtime-state snapshot in a tech-support bundle,
# so it is also flushed on this interval while the app runs.
_CURRENT_JSON_FLUSH_INTERVAL_S = 300


class LumaViewProApp(TooltipMixin, App):
    """Main application class -- build, start, stop, tooltips."""

    kv_file = 'ui/lumaviewpro.kv'

    # kv mirrors of the session's run-state derivations, published by
    # the ONE run-state listener below (worker-side truth lives on the
    # session; a kv binding cannot read it directly). run_lockout
    # carries the session derivation of the same name (a run or its
    # post-run file drain); recording_active carries a LIVE manual
    # recording; controls_locked is the full-surface lock. Never add a
    # second per-site flag -- bind to these.
    run_lockout = BooleanProperty(False)
    recording_active = BooleanProperty(False)
    controls_locked = BooleanProperty(False)

    def publish_run_state(self, dt=0):
        """Write the three kv mirrors from the session derivations.

        One closure writes all three, in fail-safe order: Kivy
        dispatches bindings synchronously inside each setattr, so a
        handler observing a torn pair must see OVER-locked, never
        under-locked -- the tightening property writes first on lock,
        last on unlock.
        """
        session = ctx.session
        run_lockout = session.run_lockout
        recording = session.exclusive_activity == 'recording' and session.recording_capturing
        locked = session.controls_locked
        if locked:
            self.controls_locked = True
            self.run_lockout = run_lockout
            self.recording_active = recording
        else:
            self.run_lockout = run_lockout
            self.recording_active = recording
            self.controls_locked = False

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

        # The ONE run-state listener: session transitions (claim
        # grant/release, file-drain exit, scope rebind) schedule a
        # single main-thread closure that re-reads the derivations at
        # fire time and writes the three kv mirrors. Level-read at
        # fire time means out-of-order delivery degrades to bounded
        # staleness, never a permanently wrong publish; registration
        # itself level-syncs the mirrors to current truth.
        ctx.session.add_run_state_listener(lambda: Clock.schedule_once(self.publish_run_state, 0))

        # Slow idle refresh (1Hz) for display elements that may change without motion
        # (e.g., labware selection, stage offset changes)
        Clock.schedule_interval(ctx.stage.draw_labware, 1.0)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 1.0)
        Clock.schedule_once(functools.partial(ctx.image_settings.set_expanded_layer, 'BF'), 0.2)

        # Periodic current.json snapshot so a hard kill / crash still leaves a
        # recent runtime-state file for tech-support bundles (it is otherwise
        # only written at clean shutdown).
        Clock.schedule_interval(self._flush_current_json, _CURRENT_JSON_FLUSH_INTERVAL_S)

        # Stage B1: publish Kivy-side layer state to scope_display_thread at
        # 30Hz. The thread cannot read Kivy widget attrs from a non-UI
        # thread (executors must stay GUI-agnostic). This callback reads
        # get_active_layer_config() + engineering-mode open-layer and
        # pushes them onto the thread; the thread reads under _config_lock
        # at each frame start. Staleness is bounded by 33ms (one tick).
        def _publish_layer_config(dt):
            if ctx is None or ctx.scope_display_thread is None:
                return
            active_layer = None
            active_layer_config = None
            open_layer = None
            try:
                from modules.config_ui_getters import get_active_layer_config

                active_layer, active_layer_config = get_active_layer_config()
            except Exception as e:
                logger.debug(
                    '[LVP Main  ] _publish_layer_config: '
                    'get_active_layer_config failed; using defaults this '
                    'tick: %s: %s',
                    type(e).__name__,
                    e,
                )
            # During a protocol scan, override the accordion-derived
            # active layer with the currently-executing step's color
            # so the live preview's false-color tint matches the
            # firing LED. Without this, every preview frame is tinted
            # with whatever layer the user had open when they pressed
            # Run (the source of the "every image looks Red" reports
            # on simulator where the camera doesn't vary intensity
            # by channel).
            runner = getattr(ctx, 'sequenced_capture_runner', None)
            if runner is not None:
                try:
                    curr_color = runner.current_step_color()
                except Exception:
                    curr_color = None
                if curr_color is not None:
                    try:
                        from modules.config_helpers import get_layer_configs

                        cfgs = get_layer_configs(ctx.settings, [curr_color])
                        active_layer = curr_color
                        active_layer_config = cfgs.get(curr_color, active_layer_config)
                    except Exception as e:
                        logger.debug(
                            '[LVP Main  ] _publish_layer_config: '
                            'get_layer_configs(%s) failed; sticking with '
                            'accordion-derived layer: %s: %s',
                            curr_color,
                            type(e).__name__,
                            e,
                        )
            if ctx.engineering_mode and ctx.image_settings is not None:
                import modules.common_utils as _cu

                for layer in _cu.get_layers():
                    accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
                    if not accordion_item_obj.collapse:
                        open_layer = layer
                        break
            ctx.scope_display_thread.update_layer_config(
                active_layer,
                active_layer_config,
                open_layer,
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
                            f'[INIT      ] {layer:6s}: gain={ls.get("gain_db", "?"):>6}, '
                            f'exp={ls.get("exp_ms", "?"):>8}ms, ill={ls.get("ill_ma", "?"):>6}mA, '
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

            # On startup stay where homing left the stage -- do NOT drive to
            # protocol step 1. A loaded protocol's steps stay available (the
            # steps table is rendered by load_protocol) and the user navigates
            # to them explicitly. Apply the default BF layer's saved settings,
            # identical to the no-protocol path, so there is no stage motion
            # either way.
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

        # The session owns the metrics lifecycle (it holds the injected
        # KivyClockScheduler and restarts metrics on the new scope at
        # every reconnect); settings.profiling.metrics_interval_s
        # overrides the default 3600 s cadence -- set 30-60 s for
        # short-soak leak hunts (gen2_depth + handle/thread counts are
        # usable signals at sub-minute granularity; hourly is fine for
        # production).
        ctx.session.start_metrics()

        # The atexit emergency-shutdown hook is registered in Lumascope.__init__
        # so every Lumascope user gets the same safety net automatically.

        # Capture the settled startup footprint a few seconds after on_start so
        # the camera/UI have finished initializing. No-op unless the memory
        # profiler is enabled.
        from lib import memory_profile

        Clock.schedule_once(lambda dt: memory_profile.snapshot('cold_start_done'), 5.0)

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

        # Every executor handle lives on ctx; if build() never completed there
        # is nothing to tear down. Stop order is preserved exactly (consumer
        # threads before the lanes they consume) -- only the source of each
        # handle changed from a module global to ctx.
        if ctx is None:
            logger.info('[LVP Main  ] Threads shut down.')
            return

        if ctx.autofocus_thread is not None:
            ctx.autofocus_thread.stop(timeout=2.0)

        if ctx.scope_display_thread is not None:
            ctx.scope_display_thread.stop()

        if ctx.protocol_thread is not None:
            ctx.protocol_thread.stop(timeout=2.0)

        if ctx.io_executor is not None:
            ctx.io_executor.shutdown(wait=False)

        if ctx.camera_executor is not None:
            ctx.camera_executor.shutdown(wait=False)

        if ctx.file_io_executor is not None:
            ctx.file_io_executor.shutdown(wait=False)

        if ctx.worker_pool is not None:
            ctx.worker_pool.shutdown(wait=False)

        logger.info('[LVP Main  ] Threads shut down.')

    def build(self) -> 'MainDisplay':
        """Kivy lifecycle hook: construct the widget tree and return the root widget."""
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        # Every entry point emits the same launch fingerprint so REST API, headless
        # test runner, and CLI tools all get identical environment lines.
        from lvp_logger import log_environment_banner
        from modules.app_environment import camera_sdk_probe

        # Pass the install directory (script_path), not the per-user data
        # directory (source_path). version.txt and .git_archival.txt ship next
        # to the executable; on an installed build source_path points at the
        # Documents data folder, which has no version.txt, so the banner would
        # report Built/Branch/CommitGUID as "unknown". On a source/dev run the
        # two paths are identical.
        log_environment_banner(script_path, version, camera_sdk_probe())

        # Lock was claimed in __main__ before any Kivy import (issue #559);
        # keep a strong ref here so the bound socket survives for the
        # lifetime of the app. The instance is a module-global below;
        # take it onto self for symmetry with the rest of the app state.
        self._lvp_lock = _lvp_lock_singleton

        # video_creation_controls, stitch_controls, zprojection_controls, and
        # composite_gen_controls register themselves on ctx in their __init__.
        global Window
        global ctx

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

        from ui.notification_popup import notification_popup_bridge

        notifications.add_listener(
            notification_popup_bridge,
            # NOTICE (not WARNING) so user-facing status of long unattended
            # operations crosses the bridge; INFO stays log-only.
            min_severity=Severity.DEBUG if ENGINEERING_MODE else Severity.NOTICE,
        )

        try:
            from kivy.core.window import Window

            # Window min size uses SDL point coordinates -- do NOT use dp()
            Window.minimum_width = 1024
            Window.minimum_height = 600
            Window.bind(on_resize=self._on_resize)
            Window.bind(on_request_close=self.on_request_close)
            # Window-level lifecycle bindings -- log every event the OS /
            # window manager / global keyboard shortcut can deliver
            # outside any registered widget. Without these, a shutdown
            # triggered by Alt-F4 / window-X / OS-close leaves the GUI
            # log silent and post-mortem cannot name the trigger.
            Window.bind(on_close=self._on_window_close)
            Window.bind(on_keyboard=self._on_window_keyboard)
            # SDL2-only events: minimize / maximize / restore. Bind under
            # try/except so non-SDL2 window providers (rare) don't crash.
            # Handler names are constructed dynamically here, so
            # _on_window_minimize/_maximize/_restore have no static
            # references -- dead-code scanners must not flag them.
            for _evt in ('on_minimize', 'on_maximize', 'on_restore'):
                try:
                    Window.bind(**{_evt: getattr(self, f'_on_window_{_evt[3:]}')})
                except Exception as _e:
                    logger.debug(f'[LVP Main  ] Window.bind({_evt}) failed: {_e}')
            Window.bind(focus=self._on_window_focus)
            # camera_type='auto' lets the registry pick by priority (Pylon -> IDS
            # -> FX2). The legacy settings['camera_type'] field is vestigial.
            lumaview = MainDisplay(camera_type='auto', simulate=simulate_mode)
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
        # Clock.schedule_once is passed as the UI dispatcher so executors can post
        # callbacks to the Kivy main thread without importing Kivy themselves.
        from kivy.clock import Clock

        from modules.scheduler import KivyClockScheduler

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

        # A crash in a pre-engine release can strand a multi-GB recording
        # scratch in the live folder; sweep it before anything records.
        from modules.manual_recording import sweep_recording_scratch

        sweep_recording_scratch(settings['live_folder'])

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

        # GUI-independent scope session; persisted to ctx.session so
        # other methods read off ctx. The session composes the ONE
        # sequenced-capture engine from the injected executors, AF
        # pair, and protocol thread -- and its run-state derivations
        # need the file-drain fact, so the FILE executor handle rides
        # the injection list too. Constructing the session also services
        # the scope (executor registration, bundle, source path) -- the
        # session owns scope bring-up so a reconnect-built scope gets
        # the identical servicing through set_scope. The bundle is
        # handed over for that servicing only; this host keeps teardown
        # (shutdown_threads), which is why owns_executors stays False.
        scope_session = ScopeSession(
            settings=settings,
            scope=lumaview.scope,
            io_executor=io_executor,
            camera_executor=camera_executor,
            wellplate_loader=wellplate_loader,
            coordinate_transformer=coordinate_transformer,
            objective_helper=objective_helper,
            source_path=source_path,
            executor_bundle=executor_bundle,
            file_io_executor=file_io_executor,
            protocol_thread=protocol_thread,
            autofocus_runner=autofocus_runner,
            autofocus_thread=autofocus_thread,
            z_ui_update_func=_handle_autofocus_ui,
            metrics_scheduler=KivyClockScheduler(Clock),
        )
        sequenced_capture_runner = scope_session.sequenced_capture_runner

        # Create AppContext -- central service registry
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

        # A plugin exception reaching the Kivy event loop must not take
        # down the host: plugins are separately versioned (and may not be
        # written by us), and one bad button handler in one crashed the
        # whole app at the bench. Exceptions whose traceback enters a
        # loaded plugin's package are contained -- logged, recorded in
        # the plugin's health ledger, surfaced as a popup -- and the app
        # continues. Anything NOT attributable to plugin code re-raises,
        # so core defects keep their full crash post-mortem.
        from kivy.base import ExceptionHandler, ExceptionManager

        class _PluginCrashGuard(ExceptionHandler):
            def handle_exception(self, inst):
                plugin_name = None
                try:
                    plugin_name = ctx.plugins.attribute_exception(sys.exc_info()[2])
                except Exception as e:
                    logger.debug(f'[Plugins ] crash attribution failed: {e}')
                if plugin_name is None:
                    return ExceptionManager.RAISE
                logger.exception(
                    f'[Plugins ] contained a crash from plugin {plugin_name!r}: {inst}'
                )
                try:
                    ctx.plugins.ui.record_runtime_error(plugin_name, 'ui_event', inst)
                except Exception as e:
                    logger.debug(f'[Plugins ] runtime-error record failed: {e}')
                try:
                    from modules.notification_center import notifications

                    notifications.error(
                        'Plugins',
                        'Plugin Error',
                        f'The "{plugin_name}" plugin hit an error and the action '
                        'was cancelled. The rest of the application is '
                        'unaffected. See the log for details.',
                    )
                except Exception as e:
                    logger.debug(f'[Plugins ] plugin-error popup failed: {e}')
                return ExceptionManager.PASS

        ExceptionManager.add_handler(_PluginCrashGuard())

        # Attach UI-namespace plugin mounts now that the widget tree
        # exists. Each registered (name, mount, builder) tuple is
        # invoked here; builder() returns the Kivy widget which is
        # added to the named mount point.
        motionsettings_accordion = ctx.motion_settings.ids['motionsettings_accordion_id']
        for plugin_name, mount_point, builder in ctx.plugins.ui.mounts():
            if mount_point == 'left_sidebar.accordion':
                try:
                    plugin_item = builder()
                    # The accordion itself no longer carries the exclusive-
                    # activity lock (its bind would swallow the run/stop
                    # toggles' abort clicks); runtime-mounted items inherit
                    # the lock explicitly so plugin tabs grey out like the
                    # built-in regions.
                    plugin_item.disabled = bool(self.controls_locked)
                    self.bind(
                        controls_locked=lambda _app, value, item=plugin_item: setattr(
                            item, 'disabled', value
                        )
                    )
                    motionsettings_accordion.add_widget(plugin_item)
                    logger.info(f'[LVP Main  ] Mounted {plugin_name} at {mount_point}')
                except Exception as e:
                    logger.error(
                        f'[LVP Main  ] {plugin_name} mount failed: {e}',
                        exc_info=True,
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

        # CPU profiling -- enabled via cprofile_enabled in settings.json,
        # independent of debug_mode (which otherwise silently started a
        # whole-app profiler). On exit, dumps a .profile file to logs/cprofile/
        # viewable with: pip install snakeviz && snakeviz <file>.profile
        if settings.get('cprofile_enabled', False):
            global profiling_helper
            profiling_helper = profiling_utils.ProfilingHelper()
            profiling_helper.enable()
            logger.info(
                '[LVP Main  ] cProfile enabled (cprofile_enabled=true) -- will dump on exit'
            )

        return lumaview

    def _on_resize(self, window, w: int, h: int) -> None:
        """Kivy on_resize hook: re-evaluate motion / image settings layout."""
        Clock.schedule_once(ctx.motion_settings.check_settings, 0.1)
        Clock.schedule_once(ctx.image_settings.check_settings, 0.1)

    def _on_window_close(self, *args) -> None:
        """Kivy on_close hook -- the window is closing for real."""
        gui_logger.window_event('close')

    def _on_window_keyboard(self, window, key: int, scancode: int, codepoint, modifier) -> bool:
        """Kivy on_keyboard hook -- log non-widget-consumed key events.

        Only logs keys of forensic interest: Escape (Kivy
        exit_on_escape is disabled, so this never closes the app, but
        the keypress is worth seeing), Alt-F4, and Ctrl-Q. Routine
        typing in widgets doesn't pollute the GUI log. Returns False
        so Kivy's regular handler chain continues unchanged.
        """
        mods = tuple(sorted(modifier)) if modifier else ()
        # 27 = Escape; 282-296 = F1-F15 (285 = F4, 293 = F12); 113 = Q
        if key == 27:
            gui_logger.window_event('keyboard', f'key=Escape mods={mods}')
        elif key == 285 and 'alt' in mods:
            gui_logger.window_event('keyboard', f'key=Alt-F4 mods={mods}')
        elif key == 113 and 'ctrl' in mods:
            gui_logger.window_event('keyboard', f'key=Ctrl-Q mods={mods}')
        return False

    def _on_window_minimize(self, *args) -> None:
        gui_logger.window_event('minimize')

    def _on_window_maximize(self, *args) -> None:
        gui_logger.window_event('maximize')

    def _on_window_restore(self, *args) -> None:
        gui_logger.window_event('restore')

    def _on_window_focus(self, window, focused: bool) -> None:
        """Kivy Window.focus property change -- log focus gain / loss.

        Focus changes are routine (clicking between LVP and another
        app); the value comes from knowing whether the user moved
        attention elsewhere just before a shutdown / freeze.
        """
        gui_logger.window_event('focus', f'focused={focused}')

    def on_request_close(self, *args) -> bool:
        """Kivy on_request_close hook: show a confirmation popup if a protocol is running.

        Returns:
            True to prevent window close (popup shown); False to allow close.
        """
        protocol_running = ctx.session.run_lockout
        # Crash-forensics: log the close request to BOTH the main log
        # (so post-mortem can correlate against the shutdown sequence)
        # and the GUI interactions log (so the gui-log timeline names
        # the trigger). Without this line, an X-button / Alt-F4 close
        # produces a silent shutdown -- the gap that prompted this hook.
        logger.info(f'[LVP Main  ] on_request_close fired; protocol_running={protocol_running}')
        gui_logger.window_event('close-requested', f'protocol_running={protocol_running}')
        if protocol_running:
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

        recording = ctx.session.manual_recording
        runner = ctx.sequenced_capture_runner
        protocol_tail_busy = runner is not None and runner.video_drain_busy
        if recording.is_busy or protocol_tail_busy:
            # Queued video frames -- a manual recording's, or a finished
            # run's video-step tail -- are still being written to their
            # final artifacts. A silent block reads as a hang and a
            # silent close eats the tail of the recording, so the close
            # shows drain progress with one explicit discard escape.
            logger.info('[LVP Main  ] Close requested during video drain; showing progress')
            Clock.schedule_once(lambda dt: self._close_with_drain_progress())
            return True  # Prevent window from closing

        # No exclusive activity - allow normal close
        return False

    def _close_with_drain_progress(self) -> None:
        """PR flow for closing mid-drain: stop, show drain progress, exit
        when the finish lands (or on explicit discard). Covers both drain
        sources -- the manual recording and a run's video-step tail."""
        from ui.notification_popup import show_blocking_progress_popup

        recording = ctx.session.manual_recording
        runner = ctx.sequenced_capture_runner
        recording.stop()

        def _busy() -> bool:
            return recording.is_busy or (runner is not None and runner.video_drain_busy)

        def _pending() -> int:
            tail = runner.video_pending_writes if runner is not None else 0
            return recording.pending_writes + tail

        def _discard(*_a):
            recording.discard_pending()
            if runner is not None:
                runner.discard_video_pending()

        popup, set_message = show_blocking_progress_popup(
            title='Finishing Video Writes',
            message='Finishing video writes...',
            action_text='Discard Remaining Frames',
            on_action=_discard,
        )

        def _watch(dt):
            if _busy():
                set_message(f'Finishing video writes -- {_pending()} frames remaining.')
                return
            Clock.unschedule(self._drain_close_watch)
            self._drain_close_watch = None
            popup.dismiss()
            self.stop()

        self._drain_close_watch = Clock.schedule_interval(_watch, 0.2)

    def _flush_current_json(self, dt: float) -> None:
        """Periodic current.json snapshot (Clock interval callback).

        Mirrors the on_stop save so a hard kill / crash still leaves recent
        runtime state on disk. The hardware-presence gate inside
        save_settings prevents overwriting real per-channel values with
        slider defaults when no hardware was connected this session.
        """
        try:
            ctx.motion_settings.ids['microscope_settings_id'].save_settings('./data/current.json')
        except Exception:
            logger.exception('[LVP Main  ] periodic current.json flush failed')

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
            ctx.session.stop_metrics()
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] metrics stop failed during shutdown: {e}')

        ctx.motion_settings.ids['protocol_settings_id'].cancel_all_protocols()
        # The abort above only signals; the hardware teardown (LED off,
        # camera restore, return-to-position) runs on the protocol thread.
        # Shutdown tears the executors down right after this block, so wait
        # -- bounded -- for that cleanup to finish before proceeding. Per
        # PERFORMANCE_BUDGETS.md row shutdown_protocol_cleanup_wait_s. The
        # leds_off drain below is the belt-and-suspenders if it times out.
        try:
            if ctx.sequenced_capture_runner is not None and not (
                ctx.sequenced_capture_runner.wait_for_run_idle(timeout_s=30.0)
            ):
                logger.warning(
                    '[LVP Main  ] protocol cleanup still in flight after 30 s '
                    'shutdown wait; proceeding with teardown anyway'
                )
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] shutdown cleanup wait failed: {e}')

        # Stop the scope-display thread BEFORE the executor cascade --
        # otherwise the FPS-paced loop submits work against a half-
        # disconnected scope and floods the shutdown log.
        try:
            if ctx.scope_display is not None:
                ctx.scope_display.stop()
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] scope_display stop during shutdown failed: {e}')

        # Drain LEDs through io_executor BEFORE shutdown_threads tears
        # the executor down. Routing the leds_off through the same
        # serial-bus serialization lane as the rest of LED writes
        # prevents the ad-hoc-Thread-vs-in-flight-IOTask race that
        # existed when this call was a bare daemon Thread. The 2 s
        # fut.result timeout preserves the prior MainThread-doesn't-
        # block-on-slow-serial behavior.
        logger.info('[LVP Main  ] lumaview.scope.illumination.leds_off()')
        try:
            from modules.sequential_io_executor import IOTask

            fut = (
                ctx.io_executor.put(
                    IOTask(action=lumaview.scope.illumination._leds_off_impl),
                    return_future=True,
                )
                if ctx.io_executor is not None
                else None
            )
            if fut is not None:
                try:
                    fut.result(timeout=2.0)
                except TimeoutError:
                    # The io_executor can still be draining protocol-abort
                    # cleanup at exit, and that cleanup turns LEDs off
                    # itself -- this expiry does not mean LEDs were left
                    # on. Log the cached channel state so the post-mortem
                    # answers that question directly.
                    states = lumaview.scope.illumination.get_led_states()
                    lit = sorted(c for c, s in states.items() if s.get('enabled'))
                    state_text = (
                        'channels still ON: ' + ', '.join(lit) if lit else 'all channels OFF'
                    )
                    logger.warning(
                        f'[LVP Main  ] shutdown leds_off still queued on '
                        f'io_executor after 2.0s; LED state cache reports '
                        f'{state_text}'
                    )
                except Exception as e:
                    logger.warning(f'[LVP Main  ] shutdown leds_off failed: {e}')
            else:
                logger.warning('[LVP Main  ] io_executor unavailable for shutdown leds_off')
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] leds_off submission failed during shutdown: {e}')

        self.shutdown_threads()

        # Considered removing this stop_motion() call, since disconnect() below calls
        # stop_motion() as its first step; rejected because shutdown_threads ran BEFORE
        # disconnect, and any in-flight motion should stop before we tear down the
        # executors that own the move callbacks. Revisit if shutdown_threads and disconnect
        # are consolidated into one teardown.
        lumaview.scope.motion.stop_motion()

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
