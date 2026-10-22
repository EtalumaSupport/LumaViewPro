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

import concurrent.futures
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
    from modules.app_environment import take_sim_launch_value

    def _take_stall(flag):
        try:
            value = take_sim_launch_value(sys.argv, flag, simulate=simulate_mode)
            if value is None:
                return None
            from drivers.simulated_camera import SimulatedStall

            _after, _for = value.split(',')
            return SimulatedStall(after_s=float(_after), for_s=float(_for))
        except ValueError as _stall_error:
            sys.exit(f'--{flag}=AFTER,FOR (seconds): {_stall_error}')

    # Each simulator flag is read here, beside --simulate, so a malformed
    # value stops the launch before anything starts.
    # --sim-camera-stall=AFTER,FOR stops the simulated camera's frames AFTER
    # seconds into the session, for FOR seconds, the camera staying connected
    # and grabbing: a stalled stream, shown without hardware.
    sim_camera_stall = _take_stall('sim-camera-stall')
    # --sim-file-stall=AFTER,FOR holds the file lane's worker AFTER seconds
    # into the session, for FOR seconds: a save drive that stops answering,
    # so a stalled file writer is shown without one.
    sim_file_stall = _take_stall('sim-file-stall')
    # --sim-focal-spread=UM draws the simulated sample's focus once within UM
    # of where bring-up leaves the stage, so the session needs a real
    # autofocus; without it the sample is in focus there. The simulated
    # camera's alone: a model simulated with an FX2 refuses it.
    sim_focal_spread_um = 0.0
    try:
        _spread = take_sim_launch_value(sys.argv, 'sim-focal-spread', simulate=simulate_mode)
        if _spread is not None:
            sim_focal_spread_um = float(_spread)
            if not 0 < sim_focal_spread_um < float('inf'):
                raise ValueError(f'{_spread!r} is not a distance above 0')
    except ValueError as _spread_error:
        sys.exit(f'--sim-focal-spread=UM: {_spread_error}')
    # --sim-walk=<file> performs a sim walk's steps on the GUI once bring-up
    # has finished (ui/sim_walk.py; what it waits on is _bring_up_owes
    # below). Simulator only: a walk on hardware would move a real stage.
    from ui.sim_walk_file import take_walk_flag

    try:
        sim_walk = take_walk_flag(sys.argv, simulate=simulate_mode)
    except (OSError, ValueError) as _walk_error:
        sys.exit(f'--sim-walk=<file>: {_walk_error}')

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

    from lvp_logger import debug, install_crash_hooks, log_dir, logger

    # LumaViewPro's crash policy: an uncaught exception is recorded in the
    # log, before settings, Kivy, the plugins or the app can raise one.
    install_crash_hooks()

    DEBUG_MODE = debug

    # The installer logs to the user TEMP folder, which no support bundle
    # collects and Windows eventually sweeps -- so an install that failed
    # to replace a file cannot be told apart from an application defect.
    from modules.app_environment import capture_installer_logs

    capture_installer_logs(log_dir)

    print(f'LumaViewPro {version}')

    if DEBUG_MODE:
        logger.info('[LVP Main  ] Debug mode is enabled.')

    # Start the memory profiler before the heavy module/driver/ui imports so
    # tracemalloc captures the resident baseline, not just post-startup growth.
    # No-op unless memory_profile_enabled is set in the live settings.
    from lib import memory_profile

    memory_profile.start(source_path)

    # A settings or installation file that cannot be used stops the launch by
    # escaping: the crash hook logs it, and the packaged build's bootloader
    # shows its message. Never a SystemExit here, which the bootloader treats
    # as a deliberate stop and shows nothing for.
    from modules.settings_init import load_lvp_settings

    load_lvp_settings(logger, source_path)

    from modules.settings_init import settings

    import modules.app_context as app_context
    import modules.common_utils as common_utils
    import modules.config_helpers as config_helpers
    import modules.lvp_lock as lvp_lock
    import modules.profiling_utils as profiling_utils
    from modules.app_context import AppContext
    from modules.scope_session import ScopeSession

    global profiling_helper
    profiling_helper = None

    from modules.path_utils import app_runtime

    if app_runtime().frozen:
        import pyi_splash  # type: ignore

        pyi_splash.update_text('')

    # Disable Kivy's own file logging (LVP has its own RotatingFileHandler)
    os.environ['KIVY_NO_CONSOLELOG'] = '1'
    os.environ['KIVY_NO_FILELOG'] = '1'
    # Kivy reads no command line: its parser exits on an argument it does not
    # know, and a packaged build's bootloader shows nothing for that exit.
    # LumaViewPro's own flags are read above, before Kivy loads.
    os.environ['KIVY_NO_ARGS'] = '1'

    # Single-instance lock check BEFORE any Kivy import.
    # When the check lived inside App.build(), Kivy had already
    # initialized SDL2 and opened a native window by the time the
    # loser reached sys.exit, producing duplicate visible Kivy
    # windows on double-launch. Run the check now while only
    # tkinter is alive (used above for the Python-version dialog).
    _lvp_lock_singleton = lvp_lock.take_instance_lock(source_path)
    if _lvp_lock_singleton is None:
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
            # The dialog's parent is an invisible 1 px window centred across
            # the screen, a third of the way down, not a withdrawn one: macOS
            # attaches the alert to its parent, and a withdrawn root sits at
            # (0, 0), so the alert opened in the corner, half off screen.
            # Moving a withdrawn root does not move it, and no parent at all
            # lets the alert open behind the existing LVP window.
            _root.overrideredirect(True)
            _root.geometry(
                f'1x1+{_root.winfo_screenwidth() // 2}+{_root.winfo_screenheight() // 3}'
            )
            _root.attributes('-alpha', 0.0)
            # Force the dialog to the foreground. Without this the messagebox
            # can open behind the existing LVP window and get buried, so the
            # user never sees why the second launch silently did nothing.
            _root.attributes('-topmost', True)
            _root.update()
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
    from kivy.base import stopTouchApp
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

    # Imported for its side effect: defining the class registers PickSpinner
    # with Kivy's Factory, which the kv rules name it through.
    import ui.pick_spinner

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

from modules import gui_logger, live_work
from modules.app_config import (
    load_mode as _load_mode,
)
from modules.exceptions import (
    CatalogueNameRefusedError,
    ObjectiveUnknownError,
    ScopeModelUnknownError,
)
from modules.settings_init import fall_back_to_template

# Kivy Factory imports: the classes below are referenced from ui/lumaviewpro.kv
# (and other .kv files Kivy loads at startup). Kivy's Builder.apply() resolves
# class names via Factory.get(), which only finds a class once it has been
# imported into a Python module. ruff F401 cannot see .kv references and would
# strip these as unused; the per-file ignore in pyproject.toml ([tool.ruff.lint
# .per-file-ignores] "lumaviewpro.py" lists F401) silences that. Do not remove
# any of these without first grepping ui/*.kv for the class name.
from ui.file_dialogs import FileChooseBTN, FileSaveBTN, FolderChooseBTN
from ui.histogram import Histogram
from ui.image_settings import (
    AccordionItemXyStageControl,
    ChannelDrawer,
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
    LoggedAccordionItem,
    _handle_ui_update_for_axis,
    draw_shared_run_displays,
    run_reported,
    run_unasked,
)

# Imported for kv: <MainDisplay> builds a HomingBanner by name.
from ui.homing_banner import HomingBanner
from ui.vertical_control import VerticalControl
from ui.zstack import ZStack


def _work_item_text(item) -> str:
    """One piece of the session's live work, as the close's popups show it."""
    if item.percent is not None:
        return f'{item.name} ({item.percent:.0f}%)'
    if item.left is not None:
        return f'{item.name} ({item.left} left)'
    return item.name


class LumaViewProApp(TooltipMixin, App):
    """Main application class -- build, start, stop, tooltips."""

    kv_file = 'ui/lumaviewpro.kv'

    # The saved protocol is loaded once, after the startup objective
    # question settles. A plain attribute rather than a kv property: no
    # widget binds to it, and it exists only so the several paths that can
    # ask the objective question cannot each re-load over the user's work.
    _persisted_protocol_loaded = False

    # The Session's lock record (``ControlsLock``) and its live work
    # (``LiveWork``) as of the last run-state edge, written whole by the ONE
    # run-state listener below; a kv binding cannot read the Session
    # directly. A greyed control binds ``app.lock.run_lockout`` (the run
    # surface) or ``app.lock.controls_locked`` (the whole surface) -- never
    # a per-site flag. A run control binds its own held flag instead, drawn
    # from the Session for the run that control started. rebind, so a
    # binding through a record's field re-evaluates when the record is
    # replaced. Written in build() before any widget binds them.
    lock = ObjectProperty(None, rebind=True)
    live_work = ObjectProperty(None, rebind=True)
    # The control session held (``ScopeSession.control``), or None: what the
    # remote-control banner and its Take back control button show, written
    # with the two records on every edge. Bound by nothing that locks.
    control = ObjectProperty(None, allownone=True)

    # The thread the window's close runs the session's close on, or None
    # before one has begun. Declared here so the close handler can read it
    # before any close has assigned it -- the alternative is every reader
    # defending itself with getattr, which is how one of them eventually
    # forgets.
    _close_thread = None

    # The REST server served beside the window (``rest.server.Server``), or
    # None while none has been started; every close ends the session
    # through it.
    _rest_server = None

    def _on_run_state(self) -> None:
        """The window's one run-state listener: draw the edge on the clock.

        Session transitions (claim grant/release, file-drain exit, scope
        rebind) schedule a single main-thread closure that re-reads the
        Session at fire time and writes the two records. Level-read at
        fire time means out-of-order delivery degrades to bounded staleness,
        never a permanently wrong publish; registration itself level-syncs
        the records to current truth. The draw is contained: a raise out of a
        clock callback closes the application. A member, not a lambda, so the
        close can remove it.
        """
        Clock.schedule_once(lambda dt: run_unasked(self.publish_run_state, 'RUN_STATE'), 0)

    def publish_run_state(self, dt: float = 0) -> None:
        """Write the Session's lock record and live work, then draw the edge.

        The lock is one record, so its two surfaces change in one write and
        no binding sees one without the other. The live work is written
        after it; nothing locks on it.
        """
        session = ctx.session
        self.lock = session.controls_lock
        self.live_work = session.live_work
        self.control = session.control
        gui_logger.display(
            'CONTROLS_LOCKED',
            f'{self.lock.controls_locked} by={self.lock.named} recording={self.lock.recording}',
        )
        gui_logger.display('HOMING_BANNER', self.live_work.has(live_work.HOME))
        self._draw_run_controls()
        # The objective question is withheld while an activity holds the
        # scope; this edge is the one that fires when the hold ends, so the
        # question is asked again here rather than at the next turret move.
        ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state()

    def _draw_run_controls(self) -> None:
        """Redraw each run control from its own run, on the same edge.

        A run control's button shows what the engine says of the run it
        started, and this edge is the one that fires after a run is back
        to idle -- so a run that ends on its own is drawn ended here, and
        nowhere else has to guess when. A control styles only its own
        button; what every run shares is drawn once, after them.
        """
        ctx.lumaview.draw_composite_button()
        ctx.lumaview.draw_record_button()
        ctx.motion_settings.ids['protocol_settings_id'].draw_protocol_buttons()
        ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id'].draw_zstack_button()
        ctx.motion_settings.ids['verticalcontrol_id'].draw_autofocus_button()
        draw_shared_run_displays(self.lock, self.live_work)

    def on_start(self) -> None:
        """Kivy lifecycle hook: fires after build() and before the main loop runs."""
        # Read scope through ctx so widget rebuilds (LS850 <-> LS620) don't strand it.
        lumaview = ctx.lumaview

        # The bridge subscribes the GUI to the scope's state events. It
        # lives in ui/ because every one of its handlers ends in a widget
        # write; a non-GUI host subscribes to the same events itself.
        from ui.listener_bridge import UIListenerBridge

        ctx.ui_listener_bridge = UIListenerBridge(
            scope=lumaview.scope,
            ctx=ctx,
            stage=ctx.stage,
        )
        ctx.ui_listener_bridge.register_all()

        ctx.session.add_run_state_listener(self._on_run_state)

        # Slow idle refresh (1Hz) for display elements that may change without motion
        # (e.g., labware selection, stage offset changes)
        Clock.schedule_interval(ctx.stage.draw_labware, 1.0)
        Clock.schedule_interval(ctx.motion_settings.update_xy_stage_control_gui, 1.0)

        # Publish the shown channel to scope_display_thread at 30Hz, for
        # the preview's false-colour tint and, in engineering mode, the
        # image stats. The Session's shown channel follows a run's steps as
        # they light, so the tint matches the firing LED. The thread reads it
        # under _config_lock at each frame start; staleness is bounded by
        # 33ms (one tick).
        def _publish_layer_config(dt):
            if ctx is None or ctx.scope_display_thread is None:
                return
            layer = ctx.session.shown_channel.layer
            ctx.scope_display_thread.update_layer_config(
                layer,
                layer if ctx.session.engineering_mode else None,
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

                settings = ctx.settings
                for layer in common_utils.get_layers():
                    ls = settings[layer]
                    logger.info(
                        f'[INIT      ] {layer:6s}: gain={ls["gain_db"]:>6}, '
                        f'exp={ls["exposure_ms"]:>8}ms, ill={ls["illumination_ma"]:>6}mA, '
                        f'af={ls["autofocus"]}, acquire={ls["acquire"]}'
                    )

            # Hide the composite-threshold and false-colour controls on the
            # transmitted channels, which have no colour to composite.
            #
            # This does NOT bound illumination, whatever its name suggests:
            # the transmitted slider cap is applied by the capability sync,
            # which the settings panel already runs at startup.
            try:
                ctx.image_settings.update_transmitted()
            except Exception as e:  # grain: ignore NAKED_EXCEPT
                logger.warning(f'[INIT      ] update_transmitted skipped: {e}')

            # On startup stay where homing left the stage -- do NOT drive to
            # protocol step 1. A loaded protocol's steps stay available (the
            # steps table is rendered by load_protocol) and the user navigates
            # to them explicitly. Show the Session's shown channel (BF in a
            # fresh session) at its saved settings, identical to the
            # no-protocol path, so there is no stage motion either way.
            shown = ctx.session.shown_channel.layer
            ctx.image_settings.layer_lookup(layer=shown).apply_settings()

        if sim_walk is not None:
            from ui.sim_walk import SimWalk

            _walk_path, _walk_steps = sim_walk
            # Held on the app: the Clock keeps only weak references to the
            # walk's bound methods. Pictures land beside the walk file.
            self._sim_walk = SimWalk(
                _walk_steps,
                source=str(_walk_path),
                bring_up_owes=self._bring_up_owes,
                in_flight=ctx.session.in_flight_named,
                shot_dir=_walk_path.parent,
            )
            self._sim_walk.start()

        # MetricsLogger owns the executor watchdog, system metrics, and camera-temp
        # logging so adding a new periodic metric only requires editing one module.

        logger.info('[LVP Main  ] LumaViewProApp.on_start()')

        def after_the_startup_motion() -> None:
            # Everything here followed the startup motion when it blocked
            # this thread, and still does, in the same order, with the scope
            # free: several of these do nothing while the scope is held.
            move_home_cb('ALL')
            ctx.image_settings.draw_shown_channel()
            complete_initialization(None)
            ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state(prompt=False)

            # Both startup questions may only fire once the session is up and
            # the frame has rendered; each helper owns its own deferral. The
            # settings question comes first -- while it is unresolved, every
            # save is refused and the objective prompt suppresses itself.
            self._ask_about_rejected_settings()
            self._prompt_objective_if_needed()

            # Objective and LEDs are set by scope.initialize() during load_settings();
            # The shown channel's settings are applied by complete_initialization().

            # Once-per-startup environment + dependency fingerprint. Pairs
            # with the per-tick [PDH METRICS] / [BUFFER METRICS] surface so
            # post-mortem can correlate a problem against the exact host
            # state (OS, Python, Pylon SDK, Defender state, etc.) without
            # the noise of repeating those facts every tick.
            config_helpers.log_environment_once()

            # The session owns the metrics lifecycle (it owns the session
            # scheduler and restarts metrics on the new scope at
            # every reconnect). Cadence is hourly in production and 60 s in
            # engineering mode; settings.profiling.metrics_interval_s
            # overrides both (gen2_depth + handle/thread counts are the
            # signals worth sub-minute granularity on a short leak hunt).
            ctx.session.enable_metrics(display=ctx.scope_display)

            # The atexit emergency-shutdown hook is registered in Lumascope.__init__
            # so every Lumascope user gets the same safety net automatically.

            # Capture the settled startup footprint a few seconds after the
            # start has finished, so the camera/UI have finished initializing.
            # No-op unless the memory profiler is enabled.
            from lib import memory_profile

            Clock.schedule_once(lambda dt: memory_profile.snapshot('cold_start_done'), 5.0)

        def startup_motion_settled(done: concurrent.futures.Future) -> None:
            from modules.exceptions import SessionClosingError

            # The app is closing: a shutdown dropped the motion, the close
            # began before it took the scope, or it began while it ran.
            if done.cancelled() or isinstance(
                done.exception(), (concurrent.futures.CancelledError, SessionClosingError)
            ):
                return
            live = ctx.session.live_work
            if live.closing or live.closed:
                return
            # Raised here, on this thread, where the app's one exception
            # handler takes it, as it took it from on_start when the motion
            # blocked; a reported failed home settles with None.
            done.result()
            after_the_startup_motion()

        # ScopeSession owns startup orchestration so REST API, headless tools and
        # the GUI all hit the same path. The window draws and stays live while
        # the startup motion runs; the scope is the motion's until it ends.
        from ui.ui_helpers import move_home_cb, set_title_event_text

        started = ctx.session.start_application_session(disable_homing=disable_homing)
        if not started.done():
            set_title_event_text('Homing, please wait...')
        started.add_done_callback(
            lambda done: Clock.schedule_once(lambda dt: startup_motion_settled(done), 0)
        )

        if app_runtime().frozen:
            pyi_splash.close()

    def _prompt_objective_if_needed(self) -> None:
        """Ask the objective question when the objective is unknowable.

        Clock-deferred: a popup opened before the frame has rendered
        appears under no window and is never seen. Called at startup and
        again when the provisional-settings dialog resolves -- while
        settings were provisional the question was suppressed because
        its answer could not be kept.

        The persisted protocol load is hung on the answer. Whether the
        scope can perform that protocol depends on what the turret
        carries, and on a turreted scope the slot at the current position
        is only assigned once this question is answered -- so loading
        first meant judging the protocol against a configuration that was
        about to change.
        """
        vertical_control = ctx.motion_settings.ids['verticalcontrol_id']
        Clock.schedule_once(
            lambda dt: vertical_control.prompt_if_objective_unknown(
                on_resolved=self._load_persisted_protocol_once
            ),
            0,
        )

    def _bring_up_owes(self) -> list[str]:
        """What bring-up has not finished, for a sim walk waiting to start.

        ``ctx.ready`` is a timer, 0.3 s after on_start, and nothing it names
        has finished: the saved protocol loads once the objective question
        resolves, and the display shows a frame once the display thread
        delivers one. A walk started on the timer pressed a control
        mid-layout and shot the black placeholder.

        While the objective question is showing, the load is owed by the
        walk's answer, not by bring-up: a walk of that question is the
        one that answers it, so a popup up is bring-up waiting on the walk.
        """
        from ui.sim_walk import open_popups

        owed = []
        if not ctx.ready:
            owed.append('initialization')
        if not self._persisted_protocol_loaded and not open_popups():
            owed.append('the saved protocol load')
        if ctx.scope_display.frames_shown == 0:
            owed.append('a displayed frame')
        return owed

    def _load_persisted_protocol_once(self) -> None:
        """Load the saved protocol, the first time the objective settles.

        Latched because the objective question is asked from more than one
        place: this startup path, the provisional-settings dialog
        resolving, and the turret arriving at an unassigned slot. Each is
        a legitimate reason to ask again; none is a reason to re-load the
        saved protocol over whatever the user has done since, which a
        naive continuation would do on every one of them.

        The latch is set BEFORE the load rather than after, so a load that
        raises does not leave the door open for the next question to try
        again -- the panel reports its own failure and keeps an empty
        protocol.
        """
        if self._persisted_protocol_loaded:
            return
        self._persisted_protocol_loaded = True
        ctx.motion_settings.ids['protocol_settings_id'].load_persisted_protocol()

    def _ask_about_rejected_settings(self) -> None:
        """Let the user choose what happens to a current.json we could not read.

        Confirm is the destructive branch and cancel is the safe one, which
        is not arbitrary: show_confirmation_popup treats a programmatic or
        lifecycle dismiss as cancel, so anything that closes this dialog
        without a human deciding must leave the file alone.

        Clock-deferred: a popup opened before the frame has rendered is
        painted under the app root once it attaches -- open, invisible,
        and unanswerable. A dialog, not the notification bus: the bus is
        one-way and renders a single OK button, so a question posted
        there would be acknowledged and never answered, leaving settings
        unsaveable for the rest of the session and every session after.

        A failure to present the question is itself fatal. Carrying on would
        mean running with saves silently disabled -- the user changes settings
        all session and loses every one of them at exit, with nothing said.
        """
        if not ctx.session.settings_are_provisional():
            return

        reason = ctx.session.bring_up_record().settings_set_aside.reason

        def _after_revert():
            # The retire is a file rename that can fail under a Windows
            # AV/indexer lock. The session still says provisional then, so
            # the question is put back -- nothing is saved until it is
            # resolved. Otherwise settings can be kept again, so the
            # objective question suppressed while they were provisional is
            # asked now.
            if ctx.session.settings_are_provisional():
                self._ask_about_rejected_settings()
            else:
                self._prompt_objective_if_needed()

        def _revert():
            run_reported(ctx.session.retire_rejected_settings, _after_revert, 'USE_DEFAULTS')

        def _quit():
            logger.warning('[LVP Main  ] user chose to repair settings; exiting without saving')
            self.stop()

        def _present(_dt):
            try:
                from ui.notification_popup import show_confirmation_popup

                # The reason string already leads with the file path, so the
                # message does not repeat it; the two buttons carry the
                # question, so the body only states the stakes.
                show_confirmation_popup(
                    title='Settings file could not be used',
                    message=(
                        f'{reason}\n\n'
                        'Your file has not been changed, and nothing will be '
                        'saved until you choose. "Use defaults" keeps the old '
                        'file alongside it for support.'
                    ),
                    confirm_text='Use defaults',
                    cancel_text='Quit and repair',
                    on_confirm=_revert,
                    on_cancel=_quit,
                )
            except Exception:
                logger.critical(
                    '[LVP Main  ] could not ask about the rejected settings '
                    'file; refusing to run with saving disabled',
                    exc_info=True,
                )
                self.stop()

        Clock.schedule_once(_present, 0)

    def build(self) -> 'MainDisplay':
        """Kivy lifecycle hook: construct the widget tree and return the root widget."""
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        # Every entry point emits the same launch fingerprint so REST API, headless
        # test runner, and CLI tools all get identical environment lines.
        from lvp_logger import log_environment_banner
        from modules.app_environment import camera_sdk_probe

        # Pass the install directory (script_path), not the per-user data
        # directory (source_path). version.txt and build_id.txt ship next
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

        # AppContext binds these two as kwargs below; declared as locals here
        # so the kwargs don't NameError at runtime.
        live_histo_setting = False
        last_save_folder = None

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
        ENGINEERING_MODE = _load_mode(settings)

        stage = Stage()

        # The popups are one subscriber to the session's outcomes, given to
        # the factory below so it hears bring-up: the factory registers it
        # before it builds the scope, whose boards and camera report what
        # failed to come up while they connect. The bridge opens what the
        # API says is shown and decides nothing more.
        from ui.notification_popup import notification_popup_bridge

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

        # Clock.schedule_once is the process's one UI dispatcher: the
        # executor lanes, a run's deliveries and the listener bridge post
        # callbacks to the Kivy main thread through it without importing
        # Kivy themselves. The Clock delivers on the main thread, the one
        # Kivy runs on.
        from kivy.clock import Clock

        from modules.kivy_utils import UiDispatcher

        ScopeSession.set_ui_dispatcher(
            UiDispatcher(schedule=Clock.schedule_once, thread=threading.main_thread())
        )

        # The Session composes the instrument -- the scope (the camera
        # registry picks by priority, Pylon -> IDS -> FX2, reading the
        # labware and objective catalogues), the executor topology and the autofocus
        # pair -- and brings the scope up (configure from settings,
        # then release the camera start gate) before it returns. What
        # only this host knows goes in by name. The pre-release
        # warning is gated off: the GUI ships in the same commit as
        # the API, so it has nothing to tell it and would only reach
        # the user's console. A raise inside the factory tears down
        # what it had started before it reaches here.
        def _compose(from_settings):
            return ScopeSession.create(
                settings=from_settings,
                source_path=source_path,
                simulate=simulate_mode,
                warn_pre_release=False,
                engineering_mode=ENGINEERING_MODE,
                no_engineering=no_engineering,
                sim_camera_stall=sim_camera_stall,
                sim_file_stall=sim_file_stall,
                sim_focal_spread_um=sim_focal_spread_um,
                outcome_listener=notification_popup_bridge,
                settings_file=ScopeSession.user_settings_file(source_path),
            )

        # A stored value the scope refuses to be configured from -- a model
        # the installation's catalogue does not list, an objective it does
        # not hold -- reaches here as that refusal, and there is nothing
        # above build() to catch it, so without this the app does not
        # launch at all. Come up on the shipped template instead, the same
        # recovery settings_init already runs for an unreadable
        # current.json. Only these: a program or installation defect is
        # not repaired by the template, which would hide it behind a
        # provisional session. The user's file is NOT repaired: a value we
        # cannot interpret is not a value we may overwrite. The fallback
        # writes the one log line for the rejection, carrying its cause; a
        # raise out of here is the process's crash, recorded once by the
        # crash hook.
        try:
            scope_session = _compose(settings)
        except (
            ScopeModelUnknownError,
            CatalogueNameRefusedError,
            ObjectiveUnknownError,
        ) as unusable:
            # Republishes the store IN PLACE and marks the session
            # provisional, so `settings` below is the template and every
            # save raises until the user resolves it. Reassigning the name
            # here instead would strand every other holder of this dict on
            # the rejected values.
            fall_back_to_template(logger, source_path, str(unusable))
            scope_session = _compose(settings)
        self.lock = scope_session.controls_lock
        self.live_work = scope_session.live_work
        self.control = scope_session.control
        lumaview = MainDisplay(scope=scope_session.scope)
        cell_count_method = scope_session.post_processing.default_cell_count_method()
        cell_count_content = CellCountControls(
            method=cell_count_method,
            size_scale=scope_session.post_processing.cell_count_size_scale(cell_count_method),
        )
        graphing_controls = GraphingControls()

        # A crash in a pre-engine release can strand a multi-GB recording
        # scratch in the live folder; sweep it before anything records.
        from modules.manual_recording import sweep_recording_scratch

        sweep_recording_scratch(settings['live_folder'])

        # Create AppContext -- central service registry. Every handle is
        # the session's object, read off it: one store, no second
        # construction.
        ctx = AppContext(
            scope=scope_session.scope,
            lumaview=lumaview,
            session=scope_session,
            sequenced_capture_runner=scope_session.sequenced_capture_runner,
            autofocus_runner=scope_session.autofocus_runner,
            version=version,
            source_path=source_path,
            io_executor=scope_session.io_executor,
            camera_executor=scope_session.camera_executor,
            protocol_thread=scope_session.protocol_thread,
            file_io_executor=scope_session.file_io_executor,
            post_processing_executor=scope_session.post_processing.lane,
            autofocus_thread=scope_session.autofocus_thread,
            scope_display_thread=scope_session.executor_bundle.scope_display_thread,
            worker_pool=scope_session.executor_bundle.worker_pool,
            stage=stage,
            cell_count_content=cell_count_content,
            graphing_controls=graphing_controls,
            show_tooltips=show_tooltips,
            live_histo_setting=live_histo_setting,
            last_save_folder=last_save_folder,
            live_view_fps=live_view_fps,
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
        ctx.motion_settings.ids['microscope_settings_id'].load_settings()

        # Creates and manages Tooltips
        self.init_tooltips(lumaview)

        # The session's plugins: the installed ones (the engineering plugin,
        # dev/bench-only; customer installs find nothing in the group), then
        # the built-ins. Each is handed the session.
        ctx.session.load_plugins()

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
                    plugin_name = ctx.session.plugins.attribute_exception(sys.exc_info()[2])
                except Exception as e:
                    logger.debug(f'[Plugins ] crash attribution failed: {e}')
                if plugin_name is None:
                    return ExceptionManager.RAISE
                ctx.session.plugins.record_runtime_error(plugin_name, 'ui_event', inst)
                return ExceptionManager.PASS

        ExceptionManager.add_handler(_PluginCrashGuard())

        # Attach UI-namespace plugin mounts now that the widget tree
        # exists. Each registered (name, mount, builder) tuple is
        # invoked here; builder() returns the Kivy widget which is
        # added to the named mount point.
        motionsettings_accordion = ctx.motion_settings.ids['motionsettings_accordion_id']
        for plugin_name, mount_point, builder in ctx.session.plugins.ui.mounts():
            if mount_point == 'left_sidebar.accordion':
                try:
                    plugin_item = builder()
                    # The item carries no exclusive-activity lock, as the
                    # accordion carries none: a lock on the whole item would
                    # grey out a control that must stay live during a run,
                    # such as the plugin's own Stop. A plugin locks its own
                    # controls, as the built-in regions do, from the
                    # session's run-state listeners.
                    motionsettings_accordion.add_widget(plugin_item)
                    logger.info(f'[LVP Main  ] Mounted {plugin_name} at {mount_point}')
                except Exception as e:
                    ctx.session.plugins.record_runtime_error(plugin_name, 'mount', e)

        # Enable engineering-only log files (autofocus.log, api.log).
        # Read from the session since the engineering plugin may have turned
        # its engineering mode on when it loaded.
        from lvp_logger import enable_engineering_logs

        enable_engineering_logs(ctx.session.engineering_mode)

        # NotificationCenter -> UI popup bridge was registered at the
        # top of build(), BEFORE the session factory / Lumascope() /
        # hardware init, so any early hardware errors surface as popups
        # instead of being logged-only.

        # CPU profiling -- enabled via cprofile_enabled in settings.json,
        # independent of debug_mode (which otherwise silently started a
        # whole-app profiler). On exit, dumps a .profile file to logs/cprofile/
        # viewable with: pip install snakeviz && snakeviz <file>.profile
        if settings['cprofile_enabled']:
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
        """Kivy on_request_close hook: ask, naming what the session is still doing, then close.

        Always True: the window never closes by itself. The close runs on
        its own thread (``_close_the_session``) and the app stops when it
        returns.
        """
        work = ctx.session.live_work.work
        locked_by = ctx.session.controls_lock.named
        # Crash-forensics: log the close request to BOTH the main log
        # (so post-mortem can correlate against the shutdown sequence)
        # and the GUI interactions log (so the gui-log timeline names
        # the trigger). Without this line, an X-button / Alt-F4 close
        # produces a silent shutdown -- the gap that prompted this hook.
        logger.info(
            f'[LVP Main  ] on_request_close fired; locked_by={locked_by} '
            f'live_work={[item.kind for item in work]}'
        )
        gui_logger.window_event('close-requested', f'locked_by={locked_by}')

        if self._close_thread is not None:
            # A close is already running. This is a SECOND close request --
            # a Kivy Popup is modal only for in-canvas touch, so the window's
            # X still reaches here while the progress popup is up. Logged
            # above and then ignored: a second close would only wait for the
            # first.
            logger.info('[LVP Main  ] close already in progress; ignoring the request')
            return True

        if not work:
            Clock.schedule_once(lambda dt: self._close_the_session())
            return True

        Clock.schedule_once(
            lambda dt: show_confirmation_popup(
                title='Confirm Exit',
                message=(
                    'LumaViewPro is still doing:\n'
                    + '\n'.join(f'- {_work_item_text(item)}' for item in work)
                    + '\n\nExiting stops a running protocol or recording, keeping what it '
                    'captured, and lets the rest finish before LumaViewPro closes.'
                    '\n\nAre you sure you want to exit?'
                ),
                confirm_text='Confirm Exit',
                cancel_text='Cancel',
                on_confirm=self._close_the_session,
            )
        )
        return True

    def _close_the_session(self) -> None:
        """Close the session on a thread of its own, showing what it waits for; then stop.

        The host's part runs first, here on the Kivy thread
        (``_prepare_the_close``). The session's close then runs on its own
        thread, so this one keeps drawing the progress and the Discard
        button stays live; when the close returns, the app stops. Closed
        on this thread instead, nothing would draw while it waits.
        """
        from ui.notification_popup import show_blocking_progress_popup

        if self._close_thread is not None:
            return
        self._prepare_the_close()
        popup, set_message, set_discard_shown = show_blocking_progress_popup(
            title='Closing',
            message='Closing LumaViewPro...',
            action_text='Discard Remaining Video Frames',
            on_action=ctx.session.discard_close_drain,
            action_shown=any(item.discardable for item in ctx.session.live_work.work),
        )
        self._close_thread = threading.Thread(
            target=self._end_the_session, name='session-close', daemon=True
        )
        self._close_thread.start()

        def _watch(dt):
            if self._close_thread.is_alive():
                work = ctx.session.live_work.work
                message = (
                    'Finishing before LumaViewPro closes:\n'
                    + '\n'.join(_work_item_text(item) for item in work)
                    if work
                    else 'Closing LumaViewPro...'
                )
                set_message(message)
                set_discard_shown(any(item.discardable for item in work))
                # The percent and count left are progress; the work still
                # waited on is the stage.
                gui_logger.display(
                    'CLOSE_PROGRESS',
                    message,
                    stage=', '.join(item.name for item in work) or 'Closing LumaViewPro...',
                )
                return True
            # Returning False is what stops a Kivy interval.
            popup.dismiss()
            self.stop()
            return False

        Clock.schedule_interval(_watch, 0.2)

    def stop(self, *largs) -> None:
        """End the event loop; ``run()`` then dispatches ``on_stop``, once.

        Kivy's own ``stop()`` dispatches ``on_stop`` itself, and ``run()``
        dispatches it again when the loop it ended returns, so every exit
        made from inside the loop (Confirm Exit, the drain close, quit and
        repair) would cancel and tear down twice. A window close already
        ends the loop this way.
        """
        stopTouchApp()

    def _prepare_the_close(self) -> None:
        """The host's part of a close, on the Kivy thread, before the session's own."""
        # A client's take_control landing after the take-back below would
        # hold the scope again and refuse the popup's Discard, so the server
        # refuses every call first.
        if self._rest_server is not None:
            self._rest_server.refuse_calls()

        # A held control session is taken back first: the close's popup
        # offers its Discard before the session's own close has begun, and
        # the session would refuse a Discard that is not its holder's.
        run_unasked(ctx.session.take_back_control, 'CLOSE_TAKE_BACK_CONTROL')

        # Suppress notification-listener dispatch during shutdown so the user
        # doesn't see 30+ error toasts as queued IO tasks fail against
        # disconnecting hardware. Log lines still fire for post-mortem.
        try:
            from modules.notification_center import notifications

            notifications.set_shutting_down(True)
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] Failed to suppress notifications on shutdown: {e}')

        # The window stops listening to the scope and redrawing its position
        # before the scope is torn down: the teardown's own events would
        # schedule redraws that read a scope that is going. Here and not in
        # on_stop, which a crash skips.
        ctx.ui_listener_bridge.unregister_all()
        ctx.session.remove_run_state_listener(self._on_run_state)
        Clock.unschedule(ctx.stage.draw_labware)
        Clock.unschedule(ctx.motion_settings.update_xy_stage_control_gui)

        # Plugins released first, here on the Kivy thread their widgets live
        # on: an unregister stops its own work, and their listener
        # subscriptions and file handles need to drop before hardware
        # tear-down. A plugin's own failure is logged inside, and the rest
        # still unload. The session's close finds them unloaded.
        ctx.session.unload_plugins()

    def _end_the_session(self) -> None:
        """The session's close: inside the REST server's close while one serves, else directly.

        Inside the server's, its clients are answered and its stream carries
        what the close reports; the server runs ``_shut_the_session_down``.
        """
        server = self._rest_server
        if server is None or not server.close():
            self._shut_the_session_down()

    def _shut_the_session_down(self) -> None:
        """The session's close: it stops a live run and recording, finishes the rest, then tears down."""
        logger.info('[LVP Main  ] ctx.session.shutdown()')
        from modules.caller import Caller, calling
        from modules.exceptions import ScopeDisconnectError
        from modules.notification_center import notifications

        try:
            # On the close's own thread, or the Kivy thread from on_stop: the GUI's close.
            with calling(Caller.GUI):
                ctx.session.shutdown()
        except ScopeDisconnectError as e:
            # Every teardown step has run; the window is closing, so the
            # record is the log (notifications are muted).
            notifications.report_outcome(e, solicited=False, category='Hardware')

    def on_stop(self) -> None:
        """Kivy lifecycle hook: close the session if no close did."""
        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')

        if profiling_helper is not None:
            profiling_helper.stop()

        if not ctx.session.live_work.closed:
            # Reached without the window's close -- a platform quit -- or
            # after a close that raised part-way: the close runs here, on
            # this thread, and nothing draws while it waits. A close still
            # running on its own thread is waited for by this call.
            self._prepare_the_close()
            self._end_the_session()

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})

    # Tooltip methods provided by TooltipMixin (ui/tooltip.py)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    from modules.caller import Caller, calling

    # Every call the Kivy thread makes is the GUI's; the GUI stamps the
    # threads it hands work to where it hands it (ui/ui_helpers.py).
    with calling(Caller.GUI):
        LumaViewProApp().run()
