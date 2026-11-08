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

# This checkout's own API, whatever lumascope.pth the interpreter holds; the
# link is not followed, so a link farm stays its own launch folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# The IDS camera stack must initialize while the process is still nearly
# empty: its image-processing DLL fails initialization once the full DLL
# population (matplotlib->numpy, Kivy/SDL, cv2, pylon) is resident, so the
# preload precedes every heavy import.
from _lumascope.modules.app_environment import preload_camera_sdks

preload_camera_sdks()

import concurrent.futures
import enum
import pathlib
import threading
import typing

import matplotlib

matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict

if __name__ == '__main__':
    # LumaViewPro owns this process, so it starts the logging first, the
    # bundle on the root logger, before anything below can write a record.
    from _lumascope.lvp_logger import start_program_logging

    start_program_logging()

    disable_homing = False
    simulate_mode = '--simulate' in sys.argv
    if simulate_mode:
        sys.argv.remove('--simulate')
    no_engineering = '--no-engineering' in sys.argv
    if no_engineering:
        sys.argv.remove('--no-engineering')
    from _lumascope.modules.app_environment import take_sim_launch_value

    def _take_stall(flag):
        try:
            value = take_sim_launch_value(sys.argv, flag, simulate=simulate_mode)
            if value is None:
                return None
            from _lumascope.drivers.simulated_camera import SimulatedStall

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
    # --sim-refused-camera-write=NODE,ANSWER[,COUNT] has the simulated camera
    # refuse its next COUNT writes of NODE (1 when not given) from bring-up
    # on, so a camera that does not take a setting at bring-up is shown
    # without hardware. The simulated camera's alone: an FX2 refuses it.
    sim_refused_camera_writes = ()
    try:
        _refused = take_sim_launch_value(
            sys.argv, 'sim-refused-camera-write', simulate=simulate_mode
        )
        if _refused is not None:
            from _lumascope.modules.lumascope_api.simulation import (
                CameraWriteAnswer,
                CameraWriteNode,
                SimulatedCameraWriteRefusal,
            )

            _node, _answer, *_count = _refused.split(',')
            if len(_count) > 1:
                raise ValueError(f'{_refused!r} is not NODE,ANSWER[,COUNT]')
            sim_refused_camera_writes = (
                SimulatedCameraWriteRefusal(
                    CameraWriteNode(_node),
                    CameraWriteAnswer(_answer),
                    int(_count[0]) if _count else 1,
                ),
            )
    except ValueError as _refused_error:
        sys.exit(f'--sim-refused-camera-write=NODE,ANSWER[,COUNT]: {_refused_error}')
    # --sim-camera=absent brings the simulated scope up with its camera
    # unplugged, so a launch with no camera is shown without hardware.
    try:
        _camera = take_sim_launch_value(sys.argv, 'sim-camera', simulate=simulate_mode)
        if _camera not in (None, 'absent'):
            raise ValueError(f'{_camera!r} is not absent')
    except ValueError as _camera_error:
        sys.exit(f'--sim-camera=absent: {_camera_error}')
    # The flags above are the simulated scope the session is built as; a
    # real scope is given none.
    simulated_scope = None
    if simulate_mode:
        from _lumascope.modules.lumascope_api.simulation import SimulatedScope

        simulated_scope = SimulatedScope(
            refused_camera_writes=sim_refused_camera_writes,
            camera_stall=sim_camera_stall,
            focal_spread_um=sim_focal_spread_um,
            file_stall=sim_file_stall,
            camera_present=_camera is None,
        )
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
    from _lumascope.modules.app_environment import init_environment

    _env = init_environment()
    source_path = _env.source_path
    version = _env.version
    windows_machine = _env.windows_machine
    num_cores = _env.num_cores

    PROTOCOL_DATA_DIR_NAME = 'ProtocolData'

    ############################################################################
    # ---------------------Module Imports---------------------------------------#
    ############################################################################

    from _lumascope.lvp_logger import debug, install_crash_hooks, log_dir, logger

    # LumaViewPro's crash policy: an uncaught exception is recorded in the
    # log, before settings, Kivy, the plugins or the app can raise one.
    install_crash_hooks()

    DEBUG_MODE = debug

    # The installer logs to the user TEMP folder, which no support bundle
    # collects and Windows eventually sweeps -- so an install that failed
    # to replace a file cannot be told apart from an application defect.
    from _lumascope.modules.app_environment import capture_installer_logs

    capture_installer_logs(log_dir)

    print(f'LumaViewPro {version}')

    if DEBUG_MODE:
        logger.info('[LVP Main  ] Debug mode is enabled.')

    # A settings or installation file that cannot be used stops the launch by
    # escaping: the crash hook logs it, and the packaged build's bootloader
    # shows its message. Never a SystemExit here, which the bootloader treats
    # as a deliberate stop and shows nothing for.
    from _lumascope.modules.settings_init import load_lvp_settings

    load_lvp_settings(logger, source_path)

    from _lumascope.modules.settings_init import settings

    import _lumascope.modules.app_context as app_context
    import _lumascope.modules.common_utils as common_utils
    import _lumascope.modules.lvp_lock as lvp_lock
    from _lumascope.modules.app_context import AppContext
    from _lumascope.modules.scope_session import ScopeSession

    from _lumascope.modules.path_utils import app_runtime

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

    # The executor handles are the session's: ctx forwards each one to it
    # (ctx.<name>), and holds no copy. The bundle itself rides into the
    # session at construction (which registers it on the scope), so no
    # build()->on_start() handoff global exists anymore.
    ctx = None

else:
    # Subprocess/worker compatibility -- Kivy not available
    from _lumascope.modules.subprocess_stubs import (
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

from _lumascope.lib.build_files import build_file
from _lumascope.modules import gui_logger, live_work
from _lumascope.modules.ui_queue import schedule_ui, schedule_ui_latest
from _lumascope.modules.app_config import (
    load_mode as _load_mode,
)
from _lumascope.modules.exceptions import (
    CatalogueNameRefusedError,
    ObjectiveUnknownError,
    ScopeModelUnknownError,
)
from _lumascope.modules.settings_init import fall_back_to_template

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
from ui.channel_control import ChannelControl
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
from ui.bring_up_view import BringUpView
from ui.stage import Stage
from ui.tooltip import Tooltip, TooltipMixin
from ui.ui_helpers import (
    UI_HOLD_TICK_S,
    LoggedAccordionItem,
    _handle_ui_update_for_axis,
    draw_shared_run_displays,
    report_ui_thread_hold,
    run_reported,
    run_unasked,
    work_item_text,
)

# Imported for kv: <MainDisplay> builds a HomingBanner by name.
from ui.homing_banner import HomingBanner
from ui.vertical_control import VerticalControl
from ui.zstack import ZStack


class _Phase(enum.Enum):
    """Where the window is in its bring-up; written only on the Kivy thread."""

    # The scope is coming up on the bring-up thread; no session exists here.
    BRINGING_UP = 'bringing up'
    # A close landed during bring-up: the session went to the close, never to the window.
    HANDED_TO_THE_CLOSE = 'handed to the close'
    # The control surface is built over the session.
    FITTED = 'fitted'


class LumaViewProApp(TooltipMixin, App):
    """Main application class -- build, start, stop, tooltips."""

    kv_file = build_file('ui/lumaviewpro.kv')

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

    # The REST server served beside the window (``_lumascope.rest.server.Server``), or
    # None while none has been started; every close ends the session
    # through it.
    _rest_server = None

    # Set first in every close, on the Kivy thread: what follows the startup
    # motion is refused once it is. The session's own closing is set later,
    # on the close's thread, so it cannot answer a tick that runs between.
    _closing_begun = False

    # Bring-up's one outcome: the session the bring-up thread brought up, or
    # what it raised. Set once, on that thread; the hand-over, the close and
    # a bring-up popup's remedy all take the session from it, so nothing else
    # holds it before the window's context does.
    _bring_up: concurrent.futures.Future | None = None
    _phase = _Phase.BRINGING_UP
    # What the window shows until the fit, written from bring-up's steps.
    _bring_up_view: BringUpView | None = None
    # The close's popup (popup, set_message, set_discard_shown) while a close
    # that landed during bring-up waits for it; None otherwise.
    _bring_up_close = None

    def _on_run_state(self) -> None:
        """The window's one run-state listener: draw the edge on the UI thread.

        Session transitions (claim grant/release, file-drain exit, scope
        rebind), from any thread, post one state that re-reads the Session
        at fire time and writes the two records, so a held-up UI thread
        keeps one draw waiting. Level-read at
        fire time means out-of-order delivery degrades to bounded staleness,
        never a permanently wrong publish; registration itself level-syncs
        the records to current truth. The draw is contained: a raise out of a
        clock callback closes the application. A member, not a lambda, so the
        close can remove it.
        """
        schedule_ui_latest(
            (self, 'run_state'), lambda dt: run_unasked(self.publish_run_state, 'RUN_STATE')
        )

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
        gui_logger.display('HOMING_BANNER', self.live_work.has(live_work.WorkKind.HOME))
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
        """Kivy lifecycle hook: fires after build() and before the main loop runs; starts bring-up.

        The scope is brought up on a daemon thread, so the window draws and
        stays live meanwhile: daemon, so a crash on this thread is not held
        open until bring-up returns, with the instance lock still bound;
        every exit that must wait for it waits on ``_bring_up``.
        """
        Clock.schedule_interval(report_ui_thread_hold, UI_HOLD_TICK_S)
        self._bring_up = concurrent.futures.Future()
        threading.Thread(target=self._bring_up_on_its_thread, name='bring-up', daemon=True).start()
        self._bring_up.add_done_callback(lambda done: schedule_ui(lambda dt: self._hand_over(done)))

        logger.info('[LVP Main  ] LumaViewProApp.on_start()')

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
        mid-layout and shot the black placeholder. A scope with no camera
        connected has no frame to show, so none is owed.

        While the objective question is showing, the load is owed by the
        walk's answer, not by bring-up: a walk of that question is the
        one that answers it, so a popup up is bring-up waiting on the walk.
        """
        from _lumascope.modules.lumascope_api.bring_up import Part
        from ui.sim_walk import open_popups

        owed = []
        if not ctx.ready:
            owed.append('initialization')
        if not self._persisted_protocol_loaded and not open_popups():
            owed.append('the saved protocol load')
        if Part.CAMERA not in ctx.scope.unconnected_parts() and ctx.scope_display.frames_shown == 0:
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

    def build(self) -> 'BoxLayout':
        """Kivy lifecycle hook: set the window up and return its one root, showing the bring-up view.

        No session exists yet: ``on_start`` brings the scope up on a thread of
        its own, and the fit step (``_fit``) puts the control surface in the
        root once it is up. The root never changes, so a popup opened during
        bring-up stays above the control surface too.
        """
        logger.info('[LVP Main  ] LumaViewProApp.build()', extra={'force_error': True})

        # Lock was claimed in __main__ before any Kivy import (issue #559);
        # keep a strong ref here so the bound socket survives for the
        # lifetime of the app. The instance is a module-global below;
        # take it onto self for symmetry with the rest of the app state.
        self._lvp_lock = _lvp_lock_singleton

        global Window

        self.icon = build_file('data/icons/icon.png')

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

        from kivy.core.window import Window

        # Window min size uses SDL point coordinates -- do NOT use dp()
        Window.minimum_width = 1024
        Window.minimum_height = 600
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

        from _lumascope.modules.ui_queue import UiDispatcher

        ScopeSession.set_ui_dispatcher(
            UiDispatcher(schedule=Clock.schedule_once, thread=threading.main_thread())
        )
        # Half a second after the last resize, move, zoom or pan
        # (``_geometry_changed``), so a drag records where it settled, not
        # every step. Built before the main loop, the first that delivers one; the
        # moves, resizes and the viewer are bound to it in the fit step.
        self._geometry_settled = Clock.create_trigger(self._log_geometry, 0.5)

        self._bring_up_view = BringUpView(splash=build_file('data/icons/etaluma_splash.png'))
        root = BoxLayout()
        root.add_widget(self._bring_up_view)
        return root

    def _bring_up_on_its_thread(self) -> None:
        """The bring-up thread's one job: bring the session up as the GUI and set the outcome on ``_bring_up``.

        The thread decides nothing: what it raised is the future's, raised
        again by the hand-over on the Kivy thread.
        """
        from _lumascope.modules.caller import Caller, calling

        try:
            with calling(Caller.GUI):
                session = self._bring_up_the_session()
        except BaseException as raised:
            self._bring_up.set_exception(raised)
        else:
            self._bring_up.set_result(session)

    def _bring_up_the_session(self) -> 'ScopeSession':
        """Compose the session, on the shipped template if the stored settings cannot configure the scope.

        Runs on the bring-up thread. Each step bring-up reports is shown on
        the bring-up view, the newest one only when the Kivy thread is late.
        """
        # Read before the factory, which a plugin may then turn on; the
        # fallback's second compose starts from the same mode.
        engineering_mode = _load_mode(settings)

        # The popups are one subscriber to the session's outcomes, given to
        # the factory below so it hears bring-up: the factory registers it
        # before it builds the scope, whose boards and camera report what
        # failed to come up while they connect. The bridge opens what the
        # API says is shown and decides nothing more.
        from ui.notification_popup import notification_popup_bridge

        def on_progress(percent: float, step: str | None) -> None:
            schedule_ui_latest(
                (self, 'bring_up_step'), lambda dt: self._show_bring_up_step(percent, step)
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
        def _compose(from_settings: dict) -> ScopeSession:
            return ScopeSession.create(
                settings=from_settings,
                source_path=source_path,
                simulate=simulate_mode,
                warn_pre_release=False,
                engineering_mode=engineering_mode,
                no_engineering=no_engineering,
                simulated=simulated_scope,
                outcome_listener=notification_popup_bridge,
                settings_file=ScopeSession.user_settings_file(source_path),
                on_progress=on_progress,
            )

        # A stored value the scope refuses to be configured from -- a model
        # the installation's catalogue does not list, an objective it does
        # not hold -- reaches here as that refusal, and there is nothing
        # above the bring-up to catch it, so without this the app does not
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
            return _compose(settings)
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
            return _compose(settings)

    def _show_bring_up_step(self, percent: float, step: str | None) -> None:
        """Show bring-up's step on the bring-up view, and in the close's popup while a close waits on it."""
        self._bring_up_view.show_step(percent, step)
        if self._bring_up_close is not None:
            _popup, set_message, _set_discard_shown = self._bring_up_close
            set_message(self._closing_once_up())

    def _closing_once_up(self) -> str:
        """The close's message while it waits for bring-up: what bring-up is doing."""
        return f'Closing LumaViewPro once the scope is up.\n{self._bring_up_view.step}'

    def _hand_over(self, done: concurrent.futures.Future) -> None:
        """Hand the session bring-up made to the close, if one began while it ran, else to the fit; once, on the Kivy thread.

        A raise from bring-up is raised here, so it ends the process through
        the crash hook, as a raise from ``build()`` did. ``_closing_begun`` is
        written and read only on this thread, so a close lands either before
        this (it takes the session here) or after it (the fit has run, and
        the window's own close runs).
        """
        session = done.result()
        if self._closing_begun:
            self._phase = _Phase.HANDED_TO_THE_CLOSE
            popup, set_message, set_discard_shown = self._bring_up_close
            self._run_the_close(session, popup, set_message, set_discard_shown)
            return
        self._phase = _Phase.FITTED
        self._fit(session)

    def with_the_session(self, use: typing.Callable[['ScopeSession'], None]) -> None:
        """Hand ``use`` the session on the Kivy thread: once bring-up has brought it up, or at once if it has.

        For what is offered before the window's context exists, a bring-up
        popup's remedy. Never called when bring-up raised: that ends the
        process (``_hand_over``).
        """

        def _brought_up(done: concurrent.futures.Future) -> None:
            if done.exception() is None:
                schedule_ui(lambda dt: use(done.result()))

        self._bring_up.add_done_callback(_brought_up)

    def _fit(self, session: 'ScopeSession') -> None:
        """Build the control surface over the session and put it in the window's root; then start it.

        On the Kivy thread, in one tick, once bring-up has handed the session
        over: the widget tree, the window's context and its wiring, the
        plugins, then what the window starts once the scope is up. The
        widgets' first-tick ``_init_ui``s run in the next tick and find the
        context set.
        """
        # video_creation_controls, stitch_controls, zprojection_controls, and
        # composite_gen_controls register themselves on ctx in their __init__.
        global ctx

        # AppContext binds these two as kwargs below; declared as locals here
        # so the kwargs don't NameError at runtime.
        live_histo_setting = False
        last_save_folder = None
        stage = Stage()

        self.lock = session.controls_lock
        self.live_work = session.live_work
        self.control = session.control
        lumaview = MainDisplay(scope=session.scope)
        cell_count_method = session.post_processing.default_cell_count_method()
        cell_count_content = CellCountControls(
            method=cell_count_method,
            size_scale=session.post_processing.cell_count_size_scale(cell_count_method),
        )
        graphing_controls = GraphingControls()

        # A crash in a pre-engine release can strand a multi-GB recording
        # scratch in the live folder; sweep it before anything records.
        from _lumascope.modules.manual_recording import sweep_recording_scratch

        sweep_recording_scratch(settings['live_folder'])

        # Create AppContext -- central service registry over the session.
        ctx = AppContext.over_session(
            session,
            lumaview=lumaview,
            version=version,
            source_path=source_path,
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
        # A zoom, a pan, Fit and 1:1 all change the viewer's transform; a
        # frame of another shape changes the image's place inside the view.
        ctx.viewer.bind(transform=self._geometry_changed)
        ctx.scope_display.bind(texture_size=self._geometry_changed)
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
        from _lumascope.modules.plugins import UIMountPoint

        motionsettings_accordion = ctx.motion_settings.ids['motionsettings_accordion_id']
        for plugin_name, mount_point, builder in ctx.session.plugins.ui.mounts():
            if mount_point == UIMountPoint.LEFT_SIDEBAR_ACCORDION:
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

        # NotificationCenter -> UI popup bridge was registered at the
        # top of build(), BEFORE the session factory / Lumascope() /
        # hardware init, so any early hardware errors surface as popups
        # instead of being logged-only.

        # The bring-up view goes; the root stays, so the popups stay above.
        # Laid out now, in this tick, as the Window sized the surface when it
        # was the root: its accordions lay themselves out on their first
        # frame, and at the widget's default size they find no room.
        self.root.clear_widgets()
        self.root.add_widget(lumaview)
        self.root.do_layout()

        # Bound now that the widgets they read exist.
        from kivy.core.window import Window

        Window.bind(on_resize=self._on_resize)
        Window.bind(on_move=self._geometry_changed)

        # A window that opens at its final size sends no resize, so its
        # geometry is recorded from here.
        self._geometry_changed()
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

        # Publish the shown channel to scope_display_thread at 30Hz, in
        # engineering mode, for the image stats. The Session's shown channel
        # follows a run's steps as they light, so the stats follow the firing
        # LED. The thread reads it under _config_lock at each frame start;
        # staleness is bounded by 33ms (one tick).
        def _publish_channel_config(dt):
            if ctx is None or ctx.scope_display_thread is None:
                return
            ctx.scope_display_thread.update_channel_config(
                ctx.session.shown_channel.channel if ctx.session.engineering_mode else None,
            )

        Clock.schedule_interval(_publish_channel_config, 1.0 / 30)

        # Clear app initialization flag and apply settings for the default opened channel
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
                for channel in common_utils.get_channels():
                    ls = settings[channel]
                    logger.info(
                        f'[INIT      ] {channel:6s}: gain={ls["gain_db"]:>6}, '
                        f'exp={ls["exposure_ms"]:>8}ms, ill={ls["illumination_ma"]:>6}mA, '
                        f'af={ls["autofocus"]}, acquire={ls["acquire"]}'
                    )

            # On startup stay where homing left the stage -- do NOT drive to
            # protocol step 1. A loaded protocol's steps stay available (the
            # steps table is rendered by load_protocol) and the user navigates
            # to them explicitly. Nothing is shown here either: bring-up has
            # already shown BF and put its settings on the camera, skipping a
            # missing camera it has already reported, so showing it again here
            # writes nothing and repeats that report.

        if sim_walk is not None:
            from ui.sim_walk import SimWalk

            _walk_path, _walk_steps = sim_walk
            # Held on the app: the Clock keeps only weak references to the
            # walk's bound methods. A shot lands where it names, under the walk file's
            # folder; the picture of a stop goes beside the log.
            self._sim_walk = SimWalk(
                _walk_steps,
                source=str(_walk_path),
                bring_up_owes=self._bring_up_owes,
                in_flight=ctx.session.in_flight_named,
                simulation=ctx.scope.simulation,
                shot_dir=_walk_path.parent,
                stop_dir=pathlib.Path(log_dir),
            )
            self._sim_walk.start()

        # MetricsLogger owns the executor watchdog, system metrics, and camera-temp
        # logging so adding a new periodic metric only requires editing one module.

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
            # the shown channel's settings are applied by the Session's bring-up.

            # The session owns the metrics lifecycle (it owns the session
            # scheduler, and logs the environment fingerprint ahead of the
            # first tick). Cadence is hourly in production and 60 s in
            # engineering mode; settings.profiling.metrics_interval_s
            # overrides both (gen2_depth + handle/thread counts are the
            # signals worth sub-minute granularity on a short leak hunt).
            ctx.session.enable_metrics(display=ctx.scope_display)

            # Served after the metrics, as the headless launcher serves after
            # its own. A port already taken is reported, and the window runs
            # on without a server.
            run_unasked(self._start_the_rest_server, 'START_REST_SERVER')

            # The atexit emergency-shutdown hook is registered in Lumascope.__init__
            # so every Lumascope user gets the same safety net automatically.

        def startup_motion_settled(done: concurrent.futures.Future) -> None:
            from _lumascope.modules.exceptions import LaneCallStillRunningError, SessionClosingError

            # The window's close began, in this tick or an earlier one.
            if self._closing_begun:
                return
            # The app is closing: the close found the motion unstarted, began
            # before it took the scope, or closed its lane while it ran.
            exception = done.exception()
            if isinstance(exception, SessionClosingError) or (
                isinstance(exception, LaneCallStillRunningError)
                and exception.reason == 'lane_closed'
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
        started.add_done_callback(lambda done: schedule_ui(lambda dt: startup_motion_settled(done)))

    def _on_resize(self, window, w: int, h: int) -> None:
        """Kivy on_resize hook: re-evaluate motion / image settings layout."""
        Clock.schedule_once(ctx.motion_settings.check_settings, 0.1)
        Clock.schedule_once(ctx.image_settings.check_settings, 0.1)
        self._geometry_changed()

    def _geometry_changed(self, *args) -> None:
        """Restart the half-second wait before the window's and the live image's geometry is recorded."""
        # A pending trigger ignores a second call, so a drag would record
        # every half second; cancelling first restarts the wait.
        self._geometry_settled.cancel()
        self._geometry_settled()

    def _log_geometry(self, *args) -> None:
        """Record where the window is and where the live image is drawn in it.

        The image's rectangle is the contain-fit image inside the live view,
        carried through the viewer's zoom and pan to window pixels, measured
        from the window's top left (``gui_logger.window_geometry``, ``gui_logger.view``).
        """
        from kivy.core.window import Window
        from kivy.metrics import Metrics

        display = ctx.scope_display
        gui_logger.window_geometry(
            (int(Window.left), int(Window.top)),
            (int(Window.width), int(Window.height)),
            Metrics.density,
            (int(display.width), int(display.height)),
        )
        image_w, image_h = display.norm_image_size
        left, bottom = display.to_window(
            display.center_x - image_w / 2, display.center_y - image_h / 2
        )
        right, top = display.to_window(
            display.center_x + image_w / 2, display.center_y + image_h / 2
        )
        gui_logger.view(
            ctx.viewer.scale,
            (round(left), round(Window.height - top), round(right - left), round(top - bottom)),
        )

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
        if self._phase is not _Phase.FITTED:
            logger.info(
                f'[LVP Main  ] on_request_close fired during bring-up; '
                f'step={self._bring_up_view.step!r}'
            )
            gui_logger.window_event('close-requested', 'bringing up')
            self._close_during_bring_up()
            return True

        session = ctx.session
        work = session.live_work.work
        locked_by = session.controls_lock.named
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
            Clock.schedule_once(lambda dt: self._close_the_session(session))
            return True

        Clock.schedule_once(
            lambda dt: show_confirmation_popup(
                title='Confirm Exit',
                message=(
                    'LumaViewPro is still doing:\n'
                    + '\n'.join(f'- {work_item_text(item)}' for item in work)
                    + '\n\nExiting stops a running protocol or recording, keeping what it '
                    'captured, and lets the rest finish before LumaViewPro closes.'
                    '\n\nAre you sure you want to exit?'
                ),
                confirm_text='Confirm Exit',
                cancel_text='Cancel',
                on_confirm=lambda: self._close_the_session(session),
            )
        )
        return True

    def _close_during_bring_up(self) -> None:
        """Begin a close while the scope comes up: the window's part now, the session's once bring-up hands it over.

        The popup says what bring-up is doing; the hand-over runs the
        session's close on it (``_hand_over``). Its Discard is shown only by
        the close's watch, once the session exists.
        """
        if self._closing_begun:
            logger.info('[LVP Main  ] close already in progress; ignoring the request')
            return
        from ui.notification_popup import show_blocking_progress_popup

        self._prepare_the_close()
        self._bring_up_close = show_blocking_progress_popup(
            title='Closing',
            message=self._closing_once_up(),
            action_text='Discard Remaining Video Frames',
            on_action=lambda: self._bring_up.result().discard_close_drain(),
            action_shown=False,
        )

    def _close_the_session(self, session: 'ScopeSession') -> None:
        """Close ``session`` on a thread of its own, showing what it waits for; then stop.

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
            on_action=session.discard_close_drain,
            action_shown=any(item.discardable for item in session.live_work.work),
        )
        self._run_the_close(session, popup, set_message, set_discard_shown)

    def _run_the_close(
        self, session: 'ScopeSession', popup, set_message, set_discard_shown
    ) -> None:
        """Run ``session``'s close on its own thread, keeping the close's popup current; stop the app when it returns."""
        self._close_thread = threading.Thread(
            target=self._end_the_session, args=(session,), name='session-close', daemon=True
        )
        self._close_thread.start()

        def _watch(dt):
            if self._close_thread.is_alive():
                work = session.live_work.work
                message = (
                    'Finishing before LumaViewPro closes:\n'
                    + '\n'.join(work_item_text(item) for item in work)
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
        """The host's part of a close, on the Kivy thread, before the session's own.

        Before the fit there is no window surface, server or plugin to
        release: a close during bring-up mutes the notifications, as every
        close does, and nothing more.
        """
        self._closing_begun = True
        if self._phase is not _Phase.FITTED:
            self._mute_notifications()
            return

        # A client's take_control landing after the take-back below would
        # hold the scope again and refuse the popup's Discard, so the server
        # refuses every call first.
        if self._rest_server is not None:
            self._rest_server.refuse_calls()

        # A held control session is taken back first: the close's popup
        # offers its Discard before the session's own close has begun, and
        # the session would refuse a Discard that is not its holder's.
        run_unasked(ctx.session.take_back_control, 'CLOSE_TAKE_BACK_CONTROL')

        self._mute_notifications()

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

    def _mute_notifications(self) -> None:
        """Stop the notification popups for the rest of the process; the log still has every outcome."""
        # Suppress notification-listener dispatch during shutdown so the user
        # doesn't see 30+ error toasts as queued IO tasks fail against
        # disconnecting hardware. Log lines still fire for post-mortem.
        try:
            from _lumascope.modules.notification_center import notifications

            notifications.set_shutting_down(True)
        except Exception as e:  # grain: ignore NAKED_EXCEPT
            logger.warning(f'[LVP Main  ] Failed to suppress notifications on shutdown: {e}')

    def _start_the_rest_server(self) -> None:
        """Serve the API beside the window, on its own session, unless ``rest_api.enabled`` is false.

        On ``rest_api.address`` and ``rest_api.port`` (shipped this computer
        only), on a thread and loop of its own; ``python -m lumascope.rest`` is the
        headless way. Where it serves is logged, and shown in Advanced
        Settings.
        Only false leaves it off: any other value is the server's to judge,
        so a mistyped one is reported rather than read as off.
        """
        if ctx.session.get_setting('rest_api.enabled') is False:
            return
        from _lumascope.rest.server import serve_beside

        session = ctx.session
        self._rest_server = serve_beside(
            session, shut_down=lambda: self._shut_the_session_down(session)
        )
        logger.info(f'[LVP Main  ] REST server serving {self._rest_server.reach}')

    def rest_reach(self) -> str:
        """What the window shows of the REST server: where it serves and whom it admits, or that it does not serve."""
        if self._rest_server is not None:
            return f'Serving {self._rest_server.reach}'
        if ctx.session.get_setting('rest_api.enabled') is False:
            return 'Off'
        # Its start was refused, and reported then.
        return 'Not started: see the log'

    def _end_the_session(self, session: 'ScopeSession') -> None:
        """The session's close: inside the REST server's close while one serves, else directly.

        Inside the server's, its clients are answered and its stream carries
        what the close reports; the server runs ``_shut_the_session_down``.
        """
        server = self._rest_server
        if server is None or not server.close():
            self._shut_the_session_down(session)

    def _shut_the_session_down(self, session: 'ScopeSession') -> None:
        """The session's close: it stops a live run and recording, finishes the rest, then tears down."""
        logger.info('[LVP Main  ] session.shutdown()')
        from _lumascope.modules.caller import Caller, calling
        from _lumascope.modules.exceptions import ScopeDisconnectError
        from _lumascope.modules.notification_center import notifications

        try:
            # On the close's own thread, or the Kivy thread from on_stop: the GUI's close.
            with calling(Caller.GUI):
                session.shutdown()
        except ScopeDisconnectError as e:
            # Every teardown step has run; the window is closing, so the
            # record is the log (notifications are muted).
            notifications.report_outcome(e, solicited=False, category='Hardware')

    def on_stop(self) -> None:
        """Kivy lifecycle hook: close the session if no close did."""
        logger.info('[LVP Main  ] LumaViewProApp.on_stop()')

        # A bring-up still running is waited for: a platform quit during
        # bring-up closes the session it brings up, once it exists.
        session = self._bring_up.result()
        if not session.live_work.closed:
            # Reached without the window's close -- a platform quit -- or
            # after a close that raised part-way: the close runs here, on
            # this thread, and nothing draws while it waits. A close still
            # running on its own thread is waited for by this call.
            self._prepare_the_close()
            self._end_the_session(session)

        logger.info('[LVP Main  ] LumaViewProApp exiting.', extra={'force_error': True})

    # Tooltip methods provided by TooltipMixin (ui/tooltip.py)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    from _lumascope.modules.caller import Caller, calling

    # Every call the Kivy thread makes is the GUI's; the GUI stamps the
    # threads it hands work to where it hands it (ui/ui_helpers.py).
    with calling(Caller.GUI):
        LumaViewProApp().run()
