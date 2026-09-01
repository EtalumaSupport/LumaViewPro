# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import threading
from dataclasses import dataclass, field

from modules.plugins import PluginRegistry

# Module-level singleton -- set by LumaViewProApp.build() after construction.
# Extracted modules import this module and access `app_context.ctx` to avoid
# circular imports with lumaviewpro.py.
ctx = None

# Early registrations -- widgets that register during KV tree construction
# before ctx exists. Copied to ctx fields when ctx is created.
_early_lock = threading.Lock()
_early_registrations = {}


def register_early(name, value):
    """Register a widget before ctx is created.  Thread-safe.

    During Kivy's KV tree construction, widgets are created before
    AppContext. This stores the registration and applies it to ctx
    once ctx exists.
    """
    with _early_lock:
        if ctx is not None:
            setattr(ctx, name, value)
        else:
            _early_registrations[name] = value


def apply_early_registrations():
    """Copy early registrations to ctx. Called after ctx is created."""
    for name, value in _early_registrations.items():
        setattr(ctx, name, value)
    _early_registrations.clear()


@dataclass
class AppContext:
    """Central service registry for LumaViewPro.

    Holds references to all shared services, executors, and helpers.
    Created in LumaViewProApp.build() after all services are initialized.
    Replaces scattered global variables and prepares for ids[] chain elimination.
    """

    # Hardware
    scope: object = None  # Lumascope instance
    lumaview: object = None  # MainDisplay widget

    # Core services
    session: object = None  # ScopeSession
    sequenced_capture_runner: object = None
    autofocus_runner: object = None
    version: str = ''
    source_path: str = ''

    # Executors + long-lived threads
    io_executor: object = None
    camera_executor: object = None
    protocol_thread: object = None
    file_io_executor: object = None
    scope_display_thread: object = None
    autofocus_thread: object = None
    worker_pool: object = None

    # Helpers
    wellplate_loader: object = None
    coordinate_transformer: object = None
    objective_helper: object = None

    # UI components (set after widget tree builds in build())
    viewer: object = None  # Viewer widget (update_shader, black, white)
    scope_display: object = None  # ScopeDisplay widget
    image_settings: object = None  # ImageSettings widget
    motion_settings: object = None  # MotionSettings widget
    stage: object = None  # Stage widget
    cell_count_content: object = None
    graphing_controls: object = None
    stitch_controls: object = None
    composite_gen_controls: object = None
    video_creation_controls: object = None
    zprojection_controls: object = None
    quick_enhance_controls: object = None
    # No metrics_logger field: it mirrored scope.metrics_logger and went
    # stale at every reconnect; the session owns the metrics lifecycle.
    ui_listener_bridge: object = None  # UIListenerBridge (LVP-A-6)

    # Plugin platform: registry + entry-points discovery
    plugins: PluginRegistry = field(default_factory=PluginRegistry)

    # State
    protocol: object = None  # Protocol instance (canonical owner, not UI)
    engineering_mode: bool = False
    no_engineering: bool = (
        False  # --no-engineering CLI flag; suppresses engineering plugin auto-enable
    )
    show_tooltips: bool = False
    live_histo_setting: bool = False
    last_save_folder: str = None
    disable_homing: bool = False
    simulate_mode: bool = False
    max_exposure: float = 0.0
    max_gain: float = 0.0
    live_view_fps: int = 30
    focus_round: int = 0

    # Initialization flag (replaces _app_initializing)
    _ready: bool = False

    @property
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, value):
        self._ready = value

    @property
    def initializing(self):
        """Backward-compatible check (replaces _app_initializing)."""
        return not self._ready

    # --- Settings access: the session owns the store, this forwards ---
    # Neither the dict nor its lock is a field here. The session is built
    # before this context and is handed the one settings dict; a field
    # holding either half would let a caller construct a second store, or
    # a lock guarding only one alias of the dict, and nothing would report
    # the mismatch. Forwarding makes both unconstructible. Worker threads
    # take get_settings_snapshot() at task entry and read from that;
    # writers off the host thread use update_settings().

    def _require_session(self):
        if self.session is None:
            raise AttributeError(
                'AppContext has no session, so there is no settings store to '
                'reach. Construct the session first and pass it in.'
            )
        return self.session

    @property
    def settings(self):
        """The live settings dict, owned by the session."""
        return self._require_session().settings

    @property
    def settings_lock(self):
        """The lock guarding the settings dict, owned by the session."""
        return self._require_session().settings_lock

    def get_settings_snapshot(self):
        """A deep copy of settings taken under the lock."""
        return self._require_session().get_settings_snapshot()

    def update_settings(self, key, value):
        """Write one top-level settings key under the lock."""
        self._require_session().update_settings(key, value)
