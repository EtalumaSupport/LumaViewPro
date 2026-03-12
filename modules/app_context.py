# Copyright Etaluma, Inc.
import copy
import threading
from dataclasses import dataclass, field

# Module-level singleton — set by LumaViewProApp.build() after construction.
# Extracted modules import this module and access `app_context.ctx` to avoid
# circular imports with lumaviewpro.py.
ctx = None


@dataclass
class AppContext:
    """Central service registry for LumaViewPro.

    Holds references to all shared services, executors, and helpers.
    Created in LumaViewProApp.build() after all services are initialized.
    Replaces scattered global variables and prepares for ids[] chain elimination.
    """

    # Hardware
    scope: object = None               # Lumascope instance
    lumaview: object = None            # MainDisplay widget

    # Core services
    settings: dict = field(default_factory=dict)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    session: object = None             # ScopeSession
    sequenced_capture_executor: object = None
    autofocus_executor: object = None
    version: str = ""
    source_path: str = ""

    # Executors
    io_executor: object = None
    camera_executor: object = None
    protocol_executor: object = None
    file_io_executor: object = None
    autofocus_thread_executor: object = None
    scope_display_thread_executor: object = None
    reset_executor: object = None
    temp_ij_executor: object = None

    # Helpers
    wellplate_loader: object = None
    coordinate_transformer: object = None
    objective_helper: object = None

    # UI components (set after widget tree builds in build())
    viewer: object = None              # Viewer widget (update_shader, black, white)
    scope_display: object = None       # ScopeDisplay widget
    image_settings: object = None      # ImageSettings widget
    motion_settings: object = None     # MotionSettings widget
    stage: object = None               # Stage widget
    cell_count_content: object = None
    graphing_controls: object = None
    stitch_controls: object = None
    composite_gen_controls: object = None
    video_creation_controls: object = None
    zprojection_controls: object = None
    ij_helper: object = None

    # State
    protocol_running: object = None    # threading.Event
    engineering_mode: bool = False
    show_tooltips: bool = False
    live_histo_setting: bool = False
    last_save_folder: str = None
    disable_homing: bool = False
    simulate_mode: bool = False
    max_exposure: float = 0.0
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

    # --- Thread-safe settings access ---
    # The `settings` dict is a module-level mutable global accessed ~395 times
    # across 27+ files. Worker threads should call get_settings_snapshot() at
    # task entry to obtain an immutable copy, avoiding races with the GUI thread.
    # GUI-thread code may continue to access `settings` directly for now.

    def get_settings_snapshot(self):
        """Return a deep copy of settings under the lock.

        Worker threads should call this once at task entry and read from
        the returned snapshot rather than touching the live settings dict.
        """
        with self.settings_lock:
            return copy.deepcopy(self.settings)

    def update_settings(self, key, value):
        """Write a top-level key in settings under the lock.

        Use this for any cross-thread write to settings so the lock is
        always acquired consistently.
        """
        with self.settings_lock:
            self.settings[key] = value
