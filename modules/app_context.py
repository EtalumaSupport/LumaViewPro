# Copyright Etaluma, Inc.
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

    # Core services
    settings: dict = field(default_factory=dict)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    session: object = None             # ScopeSession

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

    # State
    protocol_running: object = None    # threading.Event
    engineering_mode: bool = False

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
