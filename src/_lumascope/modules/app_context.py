# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _lumascope.modules.lumascope_api import Lumascope
    from _lumascope.modules.scope_display_thread import ScopeDisplayThread
    from _lumascope.modules.scope_session import ScopeSession, SettingsLock
    from _lumascope.modules.sequential_io_executor import SequentialIOExecutor


# Module-level singleton -- set by LumaViewProApp.build() after construction.
# Extracted modules import this module and access `app_context.ctx` to avoid
# circular imports with lumaviewpro.py.
ctx: 'AppContext | None' = None

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

    Holds the session, whose scope, lanes and settings it forwards, and the
    GUI's widgets and display preferences.
    Created in LumaViewProApp.build() after all services are initialized.
    Replaces scattered global variables and prepares for ids[] chain elimination.
    """

    lumaview: object = None  # MainDisplay widget

    # The session, which holds the scope and every lane; the context holds
    # no copy of either (the forwarders below).
    session: 'ScopeSession | None' = None
    version: str = ''
    source_path: str = ''

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
    ui_listener_bridge: object = None  # the GUI's subscriber to scope state events

    # No plugins and no engineering flag: both are the session's
    # (ctx.session.plugins, ctx.session.engineering_mode), so every host
    # sees the plugins and the mode the GUI sees.

    # State
    show_tooltips: bool = False
    live_histo_setting: bool = False
    live_view_fps: int = 30

    # Initialization flag (replaces _app_initializing)
    _ready: bool = False

    @classmethod
    def over_session(cls, session: 'ScopeSession', **fields: object) -> 'AppContext':
        """The context over ``session``: its scope and lanes are the session's.

        The context holds the session and no copy of anything it holds: the
        scope and the lanes are read off it as they are asked for, so a host
        and a test build the same context. ``fields`` are the rest (the
        widgets, the version, the display preferences).
        """
        return cls(session=session, **fields)

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

    # --- The session's handles: the session owns them, this forwards ---
    # Neither the scope nor a lane is a field here, for the reason the
    # settings below are not: a field would be a copy, taken when the
    # context was built, that a caller could replace or that could outlive
    # the object it was copied from, and nothing would report the mismatch.

    def _require_session(self) -> 'ScopeSession':
        if self.session is None:
            raise AttributeError(
                'AppContext has no session, so there is no scope, lane or '
                'settings store to reach. Construct the session first and pass it in.'
            )
        return self.session

    @property
    def scope(self) -> 'Lumascope':
        """The session's scope."""
        return self._require_session().scope

    @property
    def io_executor(self) -> 'SequentialIOExecutor':
        """The session's IO lane."""
        return self._require_session().io_executor

    @property
    def camera_executor(self) -> 'SequentialIOExecutor':
        """The session's CAMERA lane."""
        return self._require_session().camera_executor

    @property
    def scope_display_thread(self) -> 'ScopeDisplayThread':
        """The session's display thread."""
        return self._require_session().executor_bundle.scope_display_thread

    @property
    def worker_pool(self) -> 'SequentialIOExecutor':
        """The session's worker pool."""
        return self._require_session().executor_bundle.worker_pool

    # --- Settings access: the session owns the store, this forwards ---
    # Neither the dict nor its lock is a field here. The session is built
    # before this context and is handed the one settings dict; a field
    # holding either half would let a caller construct a second store, or
    # a lock guarding only one alias of the dict, and nothing would report
    # the mismatch. Forwarding makes both unconstructible. Worker threads
    # take settings_snapshot() at task entry and read from that;
    # writers off the host thread use update_settings().

    @property
    def settings(self) -> Mapping:
        """The live settings, read-only, owned by the session."""
        return self._require_session().settings

    @property
    def settings_lock(self) -> 'SettingsLock':
        """The lock guarding the settings dict, owned by the session."""
        return self._require_session().settings_lock

    def settings_snapshot(self) -> dict:
        """A deep copy of settings taken under the lock."""
        return self._require_session().settings_snapshot()

    def update_settings(self, path: str, value: object) -> None:
        """Write one setting by its dotted path; ``ScopeSession.update_settings``."""
        self._require_session().update_settings(path, value)

    def set_live_folder(self, folder: str) -> None:
        """Make ``folder`` the live folder; ``ScopeSession.set_live_folder``."""
        self._require_session().set_live_folder(folder)

    def set_protocol_filepath(self, file_path: str) -> None:
        """Remember the protocol to open at the next start; ``ScopeSession.set_protocol_filepath``."""
        self._require_session().set_protocol_filepath(file_path)
