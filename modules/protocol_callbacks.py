# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Typed callback contract for protocol execution.

Replaces the magic-string ``callbacks`` dict with a typed dataclass.
Callers can still pass plain dicts -- use ``ProtocolCallbacks.from_dict()``
to convert.  The executor and sub-modules use attribute access instead of
``'key' in dict`` checks.

Extracted from ``sequenced_capture_runner.py`` during the
protocol-decomposition refactor.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from collections.abc import Callable


@dataclasses.dataclass
class ProtocolCallbacks:
    """All callbacks used during protocol execution.

    Every field defaults to ``None``.  The executor checks
    ``if self.cb.<field>:`` before invoking, matching the old
    ``if 'key' in callbacks:`` pattern with zero risk of KeyError.
    """

    # --- Run lifecycle ---
    protocol_iterate_pre: Callable | None = None  # (n_scans, scan_count) -> None
    run_scan_pre: Callable | None = None  # () -> None
    scan_iterate_post: Callable | None = None  # () -> None
    run_complete: Callable | None = None  # (protocol=...) -> None
    files_complete: Callable | None = None  # (protocol=...) -> None

    # --- Autofocus ---
    autofocus_in_progress: Callable | None = None  # () -> None
    autofocus_complete: Callable | None = None  # () -> None  (UI notification)
    reset_autofocus_btns: Callable | None = None  # () -> None
    restore_autofocus_state: Callable | None = None  # (layer=, value=) -> None

    # --- Motion / position ---
    move_position: Callable | None = None  # (axis: str) -> None
    go_to_step: Callable | None = None  # (**kwargs) -> None
    update_step_number: Callable | None = None  # (step: int) -> None

    # --- LED state ---
    leds_off: Callable | None = None  # () -> None
    led_state: Callable | None = None  # (layer=, enabled=) -> None

    # --- Video / title bar ---
    set_recording_title: Callable | None = None  # (elapsed_sec=..., total_sec=...) -> None
    set_writing_title: Callable | None = None  # (progress=...) -> None
    reset_title: Callable | None = None  # () -> None

    # --- Live UI (set by callers, forwarded as-is) ---
    update_scope_display: Callable | None = None  # () -> None
    pause_live_ui: Callable | None = None  # () -> None
    resume_live_ui: Callable | None = None  # () -> None

    # --- UI shader / false-color state ---
    # Each protocol step calls layer_control.apply_settings() which sets
    # the OpenGL shader white_point for that layer's false-color (Red
    # tint for Red step, etc.). Without a cleanup-time restore, the
    # last step's shader stays active and tints the live preview after
    # protocol stop. Sibling category to the LED-driver-state restore
    # (the LED hardware is cleared by leds_off; this clears the UI
    # shader). Callable signature: () -> None. Re-applies the shader
    # for the currently-open accordion (or BF default if none open).
    restore_layer_shader: Callable | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ProtocolCallbacks:
        """Build from a plain dict, ignoring unknown keys."""
        if not d:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a dict (only non-None entries), for passing
        to sub-modules that still expect a plain dict (e.g. video_capture).

        Uses field iteration instead of dataclasses.asdict() because asdict()
        calls copy.deepcopy() on values, which fails on Kivy bound methods
        (EventDispatcher can't be pickled/deepcopied).
        """
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }
