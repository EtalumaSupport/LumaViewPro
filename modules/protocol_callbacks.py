# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Typed callback contract for protocol execution.

Replaces the magic-string ``callbacks`` dict with a typed dataclass.
Callers can still pass plain dicts — use ``ProtocolCallbacks.from_dict()``
to convert.  The executor and sub-modules use attribute access instead of
``'key' in dict`` checks.

Extracted from ``sequenced_capture_executor.py`` during the
protocol-decomposition refactor.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional


@dataclasses.dataclass
class ProtocolCallbacks:
    """All callbacks used during protocol execution.

    Every field defaults to ``None``.  The executor checks
    ``if self.cb.<field>:`` before invoking, matching the old
    ``if 'key' in callbacks:`` pattern with zero risk of KeyError.
    """

    # --- Run lifecycle ---
    protocol_iterate_pre: Optional[Callable] = None   # (n_scans, scan_count) -> None
    run_scan_pre: Optional[Callable] = None            # () -> None
    scan_iterate_post: Optional[Callable] = None       # () -> None
    run_complete: Optional[Callable] = None             # (protocol=...) -> None
    files_complete: Optional[Callable] = None           # (protocol=...) -> None

    # --- Autofocus ---
    autofocus_in_progress: Optional[Callable] = None   # () -> None
    autofocus_completed: Optional[Callable] = None     # () -> None  (passed as 'complete' to AF executor)
    autofocus_complete: Optional[Callable] = None      # () -> None  (UI notification)
    reset_autofocus_btns: Optional[Callable] = None    # () -> None
    restore_autofocus_state: Optional[Callable] = None # (layer=, value=) -> None

    # --- Motion / position ---
    move_position: Optional[Callable] = None           # (axis: str) -> None
    go_to_step: Optional[Callable] = None              # (**kwargs) -> None
    update_step_number: Optional[Callable] = None      # (step: int) -> None

    # --- LED state ---
    leds_off: Optional[Callable] = None                # () -> None
    led_state: Optional[Callable] = None               # (layer=, enabled=) -> None

    # --- Video / title bar ---
    set_recording_title: Optional[Callable] = None     # (progress=...) -> None
    set_writing_title: Optional[Callable] = None       # (progress=...) -> None
    reset_title: Optional[Callable] = None             # () -> None

    # --- Live UI (set by callers, forwarded as-is) ---
    update_scope_display: Optional[Callable] = None    # () -> None
    pause_live_ui: Optional[Callable] = None           # () -> None
    resume_live_ui: Optional[Callable] = None          # () -> None

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
