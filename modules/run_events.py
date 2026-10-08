# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""What a run tells its caller as it goes, and the one delivery that tells it.

A caller of a run member passes ``events=RunEvents(...)`` with a handler for
each event it wants; the rest stay None. Every event carries its values by
value, so a handler never reads the run's own state, which a next run may
already own. The engine names no host and no widget: what a GUI does with an
event -- draw the step, hold a frame on screen, restore the shader at the
end -- is the GUI's handler, and a REST server or a script subscribes to the
same events.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from modules.kivy_utils import schedule_ui

if TYPE_CHECKING:
    from modules.protocol import Protocol
    from modules.run_outcome import RunOutcome


@dataclasses.dataclass(frozen=True)
class VideoProgress:
    """Where a video step is, as its ``video_progress`` event says.

    ``phase`` is ``'recording'`` (``elapsed_s`` of ``total_s``), ``'writing'``
    (``percent`` of the recorded frames written) or ``'ended'``, which follows
    once any recording or writing phase on every path that had one; a step
    that ends before it records sends no phase and no ``'ended'``.
    """

    phase: str
    elapsed_s: float | None = None
    total_s: float | None = None
    percent: float | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class RunEvents:
    """The handlers a run calls as it goes, each optional; a misspelt one fails here.

    Every event but ``frame_captured`` arrives through the process's UI
    dispatcher (``ScopeSession.set_ui_dispatcher``): on a GUI host on its UI
    thread, and with none set on the thread that sends it. A handler's own
    exception is reported once, as the exception it is, under the event's
    name: a refusal stays a refusal and a fault a fault. During an unattended
    run a ``step_started`` or ``frame_captured`` failure is logged, muted by
    the run's scope, and a ``run_ended`` or ``files_written`` failure is
    shown. No handler needs to guard itself.

    Attributes:
        scan_started: ``(scan_number, scans_remaining, interval)`` as a scan
            begins: the scan's 1-based number, how many scans remain with
            this one, and the protocol's period.
        scan_ended: ``(scan_number, scans_remaining, interval)`` as a scan
            ends, a stopped one included: how many remain after it.
        step_started: ``(step_idx)`` as each step begins, step 0 included,
            the index read at the step.
        frame_captured: ``(image, frames_summed, frame_significant_bits)``
            for each captured frame whose write was handed to the run's
            writer, once it was: the frame as the file gets it (a sum when
            summing), read-only. Delivered on the run's own thread, which
            waits for it, so it must not block; a GUI renders and shows it.
        video_progress: ``(progress)``, a ``VideoProgress``, as a video step
            records and writes.
        run_ended: ``(outcome, run_dir, protocol)`` once the run has let go
            of the scope and its outcome has settled: for every run at the
            release, for a composite once its merge settles, which may be
            before or after its ``files_written``. ``run_dir`` is None for a run that
            saved nothing.
        files_written: ``(run_dir, files)`` once the run's last image write
            has landed: ``files`` is ``'written'``, or ``'incomplete'`` when
            some of its images are not on disk.
    """

    scan_started: Callable[[int, int, datetime.timedelta], object] | None = None
    scan_ended: Callable[[int, int, datetime.timedelta], object] | None = None
    step_started: Callable[[int], object] | None = None
    frame_captured: Callable[[np.ndarray, int, int], object] | None = None
    video_progress: Callable[[VideoProgress], object] | None = None
    run_ended: Callable[[RunOutcome, pathlib.Path | None, Protocol], object] | None = None
    files_written: Callable[[pathlib.Path | None, str], object] | None = None


def _report(event: str, ex: Exception) -> None:
    from modules.notification_center import notifications

    notifications.report_outcome(ex, solicited=False, category=event)


def deliver(
    handler: Callable[..., object] | None,
    event: str,
    *args: object,
    delivery: contextlib.AbstractContextManager = contextlib.nullcontext(),
) -> None:
    """Hand ``handler(*args)`` to the UI dispatcher; report its exception once.

    The one reporter of a handler's failure, whatever the host: the catch is
    inside the function the dispatcher runs, around the handler, so it holds
    on a GUI host too, where the handler runs on a later tick, after this
    call has returned. ``delivery`` is entered around the handler and its
    report on the thread that runs them: a run's ``run_ended`` and
    ``files_written`` are delivered inside their handle's delivery, which
    marks the run told once both are done. With no handler, or when the
    dispatcher cannot take it, the delivery is entered and left at once, so
    a run's waits never wait on a delivery that will not come.
    """
    if handler is None:
        with delivery:
            return

    def _deliver(dt: float) -> None:
        with delivery:
            try:
                handler(*args)
            except Exception as ex:
                _report(event, ex)

    try:
        schedule_ui(_deliver, 0)
    except BaseException:
        with delivery:
            pass
        raise


def deliver_here(handler: Callable[..., object] | None, event: str, *args: object) -> None:
    """Call ``handler(*args)`` on this thread; report its exception once."""
    if handler is None:
        return
    try:
        handler(*args)
    except Exception as ex:
        _report(event, ex)
