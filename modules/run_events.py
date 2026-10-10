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

from modules.api_surface import api_fields, event_metadata
from modules.kivy_utils import schedule_ui

if TYPE_CHECKING:
    from modules.protocol import Protocol
    from modules.run_outcome import RunOutcome


@api_fields('phase', 'elapsed_s', 'total_s', 'percent')
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


@api_fields('scan_number', 'scans_remaining', 'interval')
@dataclasses.dataclass(frozen=True)
class ScanStarted:
    """A ``scan_started`` event: the scan's 1-based number, the scans left with it, the period."""

    scan_number: int
    scans_remaining: int
    interval: datetime.timedelta


@api_fields('scan_number', 'scans_remaining', 'interval')
@dataclasses.dataclass(frozen=True)
class ScanEnded:
    """A ``scan_ended`` event: the scan's 1-based number, the scans left after it, the period."""

    scan_number: int
    scans_remaining: int
    interval: datetime.timedelta


@api_fields('step_idx')
@dataclasses.dataclass(frozen=True)
class StepStarted:
    """A ``step_started`` event: the index of the step beginning."""

    step_idx: int


@api_fields('image', 'frames_summed', 'frame_significant_bits')
@dataclasses.dataclass(frozen=True)
class FrameCaptured:
    """A ``frame_captured`` event: the frame as its file gets it, how many summed, its bits."""

    image: np.ndarray
    frames_summed: int
    frame_significant_bits: int


@api_fields('outcome', 'run_dir', 'protocol')
@dataclasses.dataclass(frozen=True)
class RunEnded:
    """A ``run_ended`` event: the outcome, the run's folder (None when it saved nothing), its protocol."""

    outcome: RunOutcome
    run_dir: pathlib.Path | None
    protocol: Protocol


@api_fields('run_dir', 'files')
@dataclasses.dataclass(frozen=True)
class FilesWritten:
    """A ``files_written`` event: the run's folder, and ``'written'`` or ``'incomplete'``."""

    run_dir: pathlib.Path | None
    files: str


@api_fields(
    'scan_started',
    'scan_ended',
    'step_started',
    'frame_captured',
    'video_progress',
    'run_ended',
    'files_written',
)
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

    Each field declares its event's record (``ScanStarted``, ...), whose
    fields name the handler's arguments in order; a host that carries the
    event elsewhere builds the record from them.
    """

    scan_started: Callable[[int, int, datetime.timedelta], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(ScanStarted)
    )
    scan_ended: Callable[[int, int, datetime.timedelta], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(ScanEnded)
    )
    step_started: Callable[[int], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(StepStarted)
    )
    frame_captured: Callable[[np.ndarray, int, int], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(FrameCaptured)
    )
    video_progress: Callable[[VideoProgress], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(VideoProgress)
    )
    run_ended: Callable[[RunOutcome, pathlib.Path | None, Protocol], object] | None = (
        dataclasses.field(default=None, metadata=event_metadata(RunEnded))
    )
    files_written: Callable[[pathlib.Path | None, str], object] | None = dataclasses.field(
        default=None, metadata=event_metadata(FilesWritten)
    )


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
