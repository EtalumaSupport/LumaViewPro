# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""What a session is still doing, as one published record.

A session's work runs in several places: a run, a home, a recording or a
diagnostic holds the scope; a recording's file and a run's video steps
finish after they let it go; a finished run's images land on the file
lane and its post-run builds on threads of their own; post-processing
builds run on their own lane; a support report or a logs zip runs on its
caller's thread; a still saves on the camera lane. Each of those owners
answers for its own work; ``ScopeSession.live_work`` asks every one and
returns their answers together, so a closing host -- the GUI, a script,
the REST server -- reads one list, and the close waits on exactly it.
"""

from __future__ import annotations

import dataclasses

from modules.api_surface import api_fields

# The kinds of work a session reports, one per place work runs. A run, a
# recording, a home and a diagnostic are the claim's own kinds, so a
# holder's kind is its item's kind.
PROTOCOL = 'protocol'
RECORDING = 'recording'
HOME = 'home'
DIAGNOSTIC = 'diagnostic'
RECORDING_FINISH = 'recording_finish'
RUN_VIDEO_FINISH = 'run_video_finish'
RUN_FILES = 'run_files'
POST_RUN_STEP = 'post_run_step'
POST_PROCESSING = 'post_processing'
POST_PROCESSING_QUEUED = 'post_processing_queued'
SUPPORT_REPORT = 'support_report'
LOGS_ZIP = 'logs_zip'
STILL = 'still'


@api_fields('kind', 'name', 'left', 'percent')
@dataclasses.dataclass(frozen=True)
class WorkItem:
    """One piece of work the session is still doing.

    Attributes:
        kind: Where the work runs; one of this module's kind names.
        name: What it is, in words a person reads.
        left: How many frames, images or builds it still has to finish,
            when it counts them; None when it does not.
        percent: How far a build has got, as its own progress last said;
            None before it has said, or for work that reports none.
    """

    kind: str
    name: str
    left: int | None = None
    percent: float | None = None


@api_fields('work', 'closing', 'closed')
@dataclasses.dataclass(frozen=True)
class LiveWork:
    """Everything a session is still doing, read once.

    Each item is read from its owner as that owner answers, one after
    another rather than under one lock, so two items can describe
    slightly different moments; the close waits until a read finds none.

    Attributes:
        work: The work still under way, in the order the session asked:
            what holds the scope first, then what finishes after.
        closing: True once the session's close has begun: nothing new
            takes the scope, and the close waits for this work.
        closed: True once the session has shut down.
    """

    work: tuple[WorkItem, ...]
    closing: bool
    closed: bool
