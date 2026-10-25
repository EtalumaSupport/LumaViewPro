# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A video step's progress names its phase from one set.

``VideoProgress`` is the record a host carries the ``video_progress`` event
in, built from the handler's arguments; its phase is a ``VideoPhase``, so a
reader branches on members and a word outside the set never reads as a
phase.
"""

import pytest

from modules.exceptions import ArgumentRefusedError
from modules.run_events import VideoPhase, VideoProgress


def test_a_video_progress_holds_its_phase_as_a_member_and_refuses_any_other_word():
    built = VideoProgress('writing', progress_pct=40.0)
    assert type(built.phase) is VideoPhase
    assert built.phase == VideoPhase.WRITING == 'writing'
    with pytest.raises(ArgumentRefusedError) as refused:
        VideoProgress('saving', progress_pct=40.0)
    assert (refused.value.reason, refused.value.argument) == (
        'not_offered',
        'VideoProgress.phase',
    )
