# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run stacks each family its own format asks for.

Its stills are stacked when the Sequenced format is Hyperstack, and each
video step's frames when the recording format is; a run asking for a family
it does not have builds nothing for it and is refused nothing. Before the
recording format had a Hyperstack of its own, a run under the Sequenced
Hyperstack also stacked every video step it saved as frames, a build of
minutes nobody asked for.

Driven end to end on the simulated scope.
"""

import datetime

import pandas as pd
import pytest

from _lumascope.modules.image_mode import OutputFormat, RecordingFormat
from _lumascope.modules.protocol import Protocol
from _lumascope.modules.run_events import RunEvents
from tests.case_walk import failing_cases
from tests.protocol_fixtures import protocol_of
from tests.test_a_late_write_records_its_frame import _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 120.0


def _still():
    return _step('S0', 0, x=20.0, gain=1.0)


def _video():
    step = _step('V0', 1, x=30.0, gain=1.0)
    step['Acquire'] = 'video'
    step['Video Config'] = {'duration': 0.4}
    return step


def _two_scans(steps):
    # Two scans, so a well's stills have a time to stack along: a well of
    # one image is a single-image group, which no stack is built for.
    period = datetime.timedelta(seconds=1.0)
    return protocol_of(
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(steps),
            'period': period,
            'duration': period * 2,
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


@pytest.mark.slow
def test_each_family_is_stacked_by_its_own_format(tmp_path):
    # (Sequenced format, recording format, the run's steps) -> the stacks
    # its folder holds: of the stills, of the recording.
    H, T = OutputFormat.HYPERSTACK, OutputFormat.TIFF
    cases = [
        (H, RecordingFormat.FRAMES, (_still, _video), {'stills'}),
        (T, RecordingFormat.HYPERSTACK, (_still, _video), {'recording'}),
        (H, RecordingFormat.HYPERSTACK, (_still, _video), {'stills', 'recording'}),
        (H, RecordingFormat.FRAMES, (_video,), set()),
    ]

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):

        def check(sequenced, recording_format, steps, stacked):
            session.update_settings('image_output_format.sequenced', sequenced)
            session.update_settings('video_recording_format', recording_format)
            parent = tmp_path / f'{sequenced.name}-{recording_format}-{len(steps)}'
            pending = runner.start_protocol(
                protocol=_two_scans([make() for make in steps]),
                parent_dir=str(parent),
                events=RunEvents(),
            )
            outcome = pending.result(timeout=WAIT_S)
            assert pending.files.result(timeout=WAIT_S) is not None, 'its files never finished'
            assert outcome.status == 'completed', outcome

            stacks = [p.name for p in parent.rglob('*hyperstack*.ome.tiff')]
            found = {'recording' if 'V0' in name else 'stills' for name in stacks}
            assert found == stacked, stacks

        failed = failing_cases(
            cases, check, label=lambda case: f'{case[0].name}/{case[1]}/{len(case[2])} steps'
        )
    assert not failed, failed
