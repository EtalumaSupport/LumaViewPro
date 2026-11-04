# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol video step records at the one global limit, never a rate of its own.

A step's Video Config held an fps, 30 in every step, copied from a layer
setting no control showed, and the recorder capped the step at it below the
global Recording max FPS: a camera delivering 68.9 fps kept 900 of 2065
frames. A recording, manual or a protocol's step, keeps every frame the
camera delivers, or samples down to ``video.max_fps`` when it is set
(``video_cadence.effective_recording_fps``). The disk a step will write is
sized at that limit, else at the rate the camera is delivering
(``video_cadence.video_sizing_fps``).

The recording is driven through the Session on the simulated scope, whose
stream check measures the delivered rate as every host's does. The run is
started straight after bring-up, before the check has measured a rate, so
its start waits for the first one (``ImagingAPI.wait_for_delivered_rate``)
rather than refusing a camera that is delivering. The 10 fps
limit on a protocol step is
``test_a_rate_limited_recording_keeps_its_whole_budget.py``'s; a v10 file's
stored fps dropped at load is ``test_protocol_version_gate.py``'s; a run
refused at its start while the camera delivers nothing is
``test_run_refusal_contract.py``'s (``camera_not_delivering``).
"""

import json
import pathlib
import threading

import pytest

from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from modules.video_cadence import video_sizing_fps
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

COMPLETION_TIMEOUT = 60.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _video_step_protocol():
    row = _make_single_step_protocol().step_row(step_index=0)
    return _build_real_protocol(
        [{**row, 'Acquire': 'video', 'Exposure': 2.0, 'Video Config': {'duration': 1}}]
    )


@pytest.mark.slow
def test_a_video_step_with_no_limit_keeps_every_delivered_frame(session, tmp_path):
    done = threading.Event()

    run = session.protocol_runner().start_single_scan(
        protocol=_video_step_protocol(),
        sequence_name='every_frame',
        parent_dir=str(tmp_path),
        events=RunEvents(run_ended=lambda *ended: done.set()),
    )

    assert done.wait(COMPLETION_TIMEOUT), 'the run never ended'
    assert run.result(timeout=COMPLETION_TIMEOUT).status == 'completed'
    assert run.files.result(timeout=COMPLETION_TIMEOUT) is not None
    (path,) = pathlib.Path(tmp_path).rglob('*manifest.json')
    manifest = json.loads(path.read_text())
    assert manifest['configured_fps'] is None, 'a step with no limit records at a rate'
    frame_ids = [frame['chunks']['FrameID'] for frame in manifest['frame_index']]
    assert frame_ids, 'the step recorded no frame'
    assert frame_ids == list(range(frame_ids[0], frame_ids[0] + len(frame_ids))), (
        f'the step skipped frames the camera delivered: {frame_ids}'
    )


def test_a_recording_is_sized_at_the_limit_else_the_delivered_rate():
    assert video_sizing_fps(10, 68.9) == 10
    assert video_sizing_fps(0, 68.9) == 68.9
    assert video_sizing_fps(0, 0.0) is None
    assert video_sizing_fps(10, 0.0) == 10
