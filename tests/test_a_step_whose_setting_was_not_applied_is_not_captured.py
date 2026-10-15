# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol step whose camera setting was not applied is not captured.

A step's gain or exposure the camera did not take was reported once, and the
step captured at whatever the camera held: an image saved at a setting its
step never asked for, on a run that then ended 'completed'. An Auto_Gain step
whose arm was refused captured at manual values with auto-gain off; a video
step whose one-shot auto-gain was refused left the step uncaught.

Now such a step is a failed capture: a strike through the writer's one
counter, its ``FailedCapture`` naming the setting in the rejection's words,
and no image. Three in a row end the run, naming the last cause. Only the
arm stops an Auto_Gain step: its starting gain or exposure only seeds what
auto-gain then chooses, so one not taken is told and the step captures.

Driven end to end on the simulated scope; each refusal is the simulator's
own fault on the node, met by the first write of that node the run makes,
which in a run with no live-view arm is its first step's.
"""

import pathlib

from modules.notification_center import Severity

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_a_run_reports_what_it_captured import _run
from tests.test_composite_run_e2e import headless_settings, open_composite_session


def _images(run_parent: pathlib.Path) -> list[str]:
    return sorted(path.name for path in run_parent.rglob('*.tiff'))


def _auto_gain(step: dict) -> dict:
    return {**step, 'Auto_Gain': True}


def _errors(centre_posts):
    return [(n.title, n.message) for n in centre_posts if n.severity is Severity.ERROR]


def test_a_manual_step_whose_exposure_was_not_taken_is_not_captured(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('exposure', 'false')
        outcome = _run(
            runner,
            tmp_path / 'runs',
            [_step('C0', 0, x=20.0, gain=1.0), _step('C1', 1, x=30.0, gain=1.0)],
        )

    assert (outcome.status, outcome.captures.asked, outcome.captures.captured) == (
        'incomplete',
        2,
        1,
    ), outcome
    [failed] = outcome.captures.failed
    assert (failed.step_index, failed.step_name) == (0, 'C0')
    assert failed.cause.startswith('The camera rejected the exposure change to 10 ms.'), failed
    images = _images(tmp_path / 'runs')
    assert len(images) == 1 and 'C1' in images[0], images


def test_an_auto_gain_step_whose_arm_was_refused_is_not_captured(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('auto_gain', 'false')
        outcome = _run(
            runner,
            tmp_path / 'runs',
            [_auto_gain(_step('C0', 0, x=20.0, gain=1.0)), _step('C1', 1, x=30.0, gain=1.0)],
        )

    [failed] = outcome.captures.failed
    assert (failed.step_index, failed.cause) == (
        0,
        'The camera did not take the auto-gain change. It is still as it was.',
    )
    assert outcome.captures.captured == 1


def test_an_auto_gain_step_whose_starting_gain_was_refused_still_captures(tmp_path, centre_posts):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('gain', 'false')
        outcome = _run(runner, tmp_path / 'runs', [_auto_gain(_step('C0', 0, x=20.0, gain=1.0))])

    assert (outcome.status, outcome.captures.captured) == ('completed', 1), outcome
    assert [title for title, _ in _errors(centre_posts)] == ['Camera Setting Not Applied']


def test_a_video_step_whose_one_shot_was_refused_is_not_captured(tmp_path):
    video = {**_auto_gain(_step('V0', 0, x=20.0, gain=1.0)), 'Acquire': 'video'}
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('auto_gain_once', 'false')
        outcome = _run(runner, tmp_path / 'runs', [video])

    [failed] = outcome.captures.failed
    assert failed.cause == (
        'The camera did not take the one-shot auto-gain change. It is still as it was.'
    )
    assert outcome.captures.captured == 0


def test_three_steps_in_a_row_end_the_run_naming_the_last(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('exposure', 'false', count=3)
        outcome = _run(
            runner,
            tmp_path / 'runs',
            [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(4)],
        )

    assert (outcome.status, outcome.reason) == ('failed', 'camera_failure'), outcome
    assert outcome.message.startswith(
        'The camera failed 3 steps in a row; the last: The camera rejected the exposure '
        'change to 10 ms.'
    ), outcome.message
    assert _images(tmp_path / 'runs') == []


def test_a_run_that_saves_no_images_reports_the_setting_instead(tmp_path, centre_posts):
    # Nothing is captured to fail, so the refusal is told, once.
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        session.scope.simulation.refuse_camera_write('exposure', 'false')
        pending = runner.start_single_scan(
            protocol=_protocol([_step('C0', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
            enable_image_saving=False,
        )
        outcome = pending.result(timeout=60.0)

    assert (outcome.status, outcome.captures.failed) == ('completed', ()), outcome
    assert [title for title, _ in _errors(centre_posts)] == ['Camera Setting Not Applied']
