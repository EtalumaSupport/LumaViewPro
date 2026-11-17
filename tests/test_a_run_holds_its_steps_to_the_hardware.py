# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run whose steps ask past the hardware's limits runs them at the limits and says so once.

A step's gain or exposure outside the camera's range, an illumination above
the LED board's maximum, or a sum above the 100 frames a capture may sum,
used to refuse the whole run at its start (or, for the board's maximum,
reach the step and be refused by the LED). Now ``prepare()`` sets each such
value at its limit in the run's own copy (``ProtocolsAPI.hold_run_copy_to_hardware``):
the run, its images and its folder's protocol file use what the hardware
takes, the person's protocol keeps what it says, since the same value may
be right on the next, larger camera, and ``start()`` reports one
``ValueClampedNotice`` naming every step once it holds the scope, so a run
refused at a gate never says it clamped. A limit the hardware does not
declare narrows nothing; a sign or kind no hardware makes sense of is still
refused when the run is prepared.
"""

from __future__ import annotations

import threading

import pandas as pd
import pytest

from _lumascope.modules.exceptions import ProtocolRunRefusedError, ValueClampedNotice
from _lumascope.modules.protocol_state_machine import SequencedCaptureRunMode
from _lumascope.modules.run_events import RunEvents
from _lumascope.modules.run_outcome import RunStatus
from tests.scope_fakes import set_camera_limits
from tests.stock_channels import LED0
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)
from tests.test_run_refusal_contract import (
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
)


def _steps(**values):
    # A layer the suite's scope has; the clamp holds the value, not the layer.
    return _make_multi_step_protocol([{'name': 'the step', 'color': LED0, **values}])


def _prepare(executor, protocol, tmp_path, **overrides):
    return executor.prepare(
        **{
            'protocol': protocol,
            'run_mode': SequencedCaptureRunMode.SINGLE_SCAN,
            'sequence_name': 'held',
            'image_capture_config': _make_image_capture_config(),
            'autogain_settings': _make_autogain_settings(),
            'parent_dir': tmp_path / 'output',
            'max_scans': 1,
            **overrides,
        }
    )


def _clamp_notices(centre_posts):
    return [post.message for post in centre_posts if post.reason == ValueClampedNotice.reason]


_WHERE = f'step "the step" ({LED0})'


def test_a_gain_above_the_camera_maximum_runs_at_it_and_the_protocol_keeps_it(
    executor, scope, tmp_path
):
    maximum = scope.camera.setting_limits.gain_max_db
    protocol = _steps(gain_db=maximum + 28.0)

    plan = _prepare(executor, protocol, tmp_path)

    assert plan.protocol.step_row(step_index=0)['Gain'] == pytest.approx(maximum)
    assert protocol.step_row(step_index=0)['Gain'] == pytest.approx(maximum + 28.0)
    assert plan.clamped == [(_WHERE, 'gain_db', maximum + 28.0, maximum)]


def test_a_gain_below_a_declared_camera_floor_runs_at_it(executor, scope, tmp_path, monkeypatch):
    # The step's own judge refuses a negative gain, so the floor that matters
    # here is a camera's above 0 dB.
    set_camera_limits(monkeypatch, scope, gain_min_db=2.0)

    plan = _prepare(executor, _steps(gain_db=1.0), tmp_path)

    assert plan.clamped == [(_WHERE, 'gain_db', 1.0, 2.0)]


def test_an_exposure_above_the_camera_maximum_runs_at_it(executor, scope, tmp_path):
    maximum = scope.camera.setting_limits.exposure_max_ms

    plan = _prepare(executor, _steps(exposure_ms=maximum + 1.0), tmp_path)

    assert plan.protocol.step_row(step_index=0)['Exposure'] == pytest.approx(maximum)
    assert plan.clamped == [(_WHERE, 'exposure_ms', maximum + 1.0, maximum)]


def test_an_illumination_above_the_led_board_runs_at_its_maximum(executor, scope, tmp_path):
    cap = scope.capabilities.led_max_ma

    plan = _prepare(executor, _steps(illumination_ma=cap + 1), tmp_path)

    assert plan.protocol.step_row(step_index=0)['Illumination'] == pytest.approx(cap)
    assert plan.clamped == [(_WHERE, 'illumination_ma', cap + 1, cap)]


def test_a_sum_above_100_runs_at_100_as_a_whole_number(executor, scope, tmp_path):
    plan = _prepare(executor, _steps(sum_count=150), tmp_path)

    held = plan.protocol.step_row(step_index=0)['Sum']
    assert (held, type(held)) == (100, int)
    assert plan.clamped == [(_WHERE, 'sum', 150, 100)]


def test_values_within_every_limit_are_left_as_they_are(executor, scope, tmp_path):
    cap = scope.capabilities.led_max_ma
    gain_max = scope.camera.setting_limits.gain_max_db

    plan = _prepare(
        executor, _steps(gain_db=gain_max, illumination_ma=cap, sum_count=100), tmp_path
    )

    assert plan.clamped == []


def test_an_undeclared_floor_narrows_nothing(executor, scope, tmp_path, monkeypatch):
    set_camera_limits(monkeypatch, scope, exposure_min_ms=None)

    plan = _prepare(executor, _steps(exposure_ms=0.001), tmp_path)

    assert plan.clamped == []


def test_a_negative_current_is_still_refused(executor, scope, tmp_path):
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, _steps(illumination_ma=-1.0), tmp_path)

    assert refusal.value.reason == 'validation_failed'
    assert 'Illumination' in str(refusal.value)


def test_the_run_tells_the_clamp_once_when_it_starts_and_records_what_it_used(
    executor, scope, tmp_path, centre_posts
):
    maximum = scope.camera.setting_limits.gain_max_db
    ended = threading.Event()
    plan = _prepare(
        executor,
        _steps(gain_db=maximum + 28.0),
        tmp_path,
        events=RunEvents(run_ended=lambda *_: ended.set()),
    )
    assert _clamp_notices(centre_posts) == [], 'prepare() told a clamp before the run started'

    executor.start(plan)
    assert ended.wait(COMPLETION_TIMEOUT), 'run_ended never came'

    assert _clamp_notices(centre_posts) == [
        str(ValueClampedNotice([(_WHERE, 'gain_db', maximum + 28.0, maximum)]))
    ]
    (saved,) = (tmp_path / 'output').rglob('held.tsv')
    assert f'{maximum + 28.0:g}' not in saved.read_text()
    gains = pd.read_csv(saved, sep='\t', skiprows=_header_rows(saved))['Gain']
    assert gains.tolist() == [pytest.approx(maximum)]


def _header_rows(path) -> int:
    # The steps table follows the file's header rows, under its 'Name' row.
    for index, line in enumerate(path.read_text().splitlines()):
        if line.startswith('Name\t'):
            return index
    raise AssertionError(f'{path} has no steps table')


def test_autofocus_all_steps_runs_a_step_above_the_led_board(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        cap = session.scope.capabilities.led_max_ma
        step = _step('too_bright', 0, x=20.0, gain=1.0)
        step['Illumination'] = float(cap + 1)
        protocol = _protocol([step])

        outcome = runner.start_autofocus_all_steps(protocol).result(timeout=COMPLETION_TIMEOUT)

    assert outcome.status == RunStatus.COMPLETED, outcome
    assert protocol.step_row(step_index=0)['Illumination'] == pytest.approx(cap + 1)
