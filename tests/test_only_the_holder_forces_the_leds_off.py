# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only the activity holding the scope, or the session's close, forces the LEDs off.

``force_off`` darkens every LED past any lease, on the caller's thread,
skipping the io lane: the holder's own off for its fault and its end, and the
close's. Before, it refused no one, so any caller could darken a run's sample
around its lease and its lane. Now a caller outside the activity is refused
``'not_the_activity'`` and nothing is sent.

A run's fatal fault reaches it from every thread the run's work runs on,
including the two a video step starts: its frame writer and its finish. Each
acts under the run's taking, so the fault still darkens the sample and is
told; refused, the raise would skip the popup.
"""

from __future__ import annotations

import pytest

import _lumascope.modules.protocol_recording as protocol_recording
from _lumascope.modules.exceptions import HardwareCommandRefusedError
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _lit(scope):
    return [r.name for r in scope.illumination.get_led_states() or () if r.state == LedState.LIT]


def test_a_force_off_from_outside_the_activity_is_refused_and_sends_nothing(session):
    """No activity holds the scope, so no caller acts for one: refused, the LED stays lit."""
    session.scope.illumination.led_on('BF', 50.0)

    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.scope.illumination.force_off()

    assert refused.value.reason == 'not_the_activity'
    assert _lit(session.scope) == ['BF'], 'a refused force_off darkened the LED'


def test_the_close_forces_the_leds_off_before_it_disconnects(tmp_path, monkeypatch):
    """The close's off is admitted while the close runs: dark before the disconnect's own off."""
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    s.scope.illumination.led_on('BF', 50.0)
    lit_at_disconnect = []
    real_disconnect = s.scope.disconnect

    # A stand-in by design: the disconnect turns the LEDs off inline too, so
    # the state is read as it begins, after the close's own off.
    def disconnect(*args, **kwargs):
        lit_at_disconnect.append(_lit(s.scope))
        return real_disconnect(*args, **kwargs)

    monkeypatch.setattr(s.scope, 'disconnect', disconnect)

    s.shutdown()

    assert lit_at_disconnect == [[]], "the close's force_off left an LED lit"


def _run_a_video_step(session, tmp_path, monkeypatch, frame_write_disk):
    """Run one video step whose frame writes meet ``frame_write_disk``; return the outcome.

    A stand-in by design: a disk that fails mid-recording, which the
    simulator cannot make. The step asks the disk once before it records;
    every later ask is a frame write's, on the recording's writer thread.
    """
    asks = []
    real_probe = protocol_recording.check_disk_space_ok

    def disk(*args, **kwargs):
        asks.append(args)
        if len(asks) == 1:
            return real_probe(*args, **kwargs)
        return frame_write_disk()

    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', disk)
    base = _make_single_step_protocol().step_row(step_index=0)
    video = {**base, 'X': 20.0, 'Y': 20.0, 'Acquire': 'video', 'Name': 'video', 'Step Index': 0}
    run = session.protocol_runner().start_single_scan(
        protocol=_build_real_protocol([video]), parent_dir=str(tmp_path)
    )
    return run.result(timeout=COMPLETION_TIMEOUT)


def test_a_video_steps_disk_floor_ends_the_run_dark_and_told(
    session, tmp_path, monkeypatch, centre_posts
):
    """The floor is met on the recording's writer thread, which acts for the run."""
    outcome = _run_a_video_step(session, tmp_path, monkeypatch, lambda: (False, 1.0))

    assert (outcome.status, outcome.reason) == ('failed', 'disk_space_critical')
    shown = [n.title for n in centre_posts if n.shown and n.fatal]
    assert shown == ['Disk Space Critical'], shown
    assert _lit(session.scope) == []


class _LaneKiller(BaseException):
    """Not an Exception, so the writer thread treats it as its own death."""


def test_a_video_writers_death_ends_the_run_dark_and_told(
    session, tmp_path, monkeypatch, centre_posts
):
    """The death is told from the step's finish thread, which acts for the run."""

    def dies():
        raise _LaneKiller('scripted: the writer thread dies')

    outcome = _run_a_video_step(session, tmp_path, monkeypatch, dies)

    assert (outcome.status, outcome.reason) == ('failed', 'video_writer_died')
    shown = [n for n in centre_posts if n.shown and n.fatal]
    assert len(shown) == 1, [n.title for n in shown]
    assert _lit(session.scope) == []
