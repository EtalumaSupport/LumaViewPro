# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""While a home runs, the scope is the home's, as it is a run's.

A move asked during a home was accepted onto the io lane and ran when the
home ended -- up to 90 s later on an LS720, a stage moving on a press the
person had forgotten -- and an LED or camera request landed as late. The
controls stayed live throughout. A home now holds the session's activity
claim: every other request is refused when it is asked, naming the home,
the controls lock for its length, and they come back when it ends.
"""

from __future__ import annotations

import threading

import pytest

from modules.exceptions import HardwareCommandRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

HOME_S = 60.0
# Only a failure waits this long: a refusal comes back at once.
ASKED_S = 5.0


@pytest.fixture
def session():
    s = ScopeSession.create(
        complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
    )
    # Homed once first, so a move asked later is refused for the home in
    # flight and not because no position is known.
    s.scope.motion.home('ALL')
    yield s
    s.shutdown()


class _HeldHome:
    """The board's home, held inside the driver until released."""

    def __init__(self, scope, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        real = scope._motion_driver.home

        def held(*args, **kwargs):
            self.entered.set()
            assert self.release.wait(HOME_S), 'the held home was never released'
            return real(*args, **kwargs)

        monkeypatch.setattr(scope._motion_driver, 'home', held)


def _answer_at_once(ask):
    """What ``ask`` raised or returned, failing if it waited for the home."""
    outcome = []

    def run():
        try:
            outcome.append(('returned', ask()))
        except BaseException as e:
            outcome.append(('raised', e))

    asker = threading.Thread(target=run, daemon=True)
    asker.start()
    asker.join(ASKED_S)
    assert outcome, 'the request waited for the home instead of being answered'
    return outcome[0]


def _refused_by_the_home(ask):
    kind, value = _answer_at_once(ask)
    assert kind == 'raised' and isinstance(value, HardwareCommandRefusedError), (kind, value)
    assert 'home' in str(value).lower(), str(value)


@pytest.mark.slow
def test_a_home_holds_the_scope_until_it_ends(session, monkeypatch):
    scope = session.scope
    edges = []
    session.add_run_state_listener(lambda: edges.append(session.controls_locked))
    held = _HeldHome(scope, monkeypatch)

    home = scope.motion.start_home('ALL')
    try:
        assert held.entered.wait(HOME_S)

        assert session.controls_locked
        assert session.run_lockout_named == 'A home is in progress.'
        _refused_by_the_home(lambda: scope.motion.start_move_relative('Z', 10.0))
        _refused_by_the_home(lambda: scope.illumination.led_on('BF', 10.0))
        _refused_by_the_home(lambda: scope.imaging.set_gain_db(1.0))
        with pytest.raises(HardwareCommandRefusedError) as second:
            scope.motion.start_home('Z')
        assert second.value.reason == 'home_in_flight'
    finally:
        # Waited for even when an assertion failed, so no home is still
        # running when the session shuts down.
        held.release.set()
        home.exception(timeout=HOME_S)
    home.result(timeout=HOME_S)

    assert not session.controls_locked
    assert True in edges and edges[-1] is False
    scope.motion.start_move_relative('Z', 10.0).wait()


def test_a_home_inside_a_diagnostic_runs_under_it(session):
    # The support report homes inside its diagnostic claim: the home is the
    # diagnostic's work, so it takes no claim of its own and is not refused.
    with session.diagnostic_claim():
        session.scope.motion.home('Z')
        assert session.run_lockout_named == 'A diagnostic is in progress.'
    assert not session.controls_locked
