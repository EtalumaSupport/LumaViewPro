# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A raising run-state listener never strands the scope's claim.

The claim settles who holds the scope and then tells its listener. The call
was bare, so a listener that raised on a take left the scope held by a
taking the taker never received -- nothing could release it -- and one that
raised on a release skipped whatever the releaser did next: a run's return
to IDLE, a recording's drained signal. The listener's raise is now contained
at the claim and reported where it happened.
"""

import pytest

from modules.activity_claim import ActivityClaim
from modules.notification_center import notifications
from tests.protocol_drives import run_identity


class _ListenerError(RuntimeError):
    pass


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, **kw: seen.append((ex, kw)))
    return seen


def _raising():
    raise _ListenerError('the listener fell over')


def test_a_take_hands_the_taker_its_claim(reported):
    claim = ActivityClaim(on_transition=_raising)

    held = claim.try_claim('protocol', run=run_identity())

    assert held is not None, 'the taker never received the taking the claim is held by'
    assert claim.holder is not None and claim.holder.kind == 'protocol'
    [(ex, kw)] = reported
    assert isinstance(ex, _ListenerError) and kw['solicited'] is False


def test_a_release_completes_and_the_releaser_goes_on(reported):
    claim = ActivityClaim()
    held = claim.try_claim('recording')
    claim._on_transition = _raising

    held.release()

    assert claim.holder is None
    [(ex, _kw)] = reported
    assert isinstance(ex, _ListenerError)


def test_the_session_reports_each_listener_that_raises_and_tells_the_rest(reported):
    from tests.test_session_run_state import _make_session

    session = _make_session()
    told = []
    session._run_state_listeners.append(_raising)
    session._run_state_listeners.append(lambda: told.append(True))

    session.notify_run_state()

    assert told, 'a raising listener kept the others from hearing the change'
    [(ex, kw)] = reported
    assert isinstance(ex, _ListenerError) and kw['solicited'] is False
