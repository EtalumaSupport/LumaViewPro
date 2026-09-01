# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A cancelled RUN_END transition still applies the run's LED end-state.

The protocol queue is cleared the moment protocol-mode ends, and a run's
terminal RUN_END can be enqueued exactly then -- a routine end-of-run
ordering, not only an overlapping abort. Before this guarantee the
cancellation silently skipped the user's LEDs-at-end policy and the
cleanup then treated the end-state as undecided. The dispatcher now
re-applies RUN_END directly through the still-held lease (which
serializes against any new run). A cancelled MID-RUN transition must
stay dead: it belongs to whatever abort cleared the queue.
"""

from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.lumascope_api.illumination import LedTransition
from modules.protocol_step_runner import ProtocolStepRunner


class _CancelledFuture:
    def result(self, timeout=None):
        raise CancelledError()


class _OkFuture:
    def result(self, timeout=None):
        return None


def _make_step_runner_stub(*, future):
    lease = MagicMock()
    p = SimpleNamespace(
        _led_lease=lease,
        _io_executor=SimpleNamespace(protocol_put=lambda task, return_future=True: future),
    )
    return SimpleNamespace(_p=p), lease


def _apply(stub_self, transition):
    ProtocolStepRunner.apply_led_transition(stub_self, transition, MagicMock())


def test_cancelled_run_end_reapplies_through_the_lease():
    stub_self, lease = _make_step_runner_stub(future=_CancelledFuture())

    _apply(stub_self, LedTransition.RUN_END)

    assert lease.apply.call_count == 1, (
        'a cancelled RUN_END must re-apply directly: the queued task never '
        'ran and the end-state policy would otherwise be silently skipped'
    )
    (transition_arg, _ctx), _ = lease.apply.call_args
    assert transition_arg is LedTransition.RUN_END


def test_cancelled_mid_run_transition_stays_dead():
    stub_self, lease = _make_step_runner_stub(future=_CancelledFuture())

    with pytest.raises(CancelledError):
        _apply(stub_self, LedTransition.STEP_LIGHT)

    lease.apply.assert_not_called()


def test_completed_run_end_does_not_double_apply():
    # Behavior-preservation guard: passes before and after the fix. A
    # future that completes normally already carried the apply on the
    # queue; the dispatcher must not apply a second time.
    stub_self, lease = _make_step_runner_stub(future=_OkFuture())

    _apply(stub_self, LedTransition.RUN_END)

    lease.apply.assert_not_called()
