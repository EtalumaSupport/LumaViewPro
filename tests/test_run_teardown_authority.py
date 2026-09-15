# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for run-teardown authority.

Tearing a run down is an authority decision, and the engine owns it.
A caller says who it is; the engine compares that against the run's
owner and refuses anyone else. The refusal is the API's, so it reaches
a REST or SDK caller exactly as it reaches the GUI -- a teardown that
could only be refused by a widget would be no refusal at all.

The contract:

- reset(requester) by the run's OWNER unwinds the run.
- reset(requester) by anyone else raises ProtocolRunRefusedError with
  reason 'not_run_owner' AND LEAVES THE RUN RUNNING. This is the case
  that lost 170 of 192 protocol steps in the field: a stale autofocus
  toggle reached reset() and destroyed a scan that a rival owned.
- reset(requester) with nothing live is a silent no-op, not a refusal
  -- there is no run whose owner could be wrong.
- force_reset(reason) is the named shutdown override: it tears down a
  run it does not own, and says so at WARNING. It exists so app close
  does not have to impersonate the run's owner to get past the guard.
"""

import threading
import time

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.protocol_drives import autofocus_snapshot
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    COMPLETION_TIMEOUT,
    _make_autogain_settings,
    _make_image_capture_config,
    _make_multi_step_protocol,
    _make_tile_grid_steps,
    executor,
    executors,
    scope,
)

OWNER = 'scan'
RIVAL = 'autofocus'


def _start_run(executor, tmp_path, done):
    """Start a long-enough run owned by OWNER and wait until it is live."""
    protocol = _make_multi_step_protocol(_make_tile_grid_steps(rows=6, cols=8))

    plan = executor.prepare(
        protocol=protocol,
        run_trigger_source=OWNER,
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='teardown_authority',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={
            'run_complete': lambda **kw: done.set(),
            'go_to_step': lambda **kw: None,
            'move_position': lambda axis: None,
        },
        leds_state_at_end='off',
        autofocus_snapshot=autofocus_snapshot(),
    )
    executor.start(plan)

    deadline = time.monotonic() + 5.0
    while not executor.run_in_progress():
        assert time.monotonic() < deadline, 'run never reached in-progress'
        time.sleep(0.01)
    return plan


class TestTeardownAuthority:
    def test_rival_teardown_is_refused_and_run_survives(
        self, executor, scope, tmp_path, monkeypatch
    ):
        """The reproduced defect: a rival's stop must not destroy the run."""
        notified = []
        from modules import notification_center

        monkeypatch.setattr(
            notification_center.notifications,
            'warning',
            lambda *a, **kw: notified.append(a),
        )
        done = threading.Event()
        _start_run(executor, tmp_path, done)

        with pytest.raises(ProtocolRunRefusedError) as exc:
            executor.reset(requester=RIVAL)

        assert exc.value.reason == 'not_run_owner'
        # Notify-once, like every other refusal: the engine has already told
        # the user, so a caller reconciles its own state without re-notifying.
        assert len(notified) == 1, f'expected one notification, got {notified}'
        assert notified[0][0] == 'Protocol'
        # The holder is named, so a caller can say WHOSE run it is.
        assert exc.value.holder_trigger == OWNER
        # The point of the whole slice: the run is still running.
        assert executor.run_in_progress(), 'a refused teardown still killed the run'

        executor.force_reset(reason='test cleanup')
        assert done.wait(timeout=COMPLETION_TIMEOUT)

    def test_owner_teardown_proceeds(self, executor, scope, tmp_path):
        done = threading.Event()
        _start_run(executor, tmp_path, done)

        executor.reset(requester=OWNER)

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'owner reset did not unwind the run'

    def test_teardown_with_no_run_is_a_silent_no_op(self, executor, scope, tmp_path):
        """No run means no owner to be wrong about -- a no-op, not a refusal."""
        executor.reset(requester=RIVAL)  # must not raise

    def test_force_reset_overrides_ownership(self, executor, scope, tmp_path):
        """App close does not impersonate the owner to get past the guard."""
        done = threading.Event()
        _start_run(executor, tmp_path, done)

        executor.force_reset(reason='app shutdown')

        assert done.wait(timeout=COMPLETION_TIMEOUT), 'force_reset did not unwind the run'


class TestRequesterIsRequired:
    """Rule 50 at the boundary: a call site cannot stay unauthenticated."""

    def test_reset_without_requester_is_a_type_error(self, executor, scope, tmp_path):
        with pytest.raises(TypeError):
            executor.reset()
