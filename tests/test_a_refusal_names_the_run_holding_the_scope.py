# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refused start says WHICH run has the scope, not that some run does.

The engine is handed the holder at every one of these gates and used to
print a literal instead, so a researcher whose Z-stack was turned away by
their own protocol read "A protocol run is already in progress" and had to
work out which control to go back to. The stop refusal in ``reset()`` has
always named the holder; these are the other half of that sentence, and
they now share its phrasing rather than copying its words.

The holder is not new state. ``_run_identity`` is written by ``start()``
with the claim, and the file-drain gates read it as the just-finished
run's -- which is exactly the run whose files are still landing, so naming
it there is naming the right run.

The run is named by its kind in a person's words ('the Z-stack run'), never
by the token its starter passed ('zstack', 'api_autofocus_scan'): the token
is provenance, carried on the refusal as ``holder_trigger``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.protocol_image_writer import RunWriteBatch
from modules.protocol_state_machine import ProtocolState
from modules.run_outcome import PendingRunOutcome
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.protocol_drives import autofocus_snapshot
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    executor,
    executors,
    scope,
)
from tests.protocol_drives import run_identity


def _a_zstack_holds_the_scope(executor):
    """The state start() commits for a live z-stack run, as its fields:
    the phase, the trigger, and the handle start() returned."""
    executor._set_state(ProtocolState.RUNNING)
    executor._run_identity = run_identity('zstack', 'Z-stack')
    executor._run_outcome = PendingRunOutcome()


def _start_a_scan(executor, tmp_path):
    return executor.prepare(
        protocol=_make_single_step_protocol(),
        run_trigger_source='scan',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='names_the_holder',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        autofocus_snapshot=autofocus_snapshot(),
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={},
    )


class TestTheRefusalNamesTheHolder:
    def test_a_live_zstack_refuses_a_scan_by_name(self, executor, tmp_path):
        _a_zstack_holds_the_scope(executor)
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _start_a_scan(executor, tmp_path)
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert refusal.value.reason == 'already_running'
        assert refusal.value.message.startswith('The Z-stack run is using the microscope'), (
            f'the refusal must name the run that has the scope: {refusal.value.message!r}'
        )
        assert 'zstack' not in refusal.value.message, 'a raw trigger token reached a person'
        assert refusal.value.holder_trigger == 'zstack'
        assert 'A protocol run is already in progress' not in refusal.value.message, (
            'the old literal told every caller the same thing about a different run'
        )

    def test_the_file_drain_names_the_run_whose_files_are_landing(
        self, executor, tmp_path, monkeypatch
    ):
        """The just-finished run, which is the one still writing."""
        executor._run_identity = run_identity('zstack', 'Z-stack')
        # The finished run's write batch, closed with a write still to land:
        # a real batch over a stand-in lane that never runs it.
        lane = MagicMock()
        lane.in_flight_task_stalled.return_value = False
        still_writing = RunWriteBatch(lane)
        still_writing.submit(lambda: None, {}, what='a capture', pace_until=None)
        still_writing.close()
        monkeypatch.setattr(executor, '_write_batch', still_writing)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _start_a_scan(executor, tmp_path)

        assert refusal.value.reason == 'files_writing'
        assert refusal.value.message.startswith('The Z-stack run is still writing'), (
            f'"previous run" named no run at all: {refusal.value.message!r}'
        )
        assert refusal.value.holder_trigger == 'zstack'

    def test_a_refused_stop_still_names_the_owner(self, executor):
        """The sentence this shares: a stop naming a run that has ended,
        refused while a z-stack is live, names the z-stack."""
        _a_zstack_holds_the_scope(executor)
        an_ended_scan = PendingRunOutcome()
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                executor._reset(an_ended_scan)
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert refusal.value.reason == 'run_not_live'
        assert refusal.value.holder_trigger == 'zstack'
        assert 'The Z-stack run is using the microscope now' in refusal.value.message
        assert 'zstack' not in refusal.value.message


class TestARawTokenNeverReachesAPerson:
    def test_an_api_run_is_named_by_its_kind(self, executor, tmp_path):
        """A script's autofocus scan once read 'The api_autofocus_scan run'.
        The token is the caller's provenance; the sentence names the run."""
        executor._set_state(ProtocolState.RUNNING)
        executor._run_identity = run_identity('api_autofocus_scan', 'autofocus scan')
        executor._run_outcome = PendingRunOutcome()
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _start_a_scan(executor, tmp_path)
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert refusal.value.message.startswith('The autofocus scan run is using the microscope')
        assert 'api_' not in refusal.value.message
        assert refusal.value.holder_trigger == 'api_autofocus_scan'

    def test_a_diagnostic_refused_by_a_run_names_the_run(self):
        from modules.activity_claim import ActivityClaim
        from modules.exceptions import DiagnosticRefusedError
        from modules.scope_session import ScopeSession

        session = ScopeSession.__new__(ScopeSession)
        session.activity_claim = ActivityClaim()
        held = session.activity_claim.try_claim(
            'protocol', run=run_identity('api_zstack', 'Z-stack')
        )
        try:
            with pytest.raises(DiagnosticRefusedError) as refusal, session.diagnostic_claim():
                pass
        finally:
            held.release()

        assert refusal.value.message.startswith('The Z-stack run is using the microscope')
        assert refusal.value.holder_trigger == 'api_zstack'

    def test_a_run_claim_without_its_run_cannot_be_taken(self):
        from modules.activity_claim import ActivityClaim

        with pytest.raises(ValueError):
            ActivityClaim().try_claim('protocol')


class TestTheSentenceStillParsesWithNoHolder:
    def test_no_trigger_does_not_print_the_word_none(self, executor, tmp_path):
        """A run can hold the state without a trigger in a partly-built runner.

        Whatever the cause, a user reads this sentence, so it degrades to
        the indefinite article rather than to 'The None run'.
        """
        executor._set_state(ProtocolState.RUNNING)
        executor._run_identity = None
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _start_a_scan(executor, tmp_path)
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert 'None' not in refusal.value.message, refusal.value.message
        assert refusal.value.message.startswith('A run is using the microscope')
