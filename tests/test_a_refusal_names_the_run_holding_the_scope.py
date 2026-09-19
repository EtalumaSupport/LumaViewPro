# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refused start says WHICH run has the scope, not that some run does.

The engine is handed the holder at every one of these gates and used to
print a literal instead, so a researcher whose Z-stack was turned away by
their own protocol read "A protocol run is already in progress" and had to
work out which control to go back to. The stop refusal in ``reset()`` has
always named the holder; these are the other half of that sentence, and
they now share its phrasing rather than copying its words.

The holder is not new state. ``_run_trigger_source`` is written by
``start()`` with the claim, and the file-drain gates read it as the
just-finished run's -- which is exactly the run whose files are still
landing, so naming it there is naming the right run.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.protocol_state_machine import ProtocolState
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


def _a_zstack_holds_the_scope(executor):
    """The state start() commits for a live z-stack run, as its two fields."""
    executor._set_state(ProtocolState.RUNNING)
    executor._run_trigger_source = 'zstack'


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
        assert 'zstack' in refusal.value.message, (
            f'the refusal must name the run that has the scope: {refusal.value.message!r}'
        )
        assert 'A protocol run is already in progress' not in refusal.value.message, (
            'the old literal told every caller the same thing about a different run'
        )

    def test_the_file_drain_names_the_run_whose_files_are_landing(
        self, executor, tmp_path, monkeypatch
    ):
        """The just-finished run, which is the one still writing."""
        executor._run_trigger_source = 'zstack'
        monkeypatch.setattr(executor.file_io_executor, 'is_protocol_queue_active', lambda: True)

        with pytest.raises(ProtocolRunRefusedError) as refusal:
            _start_a_scan(executor, tmp_path)

        assert refusal.value.reason == 'files_writing'
        assert 'zstack' in refusal.value.message, (
            f'"previous run" named no run at all: {refusal.value.message!r}'
        )

    def test_a_refused_stop_still_names_the_owner(self, executor):
        """The sentence this shares; unchanged behaviour, pinned against the move."""
        _a_zstack_holds_the_scope(executor)
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                executor.reset(requester='scan')
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert refusal.value.reason == 'not_run_owner'
        assert 'zstack' in refusal.value.message


class TestTheSentenceStillParsesWithNoHolder:
    def test_no_trigger_does_not_print_the_word_none(self, executor, tmp_path):
        """A run can hold the state without a trigger in a partly-built runner.

        Whatever the cause, a user reads this sentence, so it degrades to
        the indefinite article rather than to 'The None run'.
        """
        executor._set_state(ProtocolState.RUNNING)
        executor._run_trigger_source = None
        try:
            with pytest.raises(ProtocolRunRefusedError) as refusal:
                _start_a_scan(executor, tmp_path)
        finally:
            executor._set_state(ProtocolState.IDLE)

        assert 'None' not in refusal.value.message, refusal.value.message
        assert refusal.value.message.startswith('A run is using the microscope')
