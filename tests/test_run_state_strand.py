# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a claim-refused run must not strand caller-committed state.

ProtocolRunner commits caller-side running state (the session's
protocol_running Event, the completion-event re-arm) between prepare()
and start(). The session activity claim is gated inside start(), so a
refusal for a held claim (a live video recording) raises AFTER that
commit: session.protocol_running strands True with no run to ever
clear it, and the completion event strands cleared so
wait_for_completion() blocks until timeout for a run that never
started. Both contradict the documented refusal contract ("no state
was committed ... wait_for_completion() is not armed").

The prepare-side refusal contract (refusals that raise before the
commit) is covered by tests/test_run_refusal_contract.py; this file
pins the start-side (claim-gate) refusal specifically.
"""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# Heavy deps (lvp_logger, kivy, pypylon, ids_peak, ...) are mocked by
# tests/conftest.py at module-import time. Mock settings_init before
# sequenced_capture_runner imports it. (Harness mirrors
# tests/test_run_refusal_contract.py.)
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.exceptions import ProtocolRunRefusedError

COMPLETION_TIMEOUT = 15  # seconds -- generous for CI


def _make_session_settings(tmp_path):
    return {
        'BF': {'autofocus': False},
        'PC': {'autofocus': False},
        'DF': {'autofocus': False},
        'Red': {'autofocus': False},
        'Green': {'autofocus': False},
        'Blue': {'autofocus': False},
        'Lumi': {'autofocus': False},
        'stage_offset': {'x': 0.0, 'y': 0.0},
        'live_folder': str(tmp_path),
        'protocol': {
            'autogain': {
                'target_brightness': 0.3,
                'max_duration_seconds': 1.0,
                'min_gain_db': 0.0,
                'max_gain_db': 20.0,
            },
        },
    }


def _make_single_step_protocol():
    # Mirrors tests/test_run_refusal_contract.py -- a one-step protocol
    # that passes every prepare() gate on the simulated scope.
    import datetime
    import pathlib

    import pandas as pd

    from modules.protocol import Protocol

    step = {
        'Name': 'A1_test',
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': False,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': '10x Oly',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1, 'fps': 5},
        'Stim_Config': {},
        'Step Index': 0,
    }
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': pd.DataFrame([step]),
        'period': datetime.timedelta(minutes=1.0),
        'duration': datetime.timedelta(hours=1.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    tiling_configs = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'
    return Protocol(tiling_configs_file_loc=tiling_configs, config=config)


class TestClaimRefusalLeavesNoState:
    def test_recording_held_claim_refusal_strands_nothing(self, tmp_path):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless(settings=_make_session_settings(tmp_path))
        session.start_executors()
        runner = session.create_protocol_runner()
        claim_held = False
        try:
            # A completed first run arms the completion event, so the
            # refusal below has prior state to preserve (mirrors the
            # prepare-side contract test).
            first_done = threading.Event()
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='pre_refusal_scan',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={
                    'run_complete': lambda **kw: first_done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert first_done.wait(timeout=COMPLETION_TIMEOUT), 'first run did not complete'
            assert runner.wait_for_completion(timeout=COMPLETION_TIMEOUT)
            # run_complete fires during cleanup; the claim releases at
            # cleanup END, moments later. Wait for the release before
            # claiming as the recording.
            deadline = time.monotonic() + COMPLETION_TIMEOUT
            while session.activity_claim.owner is not None:
                assert time.monotonic() < deadline, 'first run never released the claim'
                time.sleep(0.02)

            # A video recording holds the session's exclusive-activity
            # claim, exactly as the recording engine does for its whole
            # capture + drain lifetime.
            assert session.activity_claim.try_claim('recording')
            claim_held = True

            with pytest.raises(ProtocolRunRefusedError) as excinfo:
                runner.run_single_scan(
                    protocol=_make_single_step_protocol(),
                    sequence_name='claim_refused_scan',
                    parent_dir=str(tmp_path),
                    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                )
            assert excinfo.value.reason == 'exclusive_activity_running'
            # The refusal says busy-with-what: the holder's KIND rides
            # the payload (a recording has no trigger; its kind is the
            # whole answer).
            assert excinfo.value.holder == 'recording'
            assert excinfo.value.holder_trigger is None

            assert not runner.is_running()
            assert not session.is_protocol_running, (
                'a claim-refused run must not leave session.protocol_running '
                'set: no run exists to ever clear it, so every reader of the '
                'protocol-running state is wedged until app restart'
            )

            # The documented contract: a refusal does not arm
            # wait_for_completion. A caller polling it must return
            # immediately instead of blocking until timeout on a run
            # that never started.
            t0 = time.monotonic()
            assert runner.wait_for_completion(timeout=2), (
                'a claim-refused run must not leave the completion event '
                'cleared; callers polling wait_for_completion would hang '
                'on a run that never started'
            )
            assert time.monotonic() - t0 < 1.0

            # The session is not wedged: once the recording releases the
            # claim, a valid run starts and completes.
            session.activity_claim.release('recording')
            claim_held = False
            done = threading.Event()
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='post_refusal_scan',
                parent_dir=str(tmp_path),
                image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
                callbacks={
                    'run_complete': lambda **kw: done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done.wait(timeout=COMPLETION_TIMEOUT), (
                'a valid run after a claim refusal must start and complete'
            )
            assert not session.is_protocol_running
        finally:
            if claim_held:
                session.activity_claim.release('recording')
            runner.shutdown()
            session.shutdown_executors()
