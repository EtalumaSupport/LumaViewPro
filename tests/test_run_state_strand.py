# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a claim-refused run must not strand caller-committed state.

ProtocolRunner commits caller-side running state (the session's
protocol_running Event) between prepare() and start(). The session
activity claim is gated inside start(), so a refusal for a held claim
(a live video recording) raises AFTER that commit: session.protocol_
running strands True with no run to ever clear it, contradicting the
documented refusal contract ("no state was committed").

The prepare-side refusal contract (refusals that raise before the
commit) is covered by tests/test_run_refusal_contract.py; this file
pins the start-side (claim-gate) refusal specifically.
"""

import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

from tests.settings_fixtures import complete_settings

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
from tests.protocol_drives import wait_until_not_running
from tests.scope_fakes import home_sim_scope

COMPLETION_TIMEOUT = 15  # seconds -- generous for CI


def _make_session_settings(tmp_path):
    return {
        # The objective this file's protocols name: a scope with no turret
        # refuses a protocol for any glass other than the selected one.
        'objective_id': '10x Oly',
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
        'Label': 'A1_test',
        'Auto_Named': False,
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

        session = ScopeSession.create(
            complete_settings(**_make_session_settings(tmp_path)), simulate=True
        )
        # A headless session does not home: an unhomed scope refuses every
        # XY move, and the run would end on its three-strike ceiling instead.
        home_sim_scope(session.scope)
        runner = session.create_protocol_runner()
        claim_held = False
        try:
            # A completed first run arms the completion event, so the
            # refusal below has prior state to preserve (mirrors the
            # prepare-side contract test).
            first_done = threading.Event()
            first = runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='pre_refusal_scan',
                parent_dir=str(tmp_path),
                callbacks={
                    'run_complete': lambda **kw: first_done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert first_done.wait(timeout=COMPLETION_TIMEOUT), 'first run did not end'
            settled = first.wait(timeout_s=COMPLETION_TIMEOUT)
            assert settled is not None, 'the first run never reported an outcome'
            assert (settled.status, settled.reason) == ('completed', 'completed'), (
                f'the first run reported {settled.status!r} ({settled.reason!r})'
            )
            # run_complete and wait() come once the claim is released;
            # the poll confirms it before claiming as the recording.
            deadline = time.monotonic() + COMPLETION_TIMEOUT
            while session.activity_claim.owner is not None:
                assert time.monotonic() < deadline, 'first run never released the claim'
                time.sleep(0.02)
            # The completed run is still writing its files, and prepare()
            # refuses a new run until they land -- a refusal this test is
            # not about.
            while session.protocol_files_draining:
                assert time.monotonic() < deadline, 'first run never finished writing its files'
                time.sleep(0.02)

            # A video recording holds the session's exclusive-activity
            # claim, exactly as the recording engine does for its whole
            # capture + drain lifetime.
            recording = session.activity_claim.try_claim('recording')
            assert recording
            claim_held = True

            with pytest.raises(ProtocolRunRefusedError) as excinfo:
                runner.run_single_scan(
                    protocol=_make_single_step_protocol(),
                    sequence_name='claim_refused_scan',
                    parent_dir=str(tmp_path),
                )
            assert excinfo.value.reason == 'exclusive_activity_running'
            # The refusal says busy-with-what: the holder's KIND rides
            # the payload (a recording has no trigger; its kind is the
            # whole answer).
            assert excinfo.value.holder == 'recording'
            assert excinfo.value.holder_trigger is None

            assert not session.is_protocol_running, (
                'a claim-refused run must not leave session.protocol_running '
                'set: no run exists to ever clear it, so every reader of the '
                'protocol-running state is wedged until app restart'
            )

            # The session is not wedged: once the recording releases the
            # claim, a valid run starts and completes.
            recording.release()
            claim_held = False
            done = threading.Event()
            runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='post_refusal_scan',
                parent_dir=str(tmp_path),
                callbacks={
                    'run_complete': lambda **kw: done.set(),
                    'files_complete': lambda **kw: None,
                },
            )
            assert done.wait(timeout=COMPLETION_TIMEOUT), (
                'a valid run after a claim refusal must start and complete'
            )
            # run_complete comes once the claim is released; this
            # confirms it.
            assert wait_until_not_running(session)
        finally:
            if claim_held:
                recording.release()
            session.shutdown()


class TestTheHolderIsTheLiveRun:
    """Who holds the microscope is answered by the thing that knows
    whether anything holds it: the session's activity claim."""

    def test_the_claim_names_the_run_while_it_holds_the_scope(self, tmp_path):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create(
            complete_settings(**_make_session_settings(tmp_path)), simulate=True
        )
        # A headless session does not home, and a run is refused while any
        # axis position is unknown.
        home_sim_scope(session.scope)
        runner = session.create_protocol_runner()
        try:
            # Read from inside the run: run_scan_pre fires from the run
            # loop, with the claim held, so this observes the holder while
            # it holds rather than racing the run's end. run_complete no
            # longer can: it comes after the run has let go.
            observed = {}

            def _observe(**_kwargs):
                holder = session.activity_claim.holder
                observed['kind'] = holder.kind if holder is not None else None
                observed['trigger'] = holder.run_trigger_source if holder is not None else None

            run = runner.run_single_scan(
                protocol=_make_single_step_protocol(),
                sequence_name='holder_scan',
                parent_dir=str(tmp_path),
                callbacks={'run_scan_pre': _observe, 'files_complete': lambda **kw: None},
            )
            assert run.wait(timeout_s=COMPLETION_TIMEOUT) is not None

            assert observed['kind'] == 'protocol', (
                'the run did not hold the claim while it was running'
            )
            assert observed['trigger'] == 'api_scan', (
                "the claim must carry the run's own trigger, not a constant"
            )

            assert wait_until_not_running(session)
            assert session.activity_claim.holder is None
        finally:
            session.shutdown()
