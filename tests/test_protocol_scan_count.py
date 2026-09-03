"""A full-protocol run always performs at least one scan.

Period 0 and Duration 0 are the loader's single-scan markers (Manual
Z-Stack and single-shot capture write Period 0 into their TSV). The
runner's scan count is ``int(duration / period)`` guarded for a zero
period -- but ``Protocol.period()`` is a ``timedelta`` and
``timedelta(0) == 0`` is False, so the guard added in May 2026 never
fired: a Period 0 protocol raised ZeroDivisionError on Start, and a
Duration shorter than one Period produced a zero-scan run that reported
'completed' without capturing anything. The May tests passed because
they stubbed the period as an ``int``. These tests use a real Protocol.
"""

import datetime
import pathlib

import pandas as pd
import pytest

from modules.protocol import Protocol
from modules.sequenced_capture_runner import SequencedCaptureRunMode, SequencedCaptureRunner

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _step():
    return {
        'Name': 'A1_BF',
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
        'Video Config': {'fps': 5, 'duration': 5},
        'Stim_Config': {},
        'Step Index': 0,
        'Auto_Named': True,
        'Label': '',
    }


def _protocol(*, period: datetime.timedelta, duration: datetime.timedelta) -> Protocol:
    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_step()]),
            'period': period,
            'duration': duration,
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _scan_count(protocol: Protocol, max_scans=None) -> int:
    return SequencedCaptureRunner._calculate_num_scans(
        protocol=protocol,
        run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
        max_scans=max_scans,
    )


@pytest.fixture
def notices(monkeypatch):
    import modules.notification_center as notification_center

    captured = []
    monkeypatch.setattr(
        notification_center.notifications,
        'notice',
        lambda *args, **kwargs: captured.append(args),
    )
    return captured


def test_period_zero_runs_one_scan(notices):
    protocol = _protocol(period=datetime.timedelta(0), duration=datetime.timedelta(hours=1))
    assert _scan_count(protocol) == 1
    assert len(notices) == 1, 'the user is told once that the run is a single scan'


def test_period_zero_reloaded_from_tsv_runs_one_scan(tmp_path, notices):
    # The bench recipe that found the crash: a Period 0 protocol on disk.
    protocol = _protocol(period=datetime.timedelta(0), duration=datetime.timedelta(hours=1))
    path = tmp_path / 'period_zero.tsv'
    protocol.to_file(path)
    reloaded = Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS)
    assert _scan_count(reloaded) == 1


def test_duration_zero_runs_one_scan(notices):
    protocol = _protocol(period=datetime.timedelta(minutes=10), duration=datetime.timedelta(0))
    assert _scan_count(protocol) == 1
    assert len(notices) == 1


def test_duration_shorter_than_one_period_runs_one_scan(notices):
    protocol = _protocol(
        period=datetime.timedelta(hours=1), duration=datetime.timedelta(minutes=30)
    )
    assert _scan_count(protocol) == 1
    assert len(notices) == 1


def test_ordinary_protocol_count_is_unchanged_and_silent(notices):
    protocol = _protocol(
        period=datetime.timedelta(minutes=20), duration=datetime.timedelta(hours=2)
    )
    assert _scan_count(protocol) == 6
    assert _scan_count(protocol, max_scans=2) == 2
    assert notices == [], 'a multi-scan run gets no single-scan notice'


def test_exact_timedelta_arithmetic_is_kept():
    # 0.09 min / 0.09 h is exactly 60 periods in microseconds; dividing the
    # two as float seconds gives 59.999..., which truncates to 59. The
    # count must come from the timedelta division.
    protocol = _protocol(
        period=datetime.timedelta(minutes=0.09), duration=datetime.timedelta(hours=0.09)
    )
    assert _scan_count(protocol) == 60
