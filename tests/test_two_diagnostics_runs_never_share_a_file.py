# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Two diagnostics runs inside one second leave two files, or say why not.

The grab-lifecycle benchmark and the camera probe each write a JSON file named
by what was measured and when. Named to the second, two runs of the same
camera in one second shared a name and the later silently replaced the
earlier: a benchmark under ``-n auto`` read back another test's file. The
clock is frozen here so both runs ask for the same instant; the first file
must survive, and the second run must say it could not write.
"""

import datetime
import json
import types

import pytest

import modules.lumascope_api.diagnostics as diag
from modules.lumascope_api import Lumascope
from modules.lumascope_api.diagnostics import DiagnosticsAPI
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.runtime_state import RuntimeState
from tests.scope_fakes import build_scope, give_stub_lanes

_INSTANT = datetime.datetime(2026, 10, 1, 12, 0, 0, 123456)


class _FrozenDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return _INSTANT.replace(tzinfo=tz) if tz is not None else _INSTANT


@pytest.fixture
def frozen_clock(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, 'log_dir', str(tmp_path))
    monkeypatch.setattr(
        diag,
        'datetime',
        types.SimpleNamespace(
            datetime=_FrozenDatetime, UTC=datetime.UTC, timedelta=datetime.timedelta
        ),
    )
    return tmp_path


def test_two_benchmarks_in_one_instant_do_not_overwrite(frozen_clock):
    scope = build_scope(simulate=True)
    if not scope.imaging.is_streaming():
        scope.imaging.start_streaming()

    first = scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=2)
    second = scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=3)

    assert first['written_to'] is not None, first['errors']
    with open(first['written_to']) as f:
        assert json.load(f)['num_cycles'] == 2
    assert second['written_to'] is None
    assert any('Persist failed: FileExistsError' in e for e in second['errors'])
    assert len(list((frozen_clock / 'camera_timing').iterdir())) == 1


class _ProbeCamera:
    active = True

    def __init__(self):
        self.calls = 0

    def read_diagnostic_snapshot(self, duration_s, drain_camera_side_errors):
        self.calls += 1
        return {
            'connected': True,
            'supported': True,
            'camera': {'model_name': 'a2A1920', 'serial': '123'},
            'config': {},
            'probe_number': self.calls,
        }

    def get_sdk_info(self):
        return {'name': 'Basler pylon', 'version': '11.5'}


def _scope_with(camera) -> Lumascope:
    scope = Lumascope.__new__(Lumascope)
    scope.runtime_state = RuntimeState(scope)
    scope._camera_driver = camera
    give_stub_lanes(scope)
    scope.imaging = ImagingAPI.__new__(ImagingAPI)
    scope.imaging._scope = scope
    scope.diagnostics = DiagnosticsAPI(scope)
    return scope


def test_two_probes_in_one_instant_do_not_overwrite(frozen_clock):
    scope = _scope_with(_ProbeCamera())

    first = scope.diagnostics.run_pylon_diagnostic_probe(duration_s=0.0)
    second = scope.diagnostics.run_pylon_diagnostic_probe(duration_s=0.0)

    assert 'output_path' in first, first.get('errors')
    with open(first['output_path']) as f:
        assert json.load(f)['probe_number'] == 1
    assert 'output_path' not in second
    assert any('JSON write failed: FileExistsError' in e for e in second['errors'])
    assert len(list((frozen_clock / 'camera_probe').iterdir())) == 1


def test_the_name_carries_the_microseconds(frozen_clock):
    scope = build_scope(simulate=True)
    if not scope.imaging.is_streaming():
        scope.imaging.start_streaming()

    result = scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=1)

    assert result['written_to'].endswith('_20261001_120000_123456.json')
