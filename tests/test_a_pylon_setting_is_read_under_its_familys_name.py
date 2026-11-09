# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon setting a camera family names its own way is read under that name.

Basler names two settings with a ``Bsl`` prefix on ace 2 / boost / dart M/R
and without on legacy ace (resulting-acquisition-frame-rate.html,
exposure-time.html): ``BslResultingAcquisitionFrameRate`` /
``ResultingFrameRate`` and ``BslEffectiveExposureTime`` / ``ExposureTime``.
The effective exposure is the value the camera used, after its clock
rounding; ``ExposureTime`` is the value set. Each list is declared once on
``PylonCamera`` and read under two contracts:

- a live reader (the stats poller, ``get_exposure_t``,
  ``get_resulting_frame_rate``) reads the first name the camera offers; a
  failed read, or a first name offered but holding no value now, raises to
  the caller's handler and is never answered by the second name;
- the diagnostic snapshot (``_safe_node``) tries the next name on any
  answer but a value, since it reports everything.
"""

from __future__ import annotations

import logging
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from _lumascope.drivers import pyloncamera
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.drivers.pyloncamera import PylonCamera
from tests.camera_fakes import (
    RecordingNodeMap,
    bare_pylon_camera,
    diag_snapshot_pylon_camera,
    nodemaps_answer_attributes,
    run_one_stats_poll,
    stats_poll_pylon_camera,
)
from tests.log_capture import records_of


def _without(cam, *names):
    """The camera lacks ``names``: its device map raises for them, as pypylon's does."""
    present = cam.active.GetNodeMap.return_value.GetNode.side_effect

    def get_node(name):
        if name in names:
            raise genicam.LogicalErrorException(f'Node not existing: {name}')
        return present(name)

    cam.active.GetNodeMap.return_value.GetNode.side_effect = get_node
    return cam


def _poll_csv_row(cam):
    trace_mod = MagicMock()
    with mock.patch.object(pyloncamera, 'profile_trace', trace_mod):
        run_one_stats_poll(cam)
    stats_calls = [
        call for call in trace_mod.trace.call_args_list if call.args[0] == 'pylon_stats_trace.csv'
    ]
    assert stats_calls
    return stats_calls[0].args[1].split(','), stats_calls[0].args[2]


def test_safe_node_falls_back_across_names():
    """_safe_node must probe each candidate name in order so call
    sites can pass Bsl-prefixed-then-legacy pairs."""
    nodemap = RecordingNodeMap({'ResultingFrameRate': 10.0})
    value = PylonCamera._safe_node(nodemap, *PylonCamera._RESULTING_FRAME_RATE_NODES)
    assert value == 10.0
    assert nodemap.requested == ['BslResultingAcquisitionFrameRate', 'ResultingFrameRate']


def test_stats_poller_prefers_bsl_resulting_frame_rate():
    """The live fps written to pylon_stats_trace.csv must come from
    BslResultingAcquisitionFrameRate when the camera exposes it
    (canonical for ace 2 / dart M/R per Basler doc)."""
    cam = stats_poll_pylon_camera()
    cam.active.ResultingFrameRate.GetValue.return_value = 10.0
    columns, row = _poll_csv_row(cam)
    assert row[columns.index('resulting_fps')] == '30.000'


def test_stats_poller_falls_back_to_legacy_frame_rate():
    cam = _without(stats_poll_pylon_camera(), 'BslResultingAcquisitionFrameRate')
    cam.active.ResultingFrameRate.GetValue.return_value = 10.0
    columns, row = _poll_csv_row(cam)
    assert row[columns.index('resulting_fps')] == '10.000'


def test_a_failed_temperature_read_is_told_and_the_poller_runs_on(caplog):
    cam = stats_poll_pylon_camera()
    cam.active.TemperatureState.GetValue.side_effect = genicam.RuntimeException('usb gone')
    with caplog.at_level(logging.WARNING):
        columns, _row = _poll_csv_row(cam)
    assert columns, 'the cycle still writes its row'
    warned = [
        r.getMessage() for r in records_of(caplog, pyloncamera) if r.levelno == logging.WARNING
    ]
    assert any('TemperatureState' in m and 'usb gone' in m for m in warned), warned


def test_get_exposure_t_prefers_bsl_effective():
    """get_exposure_t must report the effective exposure (what the
    camera actually used, per exposure-time.html) when the Bsl node
    is exposed -- not the requested set value."""
    cam = nodemaps_answer_attributes(bare_pylon_camera())
    cam.active.BslEffectiveExposureTime.GetValue.return_value = 5000.0
    cam.active.ExposureTime.GetValue.return_value = 4000.0
    assert cam.get_exposure_t() == 5.0


def test_get_exposure_t_falls_back_to_set_value():
    cam = _without(nodemaps_answer_attributes(bare_pylon_camera()), 'BslEffectiveExposureTime')
    cam.active.ExposureTime.GetValue.return_value = 4000.0
    assert cam.get_exposure_t() == 4.0


def test_a_failed_effective_exposure_read_is_not_answered_by_the_set_value():
    """A failed read of the effective exposure is the read's failure, raised
    with the SDK's words: never the set value in its place."""
    cam = nodemaps_answer_attributes(bare_pylon_camera())
    cam.active.BslEffectiveExposureTime.GetValue.side_effect = genicam.RuntimeException('usb gone')
    cam.active.ExposureTime.GetValue.return_value = 4000.0
    with pytest.raises(CameraReadFailedError, match='usb gone'):
        cam.get_exposure_t()


def test_diag_snapshot_prefers_bsl_nodes():
    """The snapshot's exposure / frame-rate entries must come from
    the Bsl-prefixed canonical nodes when both forms are exposed."""
    cam, _nodemap, _grabber = diag_snapshot_pylon_camera(
        camera_values={
            'BslEffectiveExposureTime': 5000.0,
            'ExposureTime': 4000.0,
            'BslResultingAcquisitionFrameRate': 30.0,
            'ResultingFrameRate': 10.0,
        }
    )
    result = cam.read_diagnostic_snapshot(duration_s=0)
    assert result['config']['exposure_us'] == 5000.0
    assert result['config']['resulting_frame_rate'] == 30.0
