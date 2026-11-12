# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler camera is brought up stopping in the mode that finishes the exposure in flight.

The ``Default`` user set ``init_camera_config`` loads leaves
``BslAcquisitionStopMode`` at whatever the body shipped with. On a daA3840 at
firmware 1.1.0 that is ``AbortExposure``, under which a stop hangs 11 s in the
camera's own ``AcquisitionStop`` and, once one has, nearly every later stop
does until power-off; under ``CompleteExposure`` the same body stopped 321
times without one. So bring-up writes ``CompleteExposure`` on every camera
that offers it. A camera without the node, or one that does not take
``CompleteExposure``, keeps its own mode, says which at INFO, and that is no
step left undone: ``config_not_applied`` stays empty.

The driver's configuration runs over the suite's SDK stand-ins
(``init_configurable_pylon_camera``); the emulator test runs the production
driver against Basler's own camera emulator.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys

import pytest
from pypylon import genicam

from tests.camera_fakes import init_configurable_pylon_camera


class _StopModeNode:
    """``BslAcquisitionStopMode`` as the camera hands it: an enumeration
    offering ``offered`` (what pypylon's ``GetSymbolics`` lists), holding
    ``value``, writable or not (read by the suite's ``genicam.IsWritable``)."""

    def __init__(self, offered, value, writable=True):
        self.offered = list(offered)
        self.value = value
        self.writable = writable

    def GetSymbolics(self):
        return list(self.offered)

    def GetValue(self):
        return self.value

    def SetValue(self, value):
        self.value = value


def _configured(stop_mode):
    """A Basler camera, configured, whose ``BslAcquisitionStopMode`` is
    ``stop_mode`` (absent when None, raising as pypylon's ``GetNode`` does);
    every other node is the fake body's, as the rest of the configuration
    expects."""
    # a stand-in by design: no simulated camera runs pyloncamera.py, and the
    # subject is what the driver writes to the node the SDK hands it.
    cam = init_configurable_pylon_camera()
    get_node = cam.active.GetNodeMap.return_value.GetNode
    body = get_node.side_effect

    def with_the_stop_mode(name):
        if name != 'BslAcquisitionStopMode':
            return body(name)
        if stop_mode is None:
            raise genicam.LogicalErrorException(f'Node not existing: {name}')
        return stop_mode

    get_node.side_effect = with_the_stop_mode
    cam.init_camera_config()
    return cam


def test_a_camera_offering_complete_exposure_is_set_to_it():
    node = _StopModeNode(['CompleteExposure', 'AbortExposure'], 'AbortExposure')

    cam = _configured(node)

    assert node.value == 'CompleteExposure'
    assert cam.config_not_applied == ()


# parametrize-justified: each camera is its own node set, built per case.
@pytest.mark.parametrize(
    ('stop_mode', 'kept'),
    [
        (None, '<missing>'),
        (_StopModeNode(['AbortExposure'], 'AbortExposure'), 'AbortExposure'),
        (
            _StopModeNode(['CompleteExposure', 'AbortExposure'], 'AbortExposure', writable=False),
            'AbortExposure',
        ),
    ],
    ids=['no_node', 'not_offered', 'not_writable'],
)
def test_a_camera_that_does_not_take_complete_exposure_keeps_its_mode_and_says_which(
    stop_mode, kept, caplog
):
    with caplog.at_level(logging.INFO):
        cam = _configured(stop_mode)

    if stop_mode is not None:
        assert stop_mode.value == kept
    assert cam.config_not_applied == ()
    told = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and 'BslAcquisitionStopMode' in r.getMessage()
    ]
    assert len(told) == 1, told
    if stop_mode is not None:
        assert f'kept at {kept}' in told[0], told


_EMULATOR_BRING_UP = """
import json, logging, sys
sys.path.insert(0, 'src')
said = []
class _Record(logging.Handler):
    def emit(self, record):
        said.append([record.levelname, record.getMessage()])
try:
    import pypylon  # noqa: F401
except ImportError:
    print(json.dumps({'sdk': False})); sys.exit(0)
from _lumascope.drivers.pyloncamera import PylonCamera
_root = logging.getLogger()
_root.addHandler(_Record())
_root.setLevel(logging.INFO)
cam = PylonCamera()  # the constructor connects
if not cam.active:
    print(json.dumps({'sdk': True, 'connected': False})); sys.exit(0)
try:
    not_applied = list(cam.config_not_applied)
finally:
    cam.disconnect()
print(json.dumps({'sdk': True, 'connected': True, 'not_applied': not_applied, 'said': said}))
"""


@pytest.mark.slow
def test_the_emulator_is_brought_up_naming_its_stop_mode_with_every_step_applied():
    """The production driver against Basler's emulator (``PYLON_CAMEMU``):
    bring-up says the stop mode it wrote or the one the camera keeps, and
    names no step not applied."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-I', '-c', _EMULATOR_BRING_UP],
        cwd=str(repo),
        env={**os.environ, 'PYLON_CAMEMU': '1'},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    if not answer['sdk']:
        pytest.skip('pypylon is not installed on this host')
    if not answer['connected']:
        pytest.skip("pypylon's camera emulator did not open on this host")

    assert answer['not_applied'] == []
    told = [
        message
        for level, message in answer['said']
        if level == 'INFO' and any(line in message for line in _THE_STEP_SAYS)
    ]
    assert len(told) == 1, told


# The stop-mode step's own lines: the mode written, or the mode kept.
_THE_STEP_SAYS = (
    "BslAcquisitionStopMode.SetValue('CompleteExposure')",
    'BslAcquisitionStopMode kept at',
    'BslAcquisitionStopMode not offered',
)
