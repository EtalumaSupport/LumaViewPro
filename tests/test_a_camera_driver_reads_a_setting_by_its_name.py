# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera driver reads one of its camera's settings by the camera's own name (``Camera.read_node``).

The answer is the camera's: the value with the unit the node states (an
enumeration's symbolic name), what kind of node holds no value, or None for
a name the map does not have. A read the camera or SDK does not answer
raises ``CameraReadFailedError``, chained to the SDK's error, and is never
written down as a value.

The Pylon and IDS SDKs are not in the test process, so the drivers read the
typed stand-ins ``tests/pypylon_stub.py`` and ``tests/ids_peak_stub.py``
model on each SDK (a stand-in by design: the SDKs are not on the test host).
The Pylon driver is also read against Basler's own camera emulator, in a
child process, since the stand-in replaces pypylon in this one.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest
from pypylon import genicam

from _lumascope.drivers.camera import CameraNodeHasNoValue, CameraNodeValue
from _lumascope.drivers.exceptions import CameraReadFailedError
from tests import ids_peak_stub as ids_peak
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera


class _PylonNodeMap:
    """GetNode as pypylon's: a name the map lacks raises LogicalErrorException."""

    def __init__(self, nodes):
        self._nodes = nodes

    def GetNode(self, name):
        if name not in self._nodes:
            raise genicam.LogicalErrorException(f'Node not existing: {name}')
        return self._nodes[name]


class _IdsNodeMap:
    """HasNode and FindNode as ids_peak's: FindNode raises for a name the map lacks."""

    def __init__(self, nodes):
        self._nodes = nodes

    def HasNode(self, name):
        return name in self._nodes

    def FindNode(self, name):
        if name not in self._nodes:
            raise ids_peak.NotFoundException(name)
        return self._nodes[name]


def _pylon(nodes, node_map='device'):
    cam = bare_pylon_camera()
    getattr(cam.active, PYLON_GETTERS[node_map]).return_value = _PylonNodeMap(nodes)
    return cam


PYLON_GETTERS = {
    'device': 'GetNodeMap',
    'transport': 'GetTLNodeMap',
    'stream': 'GetStreamGrabberNodeMap',
    'instant_camera': 'GetInstantCameraNodeMap',
    'event': 'GetEventGrabberNodeMap',
}


def _ids(nodes, node_map='device'):
    cam = bare_ids_camera()
    if node_map == 'device':
        cam.remote_nodemap = _IdsNodeMap(nodes)
    else:
        cam.data_stream.NodeMaps.return_value = [_IdsNodeMap(nodes)]
    return cam


# --- Pylon --------------------------------------------------------------------


def test_a_pylon_value_node_answers_its_value_and_unit_on_each_map():
    for node_map in PYLON_GETTERS:
        cam = _pylon({'Gain': genicam.IFloat(1.5, unit='dB')}, node_map)

        assert cam.read_node('Gain', node_map) == CameraNodeValue(1.5, 'dB'), node_map


def test_a_pylon_enumeration_answers_its_symbolic_name_with_no_unit():
    cam = _pylon({'PixelFormat': genicam.IEnumeration('Mono12')})

    assert cam.read_node('PixelFormat', 'device') == CameraNodeValue('Mono12', '')


def test_a_name_the_pylon_map_lacks_answers_none():
    cam = _pylon({'Gain': genicam.IFloat(0.0, unit='dB')})

    assert cam.read_node('BslChunkTimestampSelector', 'device') is None


def test_a_pylon_node_with_no_value_answers_what_kind_it_is():
    kinds = {
        'AcquisitionStart': (genicam.ICommand(), CameraNodeHasNoValue.COMMAND),
        'Root': (genicam.ICategory(), CameraNodeHasNoValue.CATEGORY),
        'LUTValueAll': (genicam.IRegister(), CameraNodeHasNoValue.REGISTER),
        'Device': (genicam.IPort(), CameraNodeHasNoValue.PORT),
        'EnumEntry_GainAuto_Off': (genicam.IEnumEntry(), CameraNodeHasNoValue.ENUM_ENTRY),
        'Gain': (genicam.IFloat(0.0, readable=False), CameraNodeHasNoValue.NOT_READABLE_NOW),
    }
    cam = _pylon({name: node for name, (node, _kind) in kinds.items()})

    answers = {name: cam.read_node(name, 'device') for name in kinds}

    assert answers == {name: kind for name, (_node, kind) in kinds.items()}


def test_a_pylon_read_the_camera_does_not_answer_raises_chained_to_the_sdk_error():
    timeout = genicam.TimeoutException('no answer')
    cam = _pylon({'DeviceTemperature': genicam.IFloat(timeout, unit='C')})

    with pytest.raises(CameraReadFailedError) as raised:
        cam.read_node('DeviceTemperature', 'device')

    assert raised.value.reason == 'camera_read_failed'
    assert raised.value.__cause__ is timeout


def test_a_pylon_map_the_sdk_does_not_give_raises():
    cam = bare_pylon_camera()
    refused = genicam.RuntimeException('stream grabber closed')
    cam.active.GetStreamGrabberNodeMap.side_effect = refused

    with pytest.raises(CameraReadFailedError) as raised:
        cam.read_node('Statistic_Total_Buffer_Count', 'stream')

    assert raised.value.__cause__ is refused


# --- IDS ----------------------------------------------------------------------


def test_an_ids_value_node_answers_its_value_and_unit_on_each_map():
    for node_map in ('device', 'stream'):
        cam = _ids({'ExposureTime': ids_peak.FloatNode(5000.0, unit='us')}, node_map)

        assert cam.read_node('ExposureTime', node_map) == CameraNodeValue(5000.0, 'us'), node_map


def test_an_ids_enumeration_answers_its_current_entry_with_no_unit():
    cam = _ids({'PixelFormat': ids_peak.EnumerationNode('Mono10g40IDS')})

    assert cam.read_node('PixelFormat', 'device') == CameraNodeValue('Mono10g40IDS', '')


def test_a_name_the_ids_map_lacks_answers_none():
    cam = _ids({'Gain': ids_peak.FloatNode(1.0)})

    assert cam.read_node('DeviceTemperature', 'device') is None


def test_an_ids_node_with_no_value_answers_what_kind_it_is():
    kinds = {
        'DeviceReset': (ids_peak.CommandNode(), CameraNodeHasNoValue.COMMAND),
        'Root': (ids_peak.CategoryNode(), CameraNodeHasNoValue.CATEGORY),
        'LUTValueAll': (ids_peak.RegisterNode(), CameraNodeHasNoValue.REGISTER),
        'Mono8': (ids_peak.EnumerationEntryNode('Mono8'), CameraNodeHasNoValue.ENUM_ENTRY),
        'Gain': (ids_peak.FloatNode(1.0, readable=False), CameraNodeHasNoValue.NOT_READABLE_NOW),
    }
    cam = _ids({name: node for name, (node, _kind) in kinds.items()})

    answers = {name: cam.read_node(name, 'device') for name in kinds}

    assert answers == {name: kind for name, (_node, kind) in kinds.items()}


def test_an_ids_read_the_camera_does_not_answer_raises_chained_to_the_sdk_error():
    lost = ids_peak.IOException('device disconnected')
    cam = _ids({'DeviceTemperature': ids_peak.FloatNode(lost, unit='C')})

    with pytest.raises(CameraReadFailedError) as raised:
        cam.read_node('DeviceTemperature', 'device')

    assert raised.value.reason == 'camera_read_failed'
    assert raised.value.__cause__ is lost


def test_an_ids_stream_with_no_node_map_raises():
    cam = bare_ids_camera()
    cam.data_stream.NodeMaps.return_value = []

    with pytest.raises(CameraReadFailedError):
        cam.read_node('StreamDeliveredFrameCount', 'stream')


# --- Pylon against Basler's camera emulator -----------------------------------

_EMULATOR_READS = """
import json, logging, sys
sys.path.insert(0, 'src')
logging.disable(logging.CRITICAL)
try:
    import pypylon  # noqa: F401
except ImportError:
    print(json.dumps({'sdk': False})); sys.exit(0)
from _lumascope.drivers.camera import CameraNodeValue
from _lumascope.drivers.pyloncamera import PylonCamera
cam = PylonCamera()
if not cam.connect():
    print(json.dumps({'sdk': True, 'connected': False})); sys.exit(0)

def text(answer):
    if isinstance(answer, CameraNodeValue):
        return [answer.value, answer.unit]
    return None if answer is None else str(answer)

try:
    reads = {f'{m}:{n}': text(cam.read_node(n, m)) for m, n in json.loads(sys.argv[1])}
finally:
    cam.disconnect()
print(json.dumps({'sdk': True, 'connected': True, 'reads': reads}))
"""


def test_the_pylon_driver_reads_the_emulators_nodes_by_name():
    """On Basler's emulator (``PYLON_CAMEMU``) the production driver reads each
    kind of node on the map that holds it: a value with its unit, an
    enumeration's name, a name the emulator lacks, a command, the stream
    grabber's counter and the instant camera's buffer count."""
    expected = {
        'device:Gain': [0.0, 'dB'],
        'device:PixelFormat': ['Mono8', ''],
        'device:BslChunkTimestampSelector': None,
        'device:AcquisitionStart': 'command',
        'stream:Statistic_Total_Buffer_Count': [0, ''],
        'instant_camera:NumQueuedBuffers': [0, ''],
    }
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-I', '-c', _EMULATOR_READS, json.dumps([k.split(':') for k in expected])],
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

    assert answer['reads'] == expected
