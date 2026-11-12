# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon node the camera lacks is told as absent where it is asked for.

pypylon raises ``genicam.LogicalErrorException`` for a node the map does not
hold, on ``GetNode`` and on the instant camera's attribute alike; it never
answers ``None``. ``PylonCamera._find_node`` is the one lookup that answers
absence as absence (guard ``tests/guards/test_a_pylon_node_is_found_one_way``).
Each site that asks then does what its contract says for a camera without the
node, and a read or write that fails is a failure, not an absence:

- the bench knobs (``set_device_link_throughput_limit``,
  ``set_max_transfer_size``, ``set_num_max_queued_urbs``,
  ``set_bandwidth_reserve_mode``,
  ``set_gev_packet_size``, ``set_gev_inter_packet_delay``) answer ``False``
  with one INFO line, so the bench sweep calls every one on every camera; a
  write the camera does not take raises;
- the idle wait at disconnect polls ``AcquisitionActive`` alone on a camera
  without ``ExposureActive``, and cannot poll one without
  ``AcquisitionActive``;
- the stream grabber's status line names a missing node;
- the range query at connect tells a camera without ``Gain``,
  ``ExposureTime`` or ``SensorPixelWidth`` at INFO and leaves the profile's
  range unset; a failed pixel pitch read is told at WARNING;
- a failed read of a tick-rate node is told at WARNING.

Connect and the diagnostic snapshot are read against Basler's own emulator,
the camera whose node set no hand-built fake mirrors.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pypylon import genicam

from _lumascope.drivers.exceptions import CameraNoAnswerError, CameraWriteFailedError
from _lumascope.drivers.pyloncamera import PylonCamera
from tests.camera_fakes import bare_pylon_camera


class _NodeMap:
    """A nodemap holding ``nodes``; any other name raises, as pypylon's does."""

    def __init__(self, nodes=None):
        self.nodes = dict(nodes or {})

    def GetNode(self, name):
        try:
            return self.nodes[name]
        except KeyError:
            raise genicam.LogicalErrorException(f'Node not existing: {name}') from None


class _InstantCamera:
    """The instant camera's surface the lookups reach: its device and stream
    grabber nodemaps, and its attributes, which name the device map's nodes
    and raise for a node the map lacks, as pypylon's do.

    a stand-in by design: the subject is the driver's answer to what the SDK
    hands it, and no simulated camera runs pyloncamera.py.
    """

    def __init__(self, device=None, stream=None):
        self._device = _NodeMap(device)
        self._stream = _NodeMap(stream)

    def GetNodeMap(self):
        return self._device

    def GetStreamGrabberNodeMap(self):
        return self._stream

    def IsGrabbing(self):
        return False

    # The teardown a dropped camera runs at garbage collection.
    def Close(self):
        pass

    def DetachDevice(self):
        pass

    def DestroyDevice(self):
        pass

    def __getattr__(self, name):
        return self._device.GetNode(name)


def _camera(device=None, stream=None):
    cam = bare_pylon_camera()
    cam.active = _InstantCamera(device, stream)
    return cam


# Each bench knob: the call a caller makes, and the nodemap and node it writes
# (DLTL's first node; its limit is written after the mode).
_KNOBS = [
    (
        'dltl',
        lambda c: c.set_device_link_throughput_limit('Off'),
        'device',
        'DeviceLinkThroughputLimitMode',
    ),
    ('max_transfer', lambda c: c.set_max_transfer_size(262144), 'stream', 'MaxTransferSize'),
    ('queued_urbs', lambda c: c.set_num_max_queued_urbs(64), 'stream', 'NumMaxQueuedUrbs'),
    (
        'bw_reserve',
        lambda c: c.set_bandwidth_reserve_mode('Performance'),
        'device',
        'BandwidthReserveMode',
    ),
    ('packet_size', lambda c: c.set_gev_packet_size(1500), 'device', 'GevSCPSPacketSize'),
    ('packet_delay', lambda c: c.set_gev_inter_packet_delay(100), 'device', 'GevSCPD'),
]


def _knob_camera(node_map, name, node):
    return _camera(**{node_map: {name: node}})


def test_a_bench_knob_the_camera_lacks_answers_false_with_one_info_line(caplog):
    offenders = []
    for label, call, _node_map, name in _KNOBS:
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            try:
                answer = call(_camera())
            except Exception as e:  # the answer under test: a raise is named
                offenders.append(f'{label}: raised {type(e).__name__}: {e}')
                continue
        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        told = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and 'not offered' in r.getMessage()
        ]
        if answer is not False:
            offenders.append(f'{label}: answered {answer!r}')
        if loud:
            offenders.append(f'{label}: logged {loud}')
        if len({r.getMessage() for r in told}) != 1 or name not in told[0].getMessage():
            offenders.append(f'{label}: told {[r.getMessage() for r in told]}')
    assert not offenders, '\n'.join(offenders)


@pytest.mark.parametrize(
    ('failure', 'raised'),
    [
        (genicam.AccessException('Node is not writable'), CameraWriteFailedError),
        (genicam.TimeoutException('no answer'), CameraNoAnswerError),
    ],
    ids=['refused', 'timeout'],
)
def test_a_bench_knob_write_the_camera_does_not_take_raises(failure, raised):
    offenders = []
    for label, call, node_map, name in _KNOBS:
        node = MagicMock()
        node.SetValue.side_effect = failure
        try:
            answer = call(_knob_camera(node_map, name, node))
        except raised as e:
            if str(failure) not in str(e):
                offenders.append(f'{label}: {type(e).__name__} without the SDK words: {e}')
            continue
        except Exception as e:  # the answer under test: a raise is named
            offenders.append(f'{label}: raised {type(e).__name__}: {e}')
            continue
        offenders.append(f'{label}: answered {answer!r}')
    assert not offenders, '\n'.join(offenders)


def _flag(value):
    node = MagicMock()
    node.GetValue.return_value = value
    return node


def test_the_idle_wait_polls_acquisition_alone_on_a_camera_without_exposure_active():
    cam = _camera(device={'AcquisitionActive': _flag(False)})
    assert PylonCamera._wait_for_acquisition_idle(cam, timeout_s=0.5) is True


def test_the_idle_wait_cannot_poll_a_camera_without_acquisition_active():
    """Older firmware / non-Basler cameras may not expose AcquisitionActive;
    the wait answers False at once so disconnect goes on."""
    cam = _camera()
    assert PylonCamera._wait_for_acquisition_idle(cam, timeout_s=2.0) is False


def test_a_failed_idle_poll_is_told_at_warning(caplog):
    lost = MagicMock()
    lost.GetValue.side_effect = genicam.RuntimeException('usb gone')
    cam = _camera(device={'AcquisitionActive': lost})
    with caplog.at_level(logging.WARNING):
        assert PylonCamera._wait_for_acquisition_idle(cam, timeout_s=0.5) is False
    assert any('usb gone' in r.getMessage() for r in caplog.records)


def test_the_stream_grabber_status_line_names_a_missing_node(caplog):
    cam = _camera()
    with caplog.at_level(logging.DEBUG):
        cam._log_stream_grabber_status('pre-StartGrabbing')
    said = [r.getMessage() for r in caplog.records]
    assert 'pylon StreamGrabber.Status [pre-StartGrabbing]: <node missing>' in said
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failed_tick_rate_read_is_told_at_warning_and_the_read_goes_on(caplog):
    """A present tick node whose read fails is a failure, told naming the
    node; the read then goes on as for an absent one (the next name, then
    the USB default), since what a failed tick read publishes is decided
    with the other range fields."""
    failing = MagicMock()
    failing.GetValue.side_effect = genicam.RuntimeException('usb gone')
    cam = _camera(device={'GevTimestampTickFrequency': failing})
    cam.active.GetDeviceInfo = lambda: MagicMock(GetDeviceClass=lambda: 'BaslerUsb')
    with caplog.at_level(logging.WARNING):
        cam._read_timestamp_tick_frequency()
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('GevTimestampTickFrequency' in m and 'usb gone' in m for m in warned), warned
    assert cam.timestamp_tick_frequency_hz == PylonCamera._BASLER_USB3_DEFAULT_TICK_HZ


# --- connect and the snapshot against Basler's camera emulator --------------

_EMULATOR_CONNECT = """
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
cam = PylonCamera()
if not cam.connect():
    print(json.dumps({'sdk': True, 'connected': False})); sys.exit(0)
try:
    snapshot = cam.read_diagnostic_snapshot(duration_s=0.2)
    firmware = cam.device_firmware
finally:
    cam.disconnect()
print(json.dumps({'sdk': True, 'connected': True, 'firmware': firmware,
                  'errors': snapshot['errors'], 'said': said}))
"""


@pytest.mark.slow
def test_the_emulator_connects_and_snapshots_without_a_failure_for_a_node_it_lacks():
    """Basler's emulator (``PYLON_CAMEMU``) has no ``MaxTransferSize``,
    ``DeviceSerialNumber``, ``ChunkSelector`` or ``Bsl`` error nodes. Its
    connect logs no failure for any of them, stores the firmware it states
    and logs the pixel-format line; its snapshot records no error."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-I', '-c', _EMULATOR_CONNECT],
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

    failures = [
        message
        for level, message in answer['said']
        if level in ('WARNING', 'ERROR') and 'Node not existing' in message
    ]
    assert failures == []
    assert answer['firmware']
    assert any('Pixel format:' in message for _, message in answer['said'])
    assert answer['errors'] == []


def _profile():
    # a stand-in by design: the camera profile the driver loads at connect;
    # the subject is what the range query writes into it.
    return SimpleNamespace(
        gain=SimpleNamespace(total_min_db=None, total_max_db=None),
        exposure_min_us=None,
        exposure_max_us=None,
        pixel_formats=['Mono8'],
        pixel_size_um=0.0,
    )


_RANGE_NODES = ('Gain', 'ExposureTime', 'SensorPixelWidth')


def test_a_camera_without_the_range_nodes_is_told_so_and_its_profile_is_left_unset(caplog):
    cam = _camera()
    cam.profile = _profile()
    with caplog.at_level(logging.INFO):
        cam._query_dynamic_capabilities()
    said = [(r.levelno, r.getMessage()) for r in caplog.records]
    for name in _RANGE_NODES:
        assert (logging.INFO, f'[CAM Class ] {name} not offered by this camera') in [
            (level, message.split(';')[0]) for level, message in said
        ], said
    assert not [
        m for level, m in said if level >= logging.WARNING and any(n in m for n in _RANGE_NODES)
    ]
    assert (cam.profile.gain.total_max_db, cam.profile.exposure_max_us) == (None, None)
    assert cam.profile.pixel_size_um == 0.0


def test_a_failed_pixel_pitch_read_is_told_at_warning(caplog):
    failing = MagicMock()
    failing.GetValue.side_effect = genicam.RuntimeException('usb gone')
    cam = _camera(device={'SensorPixelWidth': failing})
    cam.profile = _profile()
    with caplog.at_level(logging.WARNING):
        cam._query_dynamic_capabilities()
    assert any(
        r.levelno == logging.WARNING
        and 'SensorPixelWidth' in r.getMessage()
        and 'usb gone' in r.getMessage()
        for r in caplog.records
    )
    assert cam.profile.pixel_size_um == 0.0
