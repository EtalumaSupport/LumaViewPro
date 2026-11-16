# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bench-confirmed IDS feature setters: test pattern + USB3 transfer tuning.

The nodemap probe in read_diagnostic_snapshot confirmed on the U3-34L0XCP-M
that TestPattern, U3vStreamChannelBulkTransferSize, and
U3vStreamChannelTransferRequestCount exist and are ReadWrite (while
ChunkModeActive / ChunkSelector are absent). These tests exercise the setters
now wired to those nodes against HasNode/FindNode-style fakes (the ids_peak
idiom: SetCurrentEntry() for the enum TestPattern, the DataStream nodemap's
SetValue() for the transfer-tuning scalar), and the AccessStatus ->
symbolic-name mapping used in the probe + free-run log. A write that is not
made raises, as every write_node does.
"""

from unittest.mock import MagicMock

import pytest
from ids_peak import ids_peak

import _lumascope.drivers.idscamera as idscamera
from _lumascope.drivers.exceptions import CameraWriteFailedError
from tests.camera_fakes import bare_ids_camera


class _RecordingEnumNode:
    def __init__(self):
        self.set_entry = None

    def SetCurrentEntry(self, entry):
        self.set_entry = entry


class _RecordingIntNode:
    def __init__(self):
        self.set_value = None

    def SetValue(self, value):
        self.set_value = value


class _RaisingNode:
    def SetCurrentEntry(self, entry):
        raise ids_peak.Exception('SDK rejected entry')

    def SetValue(self, value):
        raise ids_peak.Exception('SDK rejected value')


class _FindNodeMap:
    """The ids_peak nodemap shape: ``HasNode`` says whether a name is there,
    and ``FindNode`` of an absent name raises."""

    def __init__(self, nodes):
        self._nodes = nodes

    def HasNode(self, name):
        return name in self._nodes

    def FindNode(self, name):
        if name not in self._nodes:
            raise ids_peak.NotFoundException(f'absent node: {name}')
        return self._nodes[name]


# --- set_test_pattern -------------------------------------------------------


class TestSetTestPattern:
    def test_disabled_sets_off(self):
        cam = bare_ids_camera()
        node = _RecordingEnumNode()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': node})
        cam.set_test_pattern(enabled=False)
        assert node.set_entry == 'Off'

    def test_enabled_sets_named_pattern(self):
        cam = bare_ids_camera()
        node = _RecordingEnumNode()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': node})
        cam.set_test_pattern(enabled=True, pattern='ColorBar')
        assert node.set_entry == 'ColorBar'

    def test_inactive_raises(self):
        cam = bare_ids_camera()
        cam.active = False
        with pytest.raises(CameraWriteFailedError, match='no camera is open'):
            cam.set_test_pattern(enabled=True)

    def test_sdk_rejection_raises(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': _RaisingNode()})
        with pytest.raises(CameraWriteFailedError, match='TestPattern'):
            cam.set_test_pattern(enabled=True, pattern='Bogus')


# --- USB3 transfer-tuning setter (DataStream nodemap) -----------------------


class TestTransferTuningSetters:
    def test_max_transfer_size_writes_data_stream_node(self):
        cam = bare_ids_camera()
        node = _RecordingIntNode()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelBulkTransferSize': node})
        ]
        cam.set_max_transfer_size(1048576)
        assert node.set_value == 1048576

    def test_num_max_queued_urbs_writes_data_stream_node(self):
        cam = bare_ids_camera()
        node = _RecordingIntNode()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelTransferRequestCount': node})
        ]
        cam.set_num_max_queued_urbs(6)
        assert node.set_value == 6

    def test_inactive_raises(self):
        cam = bare_ids_camera()
        cam.active = False
        with pytest.raises(CameraWriteFailedError, match='no camera is open'):
            cam.set_num_max_queued_urbs(6)

    def test_absent_node_raises_not_offered(self):
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [_FindNodeMap({})]
        with pytest.raises(CameraWriteFailedError, match='does not offer'):
            cam.set_num_max_queued_urbs(3)

    def test_empty_nodemap_raises_a_write_failure(self):
        """The map a write needs is missing (a teardown racing it): a write's
        failure, never a read's, which would tell the caller nothing was sent
        after a setter's earlier writes had landed."""
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = []
        with pytest.raises(CameraWriteFailedError, match='stream node map'):
            cam.set_num_max_queued_urbs(3)

    def test_sdk_rejection_raises(self):
        # A node present but locked (grabbing) -> SetValue raises.
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelTransferRequestCount': _RaisingNode()})
        ]
        with pytest.raises(CameraWriteFailedError, match='U3vStreamChannelTransferRequestCount'):
            cam.set_num_max_queued_urbs(3)


# --- AccessStatus symbolic mapping ------------------------------------------


class TestAccessStatusName:
    def test_bench_confirmed_codes_are_named(self):
        assert idscamera._access_status_name(2) == 'WriteOnly'
        assert idscamera._access_status_name(3) == 'ReadOnly'
        assert idscamera._access_status_name(4) == 'ReadWrite'

    def test_unknown_code_is_labelled_not_guessed(self):
        # 0/1 ordering differs between GenApi and the IDS enum -> don't guess.
        assert idscamera._access_status_name(0) == 'AccessStatus(0)'
        assert idscamera._access_status_name(1) == 'AccessStatus(1)'

    def test_noninteger_falls_back_to_str(self):
        assert idscamera._access_status_name('ReadWrite') == 'ReadWrite'
