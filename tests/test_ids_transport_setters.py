# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bench-confirmed IDS feature setters: test pattern + USB3 transfer tuning.

The nodemap probe in read_diagnostic_snapshot confirmed on the U3-34L0XCP-M
that TestPattern, U3vStreamChannelBulkTransferSize, and
U3vStreamChannelTransferRequestCount exist and are ReadWrite (while
ChunkModeActive / ChunkSelector are absent). These tests exercise the setters
now wired to those nodes against FindNode-style fakes (the ids_peak idiom:
remote_nodemap.FindNode().SetCurrentEntry() for the enum TestPattern, the
DataStream nodemap .FindNode().SetValue() for the transfer-tuning scalars), and
the AccessStatus -> symbolic-name mapping used in the probe + free-run log.
"""

from unittest.mock import MagicMock

import drivers.idscamera as idscamera
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
        raise RuntimeError('SDK rejected entry')

    def SetValue(self, value):
        raise RuntimeError('SDK rejected value')


class _FindNodeMap:
    """FindNode dispatch: known names return their node, unknown names raise
    (the ids_peak shape for an absent node)."""

    def __init__(self, nodes):
        self._nodes = nodes

    def FindNode(self, name):
        if name not in self._nodes:
            raise RuntimeError(f'absent node: {name}')
        return self._nodes[name]


# --- set_test_pattern -------------------------------------------------------


class TestSetTestPattern:
    def test_disabled_sets_off(self):
        cam = bare_ids_camera()
        node = _RecordingEnumNode()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': node})
        assert cam.set_test_pattern(enabled=False) is True
        assert node.set_entry == 'Off'

    def test_enabled_sets_named_pattern(self):
        cam = bare_ids_camera()
        node = _RecordingEnumNode()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': node})
        assert cam.set_test_pattern(enabled=True, pattern='ColorBar') is True
        assert node.set_entry == 'ColorBar'

    def test_inactive_returns_false(self):
        cam = bare_ids_camera()
        cam.active = False
        assert cam.set_test_pattern(enabled=True) is False

    def test_sdk_rejection_returns_false(self):
        cam = bare_ids_camera()
        cam.remote_nodemap = _FindNodeMap({'TestPattern': _RaisingNode()})
        assert cam.set_test_pattern(enabled=True, pattern='Bogus') is False


# --- USB3 transfer-tuning setters (DataStream nodemap) ----------------------


class TestTransferTuningSetters:
    def test_max_transfer_size_writes_data_stream_node(self):
        cam = bare_ids_camera()
        node = _RecordingIntNode()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelBulkTransferSize': node})
        ]
        assert cam.set_max_transfer_size(1048576) is True
        assert node.set_value == 1048576

    def test_num_max_queued_urbs_writes_data_stream_node(self):
        cam = bare_ids_camera()
        node = _RecordingIntNode()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelTransferRequestCount': node})
        ]
        assert cam.set_num_max_queued_urbs(6) is True
        assert node.set_value == 6

    def test_inactive_returns_false(self):
        cam = bare_ids_camera()
        cam.active = False
        assert cam.set_max_transfer_size(1048576) is False

    def test_absent_node_returns_false(self):
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [_FindNodeMap({})]
        assert cam.set_num_max_queued_urbs(3) is False

    def test_empty_nodemap_returns_false(self):
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = []
        assert cam.set_max_transfer_size(1048576) is False

    def test_sdk_rejection_returns_false(self):
        # A node present but locked (grabbing) -> SetValue raises -> False.
        cam = bare_ids_camera()
        cam.data_stream = MagicMock()
        cam.data_stream.NodeMaps.return_value = [
            _FindNodeMap({'U3vStreamChannelBulkTransferSize': _RaisingNode()})
        ]
        assert cam.set_max_transfer_size(1048576) is False


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
