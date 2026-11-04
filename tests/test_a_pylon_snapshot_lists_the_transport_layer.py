# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon diagnostic snapshot lists the device transport layer's nodes by name.

The transport layer's register-access timeouts and retries are named
differently by camera and pylon release: a daA3840-45um under pylon 11.5
exposes none of ``ReadTimeout`` / ``WriteTimeout`` / ``MaxRetryCountRead``
/ ``MaxRetryCountWrite``. So the snapshot lists every node the camera's
transport layer has, with its value, rather than asking for names it may
not have; a nodemap that cannot be fetched is named in ``errors``.

The pylon SDK is not on the test host, so the transport layer is a stand-in
nodemap here, as the snapshot's camera and stream-grabber nodemaps are in
``tests/camera_fakes.py``.
"""

from tests.camera_fakes import FakeDiagNode, diag_snapshot_pylon_camera


class _NamedNode(FakeDiagNode):
    def __init__(self, name, *values):
        super().__init__(*values)
        self._name = name

    def GetName(self):
        return self._name


class _Unreadable:
    """A node with no value: a category or a command."""

    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def GetValue(self):
        raise RuntimeError('not a value node')


class _ListingNodeMap:
    def __init__(self, nodes):
        self._nodes = {node.GetName(): node for node in nodes}

    def GetNodes(self):
        return list(self._nodes.values())

    def GetNode(self, name):
        return self._nodes.get(name)


def test_every_transport_layer_node_is_listed_with_its_value():
    cam, _nodemap, _grabber = diag_snapshot_pylon_camera()
    cam.active.GetTLNodeMap.return_value = _ListingNodeMap(
        [
            _NamedNode('DeviceTransportLayerType', 'USB3Vision'),
            _NamedNode('MaxRetryCountWrite', 2),
            _NamedNode('WriteTimeout', 500),
            _Unreadable('Statistics'),
        ]
    )

    snapshot = cam.read_diagnostic_snapshot(duration_s=0)

    assert snapshot['transport_layer'] == {
        'DeviceTransportLayerType': 'USB3Vision',
        'MaxRetryCountWrite': 2,
        'WriteTimeout': 500,
        'Statistics': '<read err: RuntimeError>',
    }


def test_a_transport_layer_that_cannot_be_fetched_is_named_in_the_errors():
    cam, _nodemap, _grabber = diag_snapshot_pylon_camera()
    cam.active.GetTLNodeMap.side_effect = RuntimeError('no device TL')

    snapshot = cam.read_diagnostic_snapshot(duration_s=0)

    assert snapshot['transport_layer'] == {}
    assert any('no device TL' in e for e in snapshot['errors']), snapshot['errors']
