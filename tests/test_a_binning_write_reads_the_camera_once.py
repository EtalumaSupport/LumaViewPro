# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A binning write reads the camera's binning once and its frame size not at all.

Every node read is a round trip to the camera and a read that can fail.
``Camera.set_binning_size`` reads the binning once, ahead of the write, to
know whether the frame window still fits; the driver's own log lines name
the size asked for and written, which it holds, so a binning write on a
camera that is not grabbing makes no other binning read and no frame-size
read. (An IDS write while grabbing also makes the grab restart's free-run
rate read, a real read that sets the binned frame-time margin.)

A stand-in by design: what is counted is the SDK's node reads, which only
a stub camera shows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.camera_fakes import bare_ids_camera, bare_pylon_camera


def test_pylon_reads_the_binning_once_and_the_frame_not_at_all():
    cam = bare_pylon_camera()
    for node in ('BinningVertical', 'BinningHorizontal'):
        getattr(cam.active, node).GetValue.return_value = 1

    assert cam.set_binning_size(2) is True

    reads = {
        node: getattr(cam.active, node).GetValue.call_count
        for node in ('BinningVertical', 'BinningHorizontal', 'Width', 'Height')
    }
    assert reads == {'BinningVertical': 1, 'BinningHorizontal': 1, 'Width': 0, 'Height': 0}


def test_ids_reads_the_binning_once_and_the_frame_not_at_all():
    cam = bare_ids_camera()
    cam._max_binning = 2
    nodes = {}

    def find(name):
        if name not in nodes:
            nodes[name] = MagicMock(name=name)
            nodes[name].Value.return_value = 1
        return nodes[name]

    cam.remote_nodemap = MagicMock()
    cam.remote_nodemap.FindNode.side_effect = find

    assert cam.set_binning_size(2) is True

    reads = {
        node: nodes[node].Value.call_count if node in nodes else 0
        for node in ('BinningVertical', 'BinningHorizontal', 'Width', 'Height')
    }
    assert reads == {'BinningVertical': 1, 'BinningHorizontal': 1, 'Width': 0, 'Height': 0}
