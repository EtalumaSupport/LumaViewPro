# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon diagnostic snapshot states which instant the camera's Timestamp chunk stamps.

An ace 2 or dart offers ``BslChunkTimestampSelector`` (exposure start,
exposure end, frame start); an older camera's Timestamp chunk is the legacy
one and has no selector. The snapshot's camera configuration carries the
selector's value, or the snapshot's sentinel for a camera without it, so
whether a camera offers the choice is answered through the diagnostics
member on any connected camera.

The pylon SDK is not on the test host, so the camera nodemap is the
stand-in ``tests/camera_fakes.py`` provides for the snapshot.
"""

from tests.camera_fakes import diag_snapshot_pylon_camera


def test_a_camera_with_the_selector_states_its_instant():
    cam, _nodemap, _grabber = diag_snapshot_pylon_camera(
        {'BslChunkTimestampSelector': 'ExposureStart'}
    )

    snapshot = cam.read_diagnostic_snapshot(duration_s=0)

    assert snapshot['config']['chunk_timestamp_selector'] == 'ExposureStart'


def test_a_camera_without_the_selector_says_it_has_none():
    cam, _nodemap, _grabber = diag_snapshot_pylon_camera()

    snapshot = cam.read_diagnostic_snapshot(duration_s=0)

    assert snapshot['config']['chunk_timestamp_selector'] == '<not present>'
