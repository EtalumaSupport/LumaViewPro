# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each camera driver delivers row 0 at the top, and a camera whose orientation cannot be known does not come up.

At connect, ``orientation_owed`` (``_lumascope/drivers/camera.py``) decides
``ReverseY`` and then ``ReverseX`` from what the body has: a node it lacks
is owed, and the driver's copy turns that axis; a writable node is written
True and read back; a read-only node's own value decides (True is taken, so
the copy must not turn it again). A read or write that fails, or a read-back
that is not True, leaves the frame's orientation unknown: the error names
the step and the node, and bring-up records the camera down,
``camera_not_initialized``. It used to come up streaming in whatever order
the body held, logged at INFO.

The simulated camera's reverse nodes are declared by its ``REVERSE_NODES``
table, so every answer runs on the simulator; Pylon and IDS run the same
decision through their own readers over the suite's SDK stand-ins. A raise
from ``init_camera_config`` reaching connect's handler as
``CameraUsbOpenFailedError`` is pinned by
``test_an_ids_connect_whose_configuration_cannot_read_the_grab_state_fails_naming_the_read``.
"""

from __future__ import annotations

import pytest

from _lumascope.drivers.camera import OwedReverse
from _lumascope.drivers.exceptions import CameraReadFailedError, CameraWriteFailedError
from _lumascope.drivers.simulated_camera import SimulatedCamera, SimulatedReverseNode
from _lumascope.modules.scope_session import ScopeSession
from tests.camera_fakes import bare_ids_camera, init_configurable_pylon_camera
from tests.settings_fixtures import complete_settings


def _body(monkeypatch, reverse_y, reverse_x=SimulatedReverseNode()):
    monkeypatch.setattr(
        SimulatedCamera, 'REVERSE_NODES', {'ReverseY': reverse_y, 'ReverseX': reverse_x}
    )


def test_each_reverse_is_taken_or_owed_by_what_the_body_has(monkeypatch):
    cases = [
        (SimulatedReverseNode(), False),
        (None, True),
        (SimulatedReverseNode(writable=False, value=True), False),
        (SimulatedReverseNode(writable=False, value=False), True),
    ]
    wrong = []
    for node, owed in cases:
        _body(monkeypatch, reverse_y=node, reverse_x=node)
        camera = SimulatedCamera(width=8, height=6)
        try:
            if camera.owed_reverse != OwedReverse(rows=owed, columns=owed):
                wrong.append(f'{node}: {camera.owed_reverse}, expected owed={owed}')
        finally:
            camera.disconnect()
    assert not wrong, '\n'.join(wrong)


def test_a_body_whose_orientation_cannot_be_known_is_down_naming_the_step(monkeypatch, tmp_path):
    wrong = []
    for fails in ('read', 'access', 'write', 'not_taken'):
        _body(monkeypatch, reverse_y=SimulatedReverseNode(fails=fails))
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)), simulate=True, warn_pre_release=False
        )
        try:
            (camera,) = [p for p in session.bring_up_record().parts if p.part == 'camera']
            if (camera.up, camera.cause) != (False, 'camera_not_initialized') or (
                'the image orientation (ReverseY)' not in camera.detail
            ):
                wrong.append(f'{fails}: {camera}')
        finally:
            session.shutdown()
    assert not wrong, '\n'.join(wrong)


def test_a_pylon_body_takes_a_writable_reverse_and_owes_an_absent_one():
    cam = init_configurable_pylon_camera()
    del cam.reverse_nodes['ReverseX']

    cam.init_camera_config()

    assert cam.owed_reverse == OwedReverse(rows=False, columns=True)
    assert cam.reverse_nodes['ReverseY'].GetValue() is True


def test_a_pylon_reverse_write_the_camera_refuses_is_raised_naming_the_step():
    from pypylon import genicam

    class RefusesWrites(genicam.IBoolean):
        def SetValue(self, value):
            raise genicam.RuntimeException('comms lost')

    cam = init_configurable_pylon_camera()
    cam.reverse_nodes['ReverseY'] = RefusesWrites(False, writable=True)

    with pytest.raises(CameraWriteFailedError, match=r'the image orientation \(ReverseY\)'):
        cam.init_camera_config()
    cam.set_frame_size.assert_not_called()


def test_an_ids_body_owes_a_read_only_reverse_that_does_not_reverse():
    from ids_peak import ids_peak

    cam = _configurable_ids_camera()
    cam.reverse_nodes['ReverseY'] = ids_peak.BooleanNode(False, writable=False)

    cam.init_camera_config()

    assert cam.owed_reverse == OwedReverse(rows=True, columns=False)


def test_an_ids_reverse_whose_access_read_fails_is_raised_naming_the_step():
    """The access read the old step folded into "not writable" and passed over."""
    from ids_peak import ids_peak

    cam = _configurable_ids_camera()
    cam.reverse_nodes['ReverseY'] = ids_peak.BooleanNode(
        False, writable=ids_peak.TimeoutException('GC_ERR_TIMEOUT')
    )

    with pytest.raises(CameraReadFailedError, match=r'the image orientation \(ReverseY\)'):
        cam.init_camera_config()
    cam.set_frame_size.assert_not_called()


def _configurable_ids_camera():
    from unittest.mock import MagicMock

    cam = bare_ids_camera()
    cam.profile = MagicMock(pixel_formats=['Mono8'])
    # Stand-ins by design: the subject is the orientation step between them.
    for name in (
        'set_pixel_format',
        'exposure_t',
        'set_frame_size',
        '_configure_free_run',
        '_log_free_run_state',
    ):
        setattr(cam, name, MagicMock())
    cam.get_supported_pixel_formats = MagicMock(return_value=('Mono8',))
    return cam
