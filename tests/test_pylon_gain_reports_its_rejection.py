# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon gain write the camera refused is reported as refused, not as applied.

``pyloncamera.gain`` caught every exception its SDK write could raise and
returned ``None`` on the way out. ``_camera_write`` treats anything that is not
``False`` as applied -- a ``None`` means "this driver cannot confirm", which is
believed -- so a REFUSED write stamped the requested value into the camera cache
and into the ``'gain'`` frame-validity chunk target. On a chunk-capable camera
the poisoned target then mismatches every frame the camera can produce and
captures are refused until the next applied gain write; on a camera without
chunk data the frames pass unverified at a gain nobody achieved.

The trigger is not the disconnect path. ``gain`` has no range reconciliation
(``exposure_t`` clamps to ``ExposureTime.Min``; ``idscamera.gain`` reconciles
against the node range), and the GenICam rejection classes are SIBLINGS of
``RuntimeException``, not subclasses -- checked against the installed pypylon,
and mirrored by ``tests/pypylon_stub.py``. So an out-of-range gain on a LIVE,
streaming camera lands in the method's bare ``except Exception``. The
engineering plugin records this from the field: a requested 29.99999986 dB
normalised to 30.0, "which the SDK rejects".

Driver-level tests assert the return contract on each path; the end-to-end test
drives the real ``ImagingAPI`` over the real driver and asserts the consequence
the contract exists for -- that the refusal is raised to an L2 caller and that
neither the cache nor the chunk target moved.
"""

import threading

import pytest
from pypylon import genicam

from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests.camera_fakes import bare_pylon_camera

# Every GenICam class a refused SetValue can arrive as that is NOT a
# RuntimeException subclass, and therefore reaches the bare handler with the
# camera still live. Each must be reported as a refusal rather than swallowed.
SIBLING_REJECTIONS = [
    'OutOfRangeException',
    'AccessException',
    'LogicalErrorException',
    'TimeoutException',
]


@pytest.fixture
def pylon_cam():
    """A real PylonCamera over a fake SDK handle, at a known gain.

    ``Gain.GetValue`` answers 0.0 so a request for any other value takes the
    write path rather than the short-circuit.
    """
    cam = bare_pylon_camera()
    cam.active.Gain.GetValue.return_value = 0.0
    # frame_validity reads frames_delivered on every invalidate, which reads
    # cam_image_handler. None is the documented never-streamed state and
    # answers 0 -- seeded here rather than in the shared bare_pylon_camera so
    # the other users of that harness keep the shape they were written against.
    cam.cam_image_handler = None
    return cam


@pytest.fixture
def pylon_imaging(pylon_cam, monkeypatch):
    """The real ImagingAPI over that driver, with the popup captured."""
    captured = []
    monkeypatch.setattr(
        'modules.lumascope_api.imaging.notifications.error',
        lambda *a, **kw: captured.append(a),
    )
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = pylon_cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, pylon_cam)
    scope.imaging = imaging
    return imaging, pylon_cam, captured


class TestEverySwallowedRejectionIsReported:
    @pytest.mark.parametrize('exc_name', SIBLING_REJECTIONS)
    def test_a_sibling_rejection_answers_refused(self, pylon_cam, exc_name):
        """Not RuntimeException, so it reaches the bare handler -- and the
        camera is still live, so the caller's next capture is real."""
        exc_cls = getattr(genicam, exc_name)
        pylon_cam.active.Gain.SetValue.side_effect = exc_cls('refused by the SDK')

        assert pylon_cam.gain(30.0) is False

    @pytest.mark.parametrize('exc_name', SIBLING_REJECTIONS)
    def test_a_sibling_rejection_does_not_mark_the_camera_gone(self, pylon_cam, exc_name):
        """A refused value is not a disconnect; only the comms path is."""
        exc_cls = getattr(genicam, exc_name)
        pylon_cam.active.Gain.SetValue.side_effect = exc_cls('refused by the SDK')

        pylon_cam.gain(30.0)

        pylon_cam._mark_disconnected.assert_not_called()

    def test_a_communication_failure_answers_refused_and_marks_the_camera_gone(self, pylon_cam):
        pylon_cam.active.Gain.SetValue.side_effect = genicam.RuntimeException('comms lost')

        assert pylon_cam.gain(30.0) is False
        pylon_cam._mark_disconnected.assert_called_once()


class TestTheAppliedPathsSayApplied:
    def test_a_completed_write_answers_applied(self, pylon_cam):
        assert pylon_cam.gain(30.0) is True

    def test_a_short_circuited_write_answers_applied(self, pylon_cam):
        """The value IS in effect, so it is applied -- the SDK write is just
        skipped. Reporting "cannot confirm" here would be a lie in the other
        direction."""
        pylon_cam.active.Gain.GetValue.return_value = 30.0

        assert pylon_cam.gain(30.0) is True
        pylon_cam.active.Gain.SetValue.assert_not_called()

    def test_an_inactive_camera_answers_not_attempted(self, pylon_cam):
        """No camera is not a refusal -- the published contract calls it a
        quiet no-op, and the API never reaches the driver on that path anyway.
        ``None`` is the one honest answer here."""
        pylon_cam.active = None

        assert pylon_cam.gain(30.0) is None


class TestARefusalIsNotRecordedAsTruth:
    """The end-to-end consequence, through the real API over the real driver."""

    def test_the_refusal_reaches_an_l2_caller(self, pylon_imaging):
        imaging, cam, _captured = pylon_imaging
        cam.active.Gain.SetValue.side_effect = genicam.OutOfRangeException('out of range')

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_gain_db(30.0)

        assert excinfo.value.setting == 'gain_db'
        assert excinfo.value.requested == 30.0

    def test_the_cache_and_the_chunk_target_stay_where_the_camera_is(self, pylon_imaging):
        imaging, cam, _captured = pylon_imaging
        imaging.set_gain_db(3.0)
        assert imaging.gain_db_cached == 3.0, 'precondition: the applied write was recorded'
        assert imaging.frame_validity.target('gain') == 3.0

        cam.active.Gain.SetValue.side_effect = genicam.OutOfRangeException('out of range')
        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(30.0)

        assert imaging.gain_db_cached == 3.0, 'a refused gain must not be cached as truth'
        assert imaging.frame_validity.target('gain') == 3.0, (
            'a chunk target no frame can carry rejects every frame the camera produces'
        )

    def test_the_user_is_told_as_well(self, pylon_imaging):
        imaging, cam, captured = pylon_imaging
        cam.active.Gain.SetValue.side_effect = genicam.OutOfRangeException('out of range')

        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(30.0)

        assert captured, 'the refusal reaches the user, not only the caller'
