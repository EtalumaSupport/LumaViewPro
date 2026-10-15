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

A write the camera did not answer is not a refusal. A ``TimeoutException`` on
the access that decides whether a setting write happened leaves every setting
writer the API reaches as ``CameraNoAnswerError``: the camera is still on the
bus, so removal is not latched, and the API may send the write again. A timeout
on a read the setter only uses to skip a redundant write is tolerated, and one
on the read-back after the write took is the write applied.
"""

import threading
from unittest.mock import PropertyMock

import pytest
from pypylon import genicam

from drivers.exceptions import CameraNoAnswerError
from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from modules.notification_center import Severity
from tests.camera_fakes import bare_pylon_camera
from tests.scope_fakes import give_stub_lanes

# Every GenICam class a refused SetValue can arrive as that is NOT a
# RuntimeException subclass, and therefore reaches the bare handler with the
# camera still live. Each must be reported as a refusal rather than swallowed.
# TimeoutException is a sibling too, but it is the camera not answering, not a
# refusal: TestAWriteTheCameraDidNotAnswer owns it.
SIBLING_REJECTIONS = [
    'OutOfRangeException',
    'AccessException',
    'LogicalErrorException',
]


@pytest.fixture
def pylon_cam():
    """A real PylonCamera over a fake SDK handle, at a known gain.

    ``Gain.GetValue`` answers 0.0 so a request for any other value takes the
    write path rather than the short-circuit. A completed ``SetValue`` moves
    what ``GetValue`` answers, as the node does, because the driver reads the
    gain back after writing it.
    """
    cam = bare_pylon_camera()
    gain = cam.active.Gain
    gain.GetValue.return_value = 0.0
    gain.SetValue.side_effect = lambda v: setattr(gain.GetValue, 'return_value', v)
    # frame_validity reads frames_delivered on every invalidate, which reads
    # cam_image_handler. None is the documented never-streamed state and
    # answers 0 -- seeded here rather than in the shared bare_pylon_camera so
    # the other users of that harness keep the shape they were written against.
    cam.cam_image_handler = None
    # The API asks is_device_removed() when a write reports refused, to tell a
    # refusal apart from a camera that vanished mid-write. __new__ skips the
    # Camera.__init__ that seeds it.
    cam._device_removed = False
    return cam


@pytest.fixture
def pylon_imaging(pylon_cam):
    """The real ImagingAPI over that driver."""
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = pylon_cam
    give_stub_lanes(scope)
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, pylon_cam)
    scope.imaging = imaging
    return imaging, pylon_cam


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
        assert pylon_cam.gain(30.0) == pytest.approx(30.0)

    def test_a_short_circuited_write_answers_applied(self, pylon_cam):
        """The value IS in effect, so it is applied -- the SDK write is just
        skipped. Reporting "cannot confirm" here would be a lie in the other
        direction."""
        pylon_cam.active.Gain.GetValue.return_value = 30.0

        assert pylon_cam.gain(30.0) == pytest.approx(30.0)
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
        imaging, cam = pylon_imaging
        cam.active.Gain.SetValue.side_effect = genicam.OutOfRangeException('out of range')

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_gain_db(30.0)

        assert excinfo.value.setting == 'gain_db'
        assert excinfo.value.requested == 30.0

    def test_the_cache_and_the_chunk_target_stay_where_the_camera_is(self, pylon_imaging):
        imaging, cam = pylon_imaging
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

    def test_the_api_shows_nothing_itself(self, pylon_imaging, centre_posts):
        imaging, cam = pylon_imaging
        cam.active.Gain.SetValue.side_effect = genicam.OutOfRangeException('out of range')

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_gain_db(30.0)

        assert not [n for n in centre_posts if n.severity == Severity.ERROR], (
            'the refusal is shown by its reporter, not by the API'
        )
        assert excinfo.value.title, 'and it carries the words the reporter shows'


def _timeout(what):
    return genicam.TimeoutException(f'{what} timed out')


def _no_answer_on_write(cam, node):
    """The camera does not answer the write of ``node``; its reads answer
    a value no request matches, so no setter short-circuits past the write."""
    getattr(cam.active, node).GetValue.return_value = -1.0
    getattr(cam.active, node).SetValue.side_effect = _timeout(f'{node}.SetValue')


def _gain(cam):
    _no_answer_on_write(cam, 'Gain')
    return cam.gain(30.0)


def _exposure_t(cam):
    cam.profile = None  # the default exposure ceiling, well above the 10 ms asked for
    cam.active.ExposureTime.Min = 20.0
    _no_answer_on_write(cam, 'ExposureTime')
    return cam.exposure_t(10.0)


def _auto_gain(cam):
    cam.active.GainAuto.SetValue.side_effect = _timeout('GainAuto.SetValue')
    return cam.auto_gain(state=False)


def _auto_exposure_t(cam):
    cam.active.ExposureAuto.SetValue.side_effect = _timeout('ExposureAuto.SetValue')
    return cam.auto_exposure_t(True)


def _set_frame_size(cam):
    for node in (cam.active.Width, cam.active.Height):
        node.GetInc.return_value = 4
        node.GetMax.return_value = 2048
        node.GetMin.return_value = 16
        node.GetValue.return_value = 2048
    cam.active.Width.SetValue.side_effect = _timeout('Width.SetValue')
    return cam.set_frame_size(1000, 1000)


def _set_binning_size(cam):
    cam.active.BinningVertical.SetValue.side_effect = _timeout('BinningVertical.SetValue')
    return cam.set_binning_size(2)


def _set_pixel_format(cam):
    cam.active.PixelFormat.GetSymbolics.return_value = ('Mono8', 'Mono12')
    cam.active.PixelFormat.GetValue.return_value = 'Mono8'
    cam.active.PixelFormat.SetValue.side_effect = _timeout('PixelFormat.SetValue')
    return cam.set_pixel_format('Mono12')


def _set_conversion_gain_mode(cam):
    cam.active.BslConversionGainMode.SetValue.side_effect = _timeout(
        'BslConversionGainMode.SetValue'
    )
    return cam.set_conversion_gain_mode('High')


def _set_line_noise_reduction(cam):
    cam.active.BslLineNoiseReductionEnable.SetValue.side_effect = _timeout(
        'BslLineNoiseReductionEnable.SetValue'
    )
    return cam.set_line_noise_reduction(True)


def _set_black_level(cam):
    _no_answer_on_write(cam, 'BlackLevel')
    return cam.set_black_level(5.0)


def _update_auto_gain_target_brightness(cam):
    _no_answer_on_write(cam, 'AutoTargetBrightness')
    return cam.update_auto_gain_target_brightness(0.5)


def _auto_gain_once(cam):
    cam.active.GainAuto.SetValue.side_effect = _timeout('GainAuto.SetValue')
    return cam.auto_gain_once(state=False)


# The twelve setting writers ImagingAPI._camera_write reaches, each driven to
# the write that decides whether the setting was sent.
SETTING_WRITERS = {
    'gain': _gain,
    'exposure_t': _exposure_t,
    'auto_gain': _auto_gain,
    'auto_exposure_t': _auto_exposure_t,
    'set_frame_size': _set_frame_size,
    'set_binning_size': _set_binning_size,
    'set_pixel_format': _set_pixel_format,
    'set_conversion_gain_mode': _set_conversion_gain_mode,
    'set_line_noise_reduction': _set_line_noise_reduction,
    'set_black_level': _set_black_level,
    'update_auto_gain_target_brightness': _update_auto_gain_target_brightness,
    'auto_gain_once': _auto_gain_once,
}


def _outcome(cam, drive):
    """What the writer did, in words a failure message can name."""
    try:
        result = drive(cam)
    except CameraNoAnswerError as e:
        outcome = f'raised CameraNoAnswerError (reason {e.reason!r})'
        if e.reason != 'camera_no_answer':
            return outcome
        if cam._mark_disconnected.called:
            return outcome + ' and marked the camera removed'
        return None
    except Exception as e:
        return f'raised {type(e).__name__}: {e}'
    return f'returned {result!r}'


class TestAWriteTheCameraDidNotAnswer:
    def test_every_setting_writer_raises_no_answer_and_keeps_the_camera(self):
        offenders = {}
        for name, drive in SETTING_WRITERS.items():
            cam = bare_pylon_camera()
            cam._device_removed = False
            outcome = _outcome(cam, drive)
            if outcome is not None:
                offenders[name] = outcome

        assert not offenders, (
            'a timeout on the write that decides the setting must raise '
            'CameraNoAnswerError and leave the camera attached: '
            + '; '.join(f'{name} {outcome}' for name, outcome in offenders.items())
        )

    def test_the_words_name_the_setting_and_carry_the_sdks_text(self, pylon_cam):
        pylon_cam.active.Gain.SetValue.side_effect = genicam.TimeoutException('USB stalled')

        with pytest.raises(CameraNoAnswerError) as excinfo:
            pylon_cam.gain(30.0)

        assert 'gain(30.0)' in str(excinfo.value)
        assert 'USB stalled' in str(excinfo.value)

    def test_a_timed_out_short_circuit_read_falls_through_to_the_write(self, pylon_cam):
        """The read only skips a redundant write; not knowing the value means
        the write is sent, and the write's own answer decides."""
        pylon_cam.active.Gain.GetValue.side_effect = [_timeout('Gain.GetValue'), 30.0]
        pylon_cam.active.Gain.SetValue.side_effect = None

        assert pylon_cam.gain(30.0) == pytest.approx(30.0)
        pylon_cam.active.Gain.SetValue.assert_called_once_with(30.0)

    def test_a_timed_out_read_back_is_the_write_applied(self, pylon_cam):
        """The node took the write; only its confirmation went unanswered."""
        pylon_cam.active.Gain.GetValue.side_effect = [0.0, _timeout('Gain.GetValue')]
        pylon_cam.active.Gain.SetValue.side_effect = None

        assert pylon_cam.gain(30.0) is True
        pylon_cam._mark_disconnected.assert_not_called()

    def test_a_timed_out_black_level_read_back_is_the_write_applied(self, pylon_cam):
        pylon_cam.active.BlackLevel.GetValue.side_effect = _timeout('BlackLevel.GetValue')

        assert pylon_cam.set_black_level(5.0) is True
        pylon_cam._mark_disconnected.assert_not_called()

    def test_an_auto_function_write_unanswered_inside_auto_gain_is_auto_gains_no_answer(
        self, pylon_cam
    ):
        _no_answer_on_write(pylon_cam, 'AutoTargetBrightness')

        with pytest.raises(CameraNoAnswerError):
            pylon_cam.auto_gain(state=True)
        pylon_cam.active.GainAuto.SetValue.assert_not_called()
        pylon_cam._mark_disconnected.assert_not_called()


def _exposure_min_unread(cam):
    cam.profile = None  # the default exposure ceiling, well above the 10 ms asked for
    type(cam.active.ExposureTime).Min = PropertyMock(side_effect=_timeout('ExposureTime.Min'))
    return cam.exposure_t(10.0)


def _frame_grid_unread(cam):
    cam.active.Width.GetInc.side_effect = _timeout('Width.GetInc')
    return cam.set_frame_size(1000, 1000)


def _conversion_gain_probe_unread(cam):
    cam.active.GetNodeMap.return_value.GetNode.side_effect = _timeout('GetNode')
    return cam.set_conversion_gain_mode('High')


def _line_noise_probe_unread(cam):
    cam.active.GetNodeMap.return_value.GetNode.side_effect = _timeout('GetNode')
    return cam.set_line_noise_reduction(True)


def _black_level_probe_unread(cam):
    cam.active.GetNodeMap.return_value.GetNode.side_effect = _timeout('GetNode')
    return cam.set_black_level(5.0)


# Reads a setter must make before it can send its write at all: the value is
# computed from them, or the node they probe is the one written.
READS_BEFORE_THE_WRITE = {
    'exposure_t (ExposureTime.Min)': _exposure_min_unread,
    'set_frame_size (the Width/Height grid)': _frame_grid_unread,
    'set_conversion_gain_mode (the node probe)': _conversion_gain_probe_unread,
    'set_line_noise_reduction (the node probe)': _line_noise_probe_unread,
    'set_black_level (the selector probe)': _black_level_probe_unread,
}


class TestAReadTheWriteNeedsWasNotAnswered:
    def test_the_setter_raises_no_answer_and_says_nothing_was_written(self):
        """A read the write cannot be made without is part of the write: the
        camera did not answer the setter, which may be sent again. The words
        say the write was never sent."""
        offenders = {}
        for name, drive in READS_BEFORE_THE_WRITE.items():
            cam = bare_pylon_camera()
            cam._device_removed = False
            outcome = _outcome(cam, drive)
            if outcome is not None:
                offenders[name] = outcome
                continue
            try:
                drive(cam)
            except CameraNoAnswerError as e:
                if 'nothing was written' not in str(e):
                    offenders[name] = f'said {str(e)!r}'

        assert not offenders, '; '.join(f'{n} {o}' for n, o in offenders.items())
