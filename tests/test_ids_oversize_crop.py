# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Wiring tests for the IDS oversize-then-crop framing path.

The pure geometry (plan_aoi / center_crop / reorient_image_center) is covered by
tests/test_aoi_geometry.py. These tests pin the IDS DRIVER wiring around it: that
set_frame_size acquires the next legal AOI up, centers it, records the crop, and
that the public frame size is the delivered (cropped) target while the hardware
AOI is reported separately; that the unpack worker actually crops; and that the
optical-center bias stays neutral until the bench collimator calibration pins the
sensor orientation.

Built on tests/camera_fakes.py (real IDSCamera via __new__ + a fake SDK), plus a
small stateful nodemap that reproduces the one GenICam interdependency that
matters here: an AOI's max width/height shrinks as its offset grows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from drivers.camera_profiles import CameraProfile
from tests.camera_fakes import bare_ids_camera

# Bench-confirmed IMX676 AOI maxima (binned-space probe, 2026-06-25): 3504x3544.
SENSOR_W, SENSOR_H = 3504, 3544


class _FakeIdsAoiNodemap:
    """FindNode-style IDS nodemap holding Width/Height/OffsetX/OffsetY state.

    Reproduces the GenICam constraint the planner relies on: an AOI's max
    width/height is the sensor extent minus its current offset, so reading the
    true sensor max requires the offsets at zero first (what set_frame_size
    does). Records every SetValue in ``writes`` for sequence assertions.
    """

    def __init__(self, sensor=(SENSOR_W, SENSOR_H), offset_step=(8, 2), fail_on=()):
        self.sensor_w, self.sensor_h = sensor
        self.off_step_x, self.off_step_y = offset_step
        self.width, self.height = sensor
        self.off_x = self.off_y = 0
        self.writes: list[tuple[str, int]] = []
        # Node names whose SetValue raises, to exercise mid-call failure paths.
        self.fail_on = set(fail_on)

    def FindNode(self, name):
        return _FakeIdsAoiNode(self, name)


class _FakeIdsAoiNode:
    def __init__(self, nm, name):
        self._nm = nm
        self._name = name

    def Value(self):
        nm = self._nm
        return {'Width': nm.width, 'Height': nm.height, 'OffsetX': nm.off_x, 'OffsetY': nm.off_y}[
            self._name
        ]

    def Minimum(self):
        return {'Width': 48, 'Height': 4, 'OffsetX': 0, 'OffsetY': 0}[self._name]

    def Maximum(self):
        nm = self._nm
        raw = {
            'Width': nm.sensor_w - nm.off_x,
            'Height': nm.sensor_h - nm.off_y,
            'OffsetX': nm.sensor_w - nm.width,
            'OffsetY': nm.sensor_h - nm.height,
        }[self._name]
        # Real GenICam floors Maximum to the node increment; model that so a
        # fixture can't mask an increment-rounding bug in the driver.
        inc = self.Increment()
        return (raw // inc) * inc

    def Increment(self):
        return {
            'OffsetX': self._nm.off_step_x,
            'OffsetY': self._nm.off_step_y,
            'Width': 48,
            'Height': 4,
        }[self._name]

    def SetValue(self, value):
        nm = self._nm
        if self._name in nm.fail_on:
            raise RuntimeError(f'simulated SDK failure on {self._name}.SetValue({value})')
        nm.writes.append((self._name, value))
        setattr(
            nm,
            {'Width': 'width', 'Height': 'height', 'OffsetX': 'off_x', 'OffsetY': 'off_y'}[
                self._name
            ],
            value,
        )


def _ids_camera_with_aoi(**nodemap_kwargs):
    cam = bare_ids_camera()
    cam.profile = CameraProfile(alignment={'width': 48, 'height': 4})
    cam.remote_nodemap = _FakeIdsAoiNodemap(**nodemap_kwargs)
    return cam


def test_set_frame_size_oversizes_centers_and_records_crop():
    """1900 cannot land on the 48-px width grid, so the AOI rounds UP to 1920,
    centers on the sensor, and the crop trims the 20-px surplus back to 1900."""
    cam = _ids_camera_with_aoi()

    assert cam.set_frame_size(1900, 1900) is True

    # Hardware AOI is the oversized acquisition; height (mult of 4) is exact.
    assert cam.get_acquired_aoi() == {'width': 1920, 'height': 1900}
    # The crop window centers the 1900 request in the 1920 AOI (10 px each side
    # in X); height has no surplus. (0,0) bias -> geometric center.
    assert cam._crop_spec == (10, 0, 1900, 1900)
    # The delivered (public) frame size is the cropped target, not the AOI.
    assert cam.get_frame_size() == {'width': 1900, 'height': 1900}


def test_request_near_sensor_max_delivers_clamped_size_truthfully():
    """When no legal AOI can supply the request (within one alignment step of the
    sensor max), set_frame_size delivers the largest legal size and reports it
    truthfully via get_frame_size, still returning True (the op applied)."""
    cam = _ids_camera_with_aoi(sensor=(1900, 1900))

    assert cam.set_frame_size(1900, 1900) is True
    # 1900 width cannot land on the 48-px grid below an 1900 max -> 1872.
    assert cam.get_frame_size() == {'width': 1872, 'height': 1900}


def test_set_frame_size_uses_nodemap_increment_not_profile():
    """The alignment step comes from the SDK nodemap (Width.Increment), not the
    profile: an unrecognized model gets a default profile (alignment 4) the
    hardware rejects (Inc=48). With a wrong profile alignment (4) but the real
    nodemap increment (48), the AOI must still land on the 48 grid."""
    cam = _ids_camera_with_aoi()
    cam.profile = CameraProfile(alignment={'width': 4, 'height': 4})  # default/wrong

    cam.set_frame_size(1900, 1900)

    # acq width on the 48 grid (from the nodemap increment), not the profile's 4.
    assert cam.get_acquired_aoi()['width'] == 1920
    assert cam.get_acquired_aoi()['width'] % 48 == 0


def test_set_frame_size_zeroes_offsets_before_reading_max():
    """Offsets must be set to 0 before Width/Height so the planner reads the true
    sensor max, not max-minus-current-offset, and the AOI fits on a fresh open."""
    cam = _ids_camera_with_aoi()
    cam.set_frame_size(1900, 1900)

    names = [name for name, _ in cam.remote_nodemap.writes]
    # OffsetX/OffsetY are zeroed before Width/Height are sized.
    assert names.index('OffsetX') < names.index('Width')
    assert names.index('OffsetY') < names.index('Height')


def test_exact_size_request_records_no_crop():
    """A request already on the grid (1920 wide) has no surplus, so plan.needs_crop
    is False and no crop window is recorded -- the unpack worker skips the slice
    and get_frame_size falls back to the acquired AOI, which equals the request."""
    cam = _ids_camera_with_aoi()

    cam.set_frame_size(1920, 1900)

    assert cam.get_acquired_aoi() == {'width': 1920, 'height': 1900}
    assert cam._crop_spec is None
    assert cam.get_frame_size() == {'width': 1920, 'height': 1900}


def test_get_frame_size_falls_back_to_aoi_before_first_set():
    """Before any set_frame_size, no crop is recorded, so the public size falls
    back to the live hardware AOI rather than returning None."""
    cam = _ids_camera_with_aoi()
    assert cam._crop_spec is None
    assert cam.get_frame_size() == {'width': SENSOR_W, 'height': SENSOR_H}


def test_max_frame_size_is_offset_independent():
    """Centering the AOI leaves a non-zero offset, but the reported sensor max
    must stay the full sensor extent (Width.Max + OffsetX), not shrink with it."""
    cam = _ids_camera_with_aoi()
    cam.set_frame_size(1900, 1900)

    # Offset is now non-zero, yet the sensor max is recovered in full.
    assert cam.remote_nodemap.off_x > 0
    assert cam.get_max_frame_size() == {'width': SENSOR_W, 'height': SENSOR_H}


def test_set_binning_size_invalidates_crop_spec():
    """Binning changes the AOI/buffer pixel dimensions, so the recorded crop
    window no longer fits. set_binning_size must clear it, or the unpack worker
    crops every rebinned frame against the stale window and drops them all."""
    cam = bare_ids_camera()
    cam._crop_spec = (10, 0, 1900, 1900)

    assert cam.set_binning_size(2) is True
    assert cam._crop_spec is None


def test_set_frame_size_failure_clears_stale_crop_spec():
    """A mid-call SDK failure (here Height.SetValue, after Width was applied)
    must not leave the previous crop window in place against the new buffer --
    set_frame_size clears the framing state on the error path."""
    cam = _ids_camera_with_aoi(fail_on=('Height',))
    cam._crop_spec = (5, 5, 100, 100)  # stale window from a prior call

    assert cam.set_frame_size(1900, 1900) is False
    assert cam._crop_spec is None


def test_optical_center_bias_is_neutral():
    """The optical-center bias is (0,0) today -- the AOI centers geometrically
    (correct for every unit). It is plumbed through plan_aoi so the upcoming
    optical-center work implements only this method's body; it must never raise
    on the neutral path (set_frame_size would swallow a raise into a silent
    failure to resize)."""
    cam = _ids_camera_with_aoi()
    assert cam._optical_center_bias() == (0, 0)
    assert cam.set_frame_size(1900, 1900) is True


def test_unpack_crops_converted_frame_to_target(monkeypatch):
    """The unpack worker crops the oversized converted frame to the recorded
    window, so the array that leaves the driver is exactly the requested size."""
    from drivers import idscamera

    handler = idscamera.ImageHandler.__new__(idscamera.ImageHandler)
    parent = bare_ids_camera()
    parent._crop_spec = (10, 0, 1900, 1900)
    parent.get_pixel_format = lambda: 'Mono12g24IDS'
    handler._parent = parent

    full = np.arange(1920 * 1900, dtype=np.uint16).reshape(1900, 1920)
    target_fmt = idscamera._ids_ipl_target('Mono12g24IDS')
    fake_img = MagicMock()
    fake_img.PixelFormat.return_value = target_fmt  # already target -> skip ConvertTo
    fake_img.get_numpy.return_value = full
    monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', lambda buffer: fake_img)

    array, significant_bits = handler._unpack(object())

    assert array.shape == (1900, 1900)
    assert significant_bits == 12
    # Exactly the centered 1900-wide window of the 1920 frame.
    assert np.array_equal(array, full[0:1900, 10:1910])
    # A contiguous copy, not a view holding the oversized source alive.
    assert array.flags['C_CONTIGUOUS']
    assert array.base is None


def test_grab_new_capture_applies_crop(monkeypatch):
    """The still-capture path (snaps / protocol scans / autofocus) goes through
    grab_new_capture, not the live _unpack worker, so it must apply the crop too
    -- otherwise saved frames are the oversized AOI, not the delivered size."""
    from drivers import idscamera

    cam = bare_ids_camera()
    cam._crop_spec = (10, 0, 1900, 1900)
    cam.get_pixel_format = lambda: 'Mono12g24IDS'
    cam.cam_image_handler = object()  # non-None gate in grab_new_capture

    full = np.zeros((1900, 1920), dtype=np.uint16)
    fake_buffer = MagicMock()
    fake_buffer.IsIncomplete.return_value = False
    cam.data_stream = MagicMock()
    cam.data_stream.WaitForFinishedBuffer.return_value = fake_buffer
    fake_img = MagicMock()
    fake_img.PixelFormat.return_value = idscamera._ids_ipl_target('Mono12g24IDS')
    fake_img.get_numpy.return_value = full
    monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', lambda b: fake_img)

    ok, _ts = cam.grab_new_capture(1.0)

    assert ok is True
    assert cam.array.shape == (1900, 1900)


def test_grab_new_capture_requeues_buffer_on_unpack_failure(monkeypatch):
    """A failed unpack (here a crop window that does not fit) must STILL re-queue
    the SDK buffer -- a buffer never returned to the pool starves the stream and
    hangs every subsequent capture."""
    from drivers import idscamera

    cam = bare_ids_camera()
    cam._crop_spec = (10, 0, 5000, 5000)  # window > frame -> center_crop raises
    cam.get_pixel_format = lambda: 'Mono12g24IDS'
    cam.cam_image_handler = object()

    full = np.zeros((1900, 1920), dtype=np.uint16)
    fake_buffer = MagicMock()
    fake_buffer.IsIncomplete.return_value = False
    cam.data_stream = MagicMock()
    cam.data_stream.WaitForFinishedBuffer.return_value = fake_buffer
    fake_img = MagicMock()
    fake_img.PixelFormat.return_value = idscamera._ids_ipl_target('Mono12g24IDS')
    fake_img.get_numpy.return_value = full
    monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', lambda b: fake_img)

    ok, _ts = cam.grab_new_capture(1.0)

    assert ok is False  # the unpack failed
    cam.data_stream.QueueBuffer.assert_called_once_with(fake_buffer)  # ...but buffer returned


def test_unpack_without_crop_spec_passes_frame_through(monkeypatch):
    """No crop recorded -> the unpack worker returns the full converted frame."""
    from drivers import idscamera

    handler = idscamera.ImageHandler.__new__(idscamera.ImageHandler)
    parent = bare_ids_camera()
    parent._crop_spec = None
    parent.get_pixel_format = lambda: 'Mono12g24IDS'
    handler._parent = parent

    full = np.zeros((1900, 1920), dtype=np.uint16)
    fake_img = MagicMock()
    fake_img.PixelFormat.return_value = idscamera._ids_ipl_target('Mono12g24IDS')
    fake_img.get_numpy.return_value = full
    monkeypatch.setattr(idscamera.ids_peak_ipl_extension, 'BufferToImage', lambda buffer: fake_img)

    array, _ = handler._unpack(object())
    assert array.shape == (1900, 1920)
