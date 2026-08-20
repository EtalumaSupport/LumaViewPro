# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression net: public camera getters never leak driver failure sentinels.

One layer up from tests/test_pylon_getter_transient_reads.py (the per-driver
net): the drivers answer a failed/transient read with a sentinel (-1 for
gain / exposure / binning, None for frame_size / pixel_format, {} for the
max/min frame-size variants) or by raising. ImagingAPI's public getters must
contain all of those at the API boundary via _validated_camera_read:

  - a validated live read updates the camera cache and is returned;
  - a failed / invalid / raising read answers with the cache's
    last-known-good value;
  - when no valid value was ever read, the documented camera-absent
    default is returned (gain -1.0, exposure 0.0, frame_size None,
    pixel_format None, width/height 0, max_width/max_height 0, binning 1).

The old behavior passed the driver sentinel straight through, so a single
flaky USB read de-binned the binning spinner (-1 -> '1x1'), crashed
get_width (TypeError on None / KeyError on {}), reclassified a Mono8 camera
as 2 bytes/pixel, and poisoned save/restore snapshots with gain -1.

Harness pattern: real ImagingAPI on a Lumascope built via __new__ (the
test_camera_write_authority.py pattern) with a scripted driver double, so
the production getter path -- including _populate_camera_cache at init --
is what runs.
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import modules.common_utils as common_utils
from drivers.camera import Camera
from modules.binning import binning_size_int_to_str
from modules.image_save import generate_image_metadata
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI

# Marker: this scripted read raises instead of returning a value.
RAISE = object()

# One consistent round of good hardware values, keyed by driver method name.
GOOD_ROUND = {
    'get_gain': 12.5,
    'get_exposure_t': 50.0,
    'get_frame_size': {'width': 1936, 'height': 1216},
    'get_max_frame_size': {'width': 3840, 'height': 2160},
    'get_min_frame_size': {'width': 64, 'height': 64},
    'get_max_exposure': 5000.0,
    'get_max_gain': 24.0,
    'get_pixel_format': 'Mono12',
    'get_binning_size': 2,
}


class _StampedFrameHandler:
    """Image-handler stub for the per-frame depth stamp: get_last_image
    reports a stored frame stamped with significant_bits, or no-frame when
    the stamp is None (nothing grabbed yet)."""

    def __init__(self, significant_bits: int | None):
        self._significant_bits = significant_bits

    def get_last_image(self):
        if self._significant_bits is None:
            return False, None, None, None
        return True, 'frame', None, self._significant_bits


class ScriptedCameraDriver:
    """Driver double whose every value read follows a per-method script.

    Each script is a list of per-call results; calls pop from the front and
    the final entry repeats forever. A RAISE entry raises RuntimeError (the
    raising-driver failure class the API boundary must contain).

    The depth surface reuses the REAL base-class rule and properties
    (Camera.format_significant_bits / significant_bits /
    last_significant_bits descriptors), so the tests exercise the one
    canonical format-name-to-depth path, not a re-implementation.
    """

    native_bit_depth = 16
    format_significant_bits = staticmethod(Camera.format_significant_bits)
    significant_bits_for_format = Camera.significant_bits_for_format
    last_stamped_significant_bits = Camera.last_stamped_significant_bits
    significant_bits = Camera.significant_bits
    last_significant_bits = Camera.last_significant_bits

    def __init__(self, scripts: dict, active: bool = True):
        self.active = active
        self.cam_image_handler = None  # tests attach a _StampedFrameHandler
        self._scripts = {name: list(values) for name, values in scripts.items()}

    def _next(self, name):
        values = self._scripts[name]
        value = values.pop(0) if len(values) > 1 else values[0]
        if value is RAISE:
            raise RuntimeError('transient read failure')
        return value

    def get_gain(self):
        return self._next('get_gain')

    def get_exposure_t(self):
        return self._next('get_exposure_t')

    def get_frame_size(self):
        return self._next('get_frame_size')

    def get_max_frame_size(self):
        return self._next('get_max_frame_size')

    def get_min_frame_size(self):
        return self._next('get_min_frame_size')

    def get_max_exposure(self):
        return self._next('get_max_exposure')

    def get_max_gain(self):
        return self._next('get_max_gain')

    def get_pixel_format(self):
        return self._next('get_pixel_format')

    def get_binning_size(self):
        return self._next('get_binning_size')


def all_reads_fail_driver() -> ScriptedCameraDriver:
    """Active driver where every value read raises, forever."""
    return ScriptedCameraDriver({name: [RAISE] for name in GOOD_ROUND})


def good_then_failing_driver() -> ScriptedCameraDriver:
    """One good round of values (consumed by the init-time cache populate),
    then every subsequent read raises."""
    return ScriptedCameraDriver({name: [GOOD_ROUND[name], RAISE] for name in GOOD_ROUND})


def steady_good_driver(overrides: dict | None = None) -> ScriptedCameraDriver:
    """Every read returns its GOOD_ROUND value forever; per-method scripts
    can be overridden for targeted failure shapes."""
    scripts = {name: [value] for name, value in GOOD_ROUND.items()}
    if overrides:
        scripts.update({name: list(values) for name, values in overrides.items()})
    return ScriptedCameraDriver(scripts)


def _build_imaging(cam) -> ImagingAPI:
    """Real ImagingAPI on a bare Lumascope (test_camera_write_authority
    pattern) so the production getter / populate path is exercised."""
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging


# --- The sweep -------------------------------------------------------------
#
# (getter name, camera-absent default, expected last-known-good after the
# one good populate round from good_then_failing_driver()).
CONVERTED_GETTERS = [
    ('get_gain_db', -1.0, 12.5),
    ('get_exposure_ms', 0.0, 50.0),
    ('get_width', 0, 1936),
    ('get_height', 0, 1216),
    ('get_max_width', 0, 3840),
    ('get_max_height', 0, 2160),
    ('get_binning_size', 1, 2),
]

# The frame-size / pixel-format live reads are ImagingAPI-internal (public
# readers use the *_cached properties), but their bodies still serve the
# validated last-known-good contract to the cache populate and the internal
# callers -- the behavior sweep covers them under their private names, while
# the public-surface classification guard below sees only public getters.
CONVERTED_PRIVATE_GETTERS = [
    ('_get_frame_size', None, {'width': 1936, 'height': 1216}),
    ('_get_pixel_format', None, 'Mono12'),
]

_SWEEP = CONVERTED_GETTERS + CONVERTED_PRIVATE_GETTERS
_SWEEP_IDS = [name for name, _, _ in _SWEEP]


def _assert_value(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert actual == expected


@pytest.mark.parametrize(('getter', 'absent', '_lkg'), _SWEEP, ids=_SWEEP_IDS)
def test_all_reads_fail_cold_cache_returns_absent_default(getter, absent, _lkg):
    # Active driver, every read raises, nothing ever validly read: the getter
    # must answer the documented camera-absent default and raise nothing.
    imaging = _build_imaging(all_reads_fail_driver())
    assert imaging._driver.active
    for _ in range(2):
        _assert_value(getattr(imaging, getter)(), absent)


@pytest.mark.parametrize(('getter', '_absent', 'lkg'), _SWEEP, ids=_SWEEP_IDS)
def test_failing_reads_after_good_populate_return_last_known_good(getter, _absent, lkg):
    # One good round is consumed by the init-time populate; every driver read
    # after that raises. The getter must keep answering the last-known-good
    # value -- never a sentinel, never an exception.
    imaging = _build_imaging(good_then_failing_driver())
    for _ in range(3):
        _assert_value(getattr(imaging, getter)(), lkg)


@pytest.mark.parametrize(('getter', 'absent', '_lkg'), _SWEEP, ids=_SWEEP_IDS)
def test_no_driver_returns_absent_default(getter, absent, _lkg):
    imaging = _build_imaging(None)
    _assert_value(getattr(imaging, getter)(), absent)


@pytest.mark.parametrize(('getter', 'absent', '_lkg'), _SWEEP, ids=_SWEEP_IDS)
def test_inactive_driver_returns_absent_default(getter, absent, _lkg):
    # Driver object present but inactive: the active gate answers the
    # camera-absent default without touching the (would-be-good) reads.
    driver = steady_good_driver()
    driver.active = False
    imaging = _build_imaging(driver)
    _assert_value(getattr(imaging, getter)(), absent)


# --- Completeness guard -----------------------------------------------------
#
# Public zero-arg get_* methods NOT converted to the validated-cache contract,
# each with the reason it is exempt. A future getter added to ImagingAPI must
# land in CONVERTED_GETTERS (with sweep coverage) or here (with a reason).
EXCLUDED = {
    'get_image': 'capture path; documented None-on-failure contract, not a cached value read',
    'get_image_from_buffer': 'capture path; returns the latest buffered frame, no SDK value read',
    'capture_and_wait': 'capture path; documented None-on-failure contract',
    'get_supported_pixel_formats': 'collection contract; documented empty tuple when inactive',
    'get_available_binning_sizes': 'profile-backed; no per-call SDK read to contain',
    'get_native_resolution': 'profile-backed; no per-call SDK read to contain',
    'get_pixel_alignment': 'profile-backed; no per-call SDK read to contain',
    'get_live_camera_settings': (
        'live-confirmed surface; deliberately the inverse contract '
        '(omits unknown rather than answering last-known-good)'
    ),
}
# significant_bits / last_significant_bits are properties (not reachable by
# the zero-arg get_* introspection below); their depth contract is covered by
# the depth-property tests in this file.


def _public_zero_arg_getters() -> set:
    """Every public ImagingAPI method named get_* callable with no args."""
    names = set()
    for name, member in inspect.getmembers(ImagingAPI, predicate=inspect.isfunction):
        if not name.startswith('get_'):
            continue
        params = [p for p in inspect.signature(member).parameters.values() if p.name != 'self']
        zero_arg = all(
            p.default is not inspect.Parameter.empty
            or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        )
        if zero_arg:
            names.add(name)
    return names


def test_every_public_zero_arg_getter_is_classified():
    converted = {name for name, _, _ in CONVERTED_GETTERS}
    actual = _public_zero_arg_getters()

    unclassified = actual - converted - set(EXCLUDED)
    assert not unclassified, (
        f'unclassified public camera getter(s) on ImagingAPI: {sorted(unclassified)}. '
        f'Either convert to the validated last-known-good contract and add to '
        f'CONVERTED_GETTERS (with sweep coverage), or add to EXCLUDED with a reason.'
    )

    # The guard must not go vacuous on a rename: every converted getter must
    # still exist as a public zero-arg method, every excluded name must still
    # be an ImagingAPI attribute, and no name may be classified twice.
    missing_converted = converted - actual
    assert not missing_converted, (
        f'CONVERTED_GETTERS entries no longer on ImagingAPI (renamed?): {sorted(missing_converted)}'
    )
    stale_excluded = [name for name in EXCLUDED if not hasattr(ImagingAPI, name)]
    assert not stale_excluded, f'EXCLUDED entries no longer on ImagingAPI: {stale_excluded}'
    assert not converted & set(EXCLUDED), 'a getter may not be both converted and excluded'


# --- Member regressions (named for the defect shape) -------------------------


def test_binning_spinner_not_debinned_by_transient_read_failure():
    # Camera at 2x2; the binning read then fails transiently. The old
    # behavior returned the driver's -1 sentinel, which binning_size_int_to_str
    # maps to '1x1' -- the UI spinner silently de-binned a 2x camera.
    imaging = _build_imaging(good_then_failing_driver())
    assert imaging.get_binning_size() == 2
    assert binning_size_int_to_str(imaging.get_binning_size()) == '2x2'


def test_pixel_format_none_read_does_not_clobber_known_mono8():
    # After a populate that knows 'Mono8', a repopulate whose format read
    # returns the None sentinel must keep the known format. The old behavior
    # cached None, and raw_bytes_per_pixel classified the camera as
    # 2 bytes/pixel (the not-Mono8 branch).
    driver = steady_good_driver({'get_pixel_format': ['Mono8', None]})
    imaging = _build_imaging(driver)
    assert imaging.pixel_format_cached == 'Mono8'
    imaging._populate_camera_cache()
    assert imaging.pixel_format_cached == 'Mono8'
    assert common_utils.raw_bytes_per_pixel(imaging.pixel_format_cached) == 1


def test_get_width_returns_zero_not_typeerror_on_cold_cache_read_failure():
    # Driver present + active, frame-size read fails, nothing cached yet.
    # The old behavior subscripted the None passthrough -> TypeError.
    imaging = _build_imaging(all_reads_fail_driver())
    assert imaging._driver.active
    assert imaging.get_width() == 0
    assert imaging.get_height() == 0


def test_get_width_returns_last_known_after_transient_failure():
    # Populate-time read fails (cold), one good getter-driven read lands
    # 1936x1216, then reads fail again: the getter answers the last-known
    # width, not a crash or a zero.
    driver = steady_good_driver({'get_frame_size': [RAISE, {'width': 1936, 'height': 1216}, RAISE]})
    imaging = _build_imaging(driver)
    assert imaging.get_width() == 1936  # the one good read
    assert imaging.get_width() == 1936  # failing read -> last-known-good
    assert imaging.get_height() == 1216


def test_get_max_width_returns_zero_not_keyerror_on_empty_dict_read():
    # The max/min frame-size drivers answer a failed read with {}. The old
    # behavior subscripted it -> KeyError. Cold cache: absent default 0.
    driver = steady_good_driver({'get_max_frame_size': [{}]})
    imaging = _build_imaging(driver)
    assert imaging.get_max_width() == 0
    assert imaging.get_max_height() == 0


def test_get_max_width_returns_last_known_after_empty_dict_read():
    driver = steady_good_driver({'get_max_frame_size': [{}, {'width': 3840, 'height': 2160}, {}]})
    imaging = _build_imaging(driver)
    assert imaging.get_max_width() == 3840  # the one good read
    assert imaging.get_max_width() == 3840  # {} sentinel -> last-known-good
    assert imaging.get_max_height() == 2160


def test_save_camera_state_snapshot_not_poisoned_by_failing_reads():
    # Input half of snapshot poisoning: after a good populate, gain/exposure
    # reads fail; the snapshot must carry the last-known values, not -1 --
    # the old shape restored gain -1 after an autofocus save/restore cycle.
    imaging = _build_imaging(good_then_failing_driver())
    snapshot = imaging.save_camera_state('pre-autofocus')
    assert snapshot['gain_db'] == 12.5
    assert snapshot['exposure_ms'] == 50.0


def test_populate_none_frame_size_read_keeps_cached_geometry():
    # Populate clobber guard: a repopulate whose frame-size read returns the
    # None sentinel must leave the previously cached geometry intact. The old
    # behavior stored zero dims over the known-good size.
    driver = steady_good_driver({'get_frame_size': [{'width': 1936, 'height': 1216}, None]})
    imaging = _build_imaging(driver)
    assert imaging.frame_size_cached == {'width': 1936, 'height': 1216}
    imaging._populate_camera_cache()
    assert imaging.frame_size_cached == {'width': 1936, 'height': 1216}


# --- The live-confirmed surface (get_live_camera_settings) --------------------


def test_get_live_camera_settings_reports_live_confirmed_values():
    imaging = _build_imaging(steady_good_driver())
    assert imaging.get_live_camera_settings() == {'gain_db': 12.5, 'exposure_ms': 50.0}


def test_get_live_camera_settings_omits_failed_gain_while_getter_answers_lkg():
    # The inverse contract, side by side in the SAME state: gain reads fail
    # after one good populate, exposure reads keep succeeding. The live
    # surface must omit 'gain_db' (unknown stays unknown), while the value
    # getter keeps answering the last-known-good for control flow.
    driver = steady_good_driver({'get_gain': [12.5, RAISE]})
    imaging = _build_imaging(driver)  # populate consumes the one good gain read
    settings = imaging.get_live_camera_settings()
    assert settings == {'exposure_ms': 50.0}
    assert 'gain_db' not in settings
    assert imaging.get_gain_db() == 12.5


def test_get_live_camera_settings_empty_when_inactive():
    driver = steady_good_driver()
    driver.active = False
    imaging = _build_imaging(driver)
    assert imaging.get_live_camera_settings() == {}


# --- Write-generation guard ---------------------------------------------------


def test_authoritative_write_beats_in_flight_stale_read():
    # A getter's driver read is in flight when an authoritative cache write
    # (_commit_camera_writes: setter write-through / after-auto resync) lands.
    # The read's value is stale hardware truth from BEFORE the write; the
    # generation guard must discard it and answer the newer written value.
    driver = steady_good_driver()
    imaging = _build_imaging(driver)

    read_started = threading.Event()
    release_read = threading.Event()

    def held_gain_read():
        read_started.set()
        release_read.wait(timeout=5.0)
        return 5.0  # pre-write hardware value, stale by the time it returns

    driver.get_gain = held_gain_read

    result = {}
    reader_thread = threading.Thread(target=lambda: result.update(value=imaging.get_gain_db()))
    reader_thread.start()
    assert read_started.wait(timeout=5.0), 'driver read never started'
    imaging._commit_camera_writes({'gain_db': 20.0})  # setter write-through lands
    release_read.set()
    reader_thread.join(timeout=5.0)
    assert not reader_thread.is_alive()

    assert result['value'] == 20.0, (
        f'racing getter must answer the newer authoritative write, '
        f'not its stale hardware read; got {result["value"]}'
    )
    assert imaging.gain_db_cached == 20.0


# --- Populate resilience --------------------------------------------------------


def test_populate_survives_raising_key_and_caches_the_rest():
    # One raising key must not abort the rest of the populate round: gain and
    # exposure (and pixel format, read AFTER the raising frame-size key) are
    # still cached. The old populate body was a single try, so one raising
    # read silently dropped every remaining key.
    driver = steady_good_driver({'get_frame_size': [RAISE]})
    imaging = _build_imaging(driver)
    assert imaging.gain_db_cached == 12.5
    assert imaging.exposure_ms_cached == 50.0
    assert imaging.pixel_format_cached == 'Mono12'
    assert imaging.frame_size_cached == {'width': 0, 'height': 0}  # seed intact


# --- Read-failure observability -------------------------------------------------


def test_failed_read_warns_once_per_window(monkeypatch):
    # A failed read the getter absorbs must still be visible in the main log
    # (WARNING on the first failure per key per 5 s window), but a dead camera
    # polled at frame rate must not warn per call: the immediate second
    # failure logs at debug. lvp_logger is conftest-mocked (a MagicMock), so
    # caplog cannot observe it -- the established recorder-logger pattern from
    # test_set_exposure_time_warning_threshold.py is used instead.
    driver = steady_good_driver()
    imaging = _build_imaging(driver)

    warnings = []
    monkeypatch.setattr('modules.lumascope_api.imaging.logger', _recording_logger(warnings))

    driver._scripts['get_gain'] = [RAISE]
    imaging.get_gain_db()
    imaging.get_gain_db()  # back-to-back, well inside the 5 s window

    hits = [w for w in warnings if 'read failed' in w]
    assert len(hits) == 1, (
        f'expected exactly one WARNING containing "read failed" for two '
        f'back-to-back failures; got {len(hits)}: {hits}'
    )


# --- Saved-image metadata omission (image_save integration) ---------------------


def _metadata_scope_with_real_imaging(imaging: ImagingAPI, driver) -> SimpleNamespace:
    """The tests/test_frame_metadata_from_chunk.py stub scope, with the REAL
    ImagingAPI wired in so generate_image_metadata exercises the production
    get_live_camera_settings path. The scripted driver has no
    cam_image_handler, so the chunk-less fallback is what runs."""
    runtime_state = SimpleNamespace(
        _objective={'focal_length': 9.0},
        _labware=SimpleNamespace(config={'rows': 8, 'columns': 12, 'standard': 'SBS'}),
        _stage_offset={'x': 0, 'y': 0},
        _coordinate_transformer=SimpleNamespace(stage_to_plate=lambda **kwargs: (1.0, 2.0)),
        get_well_label=lambda: 'A1',
    )
    return SimpleNamespace(
        runtime_state=runtime_state,
        imaging=imaging,
        diagnostics=SimpleNamespace(
            get_microscope_model=lambda: 'LS720-SIM',
            get_motor_info=lambda: {'serial_number': 'SN1', 'firmware_version': 'fw'},
            get_camera_info=lambda: {'model': 'simcam'},
        ),
        illumination=SimpleNamespace(get_led_ma=lambda color: 100.0),
        _camera_driver=driver,
    )


def test_chunkless_metadata_omits_keys_when_live_reads_fail():
    # Snapshot-poisoning, output half: the cache holds a perfectly good
    # last-known gain/exposure, but this frame's live reads failed -- saved
    # metadata must omit 'gain_db' / 'exposure_time_ms' rather than record a
    # value the frame may not have been captured at.
    driver = steady_good_driver()
    imaging = _build_imaging(driver)  # good populate: cache holds 12.5 / 50.0
    driver._scripts['get_gain'] = [RAISE]
    driver._scripts['get_exposure_t'] = [RAISE]
    scope = _metadata_scope_with_real_imaging(imaging, driver)

    metadata = generate_image_metadata(scope, color='BF', x=0, y=0, z=0)

    assert 'gain_db' not in metadata, (
        f'failed live gain read must omit the key, not record '
        f'{metadata.get("gain_db")} (cached last-known-good)'
    )
    assert 'exposure_time_ms' not in metadata
    # Same state, control-flow surface: the value getters still answer LKG.
    assert imaging.get_gain_db() == 12.5
    assert imaging.get_exposure_ms() == 50.0


# --- Depth properties (significant_bits / last_significant_bits) ----------------


def test_significant_bits_holds_12_when_format_reads_fail():
    # THE Seams-1 regression: the camera is configured Mono12 (the cache
    # knows the format), then format reads fail transiently. The depth must
    # stay 12 -- the old live-derived code fell back to 16, so a fully blown
    # frame passed the saturation evidence check at 0.0% and files were
    # stamped at the wrong depth.
    driver = steady_good_driver({'get_pixel_format': ['Mono12', RAISE]})
    imaging = _build_imaging(driver)  # populate consumes the one good read
    for _ in range(2):
        assert imaging.significant_bits == 12


def test_depth_properties_report_16_with_no_camera():
    imaging = _build_imaging(None)
    assert imaging.significant_bits == 16
    assert imaging.last_significant_bits == 16


def test_significant_bits_native_fallback_when_format_never_known():
    # Active driver, every read fails, format never known: falls back to the
    # driver's container width (native_bit_depth), not a crash.
    imaging = _build_imaging(all_reads_fail_driver())
    assert imaging.significant_bits == 16


def test_last_significant_bits_prefers_per_frame_stamp():
    # A stored frame carries its own depth stamp; it wins over the
    # format-derived value (Mono12 -> 12) because the buffered frame may
    # have been captured under an older format than the camera is at now.
    driver = steady_good_driver()
    driver.cam_image_handler = _StampedFrameHandler(10)
    imaging = _build_imaging(driver)
    assert imaging.significant_bits == 12
    assert imaging.last_significant_bits == 10


def test_last_significant_bits_falls_back_to_format_derived_without_stamp():
    # No frame stored yet -> the per-frame surface answers the format-derived
    # depth (steady Mono12 -> 12), not the container width.
    driver = steady_good_driver()
    driver.cam_image_handler = _StampedFrameHandler(None)
    imaging = _build_imaging(driver)
    assert imaging.last_significant_bits == 12


def test_last_significant_bits_no_frame_fallback_survives_failed_format_read():
    # No stored frame AND the live format read now fails: the fallback must
    # route through the validated cached format (Mono12 known from populate
    # -> 12), never the driver's live-read fallback chain, which a transient
    # format failure turns into the container width (16) -- the wrong-depth
    # answer on exactly the property callers are told to prefer.
    driver = steady_good_driver({'get_pixel_format': ['Mono12', RAISE]})
    driver.cam_image_handler = _StampedFrameHandler(None)
    imaging = _build_imaging(driver)  # populate consumes the one good read
    assert imaging.last_significant_bits == 12


# --- Depth travels with the frame (protocol writer) -----------------------------


def test_writer_saves_capture_time_depth_not_save_time_rederivation(monkeypatch, tmp_path):
    # The CapturedFrame couples the frame with the depth it was captured at.
    # By save time the camera reports a DIFFERENT depth (format changed /
    # unreadable); save_image must receive the capture-time stamp. The old
    # writer re-derived depth at save time from current camera state.
    import threading as _threading
    from unittest.mock import MagicMock

    from modules.image_mode import ImageCaptureConfig
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_image_writer import CapturedFrame, ProtocolImageWriter

    scope = MagicMock()
    scope.imaging.last_significant_bits = 16  # save-time state disagrees
    scope.imaging.significant_bits = 16
    writer = ProtocolImageWriter(
        scope=scope,
        callbacks=ProtocolCallbacks(),
        aborted=_threading.Event(),
        file_io_executor=MagicMock(),
        abort_fn=lambda: None,
        fatal_abort_event=_threading.Event(),
        execution_record=None,
        leds_off_fn=lambda: None,
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('12bit_scientific'),
        timestamp_overlay=True,
        video_max_fps=0,
    )
    recorded = []
    monkeypatch.setattr(
        'modules.protocol_image_writer.save_image',
        lambda scope, **kwargs: recorded.append(kwargs) or (tmp_path / 'x.tiff'),
    )
    writer.write_capture(
        enable_image_saving=True,
        captured_image=CapturedFrame(image=np.zeros((4, 4), dtype=np.uint16), significant_bits=12),
        step={'Name': 's', 'Color': 'BF', 'X': 0.0, 'Y': 0.0, 'Z': 0.0},
        name='s_BF',
        save_folder=str(tmp_path),
        use_color='BF',
        output_format='TIFF',
    )
    assert recorded, 'write_capture must reach save_image'
    assert recorded[0]['significant_bits'] == 12, (
        f'save_image must receive the CapturedFrame depth (12), not a '
        f'save-time re-derivation; got {recorded[0]["significant_bits"]}'
    )


# --- Temp-logger un-latch (Seams-4) ----------------------------------------------


def _recording_logger(sink: list) -> SimpleNamespace:
    """Recorder standing in for the conftest-mocked lvp_logger logger."""
    return SimpleNamespace(
        warning=lambda msg, *a, **kw: sink.append(str(msg)),
        info=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        exception=lambda *a, **kw: None,
    )


def test_temp_logger_survives_transient_disconnect_and_resumes():
    # Seams-4 regression: a single transient camera_connected False during a
    # tick must NOT unschedule the temp logger. The old tick self-unscheduled
    # on the first False, permanently ending temperature logging for the rest
    # of a multi-day soak. The tick skips the sample but stays scheduled;
    # teardown belongs solely to stop_camera_temp_logging.
    driver = steady_good_driver()
    connected = {'value': True}
    driver.is_connected = lambda: connected['value']  # camera_connected reads this
    imaging = _build_imaging(driver)

    probes = []
    imaging._scope.diagnostics = SimpleNamespace(
        get_camera_temperatures_degc=lambda: probes.append(1) or {'coreboard': 42.0}
    )

    scheduled = {}
    unschedule_calls = []

    def schedule_interval_fn(fn, interval):
        scheduled['tick'] = fn
        return 'event-token'

    imaging.start_camera_temp_logging(schedule_interval_fn, unschedule_calls.append, interval_s=1.0)
    assert len(probes) == 1  # the one immediate sample at start

    connected['value'] = False  # transient disconnect for one tick
    scheduled['tick']()
    assert unschedule_calls == [], (
        'a transient disconnected tick must not unschedule the temp logger '
        '(the old behavior self-unscheduled forever on the first False)'
    )
    assert imaging._camera_temp_event == 'event-token', 'the event must survive the tick'
    assert len(probes) == 1  # sample skipped while disconnected

    connected['value'] = True  # camera back -> next tick samples again
    scheduled['tick']()
    assert len(probes) == 2, 'logging must resume after the transient disconnect'

    # Explicit owner still tears down.
    imaging.stop_camera_temp_logging()
    assert unschedule_calls == ['event-token']
    assert imaging._camera_temp_event is None


def test_disconnect_tears_down_temp_logging_schedule():
    # A scope swap (the reconnect path) discards the whole Lumascope. The
    # temp tick deliberately never self-cancels, so Lumascope.disconnect()
    # must be the teardown owner -- without it the orphaned schedule keeps
    # sampling the discarded scope and pins its object graph for the rest
    # of the session.
    driver = steady_good_driver()
    imaging = _build_imaging(driver)
    scope = imaging._scope
    scope.motion = SimpleNamespace(stop_motion=lambda: None, disconnect=lambda: None)
    scope._led_driver = SimpleNamespace(disconnect=lambda: None)
    scope._motion_driver = SimpleNamespace(disconnect=lambda: None)

    unschedule_calls = []
    imaging.start_camera_temp_logging(
        lambda fn, interval: 'event-token', unschedule_calls.append, interval_s=1.0
    )
    assert imaging._camera_temp_event == 'event-token'

    scope.disconnect()

    assert unschedule_calls == ['event-token'], (
        'disconnect() must cancel the temp-log schedule it owns; an '
        'orphaned event outlives the scope swap otherwise'
    )
    assert imaging._camera_temp_event is None


# --- Snapshot omit-if-unknown (Seams-9, restore side) ------------------------------


def test_save_camera_state_omits_never_read_fields_and_warns(monkeypatch):
    # Cold cache, every read fails: the snapshot must carry NO gain/exposure
    # keys (omit-if-unknown -- the old shape stored gain -1 and a later
    # restore drove the sentinel back toward the camera), and a WARNING
    # names each omitted field at SAVE time.
    imaging = _build_imaging(all_reads_fail_driver())
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.imaging.logger', _recording_logger(warnings))

    snapshot = imaging.save_camera_state('t')

    assert snapshot == {'tag': 't'}
    save_warnings = [w for w in warnings if 'save_camera_state' in w]
    assert len(save_warnings) == 2, save_warnings
    assert any('gain' in w for w in save_warnings)
    assert any('exposure' in w for w in save_warnings)


def test_save_camera_state_carries_both_fields_without_warning(monkeypatch):
    imaging = _build_imaging(steady_good_driver())
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.imaging.logger', _recording_logger(warnings))

    snapshot = imaging.save_camera_state('t')

    assert snapshot == {'tag': 't', 'gain_db': 12.5, 'exposure_ms': 50.0}
    assert warnings == []


def test_restore_camera_state_trimmed_snapshot_restores_only_present_fields(monkeypatch):
    # A caller-trimmed snapshot (autofocus keeps only the fields its run
    # targeted) is legitimate: restore the present field, skip the absent one,
    # and log NO warning -- save_camera_state already warned if the omission
    # was a never-read.
    imaging = _build_imaging(steady_good_driver())
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.imaging.logger', _recording_logger(warnings))
    calls = []
    monkeypatch.setattr(imaging, '_set_gain_db_impl', lambda g: calls.append(('gain', g)))
    monkeypatch.setattr(imaging, '_set_exposure_ms_impl', lambda e: calls.append(('exposure', e)))

    imaging.restore_camera_state({'tag': 't', 'exposure_ms': 50.0})

    assert calls == [('exposure', 50.0)], (
        f'only the present field may be restored; set_gain_db must not run: {calls}'
    )
    assert warnings == []


def test_restore_camera_state_empty_snapshot_is_noop(monkeypatch):
    imaging = _build_imaging(steady_good_driver())
    calls = []
    monkeypatch.setattr(imaging, '_set_gain_db_impl', lambda g: calls.append(('gain', g)))
    monkeypatch.setattr(imaging, '_set_exposure_ms_impl', lambda e: calls.append(('exposure', e)))

    imaging.restore_camera_state({})

    assert calls == []


# --- raw_bytes_per_pixel loud-input ------------------------------------------------


def test_raw_bytes_per_pixel_unknown_input_warns_once_per_distinct_value(monkeypatch):
    # A sentinel pixel format reaching the data-rate math means a consumer
    # bypassed the getter containment: classify as the 2-byte container but
    # say so -- exactly one WARNING per distinct bad value, not per call.
    # _RAW_BPP_WARNED is module state; reset it so test order cannot mask
    # the warning.
    monkeypatch.setattr(common_utils, '_RAW_BPP_WARNED', set())
    warnings = []
    monkeypatch.setattr(common_utils, 'logger', _recording_logger(warnings))

    assert common_utils.raw_bytes_per_pixel(None) == 2
    assert common_utils.raw_bytes_per_pixel(None) == 2
    hits = [w for w in warnings if 'raw_bytes_per_pixel' in w]
    assert len(hits) == 1, (
        f'two calls with the same unknown input must warn exactly once; got {hits}'
    )

    # Valid strings behave as before, silently.
    assert common_utils.raw_bytes_per_pixel('Mono8') == 1
    assert common_utils.raw_bytes_per_pixel('Mono12') == 2
    assert len([w for w in warnings if 'raw_bytes_per_pixel' in w]) == 1
