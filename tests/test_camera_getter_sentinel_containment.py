# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression net: a driver's failed-read value never becomes a value the camera holds.

The drivers answer a failed or transient read with a sentinel (-1 for gain /
exposure / binning, None for frame_size / pixel_format, {} for the max/min
frame-size variants) or by raising. The values the camera holds (its cache:
what it last reported or took) are never overwritten by one: a fresh read
that fails raises ``CameraReadFailedError`` and leaves the value held as it
was, and the API's own refreshes (bring-up's populate, the frame after a
binning write) log the failure and keep it. The fresh reads' raise is pinned
on the simulator in tests/test_a_fresh_camera_read_raises.py.

The old behavior passed the driver sentinel straight through, so a single
flaky USB read de-binned the binning spinner (-1 -> '1x1'), reclassified a
Mono8 camera as 2 bytes/pixel, and poisoned save/restore snapshots with
gain -1.

Each failed read is the simulated camera's (``scope.simulation.refuse_camera_read``),
which answers as the real driver of that node does. A read that fails before
bring-up's first populate is not yet injectable there (a simulator gap): those
tests build the camera API over a scripted driver double.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tests.protocol_drives import lent_run_claim
from tests.frame_records import frame_record, plate, unpositioned
from modules.protocol_image_writer import RunWriteBatch
from drivers.camera import Camera
from drivers.exceptions import CameraReadFailedError
from modules.exceptions import HardwareCommandRefusedError
from modules.lumascope_api import Lumascope
from modules.lumascope_api.camera import CameraAPI
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


from modules.run_outcome import EndingLatch
from tests.scope_fakes import (
    bind_settings_like_a_session,
    build_scope,
    give_camera_and_imaging,
    give_camera_capabilities,
    give_stub_lanes,
)
from modules.layer_record import release_colors
from modules.lumascope_api.frame_record import CapturedStill


class ScriptedCameraDriver:
    """Driver double whose every value read follows a per-method script.

    Each script is a list of per-call results; calls pop from the front and
    the final entry repeats forever. A RAISE entry raises
    ``CameraReadFailedError``, as a real driver's failed read does (the
    failure the API boundary must contain).

    The depth surface reuses the REAL base-class rule
    (Camera.format_significant_bits / significant_bits_for_format), so the
    tests exercise the one
    canonical format-name-to-depth path, not a re-implementation.
    """

    native_bit_depth = 16
    format_significant_bits = staticmethod(Camera.format_significant_bits)
    significant_bits_for_format = Camera.significant_bits_for_format
    # Reuses the real rule: the ordinal comes from the attached handler, and
    # is 0 when there is none -- nothing has been delivered.
    frames_delivered = Camera.frames_delivered

    def is_device_removed(self) -> bool:
        # A scripted camera is never unplugged; the API asks before writing.
        return False

    def is_connected(self) -> bool:
        # Connected while active: the camera lane asks before a command runs.
        return self.active

    def __init__(self, scripts: dict, active: bool = True):
        self.active = active
        self.cam_image_handler = None  # tests attach a _StampedFrameHandler
        self._scripts = {name: list(values) for name, values in scripts.items()}

    def _next(self, name):
        values = self._scripts[name]
        value = values.pop(0) if len(values) > 1 else values[0]
        if value is RAISE:
            raise CameraReadFailedError('transient read failure')
        return value

    def get_gain(self):
        return self._next('get_gain')

    def get_black_level(self):
        # This double reports no black level; the record carries none.
        return None

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


def _build_camera(cam) -> CameraAPI:
    """Real CameraAPI on a bare Lumascope (test_camera_write_authority
    pattern) so the production getter / populate path is exercised. The
    scope is bound to the shipped settings, whose image mode and region a
    snapshot carries."""
    scope = Lumascope.__new__(Lumascope)
    bind_settings_like_a_session(scope)
    scope._camera_driver = cam
    give_stub_lanes(scope)
    # The scripted driver declares no static facts: no model, no frame
    # maximum, no auto modes.
    give_camera_capabilities(scope, None)
    return give_camera_and_imaging(scope)


def _build_imaging(cam) -> ImagingAPI:
    """The real ImagingAPI (the capture) of a ``_build_camera`` scope."""
    return _build_camera(cam)._scope.imaging


# --- Member regressions (named for the defect shape) -------------------------


def test_pixel_format_none_read_does_not_clobber_known_mono8(sim_scope):
    # A repopulate whose format read returns the None sentinel must keep the
    # known format. The old behavior cached None, and every reader of the
    # format then saw no format at all.
    camera = sim_scope.camera
    known = camera.pixel_format
    assert known is not None
    sim_scope.simulation.refuse_camera_read('pixel_format', 'failed')
    camera._populate_camera_cache()
    assert camera.pixel_format == known


def test_save_camera_state_snapshot_not_poisoned_by_failing_reads(sim_scope):
    # Input half of snapshot poisoning: after a good populate, gain/exposure
    # reads fail; the snapshot carries the values held, not -1 -- the old
    # shape restored gain -1 after an autofocus save/restore cycle.
    camera = sim_scope.camera
    camera.set_gain_db(12.5)
    camera.set_exposure_ms(50.0)
    sim_scope.simulation.refuse_camera_read('gain', 'failed')
    sim_scope.simulation.refuse_camera_read('exposure', 'failed')
    snapshot = camera.save_camera_state('pre-autofocus')
    assert snapshot['gain_db'] == 12.5
    assert snapshot['exposure_ms'] == 50.0


def test_populate_none_frame_size_read_keeps_cached_geometry(sim_scope):
    # Populate clobber guard: a repopulate whose frame-size read returns the
    # None sentinel must leave the previously cached geometry intact. The old
    # behavior stored zero dims over the known-good size.
    camera = sim_scope.camera
    known = camera.frame_size.to_dict()
    sim_scope.simulation.refuse_camera_read('frame_size', 'failed')
    camera._populate_camera_cache()
    assert camera.frame_size.to_dict() == known


# --- Write-generation guard ---------------------------------------------------


def test_authoritative_write_beats_in_flight_stale_read(sim_camera, monkeypatch):
    # A fresh read's driver read is in flight when an authoritative cache write
    # (_commit_camera_writes: setter write-through / after-auto resync) lands.
    # The read's value is stale hardware truth from BEFORE the write; the
    # generation guard must discard it and answer the newer written value.
    camera, cam = sim_camera

    read_started = threading.Event()
    release_read = threading.Event()

    def held_gain_read():
        read_started.set()
        release_read.wait(timeout=5.0)
        return 5.0  # pre-write hardware value, stale by the time it returns

    monkeypatch.setattr(cam, 'get_gain', held_gain_read)

    result = {}
    reader_thread = threading.Thread(target=lambda: result.update(value=camera.read_gain_db()))
    reader_thread.start()
    assert read_started.wait(timeout=5.0), 'driver read never started'
    camera._commit_camera_writes({'gain_db': 20.0})  # setter write-through lands
    release_read.set()
    reader_thread.join(timeout=5.0)
    assert not reader_thread.is_alive()

    assert result['value'] == 20.0, (
        f'a racing read must answer the newer authoritative write, '
        f'not its stale hardware read; got {result["value"]}'
    )
    assert camera.gain_db == 20.0


# --- Populate resilience --------------------------------------------------------


def test_populate_survives_raising_key_and_caches_the_rest():
    # A stand-in by design: the read fails at bring-up's populate, which the
    # simulator cannot yet inject (a simulator gap).
    # One raising key must not abort the rest of the populate round: gain and
    # exposure (and pixel format, read AFTER the raising frame-size key) are
    # still cached. The old populate body was a single try, so one raising
    # read silently dropped every remaining key.
    driver = steady_good_driver({'get_frame_size': [RAISE]})
    camera = _build_camera(driver)
    assert camera.gain_db == 12.5
    assert camera.exposure_ms == 50.0
    assert camera.pixel_format == 'Mono12'
    assert camera.frame_size is None  # seed intact


# --- Depth property (significant_bits) ---------------------------------------------


def test_significant_bits_holds_12_when_format_reads_fail(sim_scope):
    # THE Seams-1 regression: the camera is configured Mono12 (the cache
    # knows the format), then format reads fail transiently. The depth must
    # stay 12 -- the old live-derived code fell back to 16, so a fully blown
    # frame passed the saturation evidence check at 0.0% and files were
    # stamped at the wrong depth.
    sim_scope.camera.set_image_mode('12bit_scientific')
    sim_scope.simulation.refuse_camera_read('pixel_format', 'failed', count=2)
    for _ in range(2):
        assert sim_scope.imaging.significant_bits == 12


def test_with_no_camera_the_published_depth_is_refused(no_hardware):
    scope = build_scope(warn_pre_release=False)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        _ = scope.imaging.significant_bits
    assert refused.value.reason == 'not_connected'


def test_significant_bits_native_fallback_when_format_never_known():
    # A stand-in by design: the format is never known only if its read fails
    # at bring-up's populate, which the simulator cannot yet inject (a
    # simulator gap).
    # Active driver, every read fails, format never known: falls back to the
    # driver's container width (native_bit_depth), not a crash.
    imaging = _build_imaging(all_reads_fail_driver())
    assert imaging.significant_bits == 16


# --- Depth travels with the frame (protocol writer) -----------------------------


def test_writer_saves_capture_time_depth_not_save_time_rederivation(
    monkeypatch, tmp_path, sim_scope
):
    # The CapturedFrame couples the frame with the depth it was captured at.
    # By save time the camera reports a DIFFERENT depth (format changed /
    # unreadable); save_image must receive the capture-time stamp. The old
    # writer re-derived depth at save time from current camera state.
    import threading as _threading
    from unittest.mock import MagicMock

    from modules.image_mode import ImageCaptureConfig
    from modules.protocol_image_writer import CapturedFrame, ProtocolImageWriter
    from modules.run_events import RunEvents

    # The simulated camera's published depth at save time disagrees with the frame's.
    assert sim_scope.imaging.significant_bits != 12
    writer = ProtocolImageWriter(
        scope=sim_scope,
        events=RunEvents(),
        aborted=_threading.Event(),
        write_batch=RunWriteBatch(MagicMock()),
        abort_fn=lambda: None,
        end_run_fatally=lambda reason, message: None,
        fatal_abort_event=_threading.Event(),
        ending=EndingLatch(),
        execution_record=None,
        leds_off_fn=lambda: None,
        is_run_in_progress_fn=lambda: True,
        image_capture_config=ImageCaptureConfig.from_image_mode('12bit_scientific'),
        timestamp_overlay=True,
        video_max_fps=0,
        engineering_mode=False,
        run_claim=lent_run_claim(),
        labware=plate(),
        to_plate=None,
        captures_asked=1,
        channel_colors=release_colors(),
    )
    recorded = []
    monkeypatch.setattr(
        'modules.protocol_image_writer.save_image',
        lambda scope, **kwargs: recorded.append(kwargs) or (tmp_path / 'x.tiff'),
    )
    writer.write_capture(
        step_index=0,
        scan_count=0,
        enable_image_saving=True,
        captured_image=CapturedFrame(
            still=CapturedStill(
                image=np.zeros((4, 4), dtype=np.uint16),
                record=frame_record(frame_significant_bits=12),
            ),
            objective_id='4x Oly',
            position=unpositioned(),
            fallback_focus=None,
        ),
        step={'Name': 's', 'Color': 'BF', 'False_Color': False, 'X': 0.0, 'Y': 0.0, 'Z': 0.0},
        name='s_BF',
        save_folder=str(tmp_path),
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
    # A stand-in by design: a camera that is away for one tick and back is
    # not a fault the simulator offers (its removal is for the session), a
    # simulator gap.
    driver = steady_good_driver()
    connected = {'value': True}
    driver.is_connected = lambda: connected['value']  # camera_connected reads this
    imaging = _build_imaging(driver)

    probes = []
    # The read answers None while no camera is active, as the API does.
    imaging._scope.diagnostics = SimpleNamespace(
        get_camera_temperatures_degc=lambda: (
            (probes.append(1) or {'coreboard': 42.0}) if connected['value'] else None
        )
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
    scope = build_scope(simulate=True, warn_pre_release=False)
    imaging = scope.imaging

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

# The shipped template's image mode and region, which a snapshot of an active
# camera carries for the restore to save back.
SHIPPED_STORED = {
    'image_mode': '8bit',
    'binning.size': '1x1',
    'frame.width': 1900,
    'frame.height': 1900,
    'frame.native_width': 1900,
    'frame.native_height': 1900,
    'camera.high_conversion_gain': False,
    'camera.line_noise_reduction': False,
}


def test_save_camera_state_records_unknown_fields_as_none_and_warns(monkeypatch):
    # Cold cache, gain and exposure reads fail: the snapshot records each as
    # None, unknown, which the restore leaves alone (the old shape stored gain
    # -1 and a later restore drove the sentinel back toward the camera), and
    # a WARNING names each unknown field at SAVE time. No bring-up wrote the
    # conversion-gain mode or line-noise reduction, so those are unknown too.
    # A stand-in by design: the cache is cold only if the reads fail at
    # bring-up's populate, which the simulator cannot yet inject (a simulator
    # gap).
    camera = _build_camera(steady_good_driver({'get_gain': [RAISE], 'get_exposure_t': [RAISE]}))
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.camera.logger', _recording_logger(warnings))

    snapshot = camera.save_camera_state('t')

    assert snapshot == {
        'tag': 't',
        'gain_db': None,
        'exposure_ms': None,
        'auto_gain_arm': None,
        'held_layer': None,
        'frame_size': {'width': 1936, 'height': 1216},
        'pixel_format': 'Mono12',
        'high_conversion_gain': None,
        'line_noise_reduction': None,
        'stored': SHIPPED_STORED,
    }
    save_warnings = [w for w in warnings if 'save_camera_state' in w]
    assert len(save_warnings) == 4, save_warnings
    for field in ('the gain is', 'the exposure is', 'high_conversion_gain', 'line_noise_reduction'):
        assert any(field in w for w in save_warnings), (field, save_warnings)


def test_save_camera_state_carries_both_fields_without_warning(sim_scope, monkeypatch):
    camera = sim_scope.camera
    # As bring-up does: a camera without either toggle is at its standard
    # setting, which the writer records without reaching the driver.
    camera._set_conversion_gain_mode_impl('Low')
    camera._set_line_noise_reduction_impl(False)
    camera.set_gain_db(12.5)
    camera.set_exposure_ms(50.0)
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.camera.logger', _recording_logger(warnings))

    snapshot = camera.save_camera_state('t')

    assert (snapshot['gain_db'], snapshot['exposure_ms']) == (12.5, 50.0)
    assert warnings == []


def test_restore_camera_state_trimmed_snapshot_restores_only_present_fields(sim_scope, monkeypatch):
    # A caller-trimmed snapshot (autofocus keeps only the fields its run
    # targeted) is legitimate: restore the present field, skip the absent one,
    # and log NO warning -- save_camera_state already warned if the omission
    # was a never-read.
    camera = sim_scope.camera
    warnings = []
    monkeypatch.setattr('modules.lumascope_api.camera.logger', _recording_logger(warnings))
    calls = []
    monkeypatch.setattr(camera, '_set_gain_db_impl', lambda g: calls.append(('gain', g)))
    monkeypatch.setattr(camera, '_set_exposure_ms_impl', lambda e: calls.append(('exposure', e)))

    camera.restore_camera_state({'tag': 't', 'exposure_ms': 50.0})

    assert calls == [('exposure', 50.0)], (
        f'only the present field may be restored; set_gain_db must not run: {calls}'
    )
    assert warnings == []


def test_restore_camera_state_empty_snapshot_is_noop(sim_scope, monkeypatch):
    camera = sim_scope.camera
    calls = []
    monkeypatch.setattr(camera, '_set_gain_db_impl', lambda g: calls.append(('gain', g)))
    monkeypatch.setattr(camera, '_set_exposure_ms_impl', lambda e: calls.append(('exposure', e)))

    camera.restore_camera_state({})

    assert calls == []
