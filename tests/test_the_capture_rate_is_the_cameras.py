# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The capture rate is the camera's, counted where frames are delivered, in every host.

``[BUFFER METRICS] capture_fps`` and the title's "Capture" were counted by the
GUI's display thread, after its duplicate-frame skip, so they were the rate the
display pulled frames: under the display's 30 fps cap the figure could not
read above 30 whatever the camera delivered (29.5 against the camera's 42.4 on
an LS850T), it froze at its last value while the display was paused, and a
headless or REST session never logged it at all. The MB/s beside it was that
rate times the frame's size in host memory, not what crossed the link.

The imaging API now owns both: frames and wire bytes are counted where every
driver stores a frame, and the API publishes their rates over the last second
from its own stream check, which runs in every host. A rate whose sample is
old -- the check stopped, the camera stopped streaming -- reads 0, never its
last value. Every session here is headless, with no display at all.
"""

from __future__ import annotations

import time
from collections import defaultdict

import pytest

from modules import app_context, config_helpers
from modules.lumascope_api.imaging import ImagingAPI
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# The stream check's cadence, shortened so a window is seen in a fraction of a second.
TICK_S = 0.2


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(ImagingAPI, '_STREAM_CHECK_INTERVAL_S', TICK_S)
    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        yield session
    finally:
        session.shutdown()


def _wait_for(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _streaming_fast(session):
    """The simulated camera at a short exposure: its 40 fps delivery ceiling."""
    imaging = session.scope.imaging
    imaging.set_exposure_ms(2.0)
    camera = session.scope._camera_driver
    camera_fps = camera.get_resulting_frame_rate()
    assert camera_fps > 30, 'the simulated camera does not outrun a 30 fps display cap'
    return imaging, camera, camera_fps


def _camera_and_api_rates(imaging, camera, span_s=1.0):
    """The camera's own delivery rate over ``span_s``, and the API's rate read across it.

    The camera's figure is its delivered-frame count over the span, so both
    describe the same frames: on a loaded host the simulator delivers fewer
    than its 40 fps (19.5 at load 31, 2026-10-09), and the API must read what
    was delivered, not what the camera was set to. The API's figure is the
    mean of its readings every half tick.
    """
    frames_before, _ = camera.delivered_counts
    started = time.monotonic()
    readings = []
    while time.monotonic() - started < span_s:
        readings.append(imaging.get_delivered_rate().frames_per_s)
        time.sleep(TICK_S / 2)
    frames_after, _ = camera.delivered_counts
    camera_rate = (frames_after - frames_before) / (time.monotonic() - started)
    return camera_rate, sum(readings) / len(readings)


class TestTheRateIsTheCameras:
    def test_the_delivered_rate_is_the_cameras_with_no_display(self, session):
        imaging, camera, _camera_fps = _streaming_fast(session)
        assert _wait_for(lambda: imaging.get_delivered_rate().frames_per_s > 0, 5.0), (
            'a session with no display read no delivered rate'
        )
        camera_rate, api_rate = _camera_and_api_rates(imaging, camera)
        assert camera_rate > 0
        assert api_rate == pytest.approx(camera_rate, rel=0.25)

    def test_the_wire_rate_is_the_bytes_each_delivered_frame_took(self, session):
        imaging, camera, _camera_fps = _streaming_fast(session)
        assert _wait_for(lambda: imaging.get_delivered_rate().frames_per_s > 0, 5.0)
        rate = imaging.get_delivered_rate()
        # Mono8 on the simulated USB3 link: one byte a pixel of the acquired window.
        assert imaging.pixel_format_cached == 'Mono8'
        acquired_bytes = camera._width * camera._height
        assert rate.bytes_per_s / rate.frames_per_s == pytest.approx(acquired_bytes, rel=1e-6)

    def test_a_rate_whose_check_stopped_reads_zero(self, session):
        imaging, _camera, _camera_fps = _streaming_fast(session)
        assert _wait_for(lambda: imaging.get_delivered_rate().frames_per_s > 0, 5.0)
        imaging.stop_stream_check()
        time.sleep(3 * TICK_S)
        rate = imaging.get_delivered_rate()
        assert rate.frames_per_s == 0
        assert rate.bytes_per_s == 0


class TestEveryHostLogsIt:
    def test_a_headless_session_logs_the_cameras_rate(self, session, monkeypatch, tmp_path):
        imaging, _camera, _camera_fps = _streaming_fast(session)
        assert _wait_for(lambda: imaging.get_delivered_rate().frames_per_s > 0, 5.0)

        # The line logs the API's rate: what get_delivered_rate answered the
        # tick, kept as it passes. Whether that rate is the camera's is
        # TestTheRateIsTheCameras'; on a loaded host one window's rate and a
        # longer average differ, so the two are not compared here.
        answered = []
        read_rate = imaging.get_delivered_rate

        def _kept():
            rate = read_rate()
            answered.append(rate)
            return rate

        monkeypatch.setattr(imaging, 'get_delivered_rate', _kept)

        class _RecordingLogger:
            def __init__(self):
                self.messages = []

            def info(self, msg):
                self.messages.append(msg)

        rec = _RecordingLogger()
        monkeypatch.setattr(config_helpers, 'metrics_logger', rec)
        monkeypatch.setattr(app_context, 'ctx', None)
        monkeypatch.setattr(config_helpers.common_utils, 'check_disk_space', lambda **k: 1.0e5)
        monkeypatch.setattr(
            config_helpers.common_utils, 'system_metrics', lambda **k: defaultdict(float)
        )
        monkeypatch.setitem(session.metrics_logger._settings, 'live_folder', str(tmp_path))

        session.metrics_logger.tick_system_metrics()

        lines = [m for m in rec.messages if '[BUFFER METRICS]' in m]
        assert lines, 'a headless session logged no [BUFFER METRICS] line'
        fields = dict(part.strip().split('=', 1) for part in lines[0].split(']', 1)[1].split('|'))
        assert len(answered) == 1, 'the tick did not read the API rate once'
        assert answered[0].frames_per_s > 0
        assert fields['capture_fps'] == f'{answered[0].frames_per_s:.1f}'
        assert 'display_fps' not in fields, 'a session with no display logged a display rate'
