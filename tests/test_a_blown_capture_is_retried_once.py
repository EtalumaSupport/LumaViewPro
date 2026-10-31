# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture that checks saturation retries a blown frame once, outside the camera lock.

``get_image(all_ones_check=True)`` measures the frame it grabbed against its
own depth. A frame at or above the blown fraction is retried with one more
grab: a clean retry is returned, a blown one or a failed grab leaves the
first frame, kept and recorded blown on the capture's record, not told. The measure needs no camera state, so it runs with the
camera lock released, where a concurrent gain or exposure write can proceed.
Moved from tests/test_audit_fixes.py (AOC-1, AOC-2) onto the simulated scope.
"""

import logging
import threading

import numpy as np

from modules.lumascope_api.imaging import ImagingAPI
from tests.camera_fakes import frames_carrying, grab_a_frame_made_after_now


def test_a_blown_frame_is_retried_and_a_clean_retry_returned(sim_scope, monkeypatch):
    blown = np.full((4, 4), 255, dtype=np.uint8)
    clean = np.zeros((4, 4), dtype=np.uint8)
    arrays = [blown, clean]
    frames_carrying(monkeypatch, sim_scope._camera_driver, lambda: arrays.pop(0))

    out = sim_scope.imaging.get_image(all_ones_check=True)

    assert not arrays, 'the blown first frame must trigger one retry grab'
    assert np.array_equal(out, clean), 'the clean retry frame must be returned'


def test_a_frame_below_the_blown_fraction_is_not_retried(sim_scope, monkeypatch):
    partial = np.zeros((4, 4), dtype=np.uint8)
    partial[:2, :] = 255  # 50% saturated, well below the blown fraction
    arrays = [partial]
    frames_carrying(monkeypatch, sim_scope._camera_driver, lambda: arrays.pop(0))

    out = sim_scope.imaging.get_image(all_ones_check=True)

    assert not arrays, 'a below-threshold frame must not trigger a retry'
    assert np.array_equal(out, partial)


def test_every_saturation_walk_runs_with_the_camera_lock_released(sim_scope, monkeypatch):
    """Probed from a helper thread inside every fraction call: each must find
    the lock acquirable."""
    imaging = sim_scope.imaging
    blown = np.full((4, 4), 255, dtype=np.uint8)
    arrays = [blown, blown]  # first and retry both blown: both walks run
    frames_carrying(monkeypatch, sim_scope._camera_driver, lambda: arrays.pop(0))
    measure = ImagingAPI.saturated_fraction
    lock_was_free = []

    def probing_fraction(frame, significant_bits):
        seen = {}

        def try_acquire():
            got = sim_scope.camera._cam_lock.acquire(blocking=False)
            if got:
                sim_scope.camera._cam_lock.release()
            seen['free'] = got

        probe = threading.Thread(target=try_acquire)
        probe.start()
        probe.join()
        lock_was_free.append(seen['free'])
        return measure(frame, significant_bits)

    monkeypatch.setattr(ImagingAPI, 'saturated_fraction', staticmethod(probing_fraction))

    imaging.get_image(all_ones_check=True)

    assert len(lock_was_free) >= 2, 'gate + retry walk must both run'
    assert all(lock_was_free), f'a saturation walk held the camera lock: {lock_was_free}'


def test_a_failed_retry_grab_keeps_the_first_frame(sim_scope, monkeypatch):
    camera = sim_scope._camera_driver
    blown = np.full((4, 4), 255, dtype=np.uint8)
    frames_carrying(monkeypatch, camera, lambda: blown)
    first = grab_a_frame_made_after_now(camera)
    grabs = iter([first, None])
    # The retry is the second grab; it gets no frame.
    monkeypatch.setattr(camera, 'grab', lambda: next(grabs))

    out = sim_scope.imaging.get_image(all_ones_check=True)

    assert np.array_equal(out, blown), 'a failed retry grab must fall back to the original frame'


def test_a_kept_blown_frame_is_on_its_record_and_not_told(sim_scope, monkeypatch, caplog):
    """The capture records a blown frame and leaves the telling to whoever
    took it: a run once per step, a still once."""
    blown = np.full((4, 4), 255, dtype=np.uint8)
    frames_carrying(monkeypatch, sim_scope._camera_driver, lambda: blown)

    with caplog.at_level(logging.WARNING):
        record = sim_scope.imaging.capture_still(all_ones_check=True).record
    assert (record.saturated_fraction, record.blown) == (1.0, True)
    told = [line for line in caplog.messages if 'saturated' in line]
    assert told == [], told


def test_a_capture_that_does_not_ask_records_no_saturation(sim_scope, monkeypatch):
    blown = np.full((4, 4), 255, dtype=np.uint8)
    frames_carrying(monkeypatch, sim_scope._camera_driver, lambda: blown)

    record = sim_scope.imaging.capture_still().record
    assert (record.saturated_fraction, record.blown) == (None, None)
