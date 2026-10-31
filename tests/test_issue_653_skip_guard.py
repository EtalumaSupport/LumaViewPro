"""Regression for #653 residual: OnImagesSkipped must guard on device-removed.

On a USB camera disconnect the SDK fires a burst of OnImagesSkipped
callbacks during teardown. Those frames were dropped by the removal, not by
LatestImageOnly grab-strategy pressure, so logging them is misleading noise
(Rule 20) -- the bench saw ~16 stray "frames discarded" lines after the
device was already marked removed. OnImageGrabbed already early-returns on
self._parent._device_removed; OnImagesSkipped must do the same.

Behavioral since the typed pypylon stub landed: the handler is
instantiated and the callback driven directly (this file was previously
the canonical source-assertion fallback).
"""

import logging

import pytest

from tests.log_capture import records_of


@pytest.fixture
def handler_and_log(caplog):
    """The handler, its camera, and the driver's camera-log lines at a level."""
    from drivers import pyloncamera

    parent = pyloncamera.PylonCamera.__new__(pyloncamera.PylonCamera)
    parent._device_removed = False
    handler = pyloncamera.ImageHandler(parent)
    caplog.set_level(logging.INFO)

    def cam_lines(level):
        return [
            r.getMessage()
            for r in records_of(caplog, pyloncamera)
            if r.name == pyloncamera._cam_log.name and r.levelno == level
        ]

    return handler, parent, cam_lines


def test_removal_skip_burst_is_silent(handler_and_log):
    """Skips fired after the device is marked removed must not log --
    they are teardown artifacts, not grab-strategy drops."""
    handler, parent, cam_lines = handler_and_log
    parent._device_removed = True
    handler.OnImagesSkipped(camera=None, countOfSkippedImages=16)
    assert cam_lines(logging.INFO) == []
    assert cam_lines(logging.WARNING) == []


def test_live_skip_is_logged(handler_and_log):
    """With the device present, a real grab-strategy drop logs once so
    the skip distribution stays visible in camera.log."""
    handler, _parent, cam_lines = handler_and_log
    handler.OnImagesSkipped(camera=None, countOfSkippedImages=3)
    assert len(cam_lines(logging.INFO)) == 1


def test_zero_count_skip_is_silent(handler_and_log):
    """The SDK can fire with countOfSkippedImages=0; nothing to report."""
    handler, _parent, cam_lines = handler_and_log
    handler.OnImagesSkipped(camera=None, countOfSkippedImages=0)
    assert cam_lines(logging.INFO) == []
