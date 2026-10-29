# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera whose stream start fails at bring-up is told once, and the scope comes up.

Bring-up ends by releasing the camera start gate. A start that fails there
raised out of bring-up, which tore the session down: an FX2 that refused
its stream START kept LumaViewPro from launching at all. The failure is now
what bring-up found about the camera, kept on its part of the bring-up
record and reported once with the start's own fault, and the scope comes
up with its camera connected and not streaming.

The refusal is armed through ``create``'s ``sim_refused_stream_starts``,
before bring-up runs, so the start refused is bring-up's own.
"""

from __future__ import annotations

import time

import pytest

import modules.video_cadence as video_cadence
from drivers.exceptions import CameraStreamStartFailedError
from modules.exceptions import CameraStreamStalledError
from modules.lumascope_api.imaging import ImagingAPI
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def heard(monkeypatch):
    # The stall bound and the check's cadence shortened, as the stall's own
    # tests do, so a stall would be reported well inside the wait below.
    monkeypatch.setattr(video_cadence, 'STALL_FLOOR_S', 0.5)
    monkeypatch.setattr(ImagingAPI, '_STREAM_CHECK_INTERVAL_S', 0.1)
    return []


@pytest.fixture
def refused_at_bring_up(heard):
    session = ScopeSession.create(
        complete_settings(microscope='LS620'),
        simulate=True,
        sim_refused_stream_starts=1,
        warn_pre_release=False,
        outcome_listener=heard.append,
    )
    yield session
    session.shutdown()


def test_the_scope_comes_up_and_the_refused_start_is_told_once(refused_at_bring_up, heard):
    told = [n for n in heard if n.reason == CameraStreamStartFailedError.reason]
    (notice,) = told
    assert notice.title == CameraStreamStartFailedError.title
    assert notice.category == 'Camera'
    assert notice.solicited is False
    assert refused_at_bring_up.scope.camera_connected
    assert not refused_at_bring_up.scope.imaging.is_streaming()
    # Three stall bounds: a stall would be reported inside one.
    time.sleep(1.5)
    assert [n for n in heard if n.title == CameraStreamStalledError.title] == []


def test_the_bring_up_record_names_the_stream_that_did_not_start(refused_at_bring_up):
    (camera,) = [p for p in refused_at_bring_up.bring_up_record().parts if p.part == 'camera']

    assert camera.up
    assert camera.cause == 'stream_not_started'
    assert 'refused' in camera.detail
    assert not camera.missing


def test_the_camera_streams_when_started_again(refused_at_bring_up):
    imaging = refused_at_bring_up.scope.imaging

    imaging.enable_streaming()

    assert imaging.is_streaming()
