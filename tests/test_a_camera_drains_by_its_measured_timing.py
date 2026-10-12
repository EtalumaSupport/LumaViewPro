# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera's frame-validity drains are its measured timing record's, and a broken record is refused.

Frame validity drains, after each hardware change, the frames the camera
was still making under the old state. The counts are measured per camera
and ship in data/camera_timing/. The record was read under a bare
``except Exception``: a missing or corrupt one logged a warning and the
camera drained at counts nobody measured for it. And the simulated camera,
built from the daA3840's profile, reported its own model name, found no
record, and drained at the shipped defaults instead of the dart's.
"""

from __future__ import annotations

import pytest

import drivers.simulated_camera as simulated_camera
from drivers import camera_profiles
from modules.exceptions import InstallationFileError
from tests.scope_fakes import build_scope


def test_the_simulated_camera_drains_at_the_darts_measured_counts():
    measured = camera_profiles.timing_record(camera_profiles.lookup_profile('daA3840-45um'))[
        'skip_frames'
    ]
    scope = build_scope(simulate=True, sim_model='LS850T')
    try:
        drains = scope.imaging.frame_validity.SKIP_FRAMES
        assert {source: drains[source] for source in ('led', 'gain', 'exposure')} == {
            source: measured[source] for source in ('led', 'gain', 'exposure')
        }
    finally:
        scope.disconnect()


def test_a_broken_timing_record_is_refused_at_bring_up(tmp_path, monkeypatch):
    broken = tmp_path / 'daA3840-45um.json'
    broken.write_text('{"skip_frames": ')
    monkeypatch.setattr(camera_profiles, 'timing_record_path', lambda profile: broken)
    # A stand-in by design: the simulated camera's own read of the record for
    # its frame pipeline is not the subject; the API's read is.
    monkeypatch.setattr(simulated_camera, 'simulated_exposure_delay_frames', lambda: 2)
    with pytest.raises(InstallationFileError) as refused:
        build_scope(simulate=True, sim_model='LS850T')
    assert refused.value.file_path == broken
