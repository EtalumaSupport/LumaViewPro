# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Turning the camera's auto-gain loop off ends the settle its arm recorded.

An arm records an ``auto_gain`` change a capture waits on for the loop to
settle. Once the loop is off there is nothing left to settle: a capture
waits only for the off's own ``gain`` change, and the frame's record says
the arm's change was met by its end, not by its count. Every caller that
turns the loop off (a protocol step's end, a run taking over a standing
arm, a video step's prologue, REST, the GUI toggle) does it through
``CameraAPI.set_auto_gain``, pinned here; a refused arm, whose loop the API
turns off itself, ends it the same way. An off the camera refused leaves
the loop running, and its settle stands.

Driven on the simulated scope through the camera and imaging members.
"""

import pytest

from _lumascope.modules import image_save, image_utils
from _lumascope.modules.channel_record import release_color
from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules.frame_validity import FrameValiditySource
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.lumascope_api.camera import AutoGainSettings, CameraAutoMode

AUTO_GAIN = AutoGainSettings(
    target_brightness=0.5,
    min_gain_db=0.0,
    max_gain_db=20.0,
    max_exposure_ms=100.0,
    min_exposure_ms=0.1,
)
# Every other source: whether the arm's change is in force is asked alone.
NOT_THE_ARM = tuple(s for s in FrameValiditySource if s is not FrameValiditySource.AUTO_GAIN)


def _judgment(record, source: str):
    return next(j for j in record.judgments if j.change.source == source)


def test_once_the_loop_is_off_a_capture_waits_only_for_the_off(sim_scope):
    camera = sim_scope.camera
    camera.set_auto_gain(CameraAutoMode.CONTINUOUS, AUTO_GAIN)
    assert not sim_scope.imaging.changes_in_force(exclude_sources=NOT_THE_ARM), (
        'precondition: the arm records a settle'
    )

    camera.set_auto_gain(CameraAutoMode.OFF)

    assert sim_scope.imaging.changes_in_force(exclude_sources=NOT_THE_ARM), (
        "the arm's settle outlived the loop"
    )
    record = sim_scope.imaging.capture_still(accept_dark=True).record
    arm = _judgment(record, 'auto_gain')
    assert arm.counts
    assert 'ended_after' in arm.fields(), 'the record does not say the off met it'
    assert len(arm.change.credits) < arm.change.rule.skip, 'the capture waited the arm out'


def test_the_saved_file_says_the_arms_change_ended(sim_scope, tmp_path):
    """The end reads back from the file as the record states it."""
    sim_scope.camera.set_auto_gain(CameraAutoMode.CONTINUOUS, AUTO_GAIN)
    sim_scope.camera.set_auto_gain(CameraAutoMode.OFF)
    still = sim_scope.imaging.capture_still(accept_dark=True, force_to_8bit=True)
    metadata = image_save.generate_image_metadata(
        sim_scope,
        channel='BF',
        plate_x_mm=12.0,
        plate_y_mm=8.0,
        stage_z_um=4321.0,
        objective_id='10x Oly',
        frame_record=still.record,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        well_label='A1',
        fallback_focus=None,
    )
    path = tmp_path / 'ended.tif'
    image_utils.write_tiff(
        data=still.image,
        file_loc=path,
        metadata=metadata,
        ome=False,
        color=release_color('BF'),
        significant_bits=8,
        save_encoding='8bit',
        white_bits=None,
    )

    saved = {j['source']: j for j in image_utils.read_file_record(path)['judgments']}

    assert saved['auto_gain'] == _judgment(still.record, 'auto_gain').fields()
    assert 'ended_after' in saved['auto_gain']


def test_a_refused_arm_leaves_no_settle_owed(sim_scope):
    """The API turns the loop off after a refused arm; the settle the arm
    recorded ends with it."""
    sim_scope.simulation.refuse_camera_write('auto_gain', 'raise')

    with pytest.raises(CameraSettingRejected):
        sim_scope.camera.set_auto_gain(CameraAutoMode.CONTINUOUS, AUTO_GAIN)

    assert sim_scope.camera.auto_gain == CameraAutoMode.OFF, 'precondition: the loop is off'
    assert sim_scope.imaging.changes_in_force(exclude_sources=NOT_THE_ARM), (
        "a refused arm's settle outlived the loop"
    )


def test_an_off_the_camera_refused_keeps_the_settle_of_the_loop_still_running(sim_scope):
    """Green before the fix as well: it pins that the end follows an off
    the camera took, never the asking."""
    camera = sim_scope.camera
    camera.set_auto_gain(CameraAutoMode.CONTINUOUS, AUTO_GAIN)
    sim_scope.simulation.refuse_camera_write('auto_gain', 'raise')

    with pytest.raises(CameraSettingRejected):
        camera.set_auto_gain(CameraAutoMode.OFF)

    assert camera.auto_gain == CameraAutoMode.CONTINUOUS, 'precondition: the loop still runs'
    assert not sim_scope.imaging.changes_in_force(exclude_sources=NOT_THE_ARM)
