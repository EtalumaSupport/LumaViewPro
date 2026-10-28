# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame's record and a run's plate, for tests that hand a frame to a save.

A save now records the frame it is given and reads nothing live, so a test
that saves or queues a frame states what the frame was taken under. One
description here, so a field added to the record is added once.
"""

import datetime

from modules.labware_loader import WellPlateLoader
from modules.lumascope_api.frame_record import (
    FrameRecord,
    FrameTime,
    LightTiming,
    SettingSource,
    TimestampSource,
)
from modules.recording_frames import FrameFact


def frame_record(**overrides) -> FrameRecord:
    """A plausible single-frame still's record with no LED lit; override any
    field by name. A ``captured_at`` given as a datetime is a still's time."""
    fields = {
        'captured_at': FrameTime(
            at=datetime.datetime(2026, 9, 29, 12, 0, 0), source=TimestampSource.HOST_WALLCLOCK
        ),
        'exposure_ms': 10.0,
        'exposure_source': SettingSource.FRAME,
        'gain_db': 1.0,
        'gain_source': SettingSource.FRAME,
        'black_level': None,
        'illumination_ma': {},
        'illumination_timing': LightTiming.BEFORE_GRAB,
        'frames_summed': 1,
        'frame_significant_bits': 8,
        'camera_timestamp_ticks': None,
        'camera_tick_hz': None,
        'frame_id': None,
        'binning_size': 1,
        'camera_model': None,
        'clock_mapping': None,
        'judgments': (),
        'saturated_fraction': None,
        'dark': None,
        'held_channel': None,
        'auto_gain': None,
    }
    fields.update(overrides)
    if isinstance(fields['captured_at'], datetime.datetime):
        fields['captured_at'] = FrameTime(
            at=fields['captured_at'], source=TimestampSource.HOST_WALLCLOCK
        )
    return FrameRecord(**fields)


def plate(key: str = '96 well microplate'):
    """A catalogue plate, as a run or a still is handed its plate."""
    return WellPlateLoader().get_plate(key)


def unpositioned() -> FrameFact:
    """A still's position fact where the scope knew no position: it states none."""
    return FrameFact(plate_x_mm=None, plate_y_mm=None, z_um=None, moving=False)


def recorded_record(**overrides) -> FrameRecord:
    """A recorded frame's record, as ``delivered_frame_record`` states one:
    its time the host's arrival, its LEDs read on arrival, never judged and
    no black level; override any field by name."""
    fields = {
        'captured_at': FrameTime(
            at=datetime.datetime(2026, 9, 29, 12, 0, 0), source=TimestampSource.HOST_ARRIVAL
        ),
        'illumination_timing': LightTiming.ON_ARRIVAL,
        'judgments': None,
        **overrides,
    }
    return frame_record(**fields)


def write_recorded(
    scope,
    image,
    path,
    *,
    channel='BF',
    save_encoding='8bit',
    false_color_on=False,
    record=None,
    fact=None,
    frame_number=0,
    objective_id='10x Oly',
):
    """Write ``image`` as a recorded frame through the one writer
    (``image_save.write_recorded_frame``), on ``scope``'s catalogue and
    identity; a uint16 frame is 12-bit unless ``record`` says otherwise."""
    from modules.image_save import write_recorded_frame
    from modules.layer_record import release_colors

    if record is None:
        record = recorded_record(frame_significant_bits=8 if image.dtype.itemsize == 1 else 12)
    fact = fact or unpositioned()
    labware = plate()
    write_recorded_frame(
        scope,
        image,
        path,
        channel=channel,
        plate_x_mm=fact.plate_x_mm,
        plate_y_mm=fact.plate_y_mm,
        stage_z_um=fact.z_um,
        stage_moving=fact.moving,
        frame_number=frame_number,
        objective_id=objective_id,
        frame_record=record,
        labware=labware,
        well_label=fact.well_label(labware),
        fallback_focus=None,
        false_color_on=false_color_on,
        channel_colors=release_colors(),
        save_encoding=save_encoding,
    )
