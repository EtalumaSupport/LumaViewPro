# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame's record and a run's plate, for tests that hand a frame to a save.

A save now records the frame it is given and reads nothing live, so a test
that saves or queues a frame states what the frame was taken under. One
description here, so a field added to the record is added once.
"""

import datetime

from modules.labware_loader import WellPlateLoader
from modules.lumascope_api.frame_record import FrameRecord


def frame_record(**overrides) -> FrameRecord:
    """A plausible single-frame record with no LED lit; override any field by name."""
    fields = {
        'captured_at': datetime.datetime(2026, 9, 29, 12, 0, 0),
        'exposure_ms': 10.0,
        'gain_db': 1.0,
        'illumination_ma': {},
        'frames_summed': 1,
        'camera_timestamp_ticks': None,
        'camera_tick_hz': None,
        'frame_id': None,
        'binning_size': 1,
        'camera_model': None,
    }
    fields.update(overrides)
    return FrameRecord(**fields)


def plate(key: str = '96 well microplate'):
    """A catalogue plate, as a run or a still is handed its plate."""
    return WellPlateLoader().get_plate(key)
