# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The metadata builder records the frame it is handed, and reads nothing live.

A file is written after its frame is taken, sometimes long after. Anything
the builder reads from the scope at that moment describes the scope at the
write: the next step's gain, the LED after the run turned it off, the well
the stage moved on to, the write's own clock. Every such fact now travels
with the frame. This guard makes a new live read fail here rather than in a
bench file months later: the camera, the LED, the stage and the plate
selection are unreachable while the builder runs, and the clock is not
consulted.
"""

import datetime
import inspect

import pytest

import modules.image_save as image_save
from modules.labware_loader import WellPlateLoader
from modules.lumascope_api.frame_record import FrameRecord


class _Unreachable:
    """Stands in for a part of the scope the builder must not touch."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise AssertionError(
            f'the metadata builder read scope.{self._name}.{attr} at write time; '
            'that fact must travel with the frame'
        )


class _ObjectiveCatalogueOnly:
    """The runtime state, with only the objective catalogue reachable."""

    def __init__(self, runtime_state):
        self._runtime_state = runtime_state

    def get_objective_info(self, objective_id):
        return self._runtime_state.get_objective_info(objective_id)

    def __getattr__(self, attr):
        raise AssertionError(
            f'the metadata builder read scope.runtime_state.{attr} at write time; '
            'the plate, well and stage position travel with the frame'
        )


def _record():
    return FrameRecord(
        captured_at=datetime.datetime(2026, 9, 29, 12, 0, 0, 123456),
        exposure_ms=12.5,
        gain_db=3.0,
        illumination_ma={'BF': 40.0},
        frames_summed=2,
        camera_timestamp_ticks=1000,
        camera_tick_hz=1_000_000_000,
        frame_id=7,
        binning_size=1,
        camera_model='simulated',
    )


def test_the_builder_records_its_frame_with_the_live_scope_unreachable(sim_scope, monkeypatch):
    objective_id = '10x Oly'
    runtime_state = sim_scope.runtime_state
    monkeypatch.setattr(sim_scope, 'runtime_state', _ObjectiveCatalogueOnly(runtime_state))
    for part in ('imaging', 'illumination', 'motion', '_camera_driver'):
        monkeypatch.setattr(sim_scope, part, _Unreachable(part))

    metadata = image_save.generate_image_metadata(
        sim_scope,
        channel='BF',
        plate_x_mm=12.0,
        plate_y_mm=8.0,
        stage_z_um=4321.0,
        objective_id=objective_id,
        frame_record=_record(),
        labware=WellPlateLoader().get_plate('6 well microplate'),
        well_label='A1',
    )

    assert metadata['exposure_time_ms'] == 12.5
    assert metadata['gain_db'] == 3.0
    assert metadata['illumination_ma'] == 40.0
    assert metadata['frames_summed'] == 2
    assert metadata['well_label'] == 'A1'
    assert metadata['timestamp_iso'] == '2026-09-29T12:00:00.123456'
    assert metadata['frame_id'] == 7
    assert metadata['instrument']['camera_model'] == 'simulated'


def test_the_builder_does_not_consult_the_clock():
    source = inspect.getsource(image_save.generate_image_metadata)
    clocks = ('.now(', '.utcnow(', 'time.time(', 'time.monotonic(', 'perf_counter(')
    assert not [c for c in clocks if c in source], (
        'the metadata builder reads a clock; a file is stamped with the time its '
        'frame was grabbed, which the frame record carries'
    )


def test_a_frame_whose_led_was_off_records_no_current(sim_scope):
    record = FrameRecord(**{**_record().__dict__, 'illumination_ma': {}})

    metadata = image_save.generate_image_metadata(
        sim_scope,
        channel='BF',
        plate_x_mm=None,
        plate_y_mm=None,
        stage_z_um=None,
        objective_id='10x Oly',
        frame_record=record,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        well_label=None,
    )

    assert 'illumination_ma' not in metadata
    assert 'well_label' not in metadata


def test_a_record_cannot_be_changed_after_the_capture():
    record = _record()
    with pytest.raises(TypeError):
        record.illumination_ma['BF'] = 0.0
