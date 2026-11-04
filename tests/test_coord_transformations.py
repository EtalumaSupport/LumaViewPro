# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""stage_to_plate and plate_to_stage are one symmetric pair, on the selected plate or a named one."""

import pytest

from modules.coord_transformations import StageOffset
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

STAGE = (50_000.0, 30_000.0)


@pytest.fixture(scope='module')
def runtime_state(tmp_path_factory):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    try:
        yield session.scope.runtime_state
    finally:
        session.shutdown()
        session.scope.disconnect()


def _round_trip(runtime_state, **frame):
    plate = runtime_state.stage_to_plate(*STAGE, **frame)
    stage = runtime_state.plate_to_stage(plate.x_mm, plate.y_mm, **frame)
    return plate, (stage.x_um, stage.y_um)


def test_a_stage_position_round_trips_on_the_selected_plate(runtime_state):
    _, stage = _round_trip(runtime_state)
    assert stage == pytest.approx(STAGE, abs=1e-6)


def test_a_stage_position_round_trips_on_another_plate_and_offset(runtime_state):
    """A protocol's own plate and a run's offset are a frame of their own:
    the pair is stated in it, not in the selection."""
    other = '6 well microplate'
    assert runtime_state.get_labware().name != other
    offset = StageOffset(x_um=1234.0, y_um=-567.0)

    plate, stage = _round_trip(runtime_state, labware_id=other, stage_offset=offset)

    assert stage == pytest.approx(STAGE, abs=1e-6)
    assert plate != runtime_state.stage_to_plate(*STAGE)
