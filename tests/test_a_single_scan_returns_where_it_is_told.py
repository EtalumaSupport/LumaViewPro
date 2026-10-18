# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A single scan given a ``PlateLocation`` returns the stage there once it ends.

``start_single_scan``'s ``return_to_position`` is stated on the protocol's
plate, X and Y in mm and Z in um; the run's cleanup sends each axis to it.
An axis given None is left where the scan left it.

Driven end to end on the simulated scope: a real session, runner, run loop
and cleanup.
"""

from __future__ import annotations

import pytest

from modules.coord_transformations import PlateLocation
from tests.test_a_late_write_records_its_frame import _protocol
from tests.test_a_run_reports_what_it_captured import _two_steps
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60
# The two steps image at X 20 and 30 mm, Y 20 mm, Z 5000 um; the return is
# somewhere none of them is.
RETURN_TO = PlateLocation(x_mm=40.0, y_mm=30.0, z_um=4000.0)


def _scan_returning_to(tmp_path, location):
    """The stage's targets once a single scan given ``location`` has ended, and the steps' Z."""
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        protocol = _protocol(_two_steps())
        outcome = runner.start_single_scan(
            protocol=protocol,
            parent_dir=str(tmp_path / 'runs'),
            enable_image_saving=False,
            return_to_position=location,
        ).result(timeout=WAIT_S)
        assert outcome.status == 'completed', outcome
        motion = session.scope.motion
        stage = session.scope.runtime_state.plate_to_stage(
            RETURN_TO.x_mm, RETURN_TO.y_mm, labware_id=protocol.labware()
        )
        return motion.get_target_positions(), stage, protocol.step(1)['Z']


def test_a_scan_returns_to_the_location_it_was_given(tmp_path):
    targets, stage, _ = _scan_returning_to(tmp_path, RETURN_TO)

    assert targets['X'] == pytest.approx(stage.x_um, abs=0.01)
    assert targets['Y'] == pytest.approx(stage.y_um, abs=0.01)
    assert targets['Z'] == pytest.approx(RETURN_TO.z_um, abs=0.01)


def test_an_axis_given_none_stays_where_the_scan_left_it(tmp_path):
    location = PlateLocation(x_mm=RETURN_TO.x_mm, y_mm=RETURN_TO.y_mm, z_um=None)

    targets, stage, last_step_z = _scan_returning_to(tmp_path, location)

    assert targets['X'] == pytest.approx(stage.x_um, abs=0.01)
    assert targets['Z'] == pytest.approx(last_step_z, abs=0.01)
