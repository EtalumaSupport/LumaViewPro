# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A plate coordinate is bounded and refused in the frame the user typed.

Typing 180 into the Y position box reported "Y position -98520.0 is
outside the travel range 0.0 to 80000.0". The GUI converted the plate mm
to stage um itself, handed the API the converted number, and the travel
check -- the first layer with the authority to refuse -- could only name
the value it had been given. The transform inverts the axis, so a
coordinate past the plate becomes a target below zero.

The bound now lives at the motion API, which knows both the labware and
the travel, and it is the REACHABLE band: the plate coordinates whose
converted target lies within travel. The labware extent alone is wider
than that band, so checking the extent would pass a coordinate the stage
still cannot serve and hand the user a second refusal in the other frame
for the same mistake.
"""

from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from modules.coord_transformations import CoordinateTransformer
from modules.exceptions import PositionOutOfRangeError

# The shipped 96-well geometry and a stock LS850T's travel. The reachable
# plate band these produce -- X 2.26-122.26, Y 1.48-81.48 -- is the
# quantity under test, so these are written out rather than imported: a
# defaults change should fail this file loudly, not silently retarget it.
PLATE_DIMENSIONS = {'x': 127.76, 'y': 85.48}
STAGE_OFFSET = {'x': 5500.0, 'y': 4000.0}
AXIS_TRAVEL = {'X': {'min': 0.0, 'max': 120000.0}, 'Y': {'min': 0.0, 'max': 80000.0}}

REACHABLE = {'X': (2.26, 122.26), 'Y': (1.48, 81.48)}


@pytest.fixture
def motion():
    """A MotionAPI with just enough scope to resolve a plate coordinate."""
    from modules.lumascope_api.motion import MotionAPI

    from tests.scope_fakes import spec_scope

    labware = MagicMock()
    labware.get_dimensions.return_value = PLATE_DIMENSIONS

    scope = spec_scope()
    scope.runtime_state.get_labware.return_value = labware
    scope.runtime_state.get_stage_offset.return_value = STAGE_OFFSET

    transformer = CoordinateTransformer()

    def _plate_to_stage_axis(axis, plate_mm):
        sx, sy = transformer.plate_to_stage(
            labware=labware,
            stage_offset=STAGE_OFFSET,
            px=plate_mm if axis == 'X' else 0,
            py=plate_mm if axis == 'Y' else 0,
        )
        return sx if axis == 'X' else sy

    scope.runtime_state.plate_to_stage_axis.side_effect = _plate_to_stage_axis

    api = MotionAPI.__new__(MotionAPI)
    api._scope = scope
    api.get_axis_limits = lambda axis: AXIS_TRAVEL.get(axis)
    return api


class TestRefusalNamesWhatTheUserTyped:
    def test_out_of_plate_entry_reports_the_typed_number(self, motion):
        """The message carries 180, not the -98520.0 it converts to."""
        with pytest.raises(PositionOutOfRangeError) as exc:
            motion._plate_target_to_stage('Y', 180.0, ignore_limits=False)

        assert '180' in str(exc.value)
        assert '-98520' not in str(exc.value)
        assert exc.value.bound == 'reachable range'
        assert exc.value.quantity == 'plate position'

    def test_the_bound_is_reachable_not_the_labware_extent(self, motion):
        """85.0 is ON the plate (extent 85.48) and past what Y can reach.

        This is the case an extent check would let through, to be refused
        one layer down in stage microns -- the defect, one step later.
        """
        with pytest.raises(PositionOutOfRangeError) as exc:
            motion._plate_target_to_stage('Y', 85.0, ignore_limits=False)

        assert exc.value.bound == 'reachable range'
        assert (exc.value.low, exc.value.high) == REACHABLE['Y']

    def test_a_reachable_coordinate_converts(self, motion):
        """40 mm is inside the band and comes back as stage um."""
        assert motion._plate_target_to_stage('Y', 40.0, ignore_limits=False) == pytest.approx(
            41480.0
        )


class TestEquivalentToTheTravelCheck:
    """The plate bound and the travel bound reject the same inputs.

    This is what lets the protocol paths adopt the plate frame without
    changing WHICH moves they refuse. If it ever fails, those callers
    have started refusing something they used to accept, and they come
    back out of the plate frame.
    """

    @pytest.mark.parametrize('axis', ['X', 'Y'])
    def test_band_edges_agree_with_travel(self, motion, axis):
        low, high = REACHABLE[axis]
        travel = AXIS_TRAVEL[axis]
        transformer = CoordinateTransformer()
        labware = MagicMock()
        labware.get_dimensions.return_value = PLATE_DIMENSIONS

        for plate_mm in (low - 0.5, low, low + 0.5, high - 0.5, high, high + 0.5):
            sx, sy = transformer.plate_to_stage(
                labware=labware,
                stage_offset=STAGE_OFFSET,
                px=plate_mm if axis == 'X' else 0,
                py=plate_mm if axis == 'Y' else 0,
            )
            stage = sx if axis == 'X' else sy
            travel_would_refuse = not (travel['min'] <= stage <= travel['max'])

            plate_refused = False
            try:
                motion._plate_target_to_stage(axis, plate_mm, ignore_limits=False)
            except PositionOutOfRangeError:
                plate_refused = True

            assert plate_refused == travel_would_refuse, (
                f'{axis} {plate_mm}mm -> {stage}um: plate check said '
                f'{plate_refused}, travel check says {travel_would_refuse}'
            )


class TestHatchesAndBoundaries:
    def test_ignore_limits_bypasses_the_plate_bound(self, motion):
        """Same hatch as the travel check: one bound, two units."""
        assert motion._plate_target_to_stage('Y', 180.0, ignore_limits=True) == pytest.approx(
            -98520.0
        )

    def test_plate_frame_is_refused_for_z(self, motion):
        """Z is stored and commanded in stage um; it has no plate frame."""
        with pytest.raises(ValueError, match='X and Y'):
            motion._plate_target_to_stage('Z', 5.0, ignore_limits=False)

    def test_missing_stage_offset_refuses_legibly(self):
        """Before initialize() the offset is None; that must not be a TypeError."""
        from modules.exceptions import ConfigError
        from modules.lumascope_api.runtime_state import RuntimeState

        state = RuntimeState.__new__(RuntimeState)
        state._stage_offset = None

        with pytest.raises(ConfigError, match='stage offset'):
            state._require_stage_offset()


class TestEnumeratorsKeepTheirValue:
    def test_the_transform_still_returns_an_out_of_plate_coordinate(self):
        """Protocol validation and tile generation convert in order to TEST.

        They report every out-of-range entry at once; a raise inside the
        transform would cut that short at the first one. This locks the
        transform as pure against a future blanket raise.
        """
        transformer = CoordinateTransformer()
        labware = MagicMock()
        labware.get_dimensions.return_value = PLATE_DIMENSIONS

        _, sy = transformer.plate_to_stage(
            labware=labware, stage_offset=STAGE_OFFSET, px=0, py=180.0
        )
        assert sy == pytest.approx(-98520.0)


class TestTheConversionLeftTheCallers:
    """No GUI widget or protocol module converts plate to stage any more.

    This is the architectural half of the fix, and the half a mocked unit
    test cannot otherwise reach: the defect was not that the bound was
    wrong, it was that the caller did the conversion and the API only ever
    saw the result. A new caller that converts for itself re-creates the
    defect exactly, and would be invisible to every other test here.

    Three callers are exempt, for two different reasons.

    ``modules/protocol.py`` ENUMERATES rather than commands -- protocol
    validation and tile generation convert in order to test each
    candidate and report them all -- so it legitimately wants the value.

    ``modules/protocol_step_runner.py`` converts against the labware the
    PROTOCOL stores, while the motion API's plate frame resolves labware
    from the session. Those are different stores: a run must image the
    plate it was written for even if the operator has since selected a
    different one. Routing this caller through the plate frame silently
    retargets the run, which is a worse defect than the message it would
    improve. Closing it means one store, not one more frame argument.
    """

    SANCTIONED: ClassVar[set[str]] = {
        'modules/protocol.py',
        'modules/protocol_step_runner.py',
        'modules/lumascope_api/runtime_state.py',
    }

    def test_only_the_enumerators_call_the_raw_transform(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        call = re.compile(r'\.plate_to_stage\s*\(')

        offenders = []
        for path in sorted([*root.glob('ui/**/*.py'), *root.glob('modules/**/*.py')]):
            relative = path.relative_to(root).as_posix()
            if relative in self.SANCTIONED:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if call.search(line):
                    offenders.append(f'{relative}:{number}')

        assert offenders == [], (
            'these call the raw plate->stage transform instead of passing the '
            "plate coordinate to the motion API with frame='plate': " + ', '.join(offenders)
        )
