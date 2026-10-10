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
from tests.ast_seams import REPO_ROOT

# The shipped 96-well geometry and a stock LS850T's travel. The reachable
# plate band these produce -- X 2.26-122.26, Y 1.48-81.48 -- is the
# quantity under test, so these are written out rather than read back: a
# defaults change should fail this file loudly, not silently retarget it.
PLATE_DIMENSIONS = {'x': 127.76, 'y': 85.48}
STAGE_OFFSET = {'x': 5500.0, 'y': 4000.0}
AXIS_TRAVEL = {'X': {'min': 0.0, 'max': 120000.0}, 'Y': {'min': 0.0, 'max': 80000.0}}

REACHABLE = {'X': (2.26, 122.26), 'Y': (1.48, 81.48)}


@pytest.fixture(scope='module')
def scope(sim_turreted_session):
    """The simulated LS850T with the template's 96-well plate and stage offset."""
    return sim_turreted_session.scope


class TestTheScopeCarriesTheGeometryTheBandsAssume:
    def test_the_plate_the_offset_and_the_travel_are_the_written_ones(self, scope):
        """The band constants below are derived from these three; a change
        to any of them fails here, by name, before a band case fails by number."""
        dimensions = scope.runtime_state.get_labware().get_dimensions()
        assert {key: dimensions[key] for key in PLATE_DIMENSIONS} == PLATE_DIMENSIONS
        assert scope.runtime_state.get_stage_offset() == STAGE_OFFSET
        assert {axis: dict(scope.motion.get_axis_limits(axis)) for axis in AXIS_TRAVEL} == (
            AXIS_TRAVEL
        )


class TestRefusalNamesWhatTheUserTyped:
    def test_out_of_plate_entry_reports_the_typed_number(self, scope):
        """The message carries 180, not the -98520.0 it converts to, and Y stays."""
        target_before = scope.motion.get_target_position('Y')

        with pytest.raises(PositionOutOfRangeError) as exc:
            scope.motion.move_absolute('Y', 180.0, frame='plate')

        assert '180' in str(exc.value)
        assert '-98520' not in str(exc.value)
        assert exc.value.bound == 'reachable range'
        assert exc.value.quantity == 'plate position'
        assert scope.motion.get_target_position('Y') == target_before

    def test_the_bound_is_reachable_not_the_labware_extent(self, scope):
        """85.0 is ON the plate (extent 85.48) and past what Y can reach.

        This is the case an extent check would let through, to be refused
        one layer down in stage microns -- the defect, one step later.
        """
        with pytest.raises(PositionOutOfRangeError) as exc:
            scope.motion.move_absolute('Y', 85.0, frame='plate')

        assert exc.value.bound == 'reachable range'
        assert (exc.value.low, exc.value.high) == REACHABLE['Y']

    def test_a_reachable_coordinate_converts(self, scope):
        """40 mm is inside the band and the stage goes to its stage um."""
        scope.motion.move_absolute('Y', 40.0, frame='plate')

        assert scope.motion.get_target_position('Y') == pytest.approx(41480.0)


class TestEquivalentToTheTravelCheck:
    """The plate bound and the travel bound reject the same inputs.

    This is what lets the protocol paths adopt the plate frame without
    changing WHICH moves they refuse. If it ever fails, those callers
    have started refusing something they used to accept, and they come
    back out of the plate frame.
    """

    @pytest.mark.parametrize('axis', ['X', 'Y'])
    def test_band_edges_agree_with_travel(self, scope, axis):
        low, high = REACHABLE[axis]
        motion = scope.motion

        for plate_mm in (low - 0.5, low, low + 0.5, high - 0.5, high, high + 0.5):
            stage_um = scope.runtime_state.plate_to_stage_axis(axis=axis, plate_mm=plate_mm)
            travel_refused = _refuses(motion.move_absolute, axis, stage_um)
            plate_refused = _refuses(motion.move_absolute, axis, plate_mm, frame='plate')

            assert plate_refused == travel_refused, (
                f'{axis} {plate_mm}mm -> {stage_um}um: plate check said '
                f'{plate_refused}, travel check says {travel_refused}'
            )


def _refuses(move, *args, **kwargs) -> bool:
    try:
        move(*args, **kwargs)
    except PositionOutOfRangeError:
        return True
    return False


class TestHatchesAndBoundaries:
    def test_ignore_limits_bypasses_the_plate_bound(self, scope):
        """Same hatch as the travel check: one bound, two units."""
        scope.motion.move_absolute('Y', 180.0, frame='plate', ignore_limits=True)

        assert scope.motion.get_target_position('Y') == pytest.approx(-98520.0)
        scope.motion.move_absolute('Y', 40000.0)

    def test_missing_stage_offset_refuses_legibly(self):
        """A scope no session bound has no offset; that must not be a TypeError."""
        from modules.exceptions import ConfigError
        from tests.scope_fakes import build_scope

        scope = build_scope(simulate=True)

        with pytest.raises(ConfigError, match='stage_offset'):
            scope.runtime_state.get_stage_offset()


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

    ``ProtocolsAPI.plate_to_stage`` converts against the labware the
    PROTOCOL stores, while the motion API's plate frame resolves labware
    from the session. Those are different stores: a run, and a person's
    navigation to a step, must image the plate the protocol was written
    for even if the operator has since selected a different one. Routing
    that caller through the plate frame silently retargets the run, which
    is a worse defect than the message it would improve. Closing it means
    one store, not one more frame argument. The run engine and the Session
    take their targets from that member and are not callers here.
    """

    SANCTIONED: ClassVar[set[str]] = {
        'modules/protocol.py',
        'modules/lumascope_api/protocols.py',
        'modules/lumascope_api/runtime_state.py',
    }

    def test_only_the_enumerators_call_the_raw_transform(self):
        import re

        # The transformer's own member, not the protocols API's of the
        # same name, which converts on the protocol's plate for its callers.
        call = re.compile(r'coordinate_transformer\.plate_to_stage\s*\(')

        offenders = []
        for path in sorted([*REPO_ROOT.glob('ui/**/*.py'), *REPO_ROOT.glob('modules/**/*.py')]):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in self.SANCTIONED:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if call.search(line):
                    offenders.append(f'{relative}:{number}')

        assert offenders == [], (
            'these call the raw plate->stage transform instead of passing the '
            "plate coordinate to the motion API with frame='plate': " + ', '.join(offenders)
        )
