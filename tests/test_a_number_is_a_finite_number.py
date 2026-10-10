# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A number a caller passes is a finite number, refused as ``not_a_number`` before any range.

Every owner of a numeric argument checked its range by comparison, and every
comparison with NaN is False, so NaN passed them all: ``set_gain_db(nan)``
reached the camera, a NaN stored gain was applied as the camera's maximum,
``move_absolute('Z', nan, ignore_limits=True)`` reached the drive and
un-homed Z, a NaN timeout ended a wait at once. Each owner now asks one
predicate first and refuses with ``ArgumentRefusedError('not_a_number')``,
naming the argument; its range refusal stays for a finite value outside the
range. A value that is not a number of the declared kind at all -- a
``bool``, a float for a whole number -- is the ``@api`` door's
(``tests/test_an_argument_of_another_type_is_refused_at_the_door.py``).
"""

import math

import numpy as np
import pytest

from modules.finite_number import is_finite_number
from modules.coord_transformations import CoordinateTransformer
from modules.exceptions import ArgumentRefusedError, RefusalCause
from modules.labware_loader import WellPlateLoader
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

NOT_FINITE = [math.nan, math.inf, -math.inf]


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    home_sim_scope(s.scope)
    s.scope._motion_driver.set_timing_mode('instant')
    yield s
    s.shutdown()


def _refused(call, argument):
    with pytest.raises(ArgumentRefusedError) as refused:
        call()
    assert refused.value.reason == 'not_a_number'
    assert refused.value.argument == argument
    assert refused.value.cause == RefusalCause.REQUEST
    return refused.value


@pytest.mark.parametrize(
    'value, finite',
    [
        (1, True),
        (2.5, True),
        (np.float64(3.0), True),
        (np.int64(2), True),
        (math.nan, False),
        (math.inf, False),
        (True, False),
        (np.bool_(True), False),
        ('3', False),
        (None, False),
    ],
)
def test_the_predicate(value, finite):
    assert is_finite_number(value) is finite


@pytest.mark.parametrize('value', NOT_FINITE)
def test_the_camera_setters_refuse_and_the_camera_keeps_its_values(session, value):
    imaging = session.scope.imaging
    before = (imaging.get_gain_db(), imaging.get_exposure_ms(), imaging.get_black_level())

    _refused(lambda: imaging.set_gain_db(value), 'gain_db')
    _refused(lambda: imaging.set_exposure_ms(value), 'exposure_ms')
    _refused(lambda: imaging.set_black_level(value), 'black_level')
    _refused(lambda: imaging.update_auto_gain_target_brightness(value), 'target_brightness')
    assert (imaging.get_gain_db(), imaging.get_exposure_ms(), imaging.get_black_level()) == before


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_stored_camera_value_is_refused_not_applied_at_the_maximum(session, value):
    imaging = session.scope.imaging
    _refused(lambda: imaging.applied_gain_db_for(value), 'stored_gain_db')
    _refused(lambda: imaging.applied_exposure_ms_for(value), 'stored_exposure_ms')


@pytest.mark.parametrize('value', NOT_FINITE)
def test_an_led_current_is_refused_and_nothing_lit(session, value):
    illumination = session.scope.illumination
    _refused(lambda: illumination.led_on('BF', value), 'illumination_ma')
    assert not illumination.get_led_state('BF')['enabled']


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_move_is_refused_and_z_stays_homed_even_past_the_limits(session, value):
    motion = session.scope.motion
    before = motion.get_current_position('Z')

    _refused(lambda: motion.move_absolute('Z', value, ignore_limits=True), 'position')
    _refused(lambda: motion.move_absolute('Z', value), 'position')
    _refused(lambda: motion.move_relative('Z', value), 'distance')

    assert motion.get_axis_state('Z') == AxisState.IDLE
    assert motion.get_current_position('Z') == before


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_wait_and_a_handshake_refuse_a_timeout(session, value):
    _refused(lambda: session.scope.motion.wait_until_finished_moving(value), 'timeout_s')
    _refused(lambda: session.scope.diagnostics.enter_led_engineering_mode(value), 'timeout_s')


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_camera_diagnostic_refuses_a_time(session, value):
    diagnostics = session.scope.diagnostics
    _refused(lambda: diagnostics.run_camera_bandwidth_test(2, timeout_s=value), 'timeout_s')
    _refused(
        lambda: diagnostics.run_grab_lifecycle_benchmark(1, inter_cycle_delay_ms=value),
        'inter_cycle_delay_ms',
    )
    _refused(
        lambda: diagnostics.run_grab_lifecycle_benchmark(1, slow_threshold_s=value),
        'slow_threshold_s',
    )
    _refused(lambda: diagnostics.run_pylon_diagnostic_probe(duration_s=value), 'duration_s')


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_capture_refuses_a_time(session, value):
    imaging = session.scope.imaging
    _refused(lambda: imaging.get_image(timeout_s=value), 'timeout_s')
    _refused(lambda: imaging.get_image(sum_delay_s=value), 'sum_delay_s')
    _refused(lambda: imaging.get_image(new_capture_timeout_s=value), 'new_capture_timeout_s')
    _refused(lambda: imaging.capture_and_wait(timeout_s=value), 'timeout_s')
    _refused(lambda: imaging.capture_and_wait(sum_delay_s=value), 'sum_delay_s')


@pytest.mark.parametrize('value', NOT_FINITE)
def test_plate_coordinates_and_a_well_are_refused(value):
    plate = WellPlateLoader().get_plate('96 well microplate')
    transformer = CoordinateTransformer()
    offset = {'x': 0.0, 'y': 0.0}

    _refused(lambda: transformer.plate_to_stage(plate, offset, value, 10.0), 'px')
    _refused(lambda: transformer.stage_to_plate(plate, offset, 10.0, value), 'sy')
    _refused(lambda: plate.get_well_index(value, 10.0), 'x')


@pytest.mark.parametrize('value', NOT_FINITE)
def test_a_zstack_extent_is_refused_and_the_protocol_unchanged(session, value):
    protocol = session.create_empty_protocol()
    before = protocol.steps().copy()

    _refused(
        lambda: session.apply_zstacking(
            protocol, range_um=value, step_size_um=5.0, z_reference='center'
        ),
        'z-stack range',
    )
    _refused(
        lambda: session.apply_zstacking(
            protocol, range_um=20.0, step_size_um=value, z_reference='center'
        ),
        'z-stack step_size',
    )
    assert protocol.steps().equals(before)
