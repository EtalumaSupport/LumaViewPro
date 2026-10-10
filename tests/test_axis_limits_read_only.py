"""A travel limit cannot be changed by whoever reads it.

The refusal of an out-of-travel move reads the motion driver's axis
config. When a read handed out that config itself, a caller that edited
what it got back -- ``lim['max'] += 100000`` -- moved the bound the
refusal checks, and the stage was driven past its physical travel with
no error anywhere. Every door onto the limits must hand out something
that refuses the edit, loudly, at the line that tries it.
"""

from unittest.mock import patch

import pytest

from modules.exceptions import PositionOutOfRangeError
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.scope_fakes import build_scope, home_sim_scope


@pytest.fixture(scope='module')
def motion():
    """One homed simulated scope: a refused edit lands nothing, so nothing carries over."""
    scope = home_sim_scope(build_scope(simulate=True))
    yield scope.motion
    scope.disconnect()


def test_an_edit_through_the_apis_read_is_refused_and_the_bound_holds(motion):
    """The API's one door is ``get_axis_limits``; what it hands out refuses
    the edit, and the move refusal still reads the bound it had."""
    travel_max = motion.get_axis_limits('Z')['max']

    with pytest.raises(TypeError):
        motion.get_axis_limits('Z')['max'] = 114000.0

    assert motion.get_axis_limits('Z')['max'] == travel_max
    with pytest.raises(PositionOutOfRangeError):
        motion.move_absolute('Z', travel_max + 10000)


def _config_value(board):
    board.get_axes_config()['Z']['limits']['max'] = 114000.0


def _config_limits(board):
    board.get_axes_config()['Z']['limits'] = {'min': 0.0, 'max': 114000.0}


def _config_axis(board):
    board.get_axes_config()['Z'] = {'limits': {'min': 0.0, 'max': 114000.0}}


@pytest.mark.parametrize(
    'edit', [_config_value, _config_limits, _config_axis], ids=lambda f: f.__name__
)
def test_the_simulated_driver_builds_a_read_only_config(edit):
    """The simulator builds its config in its own body, as the real and the
    null drivers do; the driver's config is the other door onto the limits."""
    from drivers.simulated_motorboard import SimulatedMotorBoard

    board = SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
    travel_max = board.get_axis_limits('Z')['max']

    with pytest.raises(TypeError):
        edit(board)

    assert board.get_axis_limits('Z')['max'] == travel_max


def test_the_hardware_driver_builds_a_read_only_config():
    """The real MotorBoard builds its config in a different body from the
    simulator's; it must refuse the same edits."""
    from drivers.motorboard import MotorBoard
    from drivers.motorconfig import MotorConfig

    with patch.object(MotorBoard, '__init__', lambda self, *a, **kw: None):
        board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
    board.motorconfig = MotorConfig(SHIPPED_MOTOR_DEFAULTS)
    board._rebuild_cached_values()

    with pytest.raises(TypeError):
        board.get_axis_limits('Z')['max'] = 114000.0
    with pytest.raises(TypeError):
        board.get_axes_config()['Z'] = {}


def test_the_null_driver_builds_a_read_only_config():
    from drivers.null_motorboard import NullMotionBoard

    board = NullMotionBoard()

    # It has no axes and no limits; its empty config still refuses a write.
    with pytest.raises(TypeError):
        board.get_axes_config()['Z'] = {}
