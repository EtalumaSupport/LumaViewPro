"""A travel limit cannot be changed by whoever reads it.

The refusal of an out-of-travel move reads the motion driver's axis
config. When a read handed out that config itself, a caller that edited
what it got back -- ``lim['max'] += 100000`` -- moved the bound the
refusal checks, and the stage was driven past its physical travel with
no error anywhere. Every door onto the limits must hand out something
that refuses the edit, loudly, at the line that tries it.
"""

import pathlib
from unittest.mock import patch

import pytest

from modules.exceptions import PositionOutOfRangeError
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session():
    s = ScopeSession.create(complete_settings(), simulate=True)
    home_sim_scope(s.scope)
    s.scope._motion_driver.set_timing_mode('instant')
    yield s
    s.shutdown()


def _edits(motion):
    """Every way a caller could try to widen Z's travel through a read."""

    def limits_value():
        motion.get_axis_limits('Z')['max'] = 114000.0

    def config_value():
        motion.get_axes_config()['Z']['limits']['max'] = 114000.0

    def config_limits():
        motion.get_axes_config()['Z']['limits'] = {'min': 0.0, 'max': 114000.0}

    def config_axis():
        motion.get_axes_config()['Z'] = {'limits': {'min': 0.0, 'max': 114000.0}}

    return [limits_value, config_value, config_limits, config_axis]


@pytest.mark.parametrize('which', range(4))
def test_an_edit_through_a_read_is_refused_and_the_bound_holds(session, which):
    motion = session.scope.motion
    travel_max = motion.get_axis_limits('Z')['max']

    with pytest.raises(TypeError):
        _edits(motion)[which]()

    assert motion.get_axis_limits('Z')['max'] == travel_max
    with pytest.raises(PositionOutOfRangeError):
        motion.move_absolute('Z', travel_max + 10000, wait_until_complete=True)


def test_the_hardware_driver_builds_a_read_only_config():
    """The real MotorBoard builds its config in a different body from the
    simulator's; it must refuse the same edits."""
    from drivers.motorboard import MotorBoard
    from drivers.motorconfig import MotorConfig

    with patch.object(MotorBoard, '__init__', lambda self, *a, **kw: None):
        board = MotorBoard()
    board.motorconfig = MotorConfig(defaults_file=pathlib.Path('data/motorconfig_defaults.json'))
    board._rebuild_cached_values()

    with pytest.raises(TypeError):
        board.get_axis_limits('Z')['max'] = 114000.0
    with pytest.raises(TypeError):
        board.get_axes_config()['Z'] = {}


def test_the_null_driver_builds_a_read_only_config():
    from drivers.null_motorboard import NullMotionBoard

    board = NullMotionBoard()

    with pytest.raises(TypeError):
        board.get_axis_limits('Z')['max'] = 114000.0
    with pytest.raises(TypeError):
        board.get_axes_config()['Z'] = {}
