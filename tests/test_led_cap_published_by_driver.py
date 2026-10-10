"""The LED current cap is published by the connected LED driver.

Before this, ``ScopeCapabilities.from_drivers`` took a ``led_max_ma``
parameter with a module-constant default that no production call site
passed, so every scope advertised 1000 mA -- including the Classic, whose
peripheral's full scale is 840. The API's range guard, the illumination
sliders and the protocol validator each carried their own copy of the
number. Now each LED driver answers ``max_ma()``, capabilities probes it,
and every bound reads capabilities.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from drivers import fx2driver
from drivers.ledboard import FIRMWARE_LED_CH_MAX_MA, LEDBoard
from drivers.null_ledboard import NullLEDBoard
from drivers.registry import led_registry
from drivers.simulated_ledboard import SimulatedLEDBoard
from modules import config_helpers
from modules.exceptions import ArgumentRefusedError
from modules.layer_record import UNRESOLVED
from modules.objectives_loader import ObjectiveLoader
from modules.protocol import Protocol
from modules.scope_capabilities import ScopeCapabilities
from tests.ast_seams import find_def, parse_module

# ---------------------------------------------------------------------------
# Every registered LED driver publishes a cap (the build-failing guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(led_registry.registered_names()))
def test_every_registered_led_driver_answers_max_ma(name):
    cls = led_registry.get(name)
    fn = getattr(cls, 'max_ma', None)
    assert callable(fn), f'{cls.__name__} does not publish max_ma()'
    value = fn(object.__new__(cls))
    assert isinstance(value, int) and value >= 0, f'{cls.__name__}.max_ma() -> {value!r}'


def test_the_four_drivers_publish_their_own_boards_ceiling():
    assert LEDBoard.max_ma(object.__new__(LEDBoard)) == FIRMWARE_LED_CH_MAX_MA == 1000
    assert SimulatedLEDBoard.max_ma(object.__new__(SimulatedLEDBoard)) == FIRMWARE_LED_CH_MAX_MA
    assert fx2driver.FX2LEDController.max_ma(object.__new__(fx2driver.FX2LEDController)) == 840
    assert NullLEDBoard().max_ma() == 0


# ---------------------------------------------------------------------------
# Capabilities asks the driver and has no parameter to forget
# ---------------------------------------------------------------------------


def _caps_with(led) -> ScopeCapabilities:
    motion = MagicMock()
    motion.detect_present_axes.return_value = ()
    motion.get_microscope_model.return_value = ''
    return ScopeCapabilities.from_drivers(
        motion=motion, led=led, camera=None, layer_identity=UNRESOLVED, scope_models={}
    )


def test_capabilities_reports_the_connected_drivers_cap():
    assert _caps_with(object.__new__(fx2driver.FX2LEDController)).led_max_ma == 840
    assert _caps_with(object.__new__(LEDBoard)).led_max_ma == 1000
    # No board came up: there is no cap, not a cap of 0 mA.
    assert _caps_with(NullLEDBoard()).led_max_ma is None


def test_a_driver_that_does_not_answer_leaves_no_legal_current():
    assert _caps_with(None).led_max_ma == 0


def test_from_drivers_has_no_led_cap_parameter():
    params = inspect.signature(ScopeCapabilities.from_drivers).parameters
    assert 'led_max_ma' not in params


def test_the_module_constant_is_gone():
    import modules.scope_capabilities as caps_module

    assert not hasattr(caps_module, 'LED_MAX_MA')


# ---------------------------------------------------------------------------
# The API guard reads the capability
# ---------------------------------------------------------------------------


def test_the_api_guard_refuses_one_over_the_boards_cap(sim_scope):
    sim_scope.capabilities = replace(sim_scope.capabilities, led_max_ma=840)
    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.illumination.led_on('Blue', 841)
    assert (refused.value.reason, refused.value.limits) == ('illumination_out_of_range', (0, 840))
    sim_scope.illumination.led_on('Blue', 840)
    sim_scope.illumination.led_off('Blue')


# ---------------------------------------------------------------------------
# The protocol validator judges by the cap its caller gives it
# ---------------------------------------------------------------------------


def _protocol_with(illumination) -> Protocol:
    import pandas as pd

    p = Protocol.__new__(Protocol)
    p._config = {
        'steps': pd.DataFrame(
            [
                {
                    'Name': 'A1',
                    'X': 10.0,
                    'Y': 10.0,
                    'Z': 100.0,
                    'Auto_Focus': False,
                    'Color': 'Blue',
                    'False_Color': False,
                    'Illumination': float(illumination),
                    'Gain': 0.0,
                    'Auto_Gain': False,
                    'Exposure': 10.0,
                    'Sum': 1,
                    'Objective': '4x',
                    'Tile': '',
                    'Z-Slice': 0,
                    'Well': 'A1',
                    'Tile Group ID': 0,
                    'Z-Stack Group ID': 0,
                    'Custom Step': False,
                    'Acquire': 'image',
                }
            ]
        ),
        'labware_id': '96 well microplate',
    }
    p._num_steps_cache = None
    return p


_CATALOGUE = ObjectiveLoader()


def test_a_step_above_the_cap_given_is_refused():
    errors = _protocol_with(900).validate_steps(_CATALOGUE, led_max_ma=840)
    assert any('Illumination must be 0-840 mA' in e for e in errors), errors


def test_a_step_at_the_cap_given_is_accepted():
    errors = _protocol_with(840).validate_steps(_CATALOGUE, led_max_ma=840)
    assert not any('Illumination' in e for e in errors), errors


def test_a_negative_current_is_refused_under_any_cap():
    errors = _protocol_with(-1).validate_steps(_CATALOGUE, led_max_ma=840)
    assert any('Illumination must be 0 or more' in e for e in errors), errors


def test_the_protocol_carries_no_cap_of_its_own():
    # A cap carried on the protocol was dropped by every copy made for a
    # run, and the gate admitted a step the LED then refused: the cap is
    # the scope's, and each validator asks its caller for it.
    assert not hasattr(Protocol, '_led_max_ma')
    assert 'led_max_ma' not in inspect.signature(Protocol).parameters
    assert 'led_max_ma' not in inspect.signature(Protocol.from_file).parameters
    for validator in (Protocol.validate_steps, Protocol.validate_for_run):
        param = inspect.signature(validator).parameters['led_max_ma']
        assert param.default is inspect.Parameter.empty, validator.__name__


# ---------------------------------------------------------------------------
# The UI bounds resolve below the GUI
# ---------------------------------------------------------------------------


def _fake_caps(cap):
    return SimpleNamespace(led_max_ma=cap)


def test_slider_bound_is_the_cap_narrowed_by_transmitted_policy():
    assert config_helpers.layer_max_illumination_ma_for_ui(_fake_caps(840), 'BF') == 50
    assert config_helpers.layer_max_illumination_ma_for_ui(_fake_caps(840), 'PC') == 50
    assert config_helpers.layer_max_illumination_ma_for_ui(_fake_caps(840), 'Blue') == 840
    assert config_helpers.layer_max_illumination_ma_for_ui(_fake_caps(1000), 'Red') == 1000


def _calls_in(fn_node) -> set[str]:
    """The dotted callee names inside one function body."""
    names = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            names.add(ast.unparse(node.func))
    return names


def _ill_slider_max_writes_in(fn_node) -> int:
    """Assignments whose target ends in ``.max`` on an ``ill_slider`` lookup."""
    hits = 0
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == 'max'
                    and 'ill_slider' in ast.unparse(target.value)
                ):
                    hits += 1
    return hits


def test_the_capability_grouping_owns_the_slider_bound():
    grouping = find_def('ui/image_settings.py', 'sync_camera_capability_ranges', 'ImageSettings')
    assert grouping is not None
    assert 'self.set_layer_illumination_ranges' in _calls_in(grouping)

    setter = find_def('ui/image_settings.py', 'set_layer_illumination_ranges', 'ImageSettings')
    assert setter is not None
    assert _ill_slider_max_writes_in(setter) == 1
    assert 'get_layer_illumination_slider_max' in _calls_in(setter)


def test_nothing_else_in_the_panel_writes_the_slider_bound():
    tree = parse_module('ui/image_settings.py')
    writers = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _ill_slider_max_writes_in(node)
    ]
    assert writers == ['set_layer_illumination_ranges'], writers


def test_the_text_box_carries_no_bound_of_its_own():
    # The board's maximum is the writer's refusal (ScopeSession.update_settings
    # hands check_write capabilities.led_max_ma); the box neither clips to a
    # UI constant nor resolves a ceiling of its own.
    tree = parse_module('ui/layer_control.py')
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert 'BF_MAX_ILLUMINATION' not in names
    assert 'get_layer_illumination_text_max' not in names
    ill_text = find_def('ui/layer_control.py', 'ill_text', 'LayerControl')
    assert ill_text is not None
    assert 'get_layer_illumination_text_max' not in _calls_in(ill_text)
