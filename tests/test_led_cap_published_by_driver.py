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
    return ScopeCapabilities.from_drivers(motion=motion, led=led, camera=None)


def test_capabilities_reports_the_connected_drivers_cap():
    assert _caps_with(object.__new__(fx2driver.FX2LEDController)).led_max_ma == 840
    assert _caps_with(object.__new__(LEDBoard)).led_max_ma == 1000
    assert _caps_with(NullLEDBoard()).led_max_ma == 0


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
    with pytest.raises(ValueError, match='0-840 mA'):
        sim_scope.illumination.led_on('Blue', 841)
    sim_scope.illumination.led_on('Blue', 840)
    sim_scope.illumination.led_off('Blue')


# ---------------------------------------------------------------------------
# The Protocol carries the cap it was built under
# ---------------------------------------------------------------------------


def _protocol_with(led_max_ma, illumination) -> Protocol:
    import pandas as pd

    p = Protocol.__new__(Protocol)
    if led_max_ma is not None:
        p._led_max_ma = led_max_ma
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
    p._objective_loader = SimpleNamespace(get_objective_info=lambda **kw: {})
    return p


def test_a_protocol_built_under_a_cap_refuses_a_step_above_it():
    errors = _protocol_with(840, 900).validate_steps()
    assert any('Illumination must be 0-840 mA' in e for e in errors), errors


def test_a_protocol_built_under_a_cap_accepts_a_step_at_it():
    errors = _protocol_with(840, 840).validate_steps()
    assert not any('Illumination' in e for e in errors), errors


def test_a_protocol_with_no_authority_checks_format_only():
    errors = _protocol_with(None, 5000).validate_steps()
    assert not any('Illumination' in e for e in errors), errors
    errors = _protocol_with(None, -1).validate_steps()
    assert any('Illumination must be 0 or more' in e for e in errors), errors


def test_the_loaders_pass_the_cap_through():
    src = inspect.getsource(Protocol.from_config)
    assert 'led_max_ma=capabilities.led_max_ma' in src
    assert 'led_max_ma' in inspect.signature(Protocol.from_file).parameters


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


def test_text_bound_lets_bf_alone_exceed_its_slider():
    assert config_helpers.layer_illumination_text_max_for_ui(_fake_caps(840), 'BF') == 500
    assert config_helpers.layer_illumination_text_max_for_ui(_fake_caps(840), 'PC') == 50
    assert config_helpers.layer_illumination_text_max_for_ui(_fake_caps(840), 'Blue') == 840
    # Never above what the board can be asked for.
    assert config_helpers.layer_illumination_text_max_for_ui(_fake_caps(200), 'BF') == 200


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


def test_the_bf_text_bound_comes_from_the_getter_not_a_ui_constant():
    tree = parse_module('ui/layer_control.py')
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert 'BF_MAX_ILLUMINATION' not in names
    ill_text = find_def('ui/layer_control.py', 'ill_text', 'LayerControl')
    assert ill_text is not None
    assert 'get_layer_illumination_text_max' in _calls_in(ill_text)
