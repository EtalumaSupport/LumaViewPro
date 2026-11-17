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

import pytest

from _lumascope.drivers import fx2driver
from _lumascope.drivers.ledboard import FIRMWARE_LED_CH_MAX_MA, LEDBoard
from _lumascope.drivers.null_ledboard import NullLEDBoard
from _lumascope.drivers.registry import led_registry
from _lumascope.drivers.simulated_ledboard import SimulatedLEDBoard
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.scope_capabilities import ScopeCapabilities
from tests.ast_seams import find_def, parse_module
from tests.case_walk import failing_cases
from tests.scope_fakes import capabilities_of
from tests.stock_channels import LED0

# ---------------------------------------------------------------------------
# Every registered LED driver publishes a cap (the build-failing guard)
# ---------------------------------------------------------------------------


def test_every_registered_led_driver_answers_max_ma():
    def check(name):
        cls = led_registry.get(name)
        fn = getattr(cls, 'max_ma', None)
        assert callable(fn), f'{cls.__name__} does not publish max_ma()'
        value = fn(object.__new__(cls))
        assert isinstance(value, int) and value >= 0, f'{cls.__name__}.max_ma() -> {value!r}'

    failed = failing_cases(sorted(led_registry.registered_names()), check)
    assert not failed, failed


def test_the_four_drivers_publish_their_own_boards_ceiling():
    assert LEDBoard.max_ma(object.__new__(LEDBoard)) == FIRMWARE_LED_CH_MAX_MA == 1000
    assert SimulatedLEDBoard.max_ma(object.__new__(SimulatedLEDBoard)) == FIRMWARE_LED_CH_MAX_MA
    assert fx2driver.FX2LEDController.max_ma(object.__new__(fx2driver.FX2LEDController)) == 840
    assert NullLEDBoard().max_ma() == 0


# ---------------------------------------------------------------------------
# Capabilities asks the driver and has no parameter to forget
# ---------------------------------------------------------------------------


def _caps_with(led) -> ScopeCapabilities:
    return capabilities_of(led=led)


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
    import _lumascope.modules.scope_capabilities as caps_module

    assert not hasattr(caps_module, 'LED_MAX_MA')


# ---------------------------------------------------------------------------
# The API guard reads the capability
# ---------------------------------------------------------------------------


def test_the_api_guard_refuses_one_over_the_boards_cap(sim_scope):
    sim_scope.capabilities = replace(sim_scope.capabilities, led_max_ma=840)
    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.illumination.led_on(LED0, 841)
    assert (refused.value.reason, refused.value.limits) == ('illumination_out_of_range', (0, 840))
    sim_scope.illumination.led_on(LED0, 840)
    sim_scope.illumination.led_off(LED0)


# ---------------------------------------------------------------------------
# The slider bound is the session's answer, shown by the GUI
# ---------------------------------------------------------------------------


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
    assert 'self.set_channel_illumination_ranges' in _calls_in(grouping)

    setter = find_def('ui/image_settings.py', 'set_channel_illumination_ranges', 'ImageSettings')
    assert setter is not None
    assert _ill_slider_max_writes_in(setter) == 1
    assert 'session.channel_slider_maxima' in _calls_in(setter)


def test_nothing_else_in_the_panel_writes_the_slider_bound():
    tree = parse_module('ui/image_settings.py')
    writers = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _ill_slider_max_writes_in(node)
    ]
    assert writers == ['set_channel_illumination_ranges'], writers


def test_the_text_box_carries_no_bound_of_its_own():
    # The board's maximum is the writer's refusal (ScopeSession.update_settings
    # hands check_write capabilities.led_max_ma); the box neither clips to a
    # UI constant nor resolves a ceiling of its own.
    tree = parse_module('ui/channel_control.py')
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert 'BF_MAX_ILLUMINATION' not in names
    assert 'get_layer_illumination_text_max' not in names
    ill_text = find_def('ui/channel_control.py', 'ill_text', 'ChannelControl')
    assert ill_text is not None
    assert 'get_layer_illumination_text_max' not in _calls_in(ill_text)
