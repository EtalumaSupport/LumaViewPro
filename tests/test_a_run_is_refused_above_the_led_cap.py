# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run with a step brighter than the scope's LED can go is refused at the gate.

The cap the run gate judged illumination by was a copy the protocol carried,
set when the scope built or loaded it. A copy for execution did not carry
it, nor did a protocol built any other way, and the GUI's Run and the API's
Autofocus All Steps both hand the gate a copy: a step past the board's
ceiling was admitted, and the run reached it before the LED refused it.

The gate now judges by the connected scope's cap, the same number the
illumination API refuses with, whatever protocol object it is handed.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ProtocolRunRefusedError
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_a_protocol_needs_its_objectives_on_the_turret import _prepare
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)


def _refused_for_illumination(refusal: ProtocolRunRefusedError, cap: int) -> None:
    assert refusal.reason == 'validation_failed', refusal
    assert f'Illumination must be 0-{cap} mA' in str(refusal), str(refusal)


def test_a_step_above_the_scopes_cap_is_refused(executor, scope, tmp_path):
    cap = scope.capabilities.led_max_ma
    protocol = _make_multi_step_protocol([{'name': 'too_bright', 'illumination_ma': cap + 1}])
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, protocol, tmp_path)
    _refused_for_illumination(refusal.value, cap)


def test_a_step_at_the_scopes_cap_is_admitted(executor, scope, tmp_path):
    cap = scope.capabilities.led_max_ma
    protocol = _make_multi_step_protocol([{'name': 'at_cap', 'illumination_ma': cap}])
    _prepare(executor, protocol, tmp_path)


def test_the_copy_the_gui_runs_is_judged_by_the_scopes_cap(executor, scope, tmp_path):
    # The GUI's Run hands prepare() copy_for_execution() of the panel's
    # protocol, not the protocol itself.
    cap = scope.capabilities.led_max_ma
    protocol = _make_multi_step_protocol([{'name': 'too_bright', 'illumination_ma': cap + 1}])
    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, protocol.copy_for_execution(), tmp_path)
    _refused_for_illumination(refusal.value, cap)


def test_autofocus_all_steps_is_judged_by_the_scopes_cap(tmp_path):
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        cap = session.scope.capabilities.led_max_ma
        step = _step('too_bright', 0, x=20.0, gain=1.0)
        step['Illumination'] = float(cap + 1)
        with pytest.raises(ProtocolRunRefusedError) as refusal:
            runner.run_autofocus_all_steps(_protocol([step]))
    _refused_for_illumination(refusal.value, cap)
