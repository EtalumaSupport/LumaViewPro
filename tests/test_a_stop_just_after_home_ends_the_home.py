# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Stop pressed just after Home ends that home, on every motor board.

The API's home reads the limit switches before it asks the driver to
home, and the driver forgets a stop that came before its home began: the
LS720's 6110 clears its abort at the start of its home, the EL-0940's
firmware takes no STOP before its HOME. A Stop in that window was lost and
the whole home ran -- about 90 s on an LS720. The API asks its own stop
generation immediately before the driver's home, so the home ends stopped
with nothing moved.
"""

from __future__ import annotations

import pytest

from modules.exceptions import HomingFailedError
from tests.scope_fakes import bind_settings_like_a_session, build_scope, record_turret_answer


@pytest.fixture(params=['LS720', 'LS850', 'LS850T'])
def homed(request):
    s = build_scope(simulate=True, sim_model=request.param, source_path='.', register_atexit=False)
    record_turret_answer(s)
    bind_settings_like_a_session(s, objective_id='10x Oly', stage_offset={'x': 5500.0, 'y': 4000.0})
    s.motion.home()
    s.motion.move_absolute('X', 30_000)
    s.motion.move_absolute('Z', 2_000)
    yield s
    s.disconnect()


def _press_stop_during_the_apis_switch_read(scope, monkeypatch):
    """The person's Stop lands while the API reads the switches before the home."""
    driver = scope._motion_driver
    read = driver.limit_switch_status
    pressed = []

    def read_then_stop(*args, **kwargs):
        answer = read(*args, **kwargs)
        if not pressed:
            pressed.append(True)
            scope.motion.stop_motion()
        return answer

    monkeypatch.setattr(driver, 'limit_switch_status', read_then_stop)


@pytest.mark.parametrize('axis', ['ALL', 'Z', 'T'])
def test_a_stop_before_the_driver_homes_ends_the_home_with_nothing_moved(homed, monkeypatch, axis):
    if axis != 'ALL' and axis not in homed.capabilities.axes:
        pytest.skip(f'this scope has no {axis}')
    before = homed.motion.get_current_position()
    _press_stop_during_the_apis_switch_read(homed, monkeypatch)

    with pytest.raises(HomingFailedError) as ended:
        homed.motion.home(axis)

    assert ended.value.reason == 'stopped'
    assert ended.value.axes == ()
    assert 'before it began' in str(ended.value)
    assert homed.motion.axes_without_position() == {}
    assert homed.motion.get_current_position() == before
