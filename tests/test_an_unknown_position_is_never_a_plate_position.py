# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position the scope does not know is never converted, and never saved by New.

Before a home, ``ScopeSession.get_current_plate_position`` converted the
cached numbers of axes with no reference into plate coordinates -- on the
simulator ``{'x': 122.26, 'y': 81.48, 'z': 0.0}`` -- and ``new_protocol``
stamped that Z into every acquiring layer with no saved focus: a saved
position nobody read. Over REST each was a 200 a client could not tell from
a real answer.

The public plate read now refuses an unknown position, and New refuses an
unknown Z only when a step would save it; a scope with no Z motor builds as
it did.
"""

from __future__ import annotations

import pytest

from modules.exceptions import AxisStateUnknownError, HardwareCommandRefusedError
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model, **kwargs):
        session = _session(tmp_path, model, **kwargs)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def _focus(session, layer, z):
    with session.settings_lock:
        session.settings[layer]['focus'] = z


# --- The plate read ------------------------------------------------------------------


def test_before_a_home_the_plate_position_is_refused(make_session):
    session = make_session('LS850', homed=False)

    with pytest.raises(AxisStateUnknownError) as exc:
        session.get_current_plate_position()

    assert set(exc.value.axes) == {'X', 'Y', 'Z'}


def test_after_a_home_the_plate_position_is_read(make_session):
    session = make_session('LS850')

    position = session.get_current_plate_position()

    assert set(position) == {'x', 'y', 'z'}


def test_a_pulled_cable_is_the_cable_not_an_unknown_position(make_session, monkeypatch):
    session = make_session('LS850')
    monkeypatch.setattr(session.scope._motion_driver, 'is_connected', lambda: False)

    with pytest.raises(HardwareCommandRefusedError) as exc:
        session.get_current_plate_position()

    assert exc.value.reason == 'not_connected'


# --- New ------------------------------------------------------------------------------


def test_before_a_home_the_config_carries_no_z(make_session):
    session = make_session('LS850', homed=False, acquiring=('BF',))

    assert session.get_sequenced_capture_config()['current_z'] is None


def test_new_with_an_unfocused_layer_before_a_home_is_refused(make_session):
    session = make_session('LS850', homed=False, acquiring=('BF',))
    _focus(session, 'BF', None)

    with pytest.raises(AxisStateUnknownError) as exc:
        session.new_protocol()

    assert set(exc.value.axes) == {'Z'}


def test_new_with_every_layer_focused_builds_before_a_home(make_session):
    session = make_session('LS850', homed=False, acquiring=('BF',))
    _focus(session, 'BF', 4321.0)

    protocol = session.new_protocol()

    assert set(protocol.steps()['Z']) == {4321.0}


def test_after_a_home_new_takes_the_z_the_scope_read(make_session):
    session = make_session('LS850', acquiring=('BF',))
    _focus(session, 'BF', None)
    session.scope.motion.move_absolute('Z', 3000.0)
    z = session.scope.motion.axis_positions()['Z'].position

    protocol = session.new_protocol()

    assert list(protocol.steps()['Z']) == pytest.approx([z] * len(protocol.steps()), abs=0.01)


@pytest.mark.parametrize('model', ['LS620', 'LS560'])
def test_a_scope_with_no_z_motor_builds_as_it_did(make_session, model):
    session = make_session(model, acquiring=('BF',))
    _focus(session, 'BF', None)

    protocol = session.new_protocol()

    assert len(protocol.steps()) > 0
