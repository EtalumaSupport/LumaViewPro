# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home that fails tells its caller why, by raising; it posts nothing.

A GUI Home press, a script, bring-up and REST all get the same thing. No
motor controller is a refusal ("Not Connected"): nothing was driven. A
driver that answers False, a driver that raises and a home whose position
could not be read are one fault, ``HomingFailedError`` ("Homing Failed"),
naming the axes left without a position and chained from the driver's
exception where there is one. The home itself shows nothing: whoever waits
on it reports it, once.

Every test drives a real ``Lumascope(simulate=True)`` through the public
``home`` member, so the dispatch, the lane and the body all run.
"""

from __future__ import annotations

import logging

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import HardwareCommandRefusedError, HomingFailedError
from modules.lumascope_api import AxisState
from tests.scope_fakes import build_scope

ROUTES = ('ALL', 'Z', 'T')
SUBJECT = {'ALL': 'Homing', 'Z': 'Z axis homing', 'T': 'Turret homing'}
DRIVER_CALL = {'ALL': 'home', 'Z': 'zhome', 'T': 'thome'}
DONE_LINE = {'ALL': 'home DONE', 'Z': 'Z home DONE', 'T': 'T home DONE'}


@pytest.fixture
def scope(monkeypatch):
    import modules.notification_center as nc

    scope = build_scope(simulate=True)
    scope.motion.home('ALL')
    posts = []
    real_notify = nc.notifications.notify

    def _record(severity, category, title, message, **kw):
        posts.append((category, title))
        return real_notify(severity, category, title, message, **kw)

    monkeypatch.setattr(nc.notifications, 'notify', _record)
    scope.posts = posts
    return scope


def _axes_of(scope, route):
    return tuple(scope.capabilities.axes) if route == 'ALL' else (route,)


@pytest.mark.parametrize('route', ROUTES)
def test_no_motor_controller_is_a_refusal_in_the_cables_words(scope, monkeypatch, route):
    monkeypatch.setattr(type(scope), 'motor_connected', property(lambda self: False))

    with pytest.raises(HardwareCommandRefusedError) as raised:
        scope.motion.home(route)

    assert raised.value.reason == 'not_connected'
    assert raised.value.title == 'Not Connected'
    assert 'motor controller is not connected' in str(raised.value)
    assert scope.posts == []


@pytest.mark.parametrize('route', ROUTES)
def test_a_driver_that_answers_false_raises_the_homing_fault(scope, monkeypatch, route):
    monkeypatch.setattr(scope._motion_driver, DRIVER_CALL[route], lambda: False)

    with pytest.raises(HomingFailedError) as raised:
        scope.motion.home(route)

    assert raised.value.reason == 'failed'
    assert raised.value.title == 'Homing Failed'
    assert str(raised.value) == f'{SUBJECT[route]} failed. Position is unknown.'
    assert raised.value.axes == _axes_of(scope, route)
    for axis in raised.value.axes:
        assert scope.motion.get_axis_state(axis) == AxisState.UNKNOWN
    assert scope.posts == []


@pytest.mark.parametrize('route', ROUTES)
def test_a_driver_that_raises_is_chained_not_rewrapped(scope, monkeypatch, route):
    cause = HardwareError('no response from motor board')

    def _raise():
        raise cause

    monkeypatch.setattr(scope._motion_driver, DRIVER_CALL[route], _raise)

    with pytest.raises(HomingFailedError) as raised:
        scope.motion.home(route)

    assert raised.value.reason == 'error'
    assert str(raised.value) == f'{SUBJECT[route]} encountered an error. Position is unknown.'
    assert raised.value.__cause__ is cause
    assert raised.value.axes == _axes_of(scope, route)
    assert scope.posts == []


@pytest.mark.parametrize('route', ROUTES)
def test_a_home_whose_position_is_unread_names_the_unread_axis(scope, route):
    unread = 'Z' if route in ('ALL', 'Z') else 'T'
    scope._motion_driver._fail_on.add(f'TARGET_R{unread}')

    with pytest.raises(HomingFailedError) as raised:
        scope.motion.home(route)

    assert raised.value.reason == 'unread'
    assert raised.value.axes == (unread,)
    assert raised.value.__cause__ is None
    assert f'position of {unread} could not be read' in str(raised.value)
    assert scope.posts == []


@pytest.mark.parametrize('route', ROUTES)
def test_the_done_line_is_logged_however_the_home_ends(scope, monkeypatch, caplog, route):
    monkeypatch.setattr(scope._motion_driver, DRIVER_CALL[route], lambda: False)

    with caplog.at_level(logging.INFO), pytest.raises(HomingFailedError):
        scope.motion.home(route)

    assert any(r.getMessage().endswith(DONE_LINE[route]) for r in caplog.records)


@pytest.mark.parametrize('route', ROUTES)
def test_a_home_that_works_returns_and_posts_nothing(scope, route):
    assert scope.motion.home(route) is None
    for axis in _axes_of(scope, route):
        assert scope.motion.get_axis_state(axis) == AxisState.IDLE
    assert scope.posts == []


def test_a_turret_home_on_a_scope_with_no_turret_homes_nothing_and_fails_nothing():
    # A board with no turret has no T position to read, so its absence is
    # not an unread position: the turret home is a success, as documented.
    scope = build_scope(simulate=True, sim_model='LS850')
    assert 'T' not in scope.capabilities.axes

    assert scope.motion.home('T') is None
