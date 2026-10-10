# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A manual scope's startup says its missing motor board was expected.

Bring-up probes for a motor board on every model, so a board that is
plugged in can correct a wrongly selected model. On an LS620 or LS560 the
probe finds nothing, as it should, and the registry used to log that at
WARNING on every start, the same line a motorized scope with an unplugged
board gets. The scope now tells the registry when its model has no motor
board, and that line is INFO there. A motorized scope, or one with no
model selected, still warns; and a board that is found but cannot be used
warns on any scope, since on a manual scope it is a surprise.
"""

import logging

import pytest

import drivers.registry as registry_module
import modules.lumascope_api._lumascope as lumascope_module
from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.registry import DriverRegistry
from tests.log_capture import capture_module_log
from tests.scope_fakes import build_scope


class _NothingFound:
    found = False


class _HeldPort:
    found = True

    def is_connected(self):
        return False


class _Null:
    pass


def _registry(real_cls) -> DriverRegistry:
    reg = DriverRegistry('motor')
    reg.register('real', priority=100)(real_cls)
    reg.register('null', priority=0)(_Null)
    return reg


def _fallback_levels(records) -> list[int]:
    return [r.levelno for r in records if 'Falling back to _Null' in r.getMessage()]


@pytest.mark.parametrize(
    ('absence_expected', 'level'),
    [(True, logging.INFO), (False, logging.WARNING)],
)
def test_nothing_detected_is_info_only_where_nothing_is_expected(
    monkeypatch, absence_expected, level
):
    records = capture_module_log(monkeypatch, registry_module)

    board = _registry(_NothingFound).create('auto', absence_expected=absence_expected)

    assert isinstance(board, _Null)
    assert _fallback_levels(records) == [level]


def test_a_board_found_but_unusable_warns_even_where_none_is_expected(monkeypatch):
    records = capture_module_log(monkeypatch, registry_module)

    _registry(_HeldPort).create('auto', absence_expected=True)

    assert _fallback_levels(records) == [logging.WARNING]


@pytest.mark.parametrize(
    ('model', 'expected'),
    [('LS620', True), ('LS560', True), ('LS850T', False), (None, False)],
    ids=['LS620', 'LS560', 'LS850T', 'no-model'],
)
def test_the_scope_tells_the_probe_whether_its_model_has_a_motor_board(
    monkeypatch, model, expected
):
    asked = []

    def create(name='auto', **kwargs):
        asked.append(kwargs.get('absence_expected'))
        return NullMotionBoard(), None

    monkeypatch.setattr(lumascope_module.motor_registry, 'create_with_fallback', create)
    # No real port is opened in the suite: the LED board is answered null.
    monkeypatch.setattr(
        lumascope_module.led_registry,
        'create_with_fallback',
        lambda name='auto', **kwargs: (NullLEDBoard(), None),
    )
    build_scope(
        simulate=False,
        camera_type='sim',
        configured_model=model,
        warn_pre_release=False,
        register_atexit=False,
    )

    assert asked == [expected]
