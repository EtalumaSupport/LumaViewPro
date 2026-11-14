# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel's fields are stored by the one channel write and nowhere else (G6).

Each of a channel's writers carried its own subset of the rules -- the LED
board's maximum, the focus travel, the auto-gain lock, the stop of
stimulation, the camera and LED apply -- so a value one writer refused
another stored, and a value one applied another left in the store alone.
``ScopeSession._write_channel_fields`` is the one write: it stores only what
is judged and applies the shown channel. Two places place whole channel
blocks at composition, before any writer runs: bring-up's move of old
channel blocks and its stop of the absent channels' acquiring.

Held two ways: the walk finds a literal channel path stored anywhere else in
the Session, and ``_store_setting`` itself refuses a channel path no store
opened (a variable path, which no walk sees).
"""

from __future__ import annotations

import ast

import pytest

from _lumascope.modules.scope_session import ScopeSession
from tests.ast_seams import parse_module, walk_defs
from tests.settings_fixtures import complete_settings

_THE_ONE_WRITE = 'ScopeSession._write_channel_fields'
_AT_COMPOSITION = frozenset(
    {'ScopeSession._place_old_channel_blocks', 'ScopeSession._stop_acquiring_absent_channels'}
)


def _channel_path_stores(fn: ast.AST) -> set[int]:
    """Lines storing a path that starts with a channel name: ``f'{channel}.<field>'``."""
    lines = set()
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_store_setting'
            and node.args
        ):
            continue
        path = node.args[0]
        if (
            isinstance(path, ast.JoinedStr)
            and isinstance(path.values[0], ast.FormattedValue)
            and len(path.values) > 1
            and isinstance(path.values[1], ast.Constant)
            and str(path.values[1].value).startswith('.')
        ):
            lines.add(node.lineno)
    return lines


def test_the_walk_sees_a_channel_path_store():
    """The instrument reports a store it must see, before its silence counts."""
    tree = ast.parse(
        'def f(self, channel, axis):\n'
        "    self._store_setting(f'{channel}.gain_db', 1.0)\n"
        "    self._store_setting(f'bookmark.{axis}', 1.0)\n"
        "    self._store_setting(f'{channel}.stim_config.enabled', False)\n"
    )
    assert _channel_path_stores(tree.body[0]) == {2, 4}


def test_only_the_one_write_stores_a_channel_path():
    tree = parse_module('src/_lumascope/modules/scope_session.py')
    found = {
        qualname: sorted(lines)
        for qualname, fn in walk_defs(tree.body)
        if qualname != _THE_ONE_WRITE
        and qualname not in _AT_COMPOSITION
        and (lines := _channel_path_stores(fn))
    }
    assert not found, f'store a channel field through {_THE_ONE_WRITE} (member: lines): {found}'


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
    yield built
    built.shutdown()


def test_a_channel_path_stored_past_the_one_write_is_refused(session):
    # Called directly: the store a variable path reaches, as update_settings' does.
    before = session.settings_snapshot()
    with session.settings_lock, pytest.raises(RuntimeError, match='one channel write'):
        session._store_setting('BF.gain_db', 7.5)
    assert session.settings_snapshot() == before
