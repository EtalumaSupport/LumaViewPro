# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer's acquire mode is set through the Session.

Whether a layer captures an image, a video or nothing decides which steps
New and Add Step build and what a composite merges. The GUI's acquire
toggle and its layer-hiding wrote the store directly, and no Session member
wrote it, so a script or REST could not choose what a protocol acquires
without writing the settings dict itself. ``ScopeSession.set_layer_acquire``
is the one writer: it refuses a layer or a mode it does not know, and a
layer set to acquire stops stimulating, as the GUI's toggle did.
"""

import ast

import pytest

from modules.exceptions import ArgumentRefusedError, SettingRefusedError
from tests.ast_seams import REPO_ROOT
from tests.test_loading_a_protocol_puts_the_scope_on_its_plate import (  # noqa: F401 -- pytest fixture
    session,
)


def _stim(session, layer):
    return session.settings[layer]['stim_config']['enabled']


class TestTheSessionSetsTheMode:
    @pytest.mark.parametrize('mode', ['image', 'video', None])
    def test_the_mode_is_stored(self, session, mode):
        session.set_layer_acquire('BF', mode)

        assert session.settings['BF']['acquire'] == mode

    @pytest.mark.parametrize('mode', ['image', 'video'])
    def test_a_layer_set_to_acquire_stops_stimulating(self, session, mode):
        with session.settings_lock:
            session.settings['Green']['stim_config']['enabled'] = True

        session.set_layer_acquire('Green', mode)

        assert _stim(session, 'Green') is False

    def test_a_layer_set_to_nothing_keeps_its_stimulation(self, session):
        with session.settings_lock:
            session.settings['Green']['stim_config']['enabled'] = True

        session.set_layer_acquire('Green', None)

        assert _stim(session, 'Green') is True

    def test_an_unknown_layer_is_refused(self, session):
        with pytest.raises(ArgumentRefusedError) as refused:
            session.set_layer_acquire('Infrared', 'image')
        assert refused.value.reason == 'layer_unknown'

        assert 'Infrared' not in session.settings

    # A list cannot be looked up in the vocabulary; it is refused, not a TypeError.
    @pytest.mark.parametrize('mode', ['still', 'none', '', 'Image'])
    def test_an_unknown_mode_is_refused_and_nothing_changes(self, session, mode):
        before = session.settings['BF']['acquire']

        with pytest.raises(SettingRefusedError) as refused:
            session.set_layer_acquire('BF', mode)

        assert (refused.value.reason, refused.value.path) == ('out_of_range', 'BF.acquire')
        assert session.settings['BF']['acquire'] == before


# The ui/ writer this guard can see that is left: the stimulation toggle,
# deferred with stimulation. Step navigation also writes a step's acquire
# into its layer, through settings[color].update(...), which this assignment
# scan does not see; it moves with tier 3's navigation item.
_UI_WRITERS_ALLOWED = {('ui/layer_control.py', 'update_stim_enable')}


def _acquire_writes(path):
    tree = ast.parse((REPO_ROOT / path).read_text())
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'acquire'
                    ):
                        yield (path, func.name)


def test_no_gui_code_writes_a_layers_acquire_mode_itself():
    found = {
        hit
        for path in sorted(
            p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / 'ui').glob('*.py')
        )
        for hit in _acquire_writes(path)
    }
    assert found == _UI_WRITERS_ALLOWED, (
        'a layer acquire mode is set through ScopeSession.set_layer_acquire; '
        f'these write it themselves: {sorted(found - _UI_WRITERS_ALLOWED)}'
    )
