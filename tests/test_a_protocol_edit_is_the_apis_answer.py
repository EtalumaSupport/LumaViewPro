# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol edit hands its request to the API and shows the answer once.

New Protocol, Modify and Add ask the API, which refuses a step or a
protocol it cannot perform (the objective is not on the turret, the
z-stack has no range) in its own words. The panel shows that refusal once,
through the boundary, and changes nothing it did not get: a refused
protocol is not adopted, and a refused step is not navigated to. What the
panel does after an accepted edit -- the move to the new step, the file
name cleared for a new protocol -- is reached only on acceptance.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())

for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
from modules.exceptions import ProtocolRunRefusedError
from tests.scope_fakes import spec_scope


def _refusal():
    return ProtocolRunRefusedError(
        reason='objective_not_on_turret',
        title='Objective Not Available',
        message='That objective is not on the turret.',
    )


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree and the navigation stubbed."""

    def __init__(self, protocol):
        self.ids = {
            'protocol_filename': SimpleNamespace(text='plate.tsv'),
            'capture_root': SimpleNamespace(text='root'),
            'step_name_input': SimpleNamespace(text=''),
            'tiling_size_spinner': SimpleNamespace(text='1x1'),
            'acquire_zstack_id': SimpleNamespace(active=False),
        }
        self._protocol = protocol
        self.curr_step = 0
        self.moves = []
        self.redraws = 0

    def go_to_step(self, step_idx, protocol=True):
        self.moves.append(step_idx)

    def update_step_ui(self):
        self.redraws += 1


def _protocol(num_steps=3):
    protocol = MagicMock()
    protocol.num_steps.return_value = num_steps
    protocol.validate_steps.return_value = []
    return protocol


@pytest.fixture
def ctx(monkeypatch):
    session = MagicMock()
    scope = spec_scope()
    context = SimpleNamespace(
        session=session,
        scope=scope,
        stage=MagicMock(),
        settings={'protocol': {'filepath': 'plate.tsv'}},
        image_settings=MagicMock(),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    monkeypatch.setattr(ps, 'require_file_writes_idle', lambda operation: True)
    monkeypatch.setattr(ps.common_utils, 'get_opened_layer', lambda _settings: 'Green')
    monkeypatch.setattr(ps, 'get_active_layer_config', lambda layer: (layer, {}))
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'button', lambda *a, **kw: None)
    return context


@pytest.fixture
def shown(monkeypatch):
    from tests.shown_outcomes import capture_shown

    return capture_shown(monkeypatch)


class TestAddStep:
    def test_an_accepted_step_is_gone_to_last(self, ctx):
        ctx.session.add_step.return_value = ['A1_Green']
        panel = _Panel(_protocol())

        panel.insert_step(after_current_step=True)

        assert panel.curr_step == 1
        assert panel.moves == [1]
        ctx.stage.set_protocol_steps.assert_called_once()

    def test_a_refused_step_is_shown_once_and_not_navigated_to(self, ctx, shown):
        ctx.session.add_step.side_effect = _refusal()
        panel = _Panel(_protocol())

        panel.insert_step(after_current_step=True)

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel.moves == [], 'a refused step has nowhere to go'
        assert panel.curr_step == 0
        assert panel.redraws == 1, 'the panel still shows its protocol as the API has it'


class TestModifyStep:
    def test_the_name_field_and_the_open_layer_reach_the_api(self, ctx):
        panel = _Panel(_protocol())
        panel.ids['step_name_input'].text = 'my_step'

        panel.modify_step()

        kwargs = ctx.session.update_step.call_args.kwargs
        assert kwargs['layer'] == 'Green'
        assert kwargs['label'] == 'my_step'

    def test_a_refused_modify_is_shown_once(self, ctx, shown):
        ctx.session.update_step.side_effect = _refusal()
        panel = _Panel(_protocol())

        panel.modify_step()

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel.redraws == 1


class TestNewProtocol:
    def test_an_accepted_protocol_is_adopted_and_has_no_file(self, ctx):
        built = _protocol()
        ctx.scope.protocols.create_protocol.return_value = built
        panel = _Panel(_protocol())

        panel.new_protocol()

        assert panel._protocol is built
        assert ctx.settings['protocol']['filepath'] == ''
        assert (panel.ids['protocol_filename'].text, panel.ids['capture_root'].text) == ('', '')
        assert panel.moves == [0]

    def test_a_refused_adoption_is_shown_once_and_changes_nothing(self, ctx, shown):
        ctx.scope.protocols.create_protocol.return_value = _protocol()
        ctx.scope.protocols.refuse_unaddressable_objectives.side_effect = _refusal()
        previous = _protocol()
        panel = _Panel(previous)

        panel.new_protocol()

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel._protocol is previous
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv'
        assert panel.ids['protocol_filename'].text == 'plate.tsv'
        assert panel.ids['capture_root'].text == 'root'
        assert panel.moves == []

    def test_a_refused_build_is_shown_once_and_adopts_nothing(self, ctx, shown):
        ctx.scope.protocols.create_protocol.side_effect = _refusal()
        previous = _protocol()
        panel = _Panel(previous)

        panel.new_protocol()

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel._protocol is previous
        assert not ctx.scope.protocols.refuse_unaddressable_objectives.called
