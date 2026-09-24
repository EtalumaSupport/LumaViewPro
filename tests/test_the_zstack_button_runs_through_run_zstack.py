# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Z-stack Acquire button runs through ``ProtocolRunner.run_zstack``.

There is one implementation of "assemble and start a z-stack". The button
once built the whole run itself -- objective, stack, capture config, the
protocol, ``prepare`` and ``start`` -- beside the member built for scripts
and REST, so the two could drift and only tests ever ran the member. Now the
button states only what a running GUI knows (the open drawer, its own
trigger token, the live engineering flag, the engineering panel's saving
switch) and the member decides everything else, every refusal included.

The button's "Z n/total" readout asks the engine for the run's step count:
the member built the protocol, so the widget never holds it, and a second
count computed from settings could disagree with the run.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ui.zstack is a Kivy widget module; conftest mocks `kivy` but not the uix
# submodules, and the class subclasses a layout (a bare MagicMock cannot be
# subclassed).
class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx
import ui.zstack as zs
from modules.exceptions import ProtocolRunRefusedError
from modules.run_outcome import PendingRunOutcome


class _Starter(zs.ZStack):
    """The real class with the widget tree stubbed, the button as a first
    click leaves it."""

    def __init__(self):
        self.button = SimpleNamespace(state='down', text='Acquire')
        self.ids = {'zstack_aqr_btn': self.button}


@pytest.fixture
def clicked(monkeypatch):
    """Click Acquire with nothing running; return what the member was asked."""
    member_runner = MagicMock()
    handle = PendingRunOutcome()
    member_runner.run_zstack.return_value = handle
    member_runner.run_dir.return_value = '/runs/zstack_1'
    engine = MagicMock()
    engine.is_live_run.return_value = False
    session = MagicMock()
    session.create_protocol_runner.return_value = member_runner
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            session=session,
            sequenced_capture_runner=engine,
            image_settings=MagicMock(),
            engineering_mode=True,
            scope_display=SimpleNamespace(start=lambda: None, stop=lambda: None),
            motion_settings=MagicMock(),
        ),
    )
    monkeypatch.setattr(zs, 'live_histo_off', lambda: None)
    monkeypatch.setattr(zs, 'live_histo_reverse', lambda: None)
    monkeypatch.setattr(zs.gui_logger, 'button', lambda *a, **kw: None)
    monkeypatch.setattr(zs.common_utils, 'get_opened_layer', lambda _settings: 'Green')
    monkeypatch.setattr(zs, 'is_image_saving_enabled', lambda: False)
    linked = []
    monkeypatch.setattr(zs, 'set_last_save_folder', lambda **kw: linked.append(kw['dir']))
    return SimpleNamespace(
        runner=member_runner, engine=engine, handle=handle, linked=linked, starter=_Starter()
    )


def test_the_button_states_only_what_the_gui_knows(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    clicked.runner.run_zstack.assert_called_once()
    kwargs = clicked.runner.run_zstack.call_args.kwargs
    assert kwargs['layer'] == 'Green'
    assert kwargs['run_trigger_source'] == 'zstack'
    assert kwargs['engineering_mode'] is True
    assert kwargs['enable_image_saving'] is False
    # Nothing else about the run is the button's to say.
    assert set(kwargs) == {
        'layer',
        'callbacks',
        'run_trigger_source',
        'engineering_mode',
        'enable_image_saving',
    }


def test_the_button_starts_no_run_of_its_own(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    assert not clicked.engine.prepare.called
    assert not clicked.engine.start.called


def test_the_stop_names_the_handle_the_member_returned(clicked):
    clicked.starter.run_zstack_acquire_from_ui()

    assert clicked.starter._zstack_run is clicked.handle
    assert clicked.linked == ['/runs/zstack_1']


def test_a_refusal_resets_the_button_and_links_no_folder(clicked):
    clicked.runner.run_zstack.side_effect = ProtocolRunRefusedError(
        reason='zstack_not_configured', title='Z-Stack Not Configured', message='m'
    )

    clicked.starter.run_zstack_acquire_from_ui()

    assert clicked.starter.button.state == 'normal'
    assert clicked.starter.button.text == 'Acquire'
    assert clicked.linked == []


def test_progress_reads_the_runs_own_count(clicked, monkeypatch):
    monkeypatch.setattr(zs, 'Clock', SimpleNamespace(schedule_once=lambda fn, _delay: fn(0)))
    clicked.engine.run_num_steps.return_value = 7

    clicked.starter.run_zstack_acquire_from_ui()
    progress = clicked.runner.run_zstack.call_args.kwargs['callbacks']['update_step_number']
    progress(3)

    assert clicked.starter.button.text == 'Z 3/7'


def test_progress_after_the_run_ended_writes_nothing(clicked, monkeypatch):
    monkeypatch.setattr(zs, 'Clock', SimpleNamespace(schedule_once=lambda fn, _delay: fn(0)))
    clicked.engine.run_num_steps.return_value = None

    clicked.starter.run_zstack_acquire_from_ui()
    before = clicked.starter.button.text
    progress = clicked.runner.run_zstack.call_args.kwargs['callbacks']['update_step_number']
    progress(3)

    assert clicked.starter.button.text == before


class TestTheEnginesStepCount:
    def test_no_run_has_no_count(self):
        from tests.protocol_drives import bare_capture_runner

        runner = bare_capture_runner()
        runner._protocol = SimpleNamespace(num_steps=lambda: 5)

        assert runner.run_num_steps() is None

    def test_a_live_run_answers_its_protocols_count(self):
        from modules.sequenced_capture_runner import ProtocolState
        from tests.protocol_drives import bare_capture_runner

        runner = bare_capture_runner()
        runner._protocol = SimpleNamespace(num_steps=lambda: 5)
        runner._set_state(ProtocolState.RUNNING)

        assert runner.run_num_steps() == 5


def test_a_missing_layer_is_named_in_words():
    """A GUI with no drawer open has no layer to name; the member's answer
    says so rather than printing None into its catalogue sentence."""
    import modules.config_helpers as config_helpers
    from modules.exceptions import ConfigError

    with pytest.raises(ConfigError, match='No layer is selected'):
        config_helpers.get_standalone_capture_config_from_settings(
            {},
            MagicMock(),
            MagicMock(),
            layer=None,
            position={},
            position_name='ZStack',
            autofocus=False,
            use_zstacking=True,
            stim_config={},
        )


class TestTheMemberCarriesWhatTheGuiStates:
    """The three things only a running GUI knows reach the engine as stated;
    a caller that states none of them gets today's headless run."""

    def test_they_reach_prepare(self):
        from tests.test_run_zstack_entry_point import _prepared, _runner

        runner = _runner()
        runner.run_zstack(
            layer='BF',
            run_trigger_source='zstack',
            engineering_mode=True,
            enable_image_saving=False,
        )

        prepared = _prepared(runner)
        assert prepared['run_trigger_source'] == 'zstack'
        assert prepared['engineering_mode'] is True
        assert prepared['enable_image_saving'] is False

    def test_a_headless_caller_keeps_its_defaults(self):
        from tests.test_run_zstack_entry_point import _prepared, _runner

        runner = _runner()
        runner.session.engineering_mode = False
        runner.run_zstack(layer='BF')

        prepared = _prepared(runner)
        assert prepared['run_trigger_source'] == 'api_zstack'
        assert prepared['engineering_mode'] is False
        assert prepared['enable_image_saving'] is True
