# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A click that does not start a run must leave nothing armed behind it.

The standalone autofocus button arms a 15-second stuck-AF bound: if the
sweep stops progressing, the timer force-aborts it rather than leaving
the lockout up until someone notices. The bound is real and stays.

What matters is WHEN it is armed. Armed before ``prepare()``, it
outlives every exit between the arm and a run actually starting -- the
engine's refusal, a raise out of the protocol builder, anything the
starter's blanket handler catches. The timer then sits live for 15
seconds with no run of its own, and its predicate cannot tell the
difference: it fires on ``run_trigger_source() == 'autofocus'``, so the
next standalone autofocus the user starts inside that window is the run
it force-aborts. A click that was refused would be reaching forward to
kill the click that was not.

So the bound is armed by the run it bounds: after ``start(plan)``
commits, where a timer can only ever exist alongside the run it belongs
to. Disarming on the refusal path would fix one of the three exits by
hand; arming on commit makes all three unrepresentable.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_boxlayout = types.ModuleType('kivy.uix.boxlayout')
_boxlayout.BoxLayout = _StubWidget
sys.modules.setdefault('kivy.uix.boxlayout', _boxlayout)

import modules.app_context as _app_ctx
import ui.vertical_control as vc
from modules.exceptions import ProtocolRunRefusedError
from tests.scope_fakes import spec_scope


REFUSAL = ProtocolRunRefusedError(
    reason='already_running',
    title='Run In Progress',
    message='A protocol run is using the microscope.',
)


def _starter(arm_log):
    """The starter's own methods, with the widget tree and cosmetics stubbed.

    An unbound call with a namespace stub, the way this module's sibling
    guards drive the completion handler: the starter reaches ``prepare``
    through the real body, which is the ordering under test.
    """
    return SimpleNamespace(
        ids={'autofocus_id': SimpleNamespace(state='down', text='Autofocus')},
        _schedule_af_safety_timer=lambda: arm_log.append('armed'),
        _unschedule_af_safety_timer=lambda: arm_log.append('disarmed'),
        _set_run_autofocus_button=lambda: None,
        _reset_run_autofocus_button_cosmetics=lambda: None,
        _cleanup_at_end_of_autofocus=lambda: None,
        _autofocus_run_complete=lambda **kw: None,
    )


@pytest.fixture
def runner():
    """Idle, so the click takes the start path rather than stop or refuse-early."""
    r = MagicMock()
    r.run_in_progress.return_value = False
    r.run_trigger_source.return_value = None
    return r


@pytest.fixture
def af_ctx(monkeypatch, runner, tmp_path):
    # Specced, not bare: this test asserts what the starter did NOT do,
    # and a double that answers any attribute at all would let a renamed
    # collaborator pass silently.
    scope = spec_scope()
    scope.protocols.create_protocol.return_value = MagicMock()
    session = MagicMock()
    session.controls_locked = False
    session.get_current_plate_position.return_value = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    session.scope.runtime_state.resolve_current_objective.return_value = ('objective', {})
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=scope,
            session=session,
            sequenced_capture_runner=runner,
            settings={'live_folder': str(tmp_path)},
            settings_lock=MagicMock(),
            engineering_mode=False,
            source_path='.',
            image_settings=MagicMock(),
        ),
    )
    # Everything between the click and prepare() that reads the widget
    # tree or the live settings; the ordering under test is untouched.
    monkeypatch.setattr(vc, 'require_file_writes_idle', lambda operation: True)
    monkeypatch.setattr(vc, 'live_histo_off', lambda: None)
    monkeypatch.setattr(vc, 'live_histo_reverse', lambda: None)
    monkeypatch.setattr(vc, 'live_display_callbacks', dict)
    monkeypatch.setattr(vc.gui_logger, 'button', lambda *a, **kw: None)
    monkeypatch.setattr(vc, 'get_selected_labware', lambda: ('labware', {}))
    monkeypatch.setattr(vc, 'get_active_layer_config', lambda layer: ('Green', {}))
    monkeypatch.setattr(vc, 'get_binning_from_ui', lambda: 1)
    monkeypatch.setattr(vc, 'get_image_capture_config_from_ui', MagicMock())
    monkeypatch.setattr(vc, 'get_auto_gain_settings', MagicMock())
    monkeypatch.setattr(vc.common_utils, 'get_opened_layer', lambda image_settings: 'Green')
    monkeypatch.setattr(vc, 'TilingConfig', MagicMock())
    monkeypatch.setattr(vc.config_helpers, 'build_sequenced_capture_config', lambda cfg: cfg)
    monkeypatch.setattr(vc.config_helpers, 'autofocus_snapshot_from_settings', MagicMock())
    monkeypatch.setattr(vc.config_helpers, 'get_sequenced_run_settings', lambda *a, **kw: {})
    return runner


class TestARefusedAutofocusClick:
    def test_it_arms_no_safety_timer(self, af_ctx, monkeypatch):
        af_ctx.prepare.side_effect = REFUSAL
        armed: list[str] = []

        vc.VerticalControl.run_autofocus_from_ui(_starter(armed))

        assert af_ctx.prepare.called, (
            'the click never reached the engine -- the test is not exercising the refusal'
        )
        assert 'armed' not in armed, (
            'a refused click starts no run, so it must leave no stuck-AF bound '
            "behind it: the next real autofocus is what that timer's predicate "
            f'would match. Timer calls: {armed}'
        )

    def test_a_started_run_still_arms_one(self, af_ctx, monkeypatch):
        armed: list[str] = []

        vc.VerticalControl.run_autofocus_from_ui(_starter(armed))

        assert af_ctx.start.called, 'the run never started -- the bound has nothing to guard'
        assert 'armed' in armed, (
            'the stuck-AF bound is the reason this timer exists; a committed '
            f'run must still get one. Timer calls: {armed}'
        )
