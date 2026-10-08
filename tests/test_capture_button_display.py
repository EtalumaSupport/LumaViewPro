# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Capture button shows a failed still as a sentence, once.

The session's capture raises, at the call or when its Future settles; the
button hands either to the one reporter and only displays. A refusal carries a
reason code for callers that branch on it, and its sentence for the person --
a user never reads 'exclusive_activity_running'.
"""

import concurrent.futures
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from modules.exceptions import CaptureError, HardwareCommandRefusedError, ObjectiveUnknownError
from modules.notification_center import Severity
from tests.shown_outcomes import capture_shown


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())
_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)


@pytest.fixture
def button_ctx(monkeypatch):
    import modules.app_context as _app_ctx
    from ui import composite_capture

    ctx = MagicMock()
    ctx.scope_display.use_bullseye = True
    ctx.scope_display.use_crosshairs = False
    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    # Headless: a settled still is read as soon as it is scheduled.
    monkeypatch.setattr(
        composite_capture,
        'Clock',
        types.SimpleNamespace(schedule_once=lambda fn, timeout=0: fn(0)),
    )
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


@pytest.fixture
def shown(monkeypatch):
    return capture_shown(monkeypatch)


def _settled_with(exc):
    future = concurrent.futures.Future()
    future.set_exception(exc)
    return future


def _press(button_ctx, *, raises=None, settles=None):
    from ui.composite_capture import CompositeCapture

    capture = button_ctx.session.manual_capture.capture
    if raises is not None:
        capture.side_effect = raises
    else:
        capture.return_value = settles
    with patch('ui.composite_capture.common_utils.get_opened_layer', return_value=None):
        CompositeCapture.live_capture(object())


def _one_notice(shown):
    assert len(shown) == 1, f'expected one notice, got {shown}'
    return shown[0]


@pytest.mark.parametrize('reason', ['exclusive_activity_running', 'capture_in_flight'])
def test_a_refusal_reads_as_a_sentence(button_ctx, shown, reason):
    _press(button_ctx, raises=HardwareCommandRefusedError(reason, 'manual_capture.capture'))

    body = _one_notice(shown).message
    assert reason not in body
    assert body.endswith('.')


def test_a_diagnostics_refusal_shows_its_own_title_and_words(button_ctx, shown):
    refused = HardwareCommandRefusedError(
        'exclusive_activity_running', 'manual_capture.capture', 'diagnostic'
    )
    _press(button_ctx, raises=refused)

    notice = _one_notice(shown)
    assert notice.title == refused.title
    assert notice.message == str(refused)


def test_a_capture_failure_shows_the_engines_cause(button_ctx, shown):
    _press(
        button_ctx,
        settles=_settled_with(CaptureError('camera inactive or not grabbing', 'no_frame_returned')),
    )

    notice = _one_notice(shown)
    assert notice.title == 'Capture Failed'
    assert notice.message == 'camera inactive or not grabbing'


def test_an_unknown_objective_shows_what_to_do(button_ctx, shown):
    exc = ObjectiveUnknownError.__new__(ObjectiveUnknownError)
    Exception.__init__(exc, 'Home the turret, then capture.')

    _press(button_ctx, settles=_settled_with(exc))

    assert _one_notice(shown).message == 'Home the turret, then capture.'


def test_anything_else_is_not_shown_raw(button_ctx, shown):
    _press(button_ctx, settles=_settled_with(KeyError('live_folder')))

    assert 'live_folder' not in _one_notice(shown).message


def test_the_button_hands_the_member_what_the_user_sees(button_ctx):
    from ui.composite_capture import CompositeCapture

    with patch('ui.composite_capture.common_utils.get_opened_layer', return_value=None):
        CompositeCapture.live_capture(object())

    kwargs = button_ctx.session.manual_capture.capture.call_args.kwargs
    assert kwargs == {
        'layer': None,
        'false_color_on': False,
        'bullseye': True,
        'crosshairs': False,
    }


def test_a_refused_press_says_why(button_ctx, shown):
    _press(
        button_ctx,
        raises=HardwareCommandRefusedError('capture_in_flight', 'manual_capture.capture'),
    )

    assert [n.severity for n in shown] == [Severity.WARNING]


def test_a_saved_still_remembers_its_folder(button_ctx, shown, tmp_path):
    done = concurrent.futures.Future()
    done.set_result([tmp_path / 'Manual' / 'live_A1_BF_000001.tiff'])
    with patch('ui.composite_capture.set_last_save_folder') as remembered:
        _press(button_ctx, settles=done)

    remembered.assert_called_once_with(dir=tmp_path / 'Manual')
    assert shown == []
