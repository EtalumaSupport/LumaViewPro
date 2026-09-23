# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Capture button shows a failed still as a sentence, once.

The session's capture raises; the button only displays. A refusal carries a
reason code for callers that branch on it and no words, so the button writes
the sentence -- a user never reads 'exclusive_activity_running'.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from modules.exceptions import CaptureError, HardwareCommandRefusedError, ObjectiveUnknownError


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())
_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)


def _shown(exc):
    from ui import composite_capture

    with patch('modules.notification_center.notifications') as notifications:
        composite_capture._show_capture_failure(exc)
    calls = notifications.warning.call_args_list + notifications.error.call_args_list
    assert len(calls) == 1, f'expected one notice, got {calls}'
    return calls[0].args


@pytest.mark.parametrize('reason', ['exclusive_activity_running', 'capture_in_flight'])
def test_a_refusal_reads_as_a_sentence(reason):
    _category, _title, body = _shown(HardwareCommandRefusedError(reason, 'manual_capture.capture'))

    assert reason not in body
    assert body.endswith('.')


def test_a_capture_failure_shows_the_engines_cause():
    _category, _title, body = _shown(
        CaptureError('camera inactive or not grabbing', 'no_frame_returned')
    )

    assert body == 'camera inactive or not grabbing'


def test_an_unknown_objective_shows_what_to_do():
    exc = ObjectiveUnknownError.__new__(ObjectiveUnknownError)
    Exception.__init__(exc, 'Home the turret, then capture.')

    _category, _title, body = _shown(exc)

    assert body == 'Home the turret, then capture.'


def test_anything_else_is_not_shown_raw():
    _category, _title, body = _shown(KeyError('live_folder'))

    assert 'live_folder' not in body


@pytest.fixture
def button_ctx():
    import modules.app_context as _app_ctx

    ctx = MagicMock()
    ctx.engineering_mode = True
    ctx.scope_display.use_bullseye = True
    ctx.scope_display.use_crosshairs = False
    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


def test_the_button_hands_the_member_what_the_user_sees(button_ctx):
    from ui.composite_capture import CompositeCapture

    CompositeCapture._capturing.clear()
    with patch('ui.composite_capture.common_utils.get_opened_layer', return_value=None):
        CompositeCapture.live_capture(object())

    kwargs = button_ctx.session.manual_capture.capture.call_args.kwargs
    assert kwargs == {
        'layer': None,
        'false_color_on': False,
        'bullseye': True,
        'crosshairs': False,
        'engineering_mode': True,
    }
    CompositeCapture._capturing.clear()


def test_a_refused_press_releases_the_composite_flag_and_says_why(button_ctx):
    from ui.composite_capture import CompositeCapture

    CompositeCapture._capturing.clear()
    button_ctx.session.manual_capture.capture.side_effect = HardwareCommandRefusedError(
        'capture_in_flight', 'manual_capture.capture'
    )
    with (
        patch('ui.composite_capture.common_utils.get_opened_layer', return_value=None),
        patch('modules.notification_center.notifications') as notifications,
    ):
        CompositeCapture.live_capture(object())

    assert not CompositeCapture._capturing.is_set()
    assert notifications.warning.call_count == 1


def test_a_saved_still_remembers_its_folder(tmp_path):
    import concurrent.futures

    from ui import composite_capture

    done = concurrent.futures.Future()
    done.set_result([tmp_path / 'Manual' / 'live_A1_BF_000001.tiff'])
    composite_capture.CompositeCapture._capturing.set()
    with patch('ui.composite_capture.set_last_save_folder') as remembered:
        composite_capture._show_capture_outcome(done)

    remembered.assert_called_once_with(dir=tmp_path / 'Manual')
    assert not composite_capture.CompositeCapture._capturing.is_set()
