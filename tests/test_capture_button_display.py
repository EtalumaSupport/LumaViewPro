# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Capture button shows a failed still as a sentence, once.

The session's capture raises, at the call or when its Future settles; the
button hands either to the one reporter and only displays. A refusal carries a
reason code for callers that branch on it, and its sentence for the person --
a user never reads 'exclusive_activity_running'.
"""

import concurrent.futures
import types
from unittest.mock import MagicMock, patch

import pytest

from modules.exceptions import (
    CaptureFailedError,
    HardwareCommandRefusedError,
    ObjectiveUnknownError,
)
from modules.scope_session import ShownChannel


@pytest.fixture
def button_ctx():
    import modules.app_context as _app_ctx

    ctx = MagicMock()
    ctx.scope_display.use_bullseye = True
    ctx.scope_display.use_crosshairs = False
    ctx.session.shown_channel = ShownChannel('BF', lit=False)
    ctx.image_settings.layer_lookup.return_value.ids = {
        'false_color': types.SimpleNamespace(active=False)
    }
    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    # Headless, no UI dispatcher: a settled still's post runs at once.
    try:
        yield ctx
    finally:
        _app_ctx.ctx = original


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
    CompositeCapture.live_capture(object())


def _one_notice(centre_posts):
    assert len(centre_posts) == 1, f'expected one notice, got {centre_posts}'
    return centre_posts[0]


@pytest.mark.parametrize('reason', ['exclusive_activity_running', 'capture_in_flight'])
def test_a_refusal_reads_as_a_sentence(button_ctx, centre_posts, reason):
    _press(button_ctx, raises=HardwareCommandRefusedError(reason, 'manual_capture.capture'))

    body = _one_notice(centre_posts).message
    assert reason not in body
    assert body.endswith('.')


def test_a_diagnostics_refusal_shows_its_own_title_and_words(button_ctx, centre_posts):
    refused = HardwareCommandRefusedError(
        'exclusive_activity_running', 'manual_capture.capture', 'diagnostic'
    )
    _press(button_ctx, raises=refused)

    notice = _one_notice(centre_posts)
    assert notice.title == refused.title
    assert notice.message == str(refused)


def test_a_capture_failure_shows_the_captures_own_words(button_ctx, centre_posts):
    failed = CaptureFailedError('no_frame')
    _press(button_ctx, settles=_settled_with(failed))

    notice = _one_notice(centre_posts)
    assert notice.title == 'Capture Failed'
    assert notice.message == str(failed)


def test_an_unknown_objective_shows_what_to_do(button_ctx, centre_posts):
    exc = ObjectiveUnknownError.__new__(ObjectiveUnknownError)
    Exception.__init__(exc, 'Home the turret, then capture.')

    _press(button_ctx, settles=_settled_with(exc))

    assert _one_notice(centre_posts).message == 'Home the turret, then capture.'


def test_anything_else_is_not_shown_raw(button_ctx, centre_posts):
    _press(button_ctx, settles=_settled_with(KeyError('live_folder')))

    assert 'live_folder' not in _one_notice(centre_posts).message


def test_the_button_hands_the_member_what_the_user_sees(button_ctx):
    from ui.composite_capture import CompositeCapture

    CompositeCapture.live_capture(object())

    kwargs = button_ctx.session.manual_capture.capture.call_args.kwargs
    assert kwargs == {
        'false_color_on': False,
        'bullseye': True,
        'crosshairs': False,
    }


def test_a_saved_still_remembers_its_folder(button_ctx, centre_posts, tmp_path):
    done = concurrent.futures.Future()
    # a stand-in by design: the button reads only the record's file.
    done.set_result([types.SimpleNamespace(file=tmp_path / 'Manual' / 'live_A1_BF_000001.tiff')])
    with patch('ui.composite_capture.set_last_save_folder') as remembered:
        _press(button_ctx, settles=done)

    remembered.assert_called_once_with(dir=tmp_path / 'Manual')
    assert centre_posts == []
