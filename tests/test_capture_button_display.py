# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Capture button shows a failed still as a sentence, once.

The session's capture raises, at the call or when its Future settles; the
button hands either to the one reporter and only displays. A refusal carries a
reason code for callers that branch on it, and its sentence for the person --
a user never reads 'exclusive_activity_running'.
"""

import concurrent.futures
import types
from unittest.mock import MagicMock

import pytest

from _lumascope.modules.exceptions import (
    CaptureFailedError,
    HardwareCommandRefusedError,
    ObjectiveUnknownError,
)


@pytest.fixture
def button_ctx():
    import _lumascope.modules.app_context as _app_ctx

    ctx = MagicMock()
    ctx.scope_display.use_bullseye = True
    ctx.scope_display.use_crosshairs = False
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

    capture = button_ctx.session.manual_capture.start_save_still
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
    _press(
        button_ctx, raises=HardwareCommandRefusedError(reason, 'manual_capture.start_save_still')
    )

    body = _one_notice(centre_posts).message
    assert reason not in body
    assert body.endswith('.')


def test_a_diagnostics_refusal_shows_its_own_title_and_words(button_ctx, centre_posts):
    refused = HardwareCommandRefusedError(
        'exclusive_activity_running', 'manual_capture.start_save_still', 'diagnostic'
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


def test_the_button_hands_the_member_only_the_overlays_asked_for(button_ctx):
    from ui.composite_capture import CompositeCapture

    CompositeCapture.live_capture(object())

    kwargs = button_ctx.session.manual_capture.start_save_still.call_args.kwargs
    assert kwargs == {
        'bullseye': True,
        'crosshairs': False,
    }


def test_a_saved_still_shows_nothing(button_ctx, centre_posts, tmp_path):
    done = concurrent.futures.Future()
    # a stand-in by design: the button reads only whether the still settled.
    done.set_result([types.SimpleNamespace(file=tmp_path / 'Manual' / 'live_A1_BF_000001.tiff')])
    _press(button_ctx, settles=done)

    assert centre_posts == []
