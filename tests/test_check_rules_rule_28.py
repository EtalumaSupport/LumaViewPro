"""Tests for tools/check_rules.py rule_28.

rule_28 keeps user-facing notification strings in researcher voice. Two
shapes are blocked:

1. Internal-ID tokens (rule tags / audit IDs / fix-N refs) in any notification
   string constant -- the original check.
2. An exception's repr or class name interpolated into a notification body
   (`{e}`, `{exc}`, `{type(ex).__name__}`, ...). The exception detail belongs
   in the paired logger call; the popup body speaks to L1 researchers. The
   original check only inspected string constants, so f-string replacement
   fields carrying the exception were invisible -- a camera-init popup shipped
   the raw `ValueError: ... registry ... null fallback` text to users.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'rule_28']


class TestRule28BlocksExceptionRepr:
    def test_bare_exception_name_blocks(self):
        src = "def f(e):\n    notifications.error('Cat', 'Title', f'Failed: {e}')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1

    def test_type_name_blocks(self):
        src = (
            'def f(ex):\n'
            "    notifications.warning('Cat', 'Title', f'Failed: {type(ex).__name__}')\n"
        )
        assert len(_violations(src, 'modules/foo.py')) == 1

    def test_bare_type_call_blocks(self):
        src = "def f(exc):\n    notifications.error('Cat', 'Title', f'Failed: {type(exc)}')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1

    def test_repr_of_exception_blocks(self):
        src = "def f(e):\n    notifications.error('Cat', 'Title', f'Failed: {repr(e)}')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1

    def test_conversion_flag_on_exception_name_blocks(self):
        src = "def f(exc):\n    notifications.error('Cat', 'Title', f'Failed: {exc!r}')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1

    def test_multiline_concatenated_fstring_blocks(self):
        # The whole reported shape: exception repr several lines below the call.
        src = (
            'def f(ex):\n'
            '    notifications.error(\n'
            "        'Camera',\n"
            "        'Setting change failed',\n"
            "        f'Could not set mode to {mode!r}: '\n"
            "        f'{type(ex).__name__}: {ex}. rest.',\n"
            '    )\n'
        )
        # Two replacement fields carry the exception: type(ex).__name__ and ex.
        assert len(_violations(src, 'modules/foo.py')) == 2

    def test_keyword_message_arg_blocks(self):
        src = "def f(e):\n    notifications.error('Cat', 'Title', message=f'Failed: {e}')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1


class TestRule28AllowsBenignInterpolation:
    def test_plain_string_passes(self):
        src = "def f():\n    notifications.error('Cat', 'Title', 'Check the USB cable.')\n"
        assert _violations(src, 'modules/foo.py') == []

    def test_non_exception_value_passes(self):
        src = (
            'def f(port, value):\n'
            "    notifications.warning('Cat', 'Title', f'Set port {port} to {value}.')\n"
        )
        assert _violations(src, 'modules/foo.py') == []

    def test_mode_repr_passes(self):
        # `{mode!r}` is the user's requested value, not an exception.
        src = "def f(mode):\n    notifications.error('Cat', 'Title', f'Could not set {mode!r}.')\n"
        assert _violations(src, 'modules/foo.py') == []

    def test_logger_call_with_exception_passes(self):
        # rule_28 governs notifications only; the exception belongs in the log.
        src = "def f(e):\n    logger.error(f'Failed: {type(e).__name__}: {e}')\n"
        assert _violations(src, 'modules/foo.py') == []


class TestRule28InternalIdStillCaught:
    def test_rule_tag_in_notification_blocks(self):
        src = "def f():\n    notifications.error('Cat', 'Title', 'See Rule 28 for details.')\n"
        assert len(_violations(src, 'modules/foo.py')) == 1
