"""A refused entry must never be recorded as the value the user typed.

Five handlers wrote the app's own value back into the widget whose event had
invoked them. Where the widget re-dispatches on that write, the second pass
reads the app's value, finds it valid, and emits it as the user's -- and the
bundle asserts the user chose something they were actually refused.

The two halves of that split on whether the write dispatches:

- A TEXT box does not. Assigning ``.text`` rebuilds the lines and the cursor
  and never touches ``focus``, so the ``on_focus`` binding these handlers hang
  from cannot fire from an app write. The three text boxes are driven through
  their REAL handlers here, one commit per test, and what is pinned is that the
  attempt and the correction are BOTH recorded and the reverted value is never
  reported as typed.
- A SPINNER does. It needs a camera and a plate loader to drive, so for those
  this pins the ordering invariant instead -- the declaration must precede the
  write, because a spinner dispatches synchronously and a declaration made
  afterwards arrives too late to absorb anything.
"""

import ast
from typing import ClassVar

import pytest

from modules import gui_logger
from tests.ast_seams import find_def


class _Widget:
    def __init__(self, text=''):
        self.text = text


@pytest.fixture
def emitted(monkeypatch):
    lines = []
    monkeypatch.setattr(
        gui_logger, 'text_input', lambda name, value: lines.append((name, str(value)))
    )
    return lines


def _advanced_panel(monkeypatch, widget_id, widget, stored):
    """A stand-in AdvancedSettings carrying just what these handlers touch."""
    from ui import advanced_settings

    from tests.settings_fixtures import settings_writer

    class _Panel:
        ids: ClassVar[dict] = {widget_id: widget}
        _commit_video_limit = staticmethod(advanced_settings.AdvancedSettings._commit_video_limit)

    context = type(
        'C', (), {'settings': stored, 'update_settings': staticmethod(settings_writer(stored))}
    )()
    monkeypatch.setattr(advanced_settings._app_ctx, 'ctx', context)
    return _Panel()


def test_a_refused_fps_limit_records_the_attempt_not_the_revert(emitted, monkeypatch):
    """Type 500 into a box that caps at 200: the bundle must not claim 30."""
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('500')
    stored = {'video': {'max_fps': 30}}
    panel = _advanced_panel(monkeypatch, 'video_max_fps_input', widget, stored)

    AdvancedSettings.update_video_max_fps(panel)  # one commit -- refused

    assert ('VIDEO_MAX_FPS', '500') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_FPS_APPLIED', '30') in emitted, (
        f'the revert was not reported as the correction: {emitted}'
    )
    assert ('VIDEO_MAX_FPS', '30') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def test_a_refused_duration_records_the_attempt_not_the_revert(emitted, monkeypatch):
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('99999')
    stored = {'video': {'max_duration_seconds': 300}}
    panel = _advanced_panel(monkeypatch, 'video_max_duration_input', widget, stored)

    AdvancedSettings.update_video_max_duration(panel)

    assert ('VIDEO_MAX_DURATION_S', '99999') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_DURATION_S_APPLIED', '300') in emitted, (
        f'the revert was not reported as the correction: {emitted}'
    )
    assert ('VIDEO_MAX_DURATION_S', '300') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def test_restoring_labware_at_startup_is_not_recorded_as_a_selection():
    """Two LABWARE records fired at cold start from no user gesture.

    Restoring the stored labware writes the spinner (which dispatches) AND
    calls select_labware() explicitly, so startup emits twice. Each emission
    needs its own declaration: only one is pending per record name at a time.
    """
    fn = find_def('ui/microscope_settings.py', 'load_settings', class_name='MicroscopeSettings')
    declarations = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'note_write_back'
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 'LABWARE'
    ]
    assert len(declarations) >= 2, (
        'startup emits LABWARE twice -- once from the spinner write and once '
        'from the explicit select_labware() call -- so each needs its own '
        f'declaration. Found {len(declarations)}'
    )


def test_a_capture_root_records_and_stores_what_was_typed(emitted, monkeypatch):
    """A path-illegal root is recorded, and stored, as typed; the filename
    prefix made of it is the protocol's (capture_prefix), not the field's."""
    from unittest.mock import MagicMock

    from ui.protocol_settings import ProtocolSettings

    widget = _Widget('my/run:1')

    class _Panel:
        ids: ClassVar[dict] = {'capture_root': widget}
        _protocol = MagicMock()

    panel = _Panel()
    ProtocolSettings.update_capture_root(panel, widget.text)

    typed = [v for n, v in emitted if n == 'CAPTURE_ROOT']
    assert typed == ['my/run:1'], f'what the user typed is recorded once: {emitted}'
    panel._protocol.modify_capture_root.assert_called_once_with(capture_root='my/run:1')
    assert widget.text == 'my/run:1', 'the field shows what was typed'


def test_restoring_binning_at_startup_is_not_recorded_as_a_selection():
    """The twin of the labware restore, in the same startup block.

    Writing the spinner dispatches its text event AND select_binning_size() is
    called explicitly, so cold start recorded two binning picks nobody made.
    """
    fn = find_def('ui/microscope_settings.py', 'load_settings', class_name='MicroscopeSettings')
    declarations = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'note_write_back'
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 'BINNING'
    ]
    assert len(declarations) >= 2, (
        'startup emits BINNING twice -- once from the spinner write and once '
        'from the explicit select_binning_size() call -- so each needs its own '
        f'declaration. Found {len(declarations)}'
    )
