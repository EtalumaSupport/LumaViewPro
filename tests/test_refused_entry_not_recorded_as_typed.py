"""A refused entry must never be recorded as the value the user typed.

Five handlers wrote the app's own value back into the widget whose event had
invoked them, without declaring the write. The widget re-dispatches; the second
pass reads the app's value, finds it valid, and emits it as the user's. The
bundle then asserts the user chose something they were actually refused.

The three text boxes are driven through their REAL handlers with a fake Clock,
so these exercise the shipped logic rather than the helper in isolation. The two
spinners need a camera and a plate loader to drive, so for those this pins the
ordering invariant instead -- the declaration must precede the write, because a
spinner dispatches synchronously and a declaration made afterwards arrives too
late to absorb anything.
"""

import ast
from typing import ClassVar

import pytest

from tests.ast_seams import find_def
from ui import ui_helpers


class _Widget:
    def __init__(self, text=''):
        self.text = text


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def clock(monkeypatch):
    scheduled = []

    class _Clock:
        @staticmethod
        def schedule_once(fn, delay):
            timer = _FakeTimer()
            scheduled.append((fn, timer))
            return timer

    monkeypatch.setattr(ui_helpers, 'Clock', _Clock)
    monkeypatch.setattr(ui_helpers, '_text_input_debounce_timers', {})
    monkeypatch.setattr(ui_helpers.gui_logger, '_write_backs', {})
    return scheduled


@pytest.fixture
def emitted(monkeypatch):
    lines = []
    monkeypatch.setattr(
        ui_helpers.gui_logger, 'text_input', lambda name, value: lines.append((name, str(value)))
    )
    return lines


def _fire(scheduled):
    for fn, timer in list(scheduled):
        if not timer.cancelled:
            fn(0)
    scheduled.clear()


def _advanced_panel(monkeypatch, widget_id, widget, stored):
    """A stand-in AdvancedSettings carrying just what these handlers touch."""
    from ui import advanced_settings

    class _Panel:
        ids: ClassVar[dict] = {widget_id: widget}

    monkeypatch.setattr(advanced_settings._app_ctx, 'ctx', type('C', (), {'settings': stored})())
    monkeypatch.setattr(advanced_settings, 'notifications', _Noop(), raising=False)
    return _Panel()


class _Noop:
    def warning(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def _silence_notifications(monkeypatch):
    from modules import notification_center

    monkeypatch.setattr(notification_center.notifications, 'warning', lambda *a, **k: None)


def test_a_refused_fps_limit_records_the_attempt_not_the_revert(clock, emitted, monkeypatch):
    """Type 500 into a box that caps at 200: the bundle must not claim 30."""
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('500')
    stored = {'video': {'max_fps': 30}}
    panel = _advanced_panel(monkeypatch, 'video_max_fps_input', widget, stored)

    AdvancedSettings.update_video_max_fps(panel)  # first commit pass -- refused
    AdvancedSettings.update_video_max_fps(panel)  # the echo: box now reads 30
    _fire(clock)

    assert ('VIDEO_MAX_FPS', '500') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_FPS', '30') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def test_a_refused_duration_records_the_attempt_not_the_revert(clock, emitted, monkeypatch):
    from ui.advanced_settings import AdvancedSettings

    widget = _Widget('99999')
    stored = {'video': {'max_duration_seconds': 300}}
    panel = _advanced_panel(monkeypatch, 'video_max_duration_input', widget, stored)

    AdvancedSettings.update_video_max_duration(panel)
    AdvancedSettings.update_video_max_duration(panel)
    _fire(clock)

    assert ('VIDEO_MAX_DURATION_S', '99999') in emitted, (
        f'the refused entry left no record of what was typed: {emitted}'
    )
    assert ('VIDEO_MAX_DURATION_S', '300') not in emitted, (
        f'the reverted value was recorded as the one the user typed: {emitted}'
    )


def _declares_before_write(rel, class_name, method, record, widget_id):
    """True when note_write_back(record, ...) precedes a write to widget_id.text."""
    fn = find_def(rel, method, class_name=class_name)
    declare_at = write_at = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == 'note_write_back'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == record
                and (declare_at is None or node.lineno < declare_at)
            ):
                declare_at = node.lineno
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == 'text'
                    and widget_id in ast.unparse(t)
                    and (write_at is None or node.lineno < write_at)
                ):
                    write_at = node.lineno
    return declare_at, write_at


def test_a_refused_binning_pick_is_not_recorded_as_a_selection():
    """Restoring the spinner dispatches its event; the restore must be declared."""
    declare_at, write_at = _declares_before_write(
        'ui/microscope_settings.py',
        'MicroscopeSettings',
        'select_binning_size',
        'BINNING',
        'binning_spinner',
    )
    assert declare_at is not None, (
        'the binning reject path restores the spinner without declaring it, so a '
        'refused pick records a SELECT BINNING the user never chose'
    )
    assert write_at is not None and declare_at < write_at, (
        'the declaration must precede the spinner write -- a spinner dispatches '
        'synchronously, so a declaration made afterwards arrives too late'
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


def test_a_sanitized_capture_root_records_what_was_typed(clock, emitted, monkeypatch):
    """Typing a path-illegal name recorded only the sanitized result."""
    from ui.protocol_settings import ProtocolSettings

    widget = _Widget('my/run:1')

    class _Panel:
        ids: ClassVar[dict] = {'capture_root': widget}
        _protocol = None

    ProtocolSettings.update_capture_root(_Panel(), widget.text)
    ProtocolSettings.update_capture_root(_Panel(), widget.text)  # the echo
    _fire(clock)

    typed = [v for n, v in emitted if n == 'CAPTURE_ROOT']
    assert 'my/run:1' in typed, (
        f'only the sanitized value was recorded; what the user typed is gone: {emitted}'
    )
    applied = [v for n, v in emitted if n == 'CAPTURE_ROOT_APPLIED']
    assert applied, f'the sanitized value must still be reported as the correction: {emitted}'


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
