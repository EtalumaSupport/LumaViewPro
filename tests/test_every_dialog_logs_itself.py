# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every dialog logs itself, and nobody has to remember to make it.

Eric, 2026-09-15: *"i want all dialogs logged. period."* and *"i would like
them to be all of the same, so there is only one logging statement in there,
not 60."*

Before this, the record was written by the wrapper functions in
`ui.notification_popup`. A dialog built any other way -- and six were, in
`microscope_settings`, `post_processing` and `progress_popup` -- opened with
no record at all, so a support bundle could not say what the user had been
looking at. Adding the call to each of those is the shape that produced the
gap in the first place: a rule every new dialog has to remember.

`Popup.open` is the choke point. A dialog that never opens was never seen,
and one that opens cannot avoid it.

These drive the real installer against a stand-in dialog class, because the
suite's mocked Kivy substitutes a stub widget with no ``open()`` and the
production class is not constructible here.
"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault('modules.settings_init', MagicMock())


class _FakeLabel:
    def __init__(self, text):
        self.text = text
        self.children = ()


class _FakeDialog:
    """Shaped like Kivy's Popup for the parts the logger touches."""

    def __init__(self, title='', content=None):
        self.title = title
        self.content = content
        self.children = ()
        self.opened = False

    def open(self, *_a, **_kw):
        self.opened = True
        return self


def _install_against(monkeypatch):
    """Run the production installer over a FRESH stand-in class.

    A fresh subclass per test: the installer wraps the class's own open(),
    and a shared class would carry every previous test's wrapper.
    """
    from ui import notification_popup

    dialog_cls = type('_FakeDialog', (_FakeDialog,), {})
    monkeypatch.setattr(notification_popup, 'Popup', dialog_cls)
    notification_popup._install_dialog_open_logging()
    return notification_popup, dialog_cls


def test_a_dialog_nobody_instrumented_still_logs(monkeypatch):
    """The whole point: a dialog built from scratch and opened directly.

    No wrapper, no helper, no call-site logging statement -- the shape that
    was previously invisible in a support bundle.
    """
    recorded = []
    notification_popup, Dialog = _install_against(monkeypatch)
    monkeypatch.setattr(notification_popup, '_log_show', lambda *a: recorded.append(a))

    dialog = Dialog(title='Zip Logs', content=_FakeLabel('Collecting logs...'))
    dialog.open()

    assert dialog.opened, 'the dialog must still open'
    assert recorded, 'a dialog opened with no record -- the bundle cannot say what was shown'
    kind, _severity, title, body = recorded[0]
    assert kind == 'dialog'
    assert title == 'Zip Logs', f'the record must name the dialog the user saw: {title!r}'
    assert 'Collecting logs' in body, f'the record must carry what it said: {body!r}'


def test_a_titleless_dialog_is_still_identifiable(monkeypatch):
    """A record saying only that 'something' opened is not worth writing."""
    recorded = []
    notification_popup, Dialog = _install_against(monkeypatch)
    monkeypatch.setattr(notification_popup, '_log_show', lambda *a: recorded.append(a))

    Dialog(title='').open()

    assert recorded[0][2] == '_FakeDialog', (
        'a dialog with no title must fall back to its class name, not an empty string'
    )


def test_logging_failure_never_blocks_the_dialog(monkeypatch):
    """A user who cannot be shown a message is worse off than a missing line."""
    notification_popup, Dialog = _install_against(monkeypatch)

    def _boom(*_a):
        raise RuntimeError('forensic log unavailable')

    monkeypatch.setattr(notification_popup, '_log_show', _boom)

    dialog = Dialog(title='Still Opens')
    dialog.open()
    assert dialog.opened, 'a failure writing the record must not stop the user seeing the dialog'


def test_installing_twice_does_not_double_log(monkeypatch):
    """Import order must not decide how many records a dialog produces."""
    recorded = []
    notification_popup, Dialog = _install_against(monkeypatch)
    monkeypatch.setattr(notification_popup, '_log_show', lambda *a: recorded.append(a))
    notification_popup._install_dialog_open_logging()  # second call, must be a no-op

    Dialog(title='Once').open()
    assert len(recorded) == 1, f'the dialog logged {len(recorded)} times, expected 1'


def test_the_logging_statement_exists_exactly_once():
    """One statement, not one per surface.

    The previous design had five, one in each wrapper, and every dialog
    outside those wrappers had none. If this count grows, the per-call-site
    pattern is back and the next dialog will be the one that forgets.
    """
    import inspect

    from ui import notification_popup

    src = inspect.getsource(notification_popup)
    calls = src.count('_log_show(')
    assert calls == 2, (
        f'expected exactly one definition and one call of _log_show, found {calls} '
        'occurrences -- a per-surface logging statement has come back'
    )


def test_the_body_summary_survives_a_contentless_dialog():
    """A dialog of arbitrary widgets has nothing to say, and that is fine."""
    from ui.notification_popup import _describe_dialog_body

    class _Bare:
        content = None
        children = ()

    assert _describe_dialog_body(_Bare()) == '(no text content)'
