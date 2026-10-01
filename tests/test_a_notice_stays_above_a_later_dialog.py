# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A notice is never hidden under a dialog opened after it.

Kivy stacks popups by when they opened: the last on top, taking the
touches. In the simulator a failed "Use defaults" posted its notice and
asked the question again in the same frame, notice first, so the question
covered it -- the person saw only the question and a button that seemed to
do nothing, while the notices piled up underneath. At startup a plugin's
"did not load" notice was covered by the settings question the same way.

These drive the real open choke point against a stand-in window and
dialog class, because the suite's mocked Kivy has no real Popup: the
stand-in opens the way Kivy's ModalView does, by adding itself to the
window, which puts it at children[0], the top.
"""


class _Window:
    """Kivy's Window for the parts the stacking uses: children[0] is on top."""

    def __init__(self):
        self.children = []

    def add_widget(self, widget):
        assert widget not in self.children, 'Kivy refuses to add a widget that has a parent'
        self.children.insert(0, widget)

    def remove_widget(self, widget):
        self.children.remove(widget)


class _Dialog:
    """Shaped like Kivy's Popup for the parts the choke point touches."""

    window = None

    def __init__(self, title='', content=None, **_kw):
        self.title = title
        self.content = content
        self.children = ()
        self._window = None

    def open(self, *_a, **_kw):
        self._window = type(self).window
        self._window.add_widget(self)
        return self

    def dismiss(self, *_a, **_kw):
        self._window.remove_widget(self)

    def bind(self, **_kw):
        pass


class _Part:
    """A layout, label or button: the helpers only build and bind them."""

    def __init__(self, *_a, **kw):
        self.text = kw.get('text', '')
        self.children = []

    def add_widget(self, widget):
        self.children.append(widget)

    def bind(self, **_kw):
        pass


def _install(monkeypatch):
    from ui import notification_popup

    window = _Window()
    dialog_cls = type('_Dialog', (_Dialog,), {'window': window})
    monkeypatch.setattr(notification_popup, 'Popup', dialog_cls)
    for part in ('BoxLayout', 'Button', 'Label'):
        monkeypatch.setattr(notification_popup, part, _Part)
    monkeypatch.setattr(notification_popup, '_log_show', lambda *a: None)
    notification_popup._install_dialog_open_logging()
    return notification_popup, dialog_cls, window


def _titles(window):
    return [w.title for w in window.children]


def test_a_notice_stays_on_top_of_a_question_opened_after_it(monkeypatch):
    popups, _dialog, window = _install(monkeypatch)

    popups.show_notification_popup('Settings File Not Replaced', 'Permission denied')
    popups.show_confirmation_popup(
        title='Settings file could not be used',
        message='not valid JSON',
        confirm_text='Use defaults',
        cancel_text='Quit and repair',
        on_confirm=lambda: None,
    )

    assert _titles(window) == ['Settings File Not Replaced', 'Settings file could not be used']


def test_two_notices_keep_their_own_order_above_a_later_question(monkeypatch):
    popups, Dialog, window = _install(monkeypatch)

    popups.show_notification_popup('Plugin Not Loaded: alpha', 'a')
    popups.show_notification_popup('Plugin Not Loaded: beta', 'b')
    Dialog(title='Select Objective').open()

    assert _titles(window) == [
        'Plugin Not Loaded: beta',
        'Plugin Not Loaded: alpha',
        'Select Objective',
    ]


def test_a_question_with_no_notice_open_is_on_top_as_before(monkeypatch):
    _popups, Dialog, window = _install(monkeypatch)

    Dialog(title='Older dialog').open()
    Dialog(title='Newer dialog').open()

    assert _titles(window) == ['Newer dialog', 'Older dialog']


def test_a_dismissed_notice_is_not_brought_back(monkeypatch):
    popups, Dialog, window = _install(monkeypatch)

    notice = popups.show_notification_popup('Plugin Not Loaded: alpha', 'a')
    notice.dismiss()
    Dialog(title='Settings file could not be used').open()

    assert _titles(window) == ['Settings file could not be used']


def test_a_notice_opened_over_a_question_is_on_top_as_before(monkeypatch):
    popups, Dialog, window = _install(monkeypatch)

    Dialog(title='Settings file could not be used').open()
    popups.show_notification_popup('Settings File Not Replaced', 'Permission denied')

    assert _titles(window) == ['Settings File Not Replaced', 'Settings file could not be used']
