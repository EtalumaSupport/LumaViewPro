"""Regression: the notice that ends an operation replaces the one that began it.

A protocol run built one hyperstack in 0.9 s and left two stacked modal popups
that had to be dismissed in reverse order -- the "Saving Hyperstacks / this can
take several minutes" dialog was still on screen after the work had finished,
sitting on top of "Hyperstacks Saved".

The two notices are a pair describing one operation, but the bus had no way to
say so: each was an independent event, and the UI bridge rendered each into its
own popup while keeping no reference to what it had opened, so nothing could
ever close anything. The missing concept was supersession, not timing.

The failure path is covered here deliberately. A sim run exercises the happy
path far better than a fake popup surface can, but never produces a failure --
which makes the failure notice the call site most likely to be forgotten, and
the one a test has to hold.
"""

import pathlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ui.notification_popup builds real Kivy widgets at import time; conftest mocks
# `kivy` but not these submodules. The bridge under test never constructs a
# widget -- show_notification_popup is replaced below -- so permissive mocks
# are enough here.
for _name in (
    'kivy.clock',
    'kivy.uix',
    'kivy.uix.boxlayout',
    'kivy.uix.button',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.uix.scrollview',
    'kivy.metrics',
    'kivy.core.window',
):
    sys.modules.setdefault(_name, MagicMock())

if 'kivy.lang' not in sys.modules:
    _lang = ModuleType('kivy.lang')
    _lang.Builder = MagicMock()
    sys.modules['kivy.lang'] = _lang

import ui.notification_popup as notification_popup
from modules.notification_center import Notification, Severity


class _FakePopup:
    """Records dismissal the way a Kivy popup would be asked to."""

    def __init__(self, title):
        self.title = title
        self.dismissed = False

    def dismiss(self):
        self.dismissed = True


@pytest.fixture
def popup_surface(monkeypatch):
    """Replace popup construction and the thread hop, keeping the real bridge.

    Clock.schedule_once is run inline: the scheduling is Kivy's business, and
    what is under test is what the callback does when it runs.
    """
    opened = []

    def _fake_show(title, message):
        popup = _FakePopup(title)
        opened.append(popup)
        return popup

    monkeypatch.setattr(notification_popup, 'show_notification_popup', _fake_show)
    monkeypatch.setattr(notification_popup, '_operation_popups', {})

    fake_clock = SimpleNamespace(schedule_once=lambda cb, *a: cb(0))
    fake_module = ModuleType('kivy.clock')
    fake_module.Clock = fake_clock
    monkeypatch.setitem(sys.modules, 'kivy.clock', fake_module)

    return opened


def _notification(title, *, operation_key='', timestamp=1.0, severity=Severity.NOTICE):
    return Notification(
        severity=severity,
        category='Post-processing',
        title=title,
        message='body',
        timestamp=timestamp,
        operation_key=operation_key,
    )


KEY = 'post-processing:Hyperstack'


class TestOutcomeReplacesTheStartNotice:
    def test_failure_supersedes_the_start_notice(self, popup_surface):
        """The pin that catches forgetting the second call site.

        A failed build must replace the "please wait" dialog exactly as a
        successful one does -- and a normal sim run never gets here.
        """
        notification_popup.notification_popup_bridge(
            _notification('Saving Hyperstacks', operation_key=KEY, timestamp=1.0)
        )
        notification_popup.notification_popup_bridge(
            _notification(
                'Hyperstack Save Failed',
                operation_key=KEY,
                timestamp=2.0,
                severity=Severity.ERROR,
            )
        )

        start, failure = popup_surface
        assert start.dismissed, 'the "please wait" dialog outlived the work it described'
        assert not failure.dismissed

    def test_a_new_run_clears_the_previous_run_s_outcome_dialog(self, popup_surface):
        """One key per operation, so a second run supersedes the first's
        leftover dialog rather than stacking on it."""
        for i, title in enumerate(('Saving Hyperstacks', 'Hyperstacks Saved')):
            notification_popup.notification_popup_bridge(
                _notification(title, operation_key=KEY, timestamp=float(i))
            )
        notification_popup.notification_popup_bridge(
            _notification('Saving Hyperstacks', operation_key=KEY, timestamp=9.0)
        )

        assert [p.dismissed for p in popup_surface] == [True, True, False]


class TestUnkeyedNotificationsAreUntouched:
    def test_no_operation_key_behaves_exactly_as_before(self, popup_surface):
        """Every existing producer sends no key; none of them may change."""
        for title in ('Motor Fault', 'Camera Lost'):
            notification_popup.notification_popup_bridge(_notification(title))

        assert len(popup_surface) == 2
        assert not any(p.dismissed for p in popup_surface)


class TestOrderingCannotCorruptTheState:
    def test_a_late_older_notice_neither_dismisses_nor_displays(self, popup_surface):
        """Kivy's clock ships compiled, so callback ordering is not a
        guarantee this code can read. The notifications' own timestamps make
        the order irrelevant."""
        notification_popup.notification_popup_bridge(
            _notification('Hyperstacks Saved', operation_key=KEY, timestamp=5.0)
        )
        notification_popup.notification_popup_bridge(
            _notification('Saving Hyperstacks', operation_key=KEY, timestamp=1.0)
        )

        assert len(popup_surface) == 1, 'a stale start notice was put back on screen'
        assert not popup_surface[0].dismissed, 'a stale notice dismissed the newer popup'

    def test_dismissing_a_popup_the_user_already_closed_does_not_raise(self, popup_surface):
        notification_popup.notification_popup_bridge(
            _notification('Saving Hyperstacks', operation_key=KEY, timestamp=1.0)
        )
        popup_surface[0].dismiss()  # user clicked OK

        notification_popup.notification_popup_bridge(
            _notification('Hyperstacks Saved', operation_key=KEY, timestamp=2.0)
        )

        assert len(popup_surface) == 2


class TestBothEndsNameTheSameOperation:
    def test_start_and_outcome_share_one_key_owner(self):
        """The two ends deriving the key separately is how they drift apart."""
        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / 'modules'
            / 'protocol_post_processor.py'
        ).read_text()
        assert source.count('operation_key=self._unattended_operation_key') == 3, (
            'the start, completion and failure notices must all take the key '
            'from the one property; a hand-spelled key at any of the three '
            'drifts and the popup stops being replaced.'
        )
