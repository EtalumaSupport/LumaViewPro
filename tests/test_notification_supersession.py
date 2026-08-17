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

import ast
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.ast_seams import parse_module

# ui.notification_popup imports Kivy widget classes at module scope; conftest
# mocks `kivy` but not these submodules.
#
# These are REAL stub classes, not MagicMocks, and that distinction is
# load-bearing across the whole suite: sys.modules entries planted here
# outlive this module, and ui/file_dialogs.py does
# `class FileChooseBTN(HoverBehavior, Button)`. A MagicMock Button makes that
# a metaclass conflict at import time, which surfaces as errors in whichever
# unrelated test file happens to import file_dialogs later.


class _StubWidget:
    def __init__(self, *args, **kwargs):
        pass


for _name, _attr in (
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.button', 'Button'),
    ('kivy.uix.label', 'Label'),
    ('kivy.uix.popup', 'Popup'),
):
    if _name not in sys.modules:
        _module = ModuleType(_name)
        setattr(_module, _attr, type(_attr, (_StubWidget,), {}))
        sys.modules[_name] = _module

sys.modules.setdefault('kivy.uix', MagicMock())

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
        """The two ends deriving the key separately is how they drift apart.

        Walks the AST rather than the text, so a reformatted or line-wrapped
        call still counts and a mention inside a comment does not.
        """
        tree = parse_module('modules/protocol_post_processor.py')
        from_the_property = 0
        hand_spelled = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != 'operation_key':
                    continue
                value = keyword.value
                if isinstance(value, ast.Attribute) and value.attr == '_unattended_operation_key':
                    from_the_property += 1
                else:
                    hand_spelled.append(node.lineno)

        assert not hand_spelled, (
            f'operation_key spelled by hand at line(s) {hand_spelled}. Both ends '
            'must take it from the one property, or the outcome notice stops '
            'matching the start notice and opens a second popup instead.'
        )
        assert from_the_property == 3, (
            f'expected the start, completion and failure notices to carry the '
            f'key; found {from_the_property}. The failure path is the one that '
            'gets forgotten, and no sim run reaches it.'
        )
