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
        tree = parse_module('modules/stack_builder.py')
        (boundary,) = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'build_hyperstacks_for_run'
        ]
        key_names = {
            target.id
            for node in ast.walk(boundary)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == 'operation_key'
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        from_the_property = 0
        hand_spelled = []
        for node in ast.walk(boundary):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != 'operation_key':
                    continue
                value = keyword.value
                if isinstance(value, ast.Name) and value.id in key_names:
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


class TestARefusalThatNamesItsRemedyIsAnOffer:
    """A notification carrying a remedy is shown as a confirmation whose confirm
    asks the Session to apply it; the bridge renders the record and decides
    nothing. It is a refusal popup like any other, so the next refusal replaces
    it rather than stacking on it."""

    @pytest.fixture
    def offers(self, popup_surface, monkeypatch):
        opened = []

        def _fake_confirm(title, message, confirm_text, cancel_text, on_confirm, on_cancel=None):
            popup = _FakePopup(title)
            popup.confirm_text = confirm_text
            popup.cancel_text = cancel_text
            popup.confirm = on_confirm
            opened.append(popup)
            return popup

        monkeypatch.setattr(notification_popup, 'show_confirmation_popup', _fake_confirm)
        return opened

    @staticmethod
    def _refusal(title, *, timestamp, remedy=None):
        from modules.notification_center import REFUSAL_OPERATION_KEY

        return Notification(
            severity=Severity.WARNING,
            category='Protocol',
            title=title,
            message='body',
            timestamp=timestamp,
            operation_key=REFUSAL_OPERATION_KEY,
            solicited=True,
            remedy=remedy,
        )

    def test_the_offer_s_confirm_applies_the_remedy_through_the_session(self, offers, monkeypatch):
        import modules.app_context as app_context
        import ui.ui_helpers as ui_helpers
        from modules.exceptions import Remedy

        remedy = Remedy(
            member='recover_file_writer',
            confirm_text='Discard 3 unsaved and unlock',
            cancel_text='Keep waiting',
        )
        session = MagicMock()
        monkeypatch.setattr(app_context, 'ctx', SimpleNamespace(session=session), raising=False)
        submitted = []
        monkeypatch.setattr(
            ui_helpers,
            'submit_reported',
            lambda call, redraw, label, **kw: submitted.append((call, label)),
        )

        notification_popup.notification_popup_bridge(
            self._refusal('File Writer Stalled', timestamp=1.0, remedy=remedy)
        )

        (offer,) = offers
        assert (offer.confirm_text, offer.cancel_text) == (
            'Discard 3 unsaved and unlock',
            'Keep waiting',
        )
        offer.confirm()
        ((call, _label),) = submitted
        call()
        session.apply_remedy.assert_called_once_with(remedy)

    def test_the_next_refusal_replaces_the_offer(self, offers, popup_surface):
        from modules.exceptions import Remedy

        remedy = Remedy(member='recover_file_writer', confirm_text='Go', cancel_text='Wait')
        notification_popup.notification_popup_bridge(
            self._refusal('File Writer Stalled', timestamp=1.0, remedy=remedy)
        )
        notification_popup.notification_popup_bridge(
            self._refusal('Files Still Writing', timestamp=2.0)
        )

        (offer,) = offers
        assert offer.dismissed, 'the offer outlived the refusal that replaced it'
        assert len(popup_surface) == 1 and not popup_surface[0].dismissed


class TestTheConfirmationHandsBackWhatItOpens:
    def test_the_popup_returned_is_the_one_opened(self, monkeypatch):
        """The supersession bookkeeping dismisses what the opener returned; a
        confirmation that returned nothing would be dismissed as None by the
        next refusal, raising inside the clock callback."""

        opened = []

        class _Widget:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def add_widget(self, widget):
                pass

            def bind(self, **kwargs):
                pass

            def open(self):
                opened.append(self)

        for name in ('Label', 'BoxLayout', 'Button', 'Popup'):
            monkeypatch.setattr(notification_popup, name, _Widget)

        returned = notification_popup.show_confirmation_popup(
            title='File Writer Stalled',
            message='body',
            confirm_text='Go',
            cancel_text='Wait',
            on_confirm=lambda: None,
        )

        assert opened == [returned]


def test_a_notification_the_api_did_not_show_opens_nothing(popup_surface):
    """Every listener hears a muted post, so the bridge receives it too; it
    opens only what the API says is shown, and a muted one never replaces a
    shown one."""
    import dataclasses

    notification_popup.notification_popup_bridge(
        _notification('Saving Hyperstacks', operation_key=KEY, timestamp=1.0)
    )
    muted = dataclasses.replace(
        _notification(
            'Hyperstack Save Failed', operation_key=KEY, timestamp=2.0, severity=Severity.ERROR
        ),
        shown=False,
    )
    notification_popup.notification_popup_bridge(muted)

    assert [popup.title for popup in popup_surface] == ['Saving Hyperstacks']
    assert not popup_surface[0].dismissed
