# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import typing

from kivy.base import EventLoop
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.uix.widget import Widget

from modules import gui_logger

logger = logging.getLogger('LVP.ui.notification_popup')


def _describe_dialog_body(popup) -> str:
    """Best-effort one-line summary of what a dialog is showing.

    Walks the content tree for Label text. A dialog built from arbitrary
    widgets may have nothing to say, and that is fine -- the title plus the
    fact that it opened is the load-bearing part.
    """
    texts: list[str] = []

    def _walk(widget, depth=0):
        if widget is None or depth > 4 or len(texts) >= 3:
            return
        # The widget's OWN text first: the commonest dialog puts a single
        # Label in as its content, so walking only the children reads past
        # the one string that says what the dialog was about.
        text = getattr(widget, 'text', None)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        for child in getattr(widget, 'children', ()):
            _walk(child, depth + 1)

    _walk(getattr(popup, 'content', None) or popup)
    return ' | '.join(texts) if texts else '(no text content)'


def _install_dialog_open_logging() -> None:
    """Make every dialog log itself when it opens, from one place.

    A dialog is a thing the user SAW, so a support bundle has to be able to
    answer what was on screen. Logging at the call sites cannot deliver
    that: it is a rule every new dialog has to remember, and the ones that
    forgot were missing from exactly the bundles where it mattered.

    ``open()`` is the choke point -- a dialog that never opens was never
    seen, and one that opens cannot avoid this, whoever built it and
    however they built it. So the statement lives here once instead of at
    each surface, and a dialog added tomorrow is logged without anyone
    remembering to do it.

    The title is read off the widget rather than taken as an argument,
    because the caller is not involved at all.

    The same choke point keeps a notice readable: see _raise_open_notices.
    """
    if getattr(Popup, '_lvp_open_logged', False):
        return
    if not hasattr(Popup, 'open'):
        # The mocked-Kivy test environment substitutes a stub widget with no
        # open(). There is no dialog to log because there is no dialog; the
        # real class always has it, so this cannot mask a production gap.
        logger.debug('dialog-open logging not installed: Popup has no open()')
        return
    original_open = Popup.open

    def _open_logged_and_ordered(self, *args, **kwargs):
        title = getattr(self, 'title', '') or type(self).__name__
        _log_show('dialog', 'INFO', title, _describe_dialog_body(self))
        opened = original_open(self, *args, **kwargs)
        if not getattr(self, _NOTICE_MARK, False):
            _raise_open_notices(getattr(self, '_window', None))
        return opened

    Popup.open = _open_logged_and_ordered
    Popup._lvp_open_logged = True


# Set on a popup that says what just happened and asks nothing of the person.
_NOTICE_MARK = '_lvp_notice'


def _raise_open_notices(window) -> None:
    """Put every notice still open back on top of the dialog that just opened.

    Kivy stacks popups by when they opened, the last on top and taking the
    touches. A notice is what the person reads first -- what just happened
    -- so a question opened after it, even in the same frame, must not
    cover it: the person sees only the question, the notice's news is lost
    underneath, and a failing answer looks like a dead button.

    The window's own children are the record of what is open, so nothing
    else tracks the notices. They go back oldest first, leaving the newest
    on top. A dialog with no window (a stand-in, or not yet attached) has
    nothing above it to fix.
    """
    if window is None:
        return
    notices = [w for w in window.children if getattr(w, _NOTICE_MARK, False)]
    # children[0] is the top, so the list runs newest to oldest.
    for notice in reversed(notices):
        window.remove_widget(notice)
        window.add_widget(notice)


_install_dialog_open_logging()


def _make_message_label(message: str) -> Label:
    """Build the popup-body Label with word-wrap enabled.

    A bare ``Label(text=...)`` does not wrap -- Kivy renders the whole
    string on one line, which overflows the popup horizontally for
    multi-sentence messages and visually leaks across the underlying UI.
    Binding ``text_size`` to the Label's own width forces a wrap to the
    Label's allocated width; centered halign + valign keeps the wrapped
    paragraph balanced in the popup body.
    """
    label = Label(text=message, halign='center', valign='middle')

    def _resize(_lbl, _size):
        label.text_size = (label.width, label.height)

    label.bind(size=_resize)
    return label


def _log_show(kind: str, severity: str, title: str, message: str):
    """Log a dialog at the moment it is shown, to BOTH the main log (for post-mortem context)
    and the GUI-interactions log (for crash forensics), so a deployed customer log captures the
    full user-visible event. Reached from one place, the patched ``Popup.open``."""
    # A popup opened before Kivy's event loop runs is painted UNDER the
    # app root once the root attaches -- created, "open", and invisible.
    # The open cannot be made illegal here (callers legitimately defer),
    # so make the state loud: mark both records and log at ERROR.
    pre_mainloop = getattr(EventLoop, 'status', None) == 'idle'
    if pre_mainloop:
        message = f'{message} (pre-mainloop)'
    line = (
        f'[Popup    ] show {kind} -- {gui_logger.one_line(title)}: {gui_logger.one_line(message)}'
    )
    if pre_mainloop:
        logger.error(line)
    else:
        logger.info(line)
    gui_logger.notification(severity, title, message, source='popup')


def _log_response(title: str, response: str):
    """Log the user's response (OK / Cancel / Ack / dismiss) to BOTH surfaces. Pairs with
    _log_show so post-mortem can tell what the user was looking at AND what they decided."""
    logger.info(
        f'[Popup    ] response {gui_logger.one_line(response)} -- {gui_logger.one_line(title)}'
    )
    gui_logger.popup_response(title, response)


def show_notification_popup(title: str, message: str):
    """Show a fire-and-forget popup with a single OK button.

    Args:
        title: Short noun phrase.
        message: One sentence on what happened plus one on what to do.

    Returns:
        The Kivy Popup instance, in case the caller needs to dismiss programmatically.
    """
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(_make_message_label(message))

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    button_layout.add_widget(Label())  # spacer
    ok_button = Button(text='OK', size_hint_x=None, width='100dp')
    button_layout.add_widget(ok_button)
    button_layout.add_widget(Label())  # spacer
    content.add_widget(button_layout)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
    )
    setattr(popup, _NOTICE_MARK, True)

    def _on_ok(*_a):
        _log_response(title, 'OK')
        popup.dismiss()

    ok_button.bind(on_release=_on_ok)
    popup.open()
    return popup


# Popups opened on behalf of a named operation, as {key: (popup, timestamp)}.
# A long unattended job announces its start and then its outcome; without a
# reference to what the start opened, the outcome could only stack a second
# modal on top of a "please wait" dialog describing work that had finished.
#
# Touched only from the Kivy thread, inside the scheduled callback below --
# the Popup does not exist until then, which is also why the bookkeeping
# cannot live in the listener. Single-threaded access, so no lock.
_operation_popups: dict = {}


def _open(n):
    """Open n as the popup its record asks for, and return it.

    A notification that carries a remedy is an offer: its confirm asks the
    Session to take the remedy, so what taking it does is the Session's
    answer, reported like any other request. Everything else is a notice.
    """
    if n.remedy is None:
        return show_notification_popup(title=n.title, message=n.message)
    remedy = n.remedy

    def _take_remedy():
        import modules.app_context as _app_ctx
        from ui.ui_helpers import submit_reported

        session = _app_ctx.ctx.session
        submit_reported(lambda: session.apply_remedy(remedy), None, 'APPLY_REMEDY')

    return show_confirmation_popup(
        title=n.title,
        message=n.message,
        confirm_text=remedy.confirm_text,
        cancel_text=remedy.cancel_text,
        on_confirm=_take_remedy,
    )


def _show_superseding(n) -> None:
    """Open n's popup, replacing any popup still open for the same operation."""
    key = n.operation_key
    if not key:
        _open(n)
        return

    recorded = _operation_popups.get(key)
    if recorded is not None:
        popup, recorded_timestamp = recorded
        # Kivy's clock core ships compiled, so the order in which two
        # zero-delay callbacks are applied is not something this code can
        # read and rely on. Comparing the notifications' own timestamps makes
        # the order irrelevant: an older notice arriving late neither
        # dismisses the newer popup nor puts its own stale message back up.
        if recorded_timestamp > n.timestamp:
            return
        popup.dismiss()  # a no-op if the user already closed it by hand
        del _operation_popups[key]

    _operation_popups[key] = (_open(n), n.timestamp)


def notification_popup_bridge(n) -> None:
    """Render a notification as a popup, on the Kivy thread.

    Notification listeners run on whichever thread produced the notification,
    so the work hops to the main thread here. Only a notification the API
    says is shown opens a popup; the others are dropped before they reach the
    supersession map, so an unshown one never replaces a shown one.
    """
    if not n.shown:
        return
    from kivy.clock import Clock

    Clock.schedule_once(lambda dt: _show_superseding(n), 0)


def show_confirmation_w_ack_popup(
    title: str, message: str, ack_button_text: str, on_ack: typing.Callable
):
    """Show a popup with one acknowledgement button that fires a callback.

    Args:
        title: Short noun phrase.
        message: What happened + what to do.
        ack_button_text: Label for the single button (e.g. "Continue").
        on_ack: Called with no args when the user clicks the button.
    """
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(_make_message_label(message))

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    ack_button = Button(text=ack_button_text)

    button_layout.add_widget(ack_button)
    content.add_widget(button_layout)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
    )

    def _on_ack(*_a):
        _log_response(title, f'ACK:{ack_button_text}')
        on_ack()
        popup.dismiss()

    ack_button.bind(on_release=_on_ack)
    popup.open()


def show_blocking_progress_popup(
    title: str,
    message: str,
    action_text: str,
    on_action: typing.Callable,
):
    """Modal progress popup: a live-updatable message + one action button.

    For operations that must visibly finish before the app can proceed
    (e.g. draining queued video writes at app close): silent blocking
    reads as a hang and silent abandonment eats data, so the popup shows
    progress and offers exactly one explicit escape action.

    Args:
        title: Short noun phrase.
        message: Initial progress text; update via the returned setter.
        action_text: The escape action's button label.
        on_action: Called with no args when the user clicks the action.

    Returns:
        (popup, set_message) -- dismiss the popup from the caller's own
        completion path; set_message(text) updates the progress line.
    """
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    label = _make_message_label(message)
    content.add_widget(label)

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    action_button = Button(text=action_text)
    button_layout.add_widget(action_button)
    content.add_widget(button_layout)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
        auto_dismiss=False,
    )

    def _on_action(*_a):
        _log_response(title, f'ACTION:{action_text}')
        on_action()

    def _set_message(text: str):
        label.text = text

    action_button.bind(on_press=_on_action)
    popup.open()
    return popup, _set_message


def show_confirmation_popup(
    title: str,
    message: str,
    confirm_text: str,
    cancel_text: str,
    on_confirm: typing.Callable,
    on_cancel: typing.Callable | None = None,
):
    """Show a blocking modal popup with confirm + cancel buttons.

    Canonical path for setup prompts and OK/Cancel decisions. Plugin and shipping-LVP modal
    prompts route through this helper rather than building raw Kivy popups.

    Args:
        title: Short noun phrase.
        message: What happened + what to do.
        confirm_text: Confirm-button label (e.g. "OK", "Continue").
        cancel_text: Cancel-button label (e.g. "Cancel", "Abort").
        on_confirm: Called with no args when the user clicks confirm.
        on_cancel: Optional. Called with no args when the user clicks cancel. If omitted, cancel
            just dismisses the popup. Useful for blocking-with-return-value adapters that need
            to release a worker thread on either path.

    Returns:
        The opened popup, so a caller can dismiss it when what it asks about
        is overtaken.
    """
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(_make_message_label(message))

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    yes_button = Button(text=confirm_text)
    no_button = Button(text=cancel_text)

    button_layout.add_widget(no_button)
    button_layout.add_widget(yes_button)
    content.add_widget(button_layout)

    # auto_dismiss=False -> click-outside-popup does NOT dismiss. The
    # docstring promises "blocking modal" and callers (e.g.
    # engineering_tab._prompt_confirm) block their worker thread on
    # an Event waiting for one of the button callbacks. A click-out
    # dismiss without callback leaves the worker hung forever, which
    # bit a char run on Windows 2026-05-08 (cap-on prompt dismissed
    # by accidental click-out, char tool stopped silently).
    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
        auto_dismiss=False,
    )

    # Track whether either button fired so on_dismiss can defensively
    # fire the cancel callback if some other code path dismisses
    # programmatically (atexit, app shutdown, future Kivy lifecycle).
    decided = {'value': False}

    def _on_confirm(*_a):
        decided['value'] = True
        _log_response(title, f'CONFIRM:{confirm_text}')
        on_confirm()
        popup.dismiss()

    def _on_cancel(*_a):
        decided['value'] = True
        _log_response(title, f'CANCEL:{cancel_text}')
        if on_cancel is not None:
            on_cancel()
        popup.dismiss()

    def _on_dismiss(*_a):
        # Defensive: if neither button was clicked but the popup
        # dismissed anyway (programmatic dismiss / lifecycle), treat
        # as cancel so blocked worker threads release.
        if decided['value']:
            return
        _log_response(title, 'CANCEL:dismissed_without_button')
        if on_cancel is not None:
            on_cancel()

    yes_button.bind(on_release=_on_confirm)
    no_button.bind(on_release=_on_cancel)
    popup.bind(on_dismiss=_on_dismiss)

    popup.open()
    return popup


class _CompactSpinnerOption(SpinnerOption):
    def __init__(self, **kwargs):
        kwargs.setdefault('font_size', '12sp')
        super().__init__(**kwargs)


class _CappedDropDown(DropDown):
    def __init__(self, **kwargs):
        kwargs.setdefault('max_height', dp(280))
        super().__init__(**kwargs)


# At most one objective question may be outstanding: the prompt is
# cancel-less and every trigger asks the same thing, so a second popup
# would stack a duplicate whose answer silently overwrites the first.
# Cleared by the popup's own dismiss, which fires before the answer is
# applied.
_objective_popup_open = False
# Continuations belonging to requests folded into the prompt already on
# screen. Showing one dialog instead of two is right; losing the second
# caller's work is not.
_objective_popup_folded: list[typing.Callable[[], None]] = []


def show_objective_selection_popup(
    title: str,
    message: str,
    objectives: list[str],
    current_objective_id: str,
    on_confirm: typing.Callable[[str], None],
    on_folded: typing.Callable[[], None],
):
    """Modal prompt asking the user which objective is in the light path.

    The pixel size derived from the objective is written into the scale
    bar and every saved image's metadata, and a wrong one cannot be told
    from a measured one afterwards -- so when the app cannot know the
    objective (first run, or a turret position with no assignment), it
    asks instead of assuming silently. There is deliberately no cancel
    path: dismissing without answering would put the app right back in
    the cannot-know state the prompt exists to resolve, so the popup
    stays until the user confirms a choice.

    Args:
        title: Short noun phrase.
        message: What happened + what confirming commits to.
        objectives: Selectable objective ids, in display order.
        current_objective_id: Pre-selected value (the proposed default).
        on_confirm: Called with the chosen objective id: applies the answer.
        on_folded: This request's continuation, run instead of ``on_confirm``
            when the request is folded into a prompt already on screen.
            The answer is applied once, by the prompt on screen, to the
            question it shows; applying it again for a folded request
            wrote one answer into two slots when the turret had moved
            between the two requests.
    """
    global _objective_popup_open, _objective_popup_folded
    if _objective_popup_open:
        # Every trigger asks the same question -- objective_question() takes
        # no arguments and is a pure read of the store -- so the prompt on
        # screen IS this caller's prompt, and a second cancel-less modal
        # would stack a duplicate whose answer overwrites the first. The
        # caller's continuation is a different matter: at startup the folded
        # caller is the one carrying the persisted-protocol load, so simply
        # returning meant the saved protocol silently never loaded, on every
        # turreted scope whose current slot is unassigned.
        logger.info('[Popup    ] objective prompt already open -- request folded into it')
        _objective_popup_folded.append(on_folded)
        return
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    # The message is sized to its text, and three equal stretches -- above
    # it, between it and the spinner, between the spinner and Confirm --
    # take the spare height, so the three gaps stay equal whatever size
    # the window gives the popup. A message that takes the spare height
    # itself centres its text in it and leaves the spinner against Confirm.
    message_label = Label(text=message, halign='center', size_hint_y=None)
    message_label.bind(
        width=lambda lbl, width: setattr(lbl, 'text_size', (width, None)),
        texture_size=lambda lbl, size: setattr(lbl, 'height', size[1]),
    )
    content.add_widget(Widget())
    content.add_widget(message_label)
    content.add_widget(Widget())

    # A Spinner built in Python gets Kivy's stock dropdown: full-height
    # option rows and an uncapped list, which for a dozen objectives
    # fills the whole screen. Sync the rows to the spinner's own height
    # (the same mechanism the kv spinners use), match their option font,
    # and cap the dropdown so a long catalogue scrolls instead of
    # growing.
    # Width is constrained too, not just height: an unconstrained Spinner
    # in this layout takes size_hint_x=1 and stretches to the whole popup,
    # which is a control several times wider than the longest objective
    # name it ever shows. The dropdown inherits the spinner's width, so
    # this sizes both. Centred because a fixed width in a vertical
    # BoxLayout otherwise pins to the left edge.
    spinner = Spinner(
        text=current_objective_id,
        values=objectives,
        size_hint=(None, None),
        width='220dp',
        height='30dp',
        pos_hint={'center_x': 0.5},
        font_size='12sp',
        sync_height=True,
        option_cls=_CompactSpinnerOption,
        dropdown_cls=_CappedDropDown,
    )
    content.add_widget(spinner)
    content.add_widget(Widget())

    confirm_button = Button(text='Confirm', size_hint_y=None, height='34dp')
    content.add_widget(confirm_button)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.4, 0.32),
        auto_dismiss=False,
    )

    def _on_confirm(*_a):
        global _objective_popup_folded
        chosen = spinner.text
        _log_response(title, f'OBJECTIVE:{chosen}')
        # Taken BEFORE the dismiss below, which clears the list: every
        # caller that asked is owed its continuation, the folded ones
        # included -- after the answer has been applied, once.
        folded, _objective_popup_folded = _objective_popup_folded, []
        popup.dismiss()
        on_confirm(chosen)
        for continuation in folded:
            continuation()

    def _on_dismiss(*_a):
        global _objective_popup_open, _objective_popup_folded
        _objective_popup_open = False
        # A prompt that goes away without being answered takes its folded
        # continuations with it. Left behind, one would fire on the NEXT
        # prompt's answer -- a startup step run against a question nobody
        # asked it about.
        _objective_popup_folded = []

    confirm_button.bind(on_release=_on_confirm)
    # Dismiss (which confirm fires before applying the answer) clears the
    # outstanding-question flag, so a failure while applying cannot leave
    # the app unable to ever ask again. The flag is set only after a
    # successful open for the same reason in the other direction.
    popup.bind(on_dismiss=_on_dismiss)
    popup.open()
    _objective_popup_open = True
