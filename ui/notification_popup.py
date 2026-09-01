# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import typing

from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner, SpinnerOption

logger = logging.getLogger('LVP.ui.notification_popup')


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
    """Log every popup at the moment it is shown, to BOTH the main log (for post-mortem context)
    and the GUI-interactions log (for crash forensics). One entry per surface so a deployed
    customer log captures the full user-visible event."""
    # A popup opened before Kivy's event loop runs is painted UNDER the
    # app root once the root attaches -- created, "open", and invisible.
    # The open cannot be made illegal here (callers legitimately defer),
    # so make the state loud: mark both records and log at ERROR.
    pre_mainloop = False
    try:
        from kivy.base import EventLoop

        pre_mainloop = getattr(EventLoop, 'status', None) == 'idle'
    except Exception:
        pass
    if pre_mainloop:
        message = f'{message} (pre-mainloop)'
    try:
        from modules import gui_logger

        line = f'[Popup    ] show {kind} -- {gui_logger.one_line(title)}: {gui_logger.one_line(message)}'
    except Exception:
        line = f'[Popup    ] show {kind} -- {title}: {message}'
    if pre_mainloop:
        logger.error(line)
    else:
        logger.info(line)
    try:
        from modules import gui_logger

        gui_logger.notification(severity, title, message, source='popup')
    except Exception:
        pass


def _log_response(title: str, response: str):
    """Log the user's response (OK / Cancel / Ack / dismiss) to BOTH surfaces. Pairs with
    _log_show so post-mortem can tell what the user was looking at AND what they decided."""
    try:
        from modules import gui_logger

        logger.info(
            f'[Popup    ] response {gui_logger.one_line(response)} -- {gui_logger.one_line(title)}'
        )
    except Exception:
        logger.info(f'[Popup    ] response {response} -- {title}')
    try:
        from modules import gui_logger

        gui_logger.popup_response(title, response)
    except Exception:
        pass


def show_notification_popup(title: str, message: str):
    """Show a fire-and-forget popup with a single OK button.

    Args:
        title: Short noun phrase.
        message: One sentence on what happened plus one on what to do.

    Returns:
        The Kivy Popup instance, in case the caller needs to dismiss programmatically.
    """
    _log_show('notification', 'INFO', title, message)
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


def _show_superseding(n) -> None:
    """Open n's popup, replacing any popup still open for the same operation."""
    key = n.operation_key
    if not key:
        show_notification_popup(title=n.title, message=n.message)
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

    _operation_popups[key] = (
        show_notification_popup(title=n.title, message=n.message),
        n.timestamp,
    )


def notification_popup_bridge(n) -> None:
    """Render a notification as a popup, on the Kivy thread.

    Notification listeners run on whichever thread produced the notification,
    so the work hops to the main thread here.
    """
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
    _log_show('confirm_ack', 'INFO', title, message)
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
    _log_show('progress', 'INFO', title, message)
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
    """
    _log_show('confirm', 'INFO', title, message)
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


def show_objective_selection_popup(
    title: str,
    message: str,
    objectives: list[str],
    current_objective_id: str,
    on_confirm: typing.Callable[[str], None],
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
        on_confirm: Called with the chosen objective id.
    """
    global _objective_popup_open
    if _objective_popup_open:
        logger.info('[Popup    ] objective prompt already open -- second request dropped')
        return
    _log_show('objective_select', 'INFO', title, message)
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(_make_message_label(message))

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

    confirm_button = Button(text='Confirm', size_hint_y=None, height='34dp')
    content.add_widget(confirm_button)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.4, 0.32),
        auto_dismiss=False,
    )

    def _on_confirm(*_a):
        _log_response(title, f'OBJECTIVE:{spinner.text}')
        popup.dismiss()
        on_confirm(spinner.text)

    def _on_dismiss(*_a):
        global _objective_popup_open
        _objective_popup_open = False

    confirm_button.bind(on_release=_on_confirm)
    # Dismiss (which confirm fires before applying the answer) clears the
    # outstanding-question flag, so a failure while applying cannot leave
    # the app unable to ever ask again. The flag is set only after a
    # successful open for the same reason in the other direction.
    popup.bind(on_dismiss=_on_dismiss)
    popup.open()
    _objective_popup_open = True
