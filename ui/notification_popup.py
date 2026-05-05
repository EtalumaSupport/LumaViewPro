# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import logging
import typing

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup

logger = logging.getLogger('LVP.ui.notification_popup')


def _log_show(kind: str, severity: str, title: str, message: str):
    """Log every popup at the moment it is shown, to BOTH the main log (for post-mortem context)
    and the GUI-interactions log (for crash forensics). One entry per surface so a deployed
    customer log captures the full user-visible event."""
    logger.info(f'[Popup    ] show {kind} -- {title}: {message}')
    try:
        from modules import gui_logger

        gui_logger.notification(severity, title, message, source='popup')
    except Exception:
        pass


def _log_response(title: str, response: str):
    """Log the user's response (OK / Cancel / Ack / dismiss) to BOTH surfaces. Pairs with
    _log_show so post-mortem can tell what the user was looking at AND what they decided."""
    logger.info(f'[Popup    ] response {response} -- {title}')
    try:
        from modules import gui_logger

        gui_logger.popup_response(title, response)
    except Exception:
        pass


def show_notification_popup(title: str, message: str):
    """Show a fire-and-forget popup with a single OK button.

    Args:
        title: Short noun phrase per Rule 28 voice.
        message: One sentence on what happened plus one on what to do.

    Returns:
        The Kivy Popup instance, in case the caller needs to dismiss programmatically.
    """
    _log_show('notification', 'INFO', title, message)
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(Label(text=message))

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


def show_confirmation_w_ack_popup(
    title: str, message: str, ack_button_text: str, on_ack: typing.Callable
):
    """Show a popup with one acknowledgement button that fires a callback.

    Args:
        title: Short noun phrase per Rule 28 voice.
        message: What happened + what to do.
        ack_button_text: Label for the single button (e.g. "Continue").
        on_ack: Called with no args when the user clicks the button.
    """
    _log_show('confirm_ack', 'INFO', title, message)
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(Label(text=message))

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
        title: Short noun phrase per Rule 28 voice.
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
    content.add_widget(Label(text=message))

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    yes_button = Button(text=confirm_text)
    no_button = Button(text=cancel_text)

    button_layout.add_widget(no_button)
    button_layout.add_widget(yes_button)
    content.add_widget(button_layout)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
    )

    def _on_confirm(*_a):
        _log_response(title, f'CONFIRM:{confirm_text}')
        on_confirm()
        popup.dismiss()

    def _on_cancel(*_a):
        _log_response(title, f'CANCEL:{cancel_text}')
        if on_cancel is not None:
            on_cancel()
        popup.dismiss()

    yes_button.bind(on_release=_on_confirm)
    no_button.bind(on_release=_on_cancel)

    popup.open()
