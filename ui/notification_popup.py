# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import typing

from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


def _log_to_gui_interactions(kind: str, title: str, message: str):
    """Best-effort forensics log of every popup the user sees."""
    try:
        from modules import gui_logger
        gui_logger.notification(kind, title, message, source="popup")
    except Exception:
        pass


def show_notification_popup(title: str, message: str):
    _log_to_gui_interactions("INFO", title, message)
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
    ok_button.bind(on_release=popup.dismiss)
    popup.open()
    return popup


def show_confirmation_w_ack_popup(title: str, message: str, ack_button_text: str, on_ack: typing.Callable):
    _log_to_gui_interactions("CONFIRM_ACK", title, message)
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

    ack_button.bind(on_release=lambda *args: (on_ack(), popup.dismiss()))

    popup.open()


def show_confirmation_popup(title: str, message: str, confirm_text: str, cancel_text: str, on_confirm):
    _log_to_gui_interactions("CONFIRM", title, message)
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

    yes_button.bind(on_release=lambda *args: (on_confirm(), popup.dismiss()))
    no_button.bind(on_release=popup.dismiss)

    popup.open()
