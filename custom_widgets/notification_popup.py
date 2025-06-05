
from kivy.uix.label import Label
from kivy.uix.popup import Popup


def show_notification_popup(title: str, message: str):
    popup = Popup(
        title=title,
        content=Label(text=message),
        size_hint=(0.6, 0.3),
    )
    popup.open()

