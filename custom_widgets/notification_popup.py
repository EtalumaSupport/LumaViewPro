
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


def show_notification_popup(title: str, message: str):
    popup = Popup(
        title=title,
        content=Label(text=message),
        size_hint=(0.6, 0.3),
    )
    popup.open()

def show_confirmation_popup(title: str, message: str, confirm_text: str, cancel_text: str, on_confirm):
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    content.add_widget(Label(text=message))

    button_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
    yes_button = Button(text=confirm_text)
    no_button = Button(text=cancel_text)

    button_layout.add_widget(yes_button)
    button_layout.add_widget(no_button)
    content.add_widget(button_layout)

    popup = Popup(
        title=title,
        content=content,
        size_hint=(0.6, 0.3),
    )

    yes_button.bind(on_release=lambda *args: (on_confirm(), popup.dismiss()))
    no_button.bind(on_release=popup.dismiss)

    popup.open()
