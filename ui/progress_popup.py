# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import threading

from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.lang import Builder


class _PopupProxy:
    """Thread-safe proxy for CustomPopup.

    All property writes are scheduled on the main thread via Clock.schedule_once.
    Callers can use ``proxy.text = ...``, ``proxy.progress = ...``, etc. from any
    thread without violating Kivy's single-thread-UI rule.
    """

    _PROXIED_ATTRS = frozenset({"text", "progress", "title", "auto_dismiss"})

    def __init__(self, popup):
        object.__setattr__(self, "_popup", popup)

    def __setattr__(self, name, value):
        if name in _PopupProxy._PROXIED_ATTRS:
            Clock.schedule_once(lambda dt, n=name, v=value: setattr(self._popup, n, v), 0)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return getattr(self._popup, name)

    def dismiss(self, *args, **kwargs):
        Clock.schedule_once(lambda dt: self._popup.dismiss(*args, **kwargs), 0)

    def open(self, *args, **kwargs):
        Clock.schedule_once(lambda dt: self._popup.open(*args, **kwargs), 0)


def show_popup(function):
    def wrap(app, *args, **kwargs):
        popup = CustomPopup()  # Instantiate CustomPopup (could add some kwargs if you wish)
        app.done = False  # Reset the app.done BooleanProperty
        app.bind(done=popup.dismiss)  # When app.done is set to True, then popup.dismiss is fired
        popup.open()  # Show popup
        proxy = _PopupProxy(popup)  # Thread-safe proxy for background use
        t = threading.Thread(target=function, args=[app, proxy, *args], kwargs=kwargs)  # Create thread
        t.start()  # Start thread
        return t

    return wrap


class CustomPopup(Popup):
    pass


kv = Builder.load_string(
"""
<CustomPopup>:
    size_hint: .6, .3
    auto_dismiss: False
    progress: 0
    text: ''
    title: ''
    
    BoxLayout:
        orientation: 'vertical'
        
        Label:
            text: root.text
            size_hint: 1, 0.8
            
        ProgressBar:
            value: root.progress
"""
)
