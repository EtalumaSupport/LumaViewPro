# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import threading

from kivy.clock import Clock
from kivy.properties import Property
from kivy.uix.popup import Popup
from kivy.lang import Builder

from lvp_logger import logger


class _PopupProxy:
    """Thread-safe proxy for CustomPopup.

    All property writes are scheduled on the main thread via Clock.schedule_once.
    Callers can use ``proxy.text = ...``, ``proxy.progress = ...``, etc. from any
    thread without violating Kivy's single-thread-UI rule.
    """

    _PROXIED_ATTRS = frozenset({'text', 'progress', 'title', 'auto_dismiss'})

    def __init__(self, popup):
        object.__setattr__(self, '_popup', popup)

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


class _HostWidgetProxy:
    """Thread-safe proxy for the host widget passed to a `@show_popup`-
    decorated method.

    A `@show_popup` method runs on a daemon Thread (see `show_popup`
    below). The decorated body writes attributes on its host widget --
    if any of those attributes are Kivy properties (BooleanProperty,
    StringProperty, NumericProperty, ObjectProperty, ...), Kivy's
    property-write path runs the bound-event dispatch on the writing
    thread. A bg-thread dispatch into a popup.dismiss / widget mutation
    callback can corrupt the Kivy property graph mid-update.

    Solution: intercept attribute writes on the host. If the attribute
    is a Kivy property (class-level Property descriptor), marshal the
    write through `Clock.schedule_once` so the dispatch happens on the
    UI thread. Non-property attributes pass through directly --
    callers are responsible for thread-safety on plain Python state.
    """

    def __init__(self, host):
        object.__setattr__(self, '_host', host)

    def __setattr__(self, name, value):
        host = self._host
        descriptor = getattr(type(host), name, None)
        if isinstance(descriptor, Property):
            Clock.schedule_once(
                lambda dt, h=host, n=name, v=value: setattr(h, n, v), 0
            )
        else:
            setattr(host, name, value)

    def __getattr__(self, name):
        return getattr(self._host, name)


def show_popup(function):
    def wrap(app, *args, **kwargs):
        popup = CustomPopup()  # Instantiate CustomPopup (could add some kwargs if you wish)
        app.done = False  # Reset the app.done BooleanProperty (main thread; no proxy)
        app.bind(done=popup.dismiss)  # When app.done is set to True, then popup.dismiss is fired
        # Progress popups are a separate path from notification_center, so log
        # open/dismiss here -- otherwise a long-running (or hung) operation behind
        # one of these popups leaves no trace in the log.
        logger.info(f'[Popup    ] progress popup opened for {function.__name__}')
        popup.bind(
            on_dismiss=lambda *_: logger.info(
                f'[Popup    ] progress popup dismissed for {function.__name__}'
            )
        )
        popup.open()  # Show popup
        proxy = _PopupProxy(popup)  # Thread-safe proxy for background use
        # Wrap the host so bg-thread Kivy property writes inside the
        # decorated function get marshalled to the UI thread.
        host_proxy = _HostWidgetProxy(app)
        t = threading.Thread(
            target=function, args=[host_proxy, proxy, *args], kwargs=kwargs, daemon=True
        )
        t.start()  # Start thread
        return t

    return wrap


class CustomPopup(Popup):
    def cancel(self):
        """User-initiated escape from a progress popup. The background operation
        runs on a daemon thread and cannot be force-killed, but dismissing the
        popup unblocks the UI -- the way out when an op hangs (e.g. ImageJ init
        with no Java). Any later result the thread posts to this popup is a
        harmless no-op once it is dismissed."""
        logger.info('[Popup    ] progress popup cancelled by user')
        self.dismiss()


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
            size_hint: 1, 0.6

        ProgressBar:
            value: root.progress
            size_hint: 1, 0.2

        Button:
            text: 'Cancel'
            size_hint: 1, 0.2
            on_release: root.cancel()
"""
)
