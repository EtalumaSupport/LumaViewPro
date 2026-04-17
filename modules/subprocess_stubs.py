# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Minimal dummy class definitions for subprocess/worker compatibility.

When lumaviewpro.py is imported as a module (not run as __main__),
Kivy is not available. These stubs provide the class hierarchy that
other modules expect so that imports succeed in subprocess workers.
"""


class App:
    def __init__(self, **kwargs): pass
    def build(self): pass
    def run(self): pass
    def on_start(self): pass
    def on_stop(self): pass


class Widget:
    def __init__(self, **kwargs):
        self.ids = {}
        self.parent = None
    def add_widget(self, widget): pass
    def remove_widget(self, widget): pass


class BoxLayout(Widget): pass
class FloatLayout(Widget): pass
class Scatter(Widget): pass
class Image(Widget): pass
class Button(Widget): pass
class ToggleButton(Widget): pass
class Label(Widget): pass
class RoundedButton(Button): pass
class RoundedToggleButton(ToggleButton): pass
class Slider(Widget): pass
class ScrollView(Widget): pass
class Popup(Widget): pass
class AccordionItem(Widget): pass


# Graphics classes
class RenderContext: pass
class Line: pass
class Color: pass
class Rectangle: pass
class Ellipse: pass
class Texture: pass


# Properties — accept arguments to match Kivy's interface
class StringProperty:
    def __init__(self, default_value="", **kwargs):
        self.default_value = default_value
    def __get__(self, obj, objtype): return self.default_value
    def __set__(self, obj, value): pass


class ObjectProperty:
    def __init__(self, default_value=None, **kwargs):
        self.default_value = default_value
    def __get__(self, obj, objtype): return self.default_value
    def __set__(self, obj, value): pass


class BooleanProperty:
    def __init__(self, default_value=False, **kwargs):
        self.default_value = default_value
    def __get__(self, obj, objtype): return self.default_value
    def __set__(self, obj, value): pass


class ListProperty:
    def __init__(self, default_value=None, **kwargs):
        self.default_value = default_value or []
    def __get__(self, obj, objtype): return self.default_value
    def __set__(self, obj, value): pass


# Other classes
class MotionEvent: pass
class Factory: pass


class Clock:
    @staticmethod
    def schedule_once(func, timeout): pass
    @staticmethod
    def schedule_interval(func, interval): pass
    @staticmethod
    def unschedule(func): pass


def dp(value): return value


# Custom widget dummies
class RangeSlider(Widget): pass
class FigureCanvasKivyAgg(Widget): pass


def show_popup(*args, **kwargs): pass
def show_notification_popup(*args, **kwargs): pass


class image_utils_kivy:
    @staticmethod
    def any_method(*args, **kwargs): pass
