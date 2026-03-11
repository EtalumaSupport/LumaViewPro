# Copyright Etaluma, Inc.
"""Rounded button widgets with hover highlighting."""

from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton

from ui.hover_behavior import HoverBehavior


class RoundedButton(HoverBehavior, Button):
    """Button with rounded corners and hover highlighting."""
    pass


class RoundedToggleButton(HoverBehavior, ToggleButton):
    """Toggle button with rounded corners and hover highlighting."""
    pass
