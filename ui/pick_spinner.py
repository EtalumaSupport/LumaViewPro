# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A spinner that reports a person's pick without taking it as its value."""

from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.spinner import Spinner

from modules import gui_logger


class PickSpinner(Spinner):
    """A spinner whose text is display only; a pick is an event.

    A plain Spinner writes the picked value into ``text``, and ``on_text``
    fires for that write and for every programmatic one alike, so a
    handler on ``on_text`` cannot tell a person's choice from the display
    being refreshed. Here a pick dispatches ``on_pick`` and leaves ``text``
    alone: whatever the pick changes, the display shows it only once the
    owner of that value has accepted it.

    ``log_name`` is the record a pick writes, ``SELECT <log_name> <option>``,
    written here before anything acts on the pick, so the record is the
    person's and no write to ``text`` can make one. ``text_follows_pick``
    shows the picked option at once, for a spinner whose ``on_text`` acts on
    whatever it shows.
    """

    __events__ = ('on_pick',)

    log_name = StringProperty('')
    text_follows_pick = BooleanProperty(False)

    def _on_dropdown_select(self, instance, data, *largs):
        self.is_open = False
        if self.log_name:
            gui_logger.select(self.log_name, data)
        self.dispatch('on_pick', data)
        if self.text_follows_pick:
            self.text = data

    def on_pick(self, value):
        pass
